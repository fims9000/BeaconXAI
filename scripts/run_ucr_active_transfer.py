#!/usr/bin/env python3
"""Frozen Q16 candidate-selection holdout on UCR Wafer and TwoPatterns.

A single existing sparse Q16 search core is shared across three policies.
TRAIN-only measured D1..D4 mask outcomes train a cheap candidate priority;
D3/D4 are *never* observed by any online search before the final mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_donor_robustness import (
    rank_confirmed_donors,exact_two_sided_mcnemar,
)
from beaconxai.contrastive import make_components
from beaconxai.datasets import apply_standardizer,load_npz_dataset
from beaconxai.donor_robust_cf import (
    search_sparse_robust_counterfactual,train_profile_donor_gains,
)
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.transfer_candidate_policy import (
    N_FEATURES,BASE_FEATURES,TransferAwareRanking,
    candidate_features,fit_candidate_transfer_prior,fixed_training_masks,
)
from beaconxai.verified_counterfactual import splice
from scripts.preprocess_ucr_active_transfer import SPECS
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv


SEED=57
TREES=120
BANK_SEED=67
BANK_SIZE=16
MASK_SEED=7714
BINS=10
TAPER=2
SUPPORT=.35
Q=16
N_SELECTED=320
TEST_SEEDS={"Wafer":5744,"TwoPatterns":5755}
METHODS=("static_sparse","learned_static","learned_online")


def stratified_threeway(y,classes,seed=SEED):
    labels=np.asarray(y,dtype=int)
    if labels.shape!=(1000,) or sorted(np.unique(labels).tolist())!=list(range(classes)):
        raise ValueError("expected UCR TRAIN=1000 and complete class labels")
    rng=np.random.default_rng(seed)
    fit,ranking,calibration=[],[],[]
    for lab in range(classes):
        ix=np.flatnonzero(labels==lab)
        if len(ix)<25:
            raise ValueError("a class is too rare for preregistered split")
        perm=rng.permutation(ix)
        nf=int(np.floor(.70*len(ix)))
        nr=int(np.floor(.15*len(ix)))
        fit.extend(map(int,perm[:nf]))
        ranking.extend(map(int,perm[nf:nf+nr]))
        calibration.extend(map(int,perm[nf+nr:]))
    groups=tuple(np.asarray(sorted(v),dtype=int)
                 for v in (fit,ranking,calibration))
    if (sum(len(v) for v in groups)!=1000
        or sorted(np.concatenate(groups).tolist())!=list(range(1000))):
        raise AssertionError("TRAIN three-way split overlaps or drops rows")
    return groups


def pick_donors(x,target,bank,screened):
    indices=rank_confirmed_donors(
        x,bank.donors[target],screened[target],target,
        bank.selected_train_indices[target],
    )
    if len(indices)<4:
        return None
    return tuple(bank.donors[target][j] for j in indices[:4])


def measure_training_candidates(
    x_rows,original,bank,screened,groups,profiles,model,counter,
    *,
    name:str,
):
    """All labels TRAIN-only, joint D1..D4 mask validity, early failing stop."""
    rng=np.random.default_rng(MASK_SEED)
    input_features=[]
    labels=[]
    n_with_four=0
    skipped_support=0
    original_calls=int(counter["examples"])
    measured=0
    for i,x in enumerate(x_rows):
        start=original[i]
        source=int(np.argmax(start))
        target=int(np.argsort(-start,kind="stable")[1])
        four=pick_donors(x,target,bank,screened)
        masks=fixed_training_masks(rng)
        if four is None:
            continue
        n_with_four+=1
        online=four[:2]
        gains=train_profile_donor_gains(
            x,online,groups,profiles,source,target,taper=TAPER
        )
        for mask in masks:
            features,support=candidate_features(
                x,online,groups,gains,start,target,mask,taper=TAPER
            )
            if float(np.max(support))>SUPPORT:
                skipped_support+=1
                continue
            valid=True
            for donor in four:
                patched=splice(x,donor,groups,mask,taper=TAPER)
                fraction=float(np.mean(np.abs(patched-x)>1e-8))
                if fraction>SUPPORT:
                    valid=False
                    break
                scores=np.asarray(model(patched[None]),dtype=float)
                measured+=1
                if scores.shape!=(1,len(start)):
                    raise AssertionError("invalid train-only classifier output")
                if int(np.argmax(scores[0]))!=target:
                    valid=False
                    break
            input_features.append(features)
            labels.append(int(valid))
    used=int(counter["examples"])-original_calls
    if used!=measured:
        raise AssertionError("offline joint-donor labels hide classifier inputs")
    if input_features:
        X=np.stack(input_features)
        y=np.asarray(labels,dtype=int)
    else:
        X=np.empty((0,N_FEATURES),dtype=float)
        y=np.empty((0,),dtype=int)
    report={
        "stage":name,
        "windows":len(x_rows),
        "eligible_four_confirmed_donor_windows":n_with_four,
        "candidate_rows":len(y),
        "successful_four_donor_masks":int(np.sum(y)),
        "filtered_online_support_masks":skipped_support,
        "offline_model_input_evaluations":used,
        "label_is_measured_joint_D1_D2_D3_D4":True,
        "candidate_features_see_D1_D2_only":True,
    }
    print(json.dumps(report),flush=True)
    return X,y,report


def evaluate_one(
    x,original,four,groups,profiles,prior,model,counter,
    *,
    method:str,
):
    if method not in METHODS:
        raise ValueError("unknown frozen Q16 policy")
    target=int(np.argsort(-original,kind="stable")[1])
    source=int(np.argmax(original))
    if four is None:
        return {
            "status":"insufficient_four_confirmed_donors",
            "online_certified":0,"selected":None,
            "online_Q":0,"local_minimality":0,
            "online_max_support":None,"fresh_D3D4_transfer":0,
            "posthoc_D3D4_queries":0,"posthoc_max_support":None,
            "ranker_calls":0,"online_feedback_probes":0,
            "search_wall_s":0.,
        }
    donors=four[:2]
    cheap=train_profile_donor_gains(
        x,donors,groups,profiles,source,target,taper=TAPER
    )
    scorer=None
    if method!="static_sparse":
        scorer=TransferAwareRanking(
            x,donors,groups,cheap,original,target,prior,
            taper=TAPER,adaptive=(method=="learned_online"),
        )
    before=int(counter["examples"])
    start=time.perf_counter()
    result=search_sparse_robust_counterfactual(
        x,original,donors,groups,cheap,model,target_class=target,
        budget=Q,maximum_modified_fraction=SUPPORT,taper=TAPER,
        candidate_priority=scorer,
    )
    elapsed=time.perf_counter()-start
    queries=int(counter["examples"])-before
    if queries!=result.queries or not 0<=queries<=Q:
        raise AssertionError("online search exceeds shared Q16 or hides input calls")
    base={
        "status":result.status,
        "online_certified":int(result.is_certified),
        "selected":None,
        "online_Q":queries,
        "local_minimality":int(result.local_minimality_checked),
        "online_max_support":None,
        "fresh_D3D4_transfer":0,
        "posthoc_D3D4_queries":0,"posthoc_max_support":None,
        "ranker_calls":scorer.call_count if scorer else 0,
        "online_feedback_probes":scorer.max_feedback if scorer else 0,
        "search_wall_s":float(elapsed),
    }
    if not result.is_certified:
        return base
    mask=tuple(map(int,result.certified_edit))
    if not mask:
        raise AssertionError("empty mask cannot switch fixed original class")
    for d in (0,1):
        if not any(
            p.components==mask and p.donor_index==d and p.targeted_flip
            for p in result.probes
        ):
            raise AssertionError("surrogate score was used as model certificate")
    online_support=np.asarray([
        float(np.mean(np.abs(splice(x,d,groups,mask,taper=TAPER)-x)>1e-8))
        for d in donors
    ])
    if np.max(online_support)>SUPPORT+1e-12:
        raise AssertionError("candidate violates online support cap")
    audit=np.stack([
        splice(x,d,groups,mask,taper=TAPER) for d in four[2:4]
    ])
    aud_support=np.mean(np.abs(audit-x[None])>1e-8,axis=(1,2))
    before_audit=int(counter["examples"])
    predictions=np.asarray(model(audit),dtype=float)
    posthoc=int(counter["examples"])-before_audit
    if posthoc!=2 or predictions.shape!=(2,len(original)):
        raise AssertionError("D3/D4 posthoc metering incorrect")
    transfer=bool(np.all(np.argmax(predictions,axis=1)==target)
                  and np.all(aud_support<=SUPPORT+1e-12))
    base.update({
        "selected":mask,
        "online_max_support":float(np.max(online_support)),
        "fresh_D3D4_transfer":int(transfer),
        "posthoc_D3D4_queries":posthoc,
        "posthoc_max_support":float(np.max(aud_support)),
    })
    return base


def paired_primary(rows):
    index={(int(r["sample"]),r["method"]):r for r in rows}
    if len(index)!=N_SELECTED*len(METHODS):
        raise AssertionError("all 960 TEST method/sample results required")
    both=learned_only=static_only=neither=0
    for i in range(N_SELECTED):
        a=int(index[i,"learned_online"]["fresh_D3D4_transfer"])
        b=int(index[i,"static_sparse"]["fresh_D3D4_transfer"])
        if a and b:
            both+=1
        elif a:
            learned_only+=1
        elif b:
            static_only+=1
        else:
            neither+=1
    assert both+learned_only+static_only+neither==N_SELECTED
    return {
        "both":both,"learned_only":learned_only,
        "static_only":static_only,"neither":neither,
        "learned_fresh":both+learned_only,
        "static_fresh":both+static_only,
        "delta_pp":100*(learned_only-static_only)/N_SELECTED,
        "exact_two_sided_mcnemar":exact_two_sided_mcnemar(
            learned_only,static_only,
        ),
        "n_test":N_SELECTED,
        "prespecified_alpha":.05,
    }


def summarize(rows):
    out=[]
    for method in METHODS:
        sample=[r for r in rows if r["method"]==method]
        if len(sample)!=N_SELECTED:
            raise AssertionError("per-method TEST size must equal 320")
        online=[r for r in sample if r["online_certified"]]
        transferred=[r for r in sample if r["fresh_D3D4_transfer"]]
        out.append({
            "method":method,"N":N_SELECTED,
            "online_certificates":len(online),
            "fresh_D3D4_transferred":len(transferred),
            "unconditional_transfer":len(transferred)/N_SELECTED,
            "conditional_transfer":(
                len(transferred)/len(online) if online else None
            ),
            "online_abstain":N_SELECTED-len(online),
            "local_1deletion_verified":sum(r["local_minimality"] for r in sample),
            "mean_online_Q":float(np.mean([r["online_Q"] for r in sample])),
            "max_online_Q":max(r["online_Q"] for r in sample),
            "mean_search_wall_s":float(np.mean([
                r["search_wall_s"] for r in sample
            ])),
            "total_external_audit_inputs":sum(
                r["posthoc_D3D4_queries"] for r in sample
            ),
            "max_online_feedback":max(r["online_feedback_probes"] for r in sample),
        })
    return out


def run(args):
    if args.dataset not in SPECS:
        raise ValueError("unregistered benchmark dataset")
    if (args.seed,args.trees,args.donors)!=(SEED,TREES,BANK_SIZE):
        raise ValueError("frozen parameters: seed57 trees120 donors16")
    ntr,nte,T,classes=SPECS[args.dataset]
    path=Path(args.npz_path)
    raw,ytrain,raw_test,ytest=load_npz_dataset(path)
    with np.load(path) as z:
        label_map=z["labels"].tolist()
        protocol=str(z["dataset_protocol"])
    if (raw.shape!=(ntr,T,1) or raw_test.shape!=(nte,T,1)
        or len(np.unique(ytrain))!=classes
        or len(np.unique(ytest))!=classes
        or f"UCR {args.dataset} official aeon 1.5.0" not in protocol):
        raise ValueError("requires pinned official UCR data; no substitution allowed")
    fit_ix,risk_ix,gate_ix=stratified_threeway(ytrain,classes)
    mu,sd,profiles,model,counter=fit_predictor(
        raw[fit_ix],ytrain[fit_ix],"extratrees",SEED,TREES
    )
    train=apply_standardizer(raw[fit_ix],mu,sd)
    risk_data=apply_standardizer(raw[risk_ix],mu,sd)
    gate_data=apply_standardizer(raw[gate_ix],mu,sd)
    test_data=apply_standardizer(raw_test,mu,sd)
    groups=make_components((T,1),BINS,"all_channels")
    bank=fit_real_donor_bank(
        train,ytrain[fit_ix],groups,max_donors=BANK_SIZE,seed=BANK_SEED
    )
    before=int(counter["examples"])
    screened={}
    for lab in sorted(bank.donors):
        screened[lab]=np.argmax(model(bank.donors[lab]),axis=1)
    screening_calls=int(counter["examples"])-before

    before=int(counter["examples"])
    risk_base=model(risk_data)
    gate_base=model(gate_data)
    train_original_calls=int(counter["examples"])-before
    if train_original_calls!=len(risk_data)+len(gate_data):
        raise AssertionError("TRAIN-only original model baseline ledger failed")
    risk_features,risk_labels,risk_stat=measure_training_candidates(
        risk_data,risk_base,bank,screened,groups,profiles,model,counter,
        name="ranking_fit",
    )
    prior=fit_candidate_transfer_prior(risk_features,risk_labels)
    gate_features,gate_labels,gate_stat=measure_training_candidates(
        gate_data,gate_base,bank,screened,groups,profiles,model,counter,
        name="ranking_calibration",
    )
    predicted=prior.predict(gate_features)
    calib_auc=(float(roc_auc_score(gate_labels,predicted))
               if len(np.unique(gate_labels))==2 else None)
    calib_ap=(float(average_precision_score(gate_labels,predicted))
              if len(np.unique(gate_labels))==2 else None)
    # All learning and decision design is now frozen; only now read TEST
    # baseline scores. TEST labels are used exclusively in final accuracy.
    before=int(counter["examples"])
    test_original=model(test_data)
    test_baseline_calls=int(counter["examples"])-before
    if test_baseline_calls!=nte:
        raise AssertionError("must pay once per full original TEST observation")
    chosen=np.sort(
        np.random.default_rng(TEST_SEEDS[args.dataset]).permutation(nte)[:N_SELECTED]
    )
    rows=[]
    online_calls=0
    external_calls=0
    missing=0
    for i,j in enumerate(chosen):
        x=test_data[int(j)]
        base=test_original[int(j)]
        source=int(np.argmax(base))
        target=int(np.argsort(-base,kind="stable")[1])
        four=pick_donors(x,target,bank,screened)
        missing+=int(four is None)
        for method in METHODS:
            result=evaluate_one(
                x,base,four,groups,profiles,prior,model,counter,method=method,
            )
            online_calls+=result["online_Q"]
            external_calls+=result["posthoc_D3D4_queries"]
            rows.append({
                "sample":i,"test_index":int(j),
                "method":method,"source_pred":source,"target_pred":target,
                "four_confirmed_train_donors":int(four is not None),
                **result,
            })
    complete=(
        screening_calls+train_original_calls+test_baseline_calls
        +risk_stat["offline_model_input_evaluations"]
        +gate_stat["offline_model_input_evaluations"]
        +online_calls+external_calls
    )
    if int(counter["examples"])!=complete:
        raise AssertionError("all classifier calls must reconcile exactly")
    primary=paired_primary(rows)
    methods=summarize(rows)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",[
        {**r,"selected":(",".join(map(str,r["selected"]))
                          if r["selected"] is not None else "")}
        for r in rows
    ])
    write_csv(out/"summary.csv",methods)
    report={
        "protocol":"artifacts/beacon_wafer_twopatterns_active_transfer_preregistered_2026-11-22.md",
        "dataset":args.dataset,
        "official_TRAIN":ntr,"official_TEST":nte,
        "test_subset_N":N_SELECTED,
        "test_subset_seed":TEST_SEEDS[args.dataset],
        "test_subset_indices":chosen.tolist(),
        "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_sha256":counter["fitted_model_sha256"],
        "scaler_sha256":counter["training_standardizer_sha256"],
        "model_fit_indices":fit_ix.tolist(),
        "rank_fit_indices":risk_ix.tolist(),
        "rank_calibration_indices":gate_ix.tolist(),
        "model_test_accuracy_full":float(np.mean(
            np.argmax(test_original,axis=1)==ytest
        )),
        "model_adequacy_gate":.7,
        "test_ground_truth_used_only_for_accuracy":True,
        "device_or_subject_disjoint_verified":False,
        "class_map":label_map,
        "frozen_model":"ExtraTrees trees120 seed57",
        "shared_Q":Q,
        "K2_model_confirmed_TRAIN_donors":True,
        "D3D4_unseen_during_all_TEST_search":True,
        "edited_fraction_max_each_donor":SUPPORT,
        "time_bins":BINS,"taper":TAPER,
        "frozen_fixed_training_masks_seed":MASK_SEED,
        "frozen_logistic":"C1 balanced liblinear seed57",
        "online_cheap_feedback_formula":"logit(p)+2/N_probes*sum(sign_flip*exp(-Hamming/1.5))",
        "rank_prior_fit_rows":prior.n_fit,
        "rank_prior_fit_joint_successes":prior.joint_successes,
        "rank_prior_constant":prior.constant,
        "rank_prior_coefficients":(
            prior.model.coef_[0].tolist()
            if prior.model is not None else None
        ),
        "rank_prior_scaler_mean":(
            prior.scaler.mean_.tolist() if prior.scaler is not None else None
        ),
        "rank_prior_scaler_scale":(
            prior.scaler.scale_.tolist() if prior.scaler is not None else None
        ),
        "feature_names":BASE_FEATURES,
        "riskfit_training":risk_stat,
        "separate_offline_calibration":gate_stat,
        "calibration_AUROC":calib_auc,
        "calibration_average_precision":calib_ap,
        "original_model_TEST_inputs":test_baseline_calls,
        "TRAIN_offline_donor_screening_inputs":screening_calls,
        "TRAIN_original_predictions":train_original_calls,
        "TOTAL_model_inputs_all_stages":complete,
        "online_Q16_evaluations_all_methods":online_calls,
        "external_posthoc_D3D4_evaluations":external_calls,
        "TEST_windows_without_four_eligible_donors":missing,
        "primary" if args.dataset=="Wafer" else "replication":primary,
        "summary":methods,
        "limitations":[
            "TwoPatterns synthetic, not real physical counterfactual",
            "UCR Wafer device/session overlap not verified",
            "Only D1/D2 online, D3/D4 are separately priced posthoc",
            "Surrogate training uses genuine offline model queries",
            "Score is not a certificate or causal/physical explanation",
            "The fixed native sparse core remains responsible for actual verification",
        ],
    }
    (out/"manifest.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8",
    )
    print(json.dumps({
        "dataset":args.dataset,
        "accuracy":report["model_test_accuracy_full"],
        "rank_fit_positives":prior.joint_successes,
        "offline_calibration_auc":calib_auc,
        "offline_calibration_ap":calib_ap,
        "contrast":primary,
        "summary":methods,
        "online_calls":online_calls,
        "external_calls":external_calls,
        "all_inputs":complete,
    },indent=2,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",required=True,choices=sorted(SPECS))
    parser.add_argument("--npz-path",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--trees",type=int,default=TREES)
    parser.add_argument("--donors",type=int,default=BANK_SIZE)
    run(parser.parse_args())


if __name__=="__main__":
    main()
