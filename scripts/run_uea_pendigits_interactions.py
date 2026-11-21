#!/usr/bin/env python3
"""TRAIN-calibrated pair interactions and adaptive Q8/Q16 on untouched UEA PenDigits.

See artifacts/beacon_pendigits_pairwise_adaptive_preregistered_2026-11-22.md.
No TEST labels, masks, outcomes or donors are used when fitting the prior.
Only genuine model evaluations count toward each method's shared online Q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_donor_robustness import (
    rank_confirmed_donors, exact_two_sided_mcnemar,
)
from beaconxai.contrastive import make_components
from beaconxai.datasets import apply_standardizer, load_npz_dataset
from beaconxai.donor_robust_cf import (
    search_sparse_robust_counterfactual,train_profile_donor_gains,
)
from beaconxai.learned_interactions import (
    fit_interaction_prior,search_learned_adaptive,
)
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.verified_counterfactual import splice,target_margin
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv


SEED=57
TREES=120
BANK_SEED=67
BANK_SIZE=16
TIME_BINS=8
TAPER=0
SUPPORT=.35
CAL_MASK_SEED=571
ALPHA=10.
BUDGETS=(8,16)
METHODS=(
    "static_sparse",
    "pairs_static",
    "pairs_adaptive",
    "additive_adaptive",
)


def stratified_official_train_split(y,seed=SEED):
    labels=np.asarray(y,dtype=int)
    if labels.shape!=(7494,) or sorted(np.unique(labels).tolist())!=list(range(10)):
        raise ValueError("Expected 7494 TRAIN rows and 10 digit classes")
    rng=np.random.default_rng(seed)
    fit,calibration=[],[]
    for label in range(10):
        eligible=np.flatnonzero(labels==label)
        if len(eligible)<=10:
            raise ValueError("Insufficient digit records to reserve ten per class")
        perm=rng.permutation(eligible)
        calibration.extend(map(int,perm[:10]))
        fit.extend(map(int,perm[10:]))
    fit=np.asarray(sorted(fit),dtype=int)
    calibration=np.asarray(sorted(calibration),dtype=int)
    if len(fit)!=7394 or len(calibration)!=100 or len(set(fit)&set(calibration)):
        raise AssertionError("TRAIN/CALIB split overlap")
    return fit,calibration


def calibration_masks(rng,n_components=TIME_BINS):
    if n_components!=8:
        raise ValueError("preregistered eight-component digit geometry")
    singles=[(j,) for j in range(n_components)]
    pairs=list(combinations(range(n_components),2))
    ix=rng.choice(len(pairs),size=20,replace=False)
    return singles+[pairs[int(i)] for i in ix]


def fit_prior_from_train_only(
    train_raw, ytrain, *,
    seed=SEED,trees=TREES,
):
    fit_ids,cal_ids=stratified_official_train_split(ytrain,seed=seed)
    raw_fit=np.asarray(train_raw)[fit_ids]
    yfit=np.asarray(ytrain)[fit_ids]
    raw_cal=np.asarray(train_raw)[cal_ids]
    mu,sd,profiles,model,counter=fit_predictor(
        raw_fit,yfit,"raw_mlp",seed,trees,
    )
    fit=apply_standardizer(raw_fit,mu,sd)
    cal=apply_standardizer(raw_cal,mu,sd)
    groups=make_components((8,2),TIME_BINS,"all_channels")
    bank=fit_real_donor_bank(
        fit,yfit,groups,max_donors=BANK_SIZE,seed=BANK_SEED,
    )
    screened={}
    for label in sorted(bank.donors):
        screened[label]=np.argmax(model(bank.donors[label]),axis=1)
    donor_screen_calls=int(counter["examples"])
    base=model(cal)
    cal_base_calls=len(cal)
    rows=[]
    attempted_masks=0
    missing_cal_donors=0
    rng=np.random.default_rng(CAL_MASK_SEED)
    before=int(counter["examples"])
    for i,x in enumerate(cal):
        original=base[i]
        source=int(np.argmax(original))
        target=int(np.argsort(-original,kind="stable")[1])
        eligible=rank_confirmed_donors(
            x,bank.donors[target],screened[target],target,
            bank.selected_train_indices[target],
        )
        if len(eligible)<2:
            missing_cal_donors+=1
            continue
        donors=[bank.donors[target][j] for j in eligible[:2]]
        gains=train_profile_donor_gains(
            x,donors,groups,profiles,source,target,taper=TAPER,
        )
        masks=calibration_masks(rng)
        baseline_margin=target_margin(original,target)
        for didx,donor in enumerate(donors):
            proposed=np.stack([
                splice(x,donor,groups,mask,taper=TAPER)
                for mask in masks
            ])
            measured=model(proposed)
            if measured.shape!=(len(masks),len(original)):
                raise AssertionError("calibration classifier geometry incorrect")
            attempted_masks+=len(masks)
            for j,mask in enumerate(masks):
                delta=target_margin(measured[j],target)-baseline_margin
                rows.append((gains[didx].copy(),mask,float(delta)))
    calibration_mask_calls=int(counter["examples"])-before
    if calibration_mask_calls!=attempted_masks:
        raise AssertionError("unaccounted offline calibration queries")
    pair=fit_interaction_prior(
        rows,n_components=TIME_BINS,pairwise=True,alpha=ALPHA,
    )
    additive=fit_interaction_prior(
        rows,n_components=TIME_BINS,pairwise=False,alpha=ALPHA,
    )
    costs={
        "fit_train_count":len(fit),"calibration_count":len(cal),
        "fit_train_original_indices":fit_ids.tolist(),
        "calibration_original_indices":cal_ids.tolist(),
        "model_screening_calls":donor_screen_calls,
        "calibration_baseline_calls":cal_base_calls,
        "calibration_mask_calls":calibration_mask_calls,
        "calibration_masks_per_donor":28,
        "missing_calibration_two_donor_windows":missing_cal_donors,
        "ridge_rows":len(rows),
        "pair_ridge_training_mae_in_sample":pair.calibration_mae,
        "additive_ridge_training_mae_in_sample":additive.calibration_mae,
        "pair_coefficient_norm":pair.pair_coefficient_norm,
        "pair_ridge_alpha":ALPHA,
        "learned_coefficients_are_not_causes":True,
    }
    if int(counter["examples"])!=(
        donor_screen_calls+cal_base_calls+calibration_mask_calls
    ):
        raise AssertionError("offline counter ledger mismatch")
    return (mu,sd,profiles,model,counter,bank,screened,groups,
            pair,additive,costs)


def single_result(method,q,x,original,donors,groups,profiles,priors,
                  model,counter):
    source=int(np.argmax(original))
    target=int(np.argsort(-original,kind="stable")[1])
    gains=train_profile_donor_gains(
        x,donors,groups,profiles,source,target,taper=TAPER,
    )
    before=int(counter["examples"])
    t0=time.perf_counter()
    if method=="static_sparse":
        outcome=search_sparse_robust_counterfactual(
            x,original,donors,groups,gains,model,target_class=target,
            budget=q,maximum_modified_fraction=SUPPORT,taper=TAPER,
        )
        cert=outcome
        updates=0
        rerankings=0
        candidates=None
    else:
        prior=priors[0] if method.startswith("pairs_") else priors[1]
        outcome=search_learned_adaptive(
            x,original,donors,groups,gains,prior,model,
            target_class=target,budget=q,maximum_modified_fraction=SUPPORT,
            taper=TAPER,adaptive=(method!="pairs_static"),
        )
        cert=outcome.certificate
        updates=outcome.posterior_updates
        rerankings=outcome.online_rerankings
        candidates=outcome.candidate_masks
    seconds=time.perf_counter()-t0
    used=int(counter["examples"])-before
    if used!=cert.queries or not 0<=used<=q:
        raise AssertionError("Q is number of actual model input examples")
    if cert.is_certified:
        mask=tuple(map(int,cert.certified_edit))
        fractions=[
            float(np.mean(np.abs(
                splice(x,d,groups,mask,taper=TAPER)-x
            )>1e-8))
            for d in donors
        ]
        if max(fractions)>SUPPORT+1e-12:
            raise AssertionError("algorithm returned noncompact mask")
        is_compact=1
    else:
        mask=None
        fractions=[None,None]
        is_compact=0
    return {
        "status":cert.status,"certified":int(cert.is_certified),
        "strict_joint":is_compact,
        "selected":mask,
        "model_queries":used,
        "fraction_d1":fractions[0],"fraction_d2":fractions[1],
        "both_le20":int(is_compact and max(fractions)<=.20),
        "both_le35":int(is_compact),
        "both_le50":int(is_compact),
        "minimal_on_both":int(cert.local_minimality_checked),
        "posterior_updates":updates,
        "online_rerankings":rerankings,
        "feasible_candidate_masks":candidates,
        "search_wall_seconds":float(seconds),
        "observed_any_single_donor_flip":int(cert.ever_found_single_donor_flip),
    }


def paired_primary(rows):
    lookup={(r["sample"],r["method"],r["Q"]):r for r in rows}
    table={"both":0,"adaptive_only":0,"static_only":0,"neither":0}
    for i in range(320):
        a=int(lookup[(i,"additive_adaptive",16)]["strict_joint"])
        b=int(lookup[(i,"static_sparse",16)]["strict_joint"])
        key="both" if a and b else "adaptive_only" if a else "static_only" if b else "neither"
        table[key]+=1
    assert sum(table.values())==320
    table["n_test"]=320
    table["adaptive_joint"]=table["both"]+table["adaptive_only"]
    table["static_joint"]=table["both"]+table["static_only"]
    table["effect_pp"]=100*(
        table["adaptive_only"]-table["static_only"]
    )/320
    table["exact_mcnemar_p"]=exact_two_sided_mcnemar(
        table["adaptive_only"],table["static_only"],
    )
    table["alpha"]=.05
    return table


def summarize(rows):
    table=[]
    for q in BUDGETS:
        for method in METHODS:
            subset=[r for r in rows if r["Q"]==q and r["method"]==method]
            if len(subset)!=320 or len({r["sample"] for r in subset})!=320:
                raise AssertionError("missing official test rows")
            accepted=[r for r in subset if r["strict_joint"]]
            table.append({
                "method":method,"Q":q,"n":len(subset),
                "joint_compact":len(accepted),
                "joint_le20":sum(r["both_le20"] for r in accepted),
                "minimal_on_both":sum(r["minimal_on_both"] for r in accepted),
                "mean_Q":float(np.mean([r["model_queries"] for r in subset])),
                "mean_cpu_wall_s":float(np.mean(
                    [r["search_wall_seconds"] for r in subset])),
                "mean_local_posterior_updates":float(np.mean(
                    [r["posterior_updates"] for r in subset])),
                "accepted_median_support":(
                    float(np.median([max(r["fraction_d1"],r["fraction_d2"])
                                     for r in accepted]))
                    if accepted else None
                ),
                "abstentions":320-len(accepted),
                "eligible_four_donor_extracertificates":sum(
                    r.get("audit_extra_two_valid") is not None for r in accepted),
                "extra_two_donor_transfer":sum(
                    r.get("audit_extra_two_valid")==1 for r in accepted),
            })
    return table


def run(args):
    if (args.seed,args.trees,args.donors)!=(SEED,TREES,BANK_SIZE):
        raise ValueError("frozen preregistration: seed57 trees120 donors16")
    data=Path(args.npz_path)
    raw_train,ytrain,raw_test,ytest=load_npz_dataset(data)
    with np.load(data) as z:
        labels=z["labels"].tolist()
        protocol=str(z["dataset_protocol"])
    if (raw_train.shape!=(7494,8,2) or raw_test.shape!=(3498,8,2)
        or len(set(ytrain))!=10 or len(set(ytest))!=10
        or "UEA PenDigits official TRAIN 7494 TEST 3498"
           not in protocol):
        raise ValueError("requires official UEA PenDigits 7494/3498 split")
    (mu,sd,profiles,model,counter,bank,screened,groups,
     pair,additive,offline)=fit_prior_from_train_only(
        raw_train,ytrain,seed=args.seed,trees=args.trees,
    )
    test=apply_standardizer(raw_test,mu,sd)
    original=model(test)
    test_base_calls=len(test)
    if int(counter["examples"])!=(
        offline["model_screening_calls"]+
        offline["calibration_baseline_calls"]+
        offline["calibration_mask_calls"]+
        test_base_calls
    ):
        raise AssertionError("missing model evaluations before TEST online search")
    selected_test_indices=np.sort(
        np.random.default_rng(5712).permutation(len(test))[:320]
    )
    selected_test=test[selected_test_indices]
    selected_original=original[selected_test_indices]
    records=[]
    online=0
    third_fourth_audits=0
    missing_donors=0
    priors=(pair,additive)
    for i,x in enumerate(selected_test):
        src=int(np.argmax(selected_original[i]))
        tgt=int(np.argsort(-selected_original[i],kind="stable")[1])
        eligible=rank_confirmed_donors(
            x,bank.donors[tgt],screened[tgt],tgt,
            bank.selected_train_indices[tgt],
        )
        missing_donors+=int(len(eligible)<2)
        chosen=eligible[:2]
        donors=tuple(bank.donors[tgt][j] for j in chosen)
        for q in BUDGETS:
            for method in METHODS:
                if len(donors)<2:
                    rec={
                        "status":"missing_two_model_confirmed_train_donors",
                        "certified":0,"strict_joint":0,"selected":None,
                        "model_queries":0,"fraction_d1":None,
                        "fraction_d2":None,"both_le20":0,
                        "both_le35":0,"both_le50":0,
                        "minimal_on_both":0,"posterior_updates":0,
                        "online_rerankings":0,"feasible_candidate_masks":0,
                        "search_wall_seconds":0.,
                        "observed_any_single_donor_flip":0,
                    }
                else:
                    rec=single_result(
                        method,q,x,selected_original[i],donors,groups,
                        profiles,priors,model,counter,
                    )
                online+=rec["model_queries"]
                # Secondary audit. New D3 and D4 never influence the policy,
                # and their model calls are NOT counted as online Q.
                extra_valid=None
                if q==16 and rec["strict_joint"] and len(eligible)>=4:
                    others=[bank.donors[tgt][j] for j in eligible[2:4]]
                    edits=np.stack([
                        splice(x,d,groups,rec["selected"],taper=TAPER)
                        for d in others
                    ])
                    extra_output=model(edits)
                    third_fourth_audits+=len(edits)
                    extra_valid=int(all(
                        int(np.argmax(extra_output[k]))==tgt
                        and np.mean(np.abs(edits[k]-x)>1e-8)<=SUPPORT
                        for k in range(len(others))
                    ))
                records.append({
                    "sample":i,"test_index":int(selected_test_indices[i]),
                    "Q":q,"method":method,
                    "source_pred":src,"target_pred":tgt,
                    "n_qualified_train_donors":len(eligible),
                    "train_donor_1":(
                        int(bank.selected_train_indices[tgt][chosen[0]])
                        if len(chosen)==2 else None),
                    "train_donor_2":(
                        int(bank.selected_train_indices[tgt][chosen[1]])
                        if len(chosen)==2 else None),
                    "audit_extra_two_valid":extra_valid,
                    **rec,
                })
    ledger_expected=(
        offline["model_screening_calls"]+
        offline["calibration_baseline_calls"]+
        offline["calibration_mask_calls"]+
        test_base_calls+online+third_fourth_audits
    )
    if int(counter["examples"])!=ledger_expected:
        raise AssertionError("offline, cached, search and audit model calls do not sum")
    primary=paired_primary(records)
    methods=summarize(records)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",[
        {**r,"selected":",".join(map(str,r["selected"])) if
         r["selected"] is not None else ""} for r in records
    ])
    write_csv(out/"summary.csv",methods)
    registry={
        "protocol":"artifacts/beacon_pendigits_pairwise_adaptive_preregistered_2026-11-22.md",
        "dataset":"UEA PenDigits official aeon 1.5.0",
        "primary_comparison":"additive_adaptive vs static_sparse at Q16",
        "data_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),
        "model_sha256":counter["fitted_model_sha256"],
        "scaler_sha256":counter["training_standardizer_sha256"],
        "test_records":320,"full_archive_test_records":3498,
        "sampled_test_indices":selected_test_indices.tolist(),
        "train_fit_records":7394,"calibration_records":100,
        "train_test_independent_participants_verified":False,
        "classes":labels,"model_type":"raw_mlp",
        "model_test_accuracy":float(np.mean(
            np.argmax(original,axis=1)==ytest
        )),
        "classifier_adequacy_threshold":.7,
        "test_labels_used_only_for_accuracy":True,
        "Q_values":BUDGETS,"donor_count_online":2,
        "max_modified_fraction":SUPPORT,
        "temporal_components":TIME_BINS,"taper":TAPER,
        "frozen_offline_training":offline,
        "train_surrogate_updates_from_TEST":False,
        "online_adaptive_updates_are_local_per_TEST_instance":True,
        "online_evaluations":online,
        "offline_test_donor_3_4_evaluations":third_fourth_audits,
        "cached_TEST_baseline_evaluations":test_base_calls,
        "complete_model_input_evaluations":counter["examples"],
        "missing_two_confirmed_donors":missing_donors,
        "primary":primary,"summary":methods,
        "limits":[
            "Conditional two-donor counterfactual, not causal attribution",
            "Online learned policy uses offline calibration model queries",
            "MLP trained on 7394 TRAIN digit windows; 100 different TRAIN windows calibrate surrogate",
            "No verified user/session split; no physical-validity guarantee for splice",
            "D3/D4 audit is posthoc, never part of any Q8/Q16 online budget",
            "Single fixed raw MLP architecture/seed, generalization unproven",
        ],
    }
    (out/"manifest.json").write_text(
        json.dumps(registry,indent=2,ensure_ascii=False),
        encoding="utf8",
    )
    print(json.dumps({
        "test_accuracy":registry["model_test_accuracy"],
        "adequate":registry["model_test_accuracy"]>=.7,
        "primary":primary,"summary":methods,
        "offline":offline,
        "online_calls":online,
        "posthoc_transfer_calls":third_fourth_audits,
        "missing_donors":missing_donors,
    },ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--seed",default=SEED,type=int)
    p.add_argument("--trees",default=TREES,type=int)
    p.add_argument("--donors",default=BANK_SIZE,type=int)
    run(p.parse_args())


if __name__=="__main__":
    main()
