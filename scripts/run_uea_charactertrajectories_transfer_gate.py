#!/usr/bin/env python3
"""Preregistered UEA CharacterTrajectories donor-transfer selective risk benchmark.

Uses existing two-donor compact Q16 search without duplicating the search core.
Risk-training / threshold-calibration / TEST are disjoint official TRAIN/TEST
window sets. All external donor evaluations are separately counted offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_donor_robustness import rank_confirmed_donors
from beaconxai.contrastive import make_components
from beaconxai.datasets import apply_standardizer,load_npz_dataset
from beaconxai.donor_robust_cf import (
    search_sparse_robust_counterfactual,train_profile_donor_gains,
)
from beaconxai.donor_transfer_risk import (
    FEATURE_NAMES,fit_transfer_risk,donor_transfer_features,
    calibrate_transfer_gate,
)
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.verified_counterfactual import splice,target_margin
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv

SEED=57
TREES=120
BANK_SIZE=16
BANK_SEED=67
Q=16
BINS=10
TAPER=2
MAX_EDIT=.35
N_TEST=320
SELECT_SEED=5733


def stratified_threeway(y):
    labels=np.asarray(y,dtype=int)
    if (labels.shape!=(1422,)
        or sorted(np.unique(labels).tolist())!=list(range(20))):
        raise ValueError("Expected 1422 official TRAIN rows and twenty classes")
    rng=np.random.default_rng(SEED)
    fit,risk,gate=[],[],[]
    for cls in range(20):
        ids=np.flatnonzero(labels==cls)
        if len(ids)<25:
            raise ValueError("too few examples for stratified calibration")
        perm=rng.permutation(ids)
        nf=int(np.floor(.70*len(ids)))
        nr=int(np.floor(.15*len(ids)))
        fit.extend(map(int,perm[:nf]))
        risk.extend(map(int,perm[nf:nf+nr]))
        gate.extend(map(int,perm[nf+nr:]))
    arr=tuple(np.asarray(sorted(a),dtype=int) for a in (fit,risk,gate))
    if (sum(len(a) for a in arr)!=1422
        or len(set(arr[0])&set(arr[1]))
        or len(set(arr[0])&set(arr[2]))
        or len(set(arr[1])&set(arr[2]))):
        raise AssertionError("TRAIN calibration leakage")
    return arr


def empty_record(sample,original_index,source,target,stage,reason):
    return {
        "stage":stage,"sample":sample,"original_index":int(original_index),
        "source":source,"target":target,"status":reason,
        "online_certified":0,"selected":None,"features":None,
        "online_queries":0,"posthoc_queries":0,
        "failure_D3_D4":None,"transfer_D3_D4":0,
        "fraction_max":None,"minimal_on_K2":0,
        "model_search_wall_s":0.,
    }


def explain_one(x,baseline,bank,screened,groups,profiles,model,counter,
                *,sample,original_index,stage):
    source=int(np.argmax(baseline))
    target=int(np.argsort(-baseline,kind="stable")[1])
    ranked=rank_confirmed_donors(
        x,bank.donors[target],screened[target],target,
        bank.selected_train_indices[target],
    )
    if len(ranked)<4:
        return empty_record(sample,original_index,source,target,stage,
                            "insufficient_four_confirmed_train_donors")
    selected=tuple(bank.donors[target][k] for k in ranked[:2])
    unseen=tuple(bank.donors[target][k] for k in ranked[2:4])
    prior=train_profile_donor_gains(
        x,selected,groups,profiles,source,target,taper=TAPER,
    )
    previous=int(counter["examples"])
    t0=time.perf_counter()
    result=search_sparse_robust_counterfactual(
        x,baseline,selected,groups,prior,model,
        target_class=target,budget=Q,
        maximum_modified_fraction=MAX_EDIT,taper=TAPER,
    )
    seconds=time.perf_counter()-t0
    used=int(counter["examples"])-previous
    if used!=result.queries or not 0<=used<=Q:
        raise AssertionError("shared K2 Q16 model-input ledger mismatch")
    if not result.is_certified:
        out=empty_record(sample,original_index,source,target,stage,result.status)
        out["online_queries"]=used
        out["model_search_wall_s"]=seconds
        return out
    mask=tuple(map(int,result.certified_edit))
    margins=[]
    for didx in (0,1):
        matching=[
            p for p in result.probes if p.components==mask
            and p.donor_index==didx and p.targeted_flip
        ]
        if not matching:
            raise AssertionError("selected mask lacks measured evidence from a donor")
        margins.append(float(matching[-1].target_margin))
    fractions=[
        float(np.mean(np.abs(
            splice(x,d,groups,mask,taper=TAPER)-x)>1e-8))
        for d in selected
    ]
    if max(fractions)>MAX_EDIT+1e-12:
        raise AssertionError("noncompact output was certified")
    features=donor_transfer_features(
        x,selected,mask,tuple(fractions),tuple(margins),
        target_margin(baseline,target),components_count=len(groups),
        online_queries=used,budget=Q,
        minimality_checked=result.local_minimality_checked,
    )
    # IMPORTANT: unseen donors are NEVER passed to policy, prior or features.
    # D3/D4 intervention scores become TRAIN risk labels or posthoc TEST audit.
    unseen_edits=np.stack([
        splice(x,d,groups,mask,taper=TAPER) for d in unseen
    ])
    audit_previous=int(counter["examples"])
    audit=np.asarray(model(unseen_edits),dtype=float)
    audit_count=int(counter["examples"])-audit_previous
    if audit_count!=2 or audit.shape!=(2,len(baseline)):
        raise AssertionError("external D3/D4 evaluations must cost exactly two")
    accepted=bool(all(
        int(np.argmax(audit[k]))==target and
        float(np.mean(np.abs(unseen_edits[k]-x)>1e-8))<=MAX_EDIT
        for k in range(2)
    ))
    return {
        "stage":stage,"sample":sample,"original_index":int(original_index),
        "source":source,"target":target,"status":result.status,
        "online_certified":1,"selected":mask,
        "features":features,
        "online_queries":used,"posthoc_queries":audit_count,
        "failure_D3_D4":int(not accepted),
        "transfer_D3_D4":int(accepted),
        "fraction_max":float(max(fractions)),
        "minimal_on_K2":int(result.local_minimality_checked),
        "model_search_wall_s":seconds,
    }


def evaluate_stage(name,data,indices,baseline,bank,screened,
                   groups,profiles,model,counter):
    records=[]
    for i,x in enumerate(data):
        rec=explain_one(
            x,baseline[i],bank,screened,groups,profiles,model,counter,
            sample=i,original_index=int(indices[i]),stage=name,
        )
        records.append(rec)
    print(json.dumps({
        "stage":name,"n":len(records),
        "certified":sum(r["online_certified"] for r in records),
        "transferred":sum(r["transfer_D3_D4"] for r in records),
        "online_Q":sum(r["online_queries"] for r in records),
        "posthoc_model_inputs":sum(r["posthoc_queries"] for r in records),
    }),flush=True)
    return records


def accepted_features(rows):
    passed=[r for r in rows if r["online_certified"]]
    if not passed:
        return (np.empty((0,9),dtype=float),
                np.empty((0,),dtype=int),
                np.empty((0,),dtype=float))
    return (
        np.stack([r["features"] for r in passed]),
        np.asarray([r["failure_D3_D4"] for r in passed],dtype=int),
        np.asarray([r["fraction_max"] for r in passed],dtype=float),
    )


def summarize_selection(records,flag):
    issued=[r for r in records if r[flag]]
    failed=sum(int(r["failure_D3_D4"]) for r in issued)
    n=len(issued)
    success=n-failed
    if n:
        lower=0. if failed==0 else float(beta.ppf(.025,failed,n-failed+1))
        upper=1. if failed==n else float(beta.ppf(.975,failed+1,n-failed))
    else:
        lower=upper=None
    return {
        "method":flag,
        "N":N_TEST,
        "issued":n,
        "abstentions":N_TEST-n,
        "issued_fraction":n/N_TEST,
        "failed_on_D3_D4":failed,
        "selective_failure_rate":failed/n if n else None,
        "failure_95pct_CP_interval":[lower,upper],
        "transfer_success":success,
        "unconditional_transferred_fraction":success/N_TEST,
        "mean_online_Q":float(np.mean([r["online_queries"] for r in records])),
        "max_online_Q":max(r["online_queries"] for r in records),
    }


def run(args):
    if (args.seed,args.trees,args.donors)!=(SEED,TREES,BANK_SIZE):
        raise ValueError("frozen protocol expects seed57 trees120 bank16")
    path=Path(args.npz_path)
    raw,ytrain,raw_test,ytest=load_npz_dataset(path)
    with np.load(path) as z:
        protocol=str(z["dataset_protocol"])
        classes=z["labels"].tolist()
    if (raw.shape!=(1422,182,3) or raw_test.shape!=(1436,182,3)
        or len(np.unique(ytrain))!=20 or len(np.unique(ytest))!=20
        or "UEA CharacterTrajectories _eq aeon 1.5.0 official" not in protocol):
        raise ValueError("requires pinned official CharacterTrajectories _eq archive")
    fit_ix,risk_ix,gate_ix=stratified_threeway(ytrain)
    fit_raw=raw[fit_ix]
    mu,sd,profiles,model,counter=fit_predictor(
        fit_raw,ytrain[fit_ix],"extratrees",SEED,TREES
    )
    fit=apply_standardizer(fit_raw,mu,sd)
    risk_x=apply_standardizer(raw[risk_ix],mu,sd)
    gate_x=apply_standardizer(raw[gate_ix],mu,sd)
    test=apply_standardizer(raw_test,mu,sd)
    groups=make_components((182,3),BINS,"all_channels")
    bank=fit_real_donor_bank(
        fit,ytrain[fit_ix],groups,max_donors=BANK_SIZE,seed=BANK_SEED
    )
    screening_start=int(counter["examples"])
    screened={}
    for lab in sorted(bank.donors):
        screened[lab]=np.argmax(model(bank.donors[lab]),axis=1)
    screening_calls=int(counter["examples"])-screening_start
    baseline_start=int(counter["examples"])
    risk_base=model(risk_x)
    gate_base=model(gate_x)
    full_test_base=model(test)
    baseline_calls=int(counter["examples"])-baseline_start
    if baseline_calls!=len(risk_x)+len(gate_x)+1436:
        raise AssertionError("cache all original predictions exactly once")
    ledger_before_online=int(counter["examples"])
    risk_rows=evaluate_stage(
        "risk_fit",risk_x,risk_ix,risk_base,bank,screened,
        groups,profiles,model,counter,
    )
    features,yfail,_=accepted_features(risk_rows)
    estimator=fit_transfer_risk(features,yfail)
    fit_scores=estimator.predict(features)

    gate_rows=evaluate_stage(
        "gate_calibration",gate_x,gate_ix,gate_base,bank,screened,
        groups,profiles,model,counter,
    )
    gate_features,gate_fail,gate_support=accepted_features(gate_rows)
    gate_scores=estimator.predict(gate_features)
    calibrated=calibrate_transfer_gate(
        fit_scores,gate_scores,gate_fail,gate_support
    )
    chosen=np.sort(
        np.random.default_rng(SELECT_SEED).permutation(len(test))[:N_TEST]
    )
    test_rows=evaluate_stage(
        "test",test[chosen],chosen,full_test_base[chosen],
        bank,screened,groups,profiles,model,counter,
    )
    for r in test_rows:
        r["base_issue"]=int(r["online_certified"])
        if r["online_certified"]:
            score=float(estimator.predict(r["features"][None])[0])
            r["predicted_failure_risk"]=score
            r["risk_issue"]=int(calibrated.accepts(score))
            r["support_issue"]=int(calibrated.accepts_support(r["fraction_max"]))
        else:
            r["predicted_failure_risk"]=None
            r["risk_issue"]=0
            r["support_issue"]=0
    methods=tuple(summarize_selection(test_rows,key)
                  for key in ("base_issue","risk_issue","support_issue"))
    search_calls=sum(r["online_queries"] for r in (
        risk_rows+gate_rows+test_rows
    ))
    audits=sum(r["posthoc_queries"] for r in (
        risk_rows+gate_rows+test_rows
    ))
    expected=screening_calls+baseline_calls+search_calls+audits
    if int(counter["examples"])!=expected:
        raise AssertionError("complete model-input cost ledger mismatch")
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    allrows=risk_rows+gate_rows+test_rows
    write_csv(out/"per_sample.csv",[
        {**r,
         "features":(",".join(map(str,r["features"]))
                     if r["features"] is not None else ""),
         "selected":(",".join(map(str,r["selected"]))
                     if r["selected"] is not None else "")}
        for r in allrows
    ])
    write_csv(out/"summary.csv",list(methods))
    conf={
        "protocol":"artifacts/beacon_charactertrajectories_transfer_gate_preregistered_2026-11-22.md",
        "model_type":"ExtraTrees",
        "dataset":"UEA CharacterTrajectories _eq",
        "archive_source_was_unequal_length":True,
        "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_sha256":counter["fitted_model_sha256"],
        "scaler_sha256":counter["training_standardizer_sha256"],
        "model_accuracy_all_official_TEST":float(
            np.mean(np.argmax(full_test_base,axis=1)==ytest)
        ),
        "adequate_model_threshold_preregistered":.7,
        "TEST_y_not_used_for_search_gate_or_fit":True,
        "unverified_independent_participants":True,
        "class_map":classes,
        "fit_indices":fit_ix.tolist(),
        "risk_fit_indices":risk_ix.tolist(),
        "gate_calibration_indices":gate_ix.tolist(),
        "TEST_selected_indices":chosen.tolist(),
        "risk_fit_count":estimator.fit_n,
        "risk_fit_failures":estimator.fit_failure_n,
        "risk_model_type":"constant_Laplace" if estimator.model is None
                          else "LogisticRegression_liblinear",
        "risk_model_constant":estimator.constant,
        "risk_feature_names":FEATURE_NAMES,
        "risk_model_coefficients":(
            estimator.model.coef_[0].tolist()
            if estimator.model is not None else None
        ),
        "risk_scaler_mean":(
            estimator.scaler.mean_.tolist()
            if estimator.scaler is not None else None
        ),
        "risk_scaler_scale":(
            estimator.scaler.scale_.tolist()
            if estimator.scaler is not None else None
        ),
        "train_fitted_thresholds":calibrated.fit_thresholds,
        "calibration_risk_threshold":calibrated.threshold,
        "calibration_accepted":calibrated.calibration_accepted,
        "calibration_failed":calibrated.calibration_failed,
        "calibration_corrected_upper":calibrated.calibration_upper,
        "support_control_threshold":calibrated.support_threshold,
        "TEST_not_used_for_threshold_selection":True,
        "risk_fit_and_calibration_disjoint":True,
        "risk_model_uses_unseen_donors_only_as_TRAIN_targets":True,
        "Q_shared_across_2_online_donors":16,
        "max_modified_fraction":MAX_EDIT,
        "n_components":BINS,
        "taper":TAPER,
        "screening_model_inputs":screening_calls,
        "cached_baseline_model_inputs":baseline_calls,
        "online_Q16_model_inputs_all_three_stages":search_calls,
        "posthoc_D3_D4_model_inputs_all_three_stages":audits,
        "all_classifier_model_input_evaluations":expected,
        "results":methods,
        "limitations":[
            "Filtering cannot increase unconditional count of verified transfers",
            "CP requires calibration-to-test exchangeability, not established by person",
            "Finite TRAIN donor transfer only; no physical or causal guarantees",
            "TEST audit calls are outside online Q16 and fully reported",
            "Posthoc risk model does not create any new model certificate",
        ],
    }
    (out/"manifest.json").write_text(
        json.dumps(conf,indent=2,ensure_ascii=False),encoding="utf8"
    )
    print(json.dumps({
        "accuracy":conf["model_accuracy_all_official_TEST"],
        "risk_fit_count":estimator.fit_n,
        "calibration_threshold":calibrated.threshold,
        "calibration_upper":calibrated.calibration_upper,
        "methods":methods,
        "online_model_calls":search_calls,
        "posthoc_model_calls":audits,
        "total_model_inputs":expected,
    },indent=2,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--trees",type=int,default=TREES)
    parser.add_argument("--donors",type=int,default=BANK_SIZE)
    run(parser.parse_args())


if __name__=="__main__":
    main()
