#!/usr/bin/env python3
"""Independent UEA BasicMotions Q8/Q16 counterfactual experiment.

Preserves the original UEA TRAIN/TEST split. Trains a frozen classifier only
on TRAIN. Prepares class statistics and real donors from TRAIN only. Uses
model-predicted A and a fixed runner-up B before performing any edits.
Compares model-verified BEACON directed/uniform patch paths, native pinned
MultiSubSpaCE (budget-adapted) and a donor-segment binary-search baseline.

No participant IDs in the public BasicMotions split. No gravity-residual
invariant, sensor-physics claim or independent human-subject generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_donor_line import search_donor_segment
from beaconxai.cf_path_search import search_counterfactual_path
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import load_npz_dataset, apply_standardizer
from beaconxai.native_cf_budget import TrainingGaussianOutlier
from beaconxai.real_donor import fit_real_donor_bank, nearest_context_index
from beaconxai.verified_counterfactual import splice
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_native_cf_baselines import UPSTREAM_SHA, run_native
from scripts.run_q16_matched_ablation import write_csv


def wilson(successes, total):
    """Descriptive binomial interval; no participant-level generality."""
    if total<1 or not 0<=successes<=total:
        raise ValueError("invalid binomial fraction")
    p=successes/total
    z=1.959963984540054
    div=1+z*z/total
    ctr=(p+z*z/(2*total))/div
    rad=z*math.sqrt(p*(1-p)/total+z*z/(4*total**2))/div
    return max(0.,ctr-rad), min(1.,ctr+rad)


def summarize(rows):
    answer=[]
    for method,q in sorted({(r["method"],r["Q"]) for r in rows}):
        group=[r for r in rows if (r["method"],r["Q"])==(method,q)]
        tests={r["sample"] for r in group}
        if len(tests)!=len(group):
            raise AssertionError("duplicated method/test-window row")
        success=[r for r in group if r["verified"]]
        n=len(group)
        lo,hi=wilson(len(success),n)
        answer.append({
            "method":method, "Q":q,
            "n_test_windows":n,
            "n_target_flips":len(success),
            "flip_fraction":len(success)/n,
            "wilson_95pct_low_descriptive":lo,
            "wilson_95pct_high_descriptive":hi,
            "verified_and_modified_le20pct":sum(
                r["verified"] and r["modified_fraction"]<=.20 for r in success),
            "verified_and_modified_le35pct":sum(
                r["verified"] and r["modified_fraction"]<=.35 for r in success),
            "verified_and_modified_le50pct":sum(
                r["verified"] and r["modified_fraction"]<=.50 for r in success),
            "mean_actual_additional_calls":float(np.mean(
                [r["additional_calls"] for r in group])),
            "median_modified_fraction_conditional":(
                float(np.median([r["modified_fraction"] for r in success]))
                if success else None),
            "median_relative_l2_conditional":(
                float(np.median([r["relative_l2"] for r in success]))
                if success else None),
            "n_model_errors":sum(r["status"]=="native_algorithm_error"
                                 for r in group),
            "n_native_batch_aborts":sum(r["status"]=="native_budget_exceeded"
                                        for r in group),
        })
    return answer


def run(args):
    root=Path(args.upstream).resolve()
    if not (root/"methods/MultiSubSpaCECF.py").is_file():
        raise FileNotFoundError("authentic native MultiSubSpaCE source absent")
    sha=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],
                                text=True).strip()
    if sha!=UPSTREAM_SHA:
        raise ValueError("wrong native MultiSubSpaCE source SHA")
    sys.path.insert(0,str(root))
    from methods.MultiSubSpaCECF import MultiSubSpaCECF

    path=Path(args.npz_path)
    train_raw,ytrain,test_raw,ytest=load_npz_dataset(path)
    with np.load(path) as data:
        protocol=str(data["dataset_protocol"])
        class_labels=data["labels"].tolist()
    if (train_raw.shape!=(40,100,6) or test_raw.shape!=(40,100,6) or
        len(set(ytrain))!=4 or len(set(ytest))!=4 or
        "participant ids unavailable" not in protocol):
        raise ValueError("only the original BasicMotions official split supported")
    qs=sorted(set(int(v) for v in args.budgets.split(",")))
    if not qs or not set(qs).issubset({8,16}) or not 1<=args.max_test<=40:
        raise ValueError("invalid Q8/Q16 or test-window limit")
    if not 1<=args.max_changed<=args.max_extended<=60:
        raise ValueError("invalid standard or extended component count")
    mu,sd,profiles,model,calls=fit_predictor(
        train_raw,ytrain,args.model,args.seed,args.trees)
    train=apply_standardizer(train_raw,mu,sd)
    test=apply_standardizer(test_raw[:args.max_test],mu,sd)
    baseline=model(test)
    scorer=TrainingGaussianOutlier(train)
    components=make_components(train.shape[1:],10,"per_channel")
    donor_bank=fit_real_donor_bank(
        train,ytrain,components,max_donors=args.donors,seed=args.seed+10
    )
    rows=[]
    prefix_diagnostics=[]
    offline_prefix_queries=0
    for i,x in enumerate(test):
        base=baseline[i]
        source=int(np.argmax(base))
        target=int(np.argsort(-base,kind="stable")[1])
        donor_id=nearest_context_index(
            x,0,donor_bank,target,components
        )
        donor=donor_bank.donors[target][donor_id]
        prior=contrastive_prior(x,source,target,profiles,components)
        directed=np.argsort(-np.abs(prior),kind="stable").astype(int).tolist()
        uniform=np.random.default_rng(
            args.seed*100003+i+10
        ).permutation(len(components)).astype(int).tolist()
        for q in qs:
            for method in ("beacon_directed","beacon_uniform",
                           "beacon_directed_extended","beacon_uniform_extended",
                           "donor_line","native_multisubspace"):
                before=int(calls["examples"])
                edit=None
                status="not_found"
                if method.startswith("beacon_"):
                    order=directed if method.startswith("beacon_directed") else uniform
                    max_groups=(
                        args.max_extended if method.endswith("_extended")
                        else args.max_changed
                    )
                    result=search_counterfactual_path(
                        x,base,donor,components,order,model,
                        target_class=target,budget=q,
                        max_changed_components=min(max_groups, len(components)),
                        seed_singletons=3,taper=args.taper,
                    )
                    if result.is_valid:
                        edit=splice(x,donor,components,result.verified_edit,
                                    taper=args.taper)
                    status=result.status
                    reported=result.calls
                    local_minimality=int(
                        result.is_valid and result.local_minimality_checked)
                elif method=="donor_line":
                    result=search_donor_segment(
                        x,donor,base,model,target_class=target,budget=q)
                    edit=result.edited if result.is_verified else None
                    status=result.status
                    reported=result.queries
                    local_minimality=0
                else:
                    edit,status,reported,_,_,_=run_native(
                        x,donor,target,model,scorer,q,
                        args.seed*100003+i,MultiSubSpaCECF,
                        n_classes=len(base),
                    )
                    local_minimality=0
                spent=int(calls["examples"])-before
                if spent!=reported or spent>q or spent<0:
                    raise AssertionError("query budget / actual model calls disagree")
                fraction=(float(np.mean(np.abs(edit-x)>1e-8))
                          if edit is not None else None)
                rel=(float(np.linalg.norm(edit-x) /
                            max(np.linalg.norm(x),np.linalg.norm(donor),1e-8))
                     if edit is not None else None)
                rows.append({
                    "sample":i,"method":method,"Q":q,
                    "source_pred":source,"target_pred":target,
                    "label_offline":int(ytest[i]),
                    "base_prediction_correct_offline":int(source==int(ytest[i])),
                    "donor_train_index":int(
                        donor_bank.selected_train_indices[target][donor_id]),
                    "verified":int(edit is not None),"status":status,
                    "additional_calls":spent,
                    "modified_fraction":fraction,
                    "relative_l2":rel,
                    "locally_minimal_verified":local_minimality,
                })
        if args.offline_prefix:
            # OFFLINE FULL-REFERENCE diagnostic, NEVER exposed to search
            # decisions, model selection, or held-out labels.
            for name,order in (("directed",directed),("uniform",uniform)):
                candidates=np.stack([
                    splice(x,donor,components,order[:n],taper=args.taper)
                    for n in range(1,len(order)+1)
                ])
                start=int(calls["examples"])
                outputs=model(candidates)
                offline_prefix_queries+=len(order)
                if int(calls["examples"])-start!=len(order):
                    raise AssertionError("offline full prefix queries undercounted")
                successes=np.flatnonzero(np.argmax(outputs,axis=1)==target)
                first=int(successes[0])+1 if len(successes) else None
                selected=candidates[first-1] if first is not None else None
                prefix_diagnostics.append({
                    "sample":i,"order":name,"target_pred":target,
                    "n_components":len(order),"first_flip_prefix_size":first,
                    "first_flip_modified_fraction":(
                        float(np.mean(np.abs(selected-x)>1e-8))
                        if selected is not None else None
                    ),
                    "full_prefix_target_flip":int(
                        int(np.argmax(outputs[-1]))==target),
                    "offline_queries":len(order),
                    "offline_only":True,
                })
    online=sum(r["additional_calls"] for r in rows)
    if calls["examples"]!=len(test)+online+offline_prefix_queries:
        raise AssertionError("some model calls were not counted")
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    table=summarize(rows)
    write_csv(output/"per_sample.csv",rows)
    write_csv(output/"summary.csv",table)
    if args.offline_prefix:
        write_csv(output/"offline_prefix_reachability.csv",prefix_diagnostics)
    manifest={
        "dataset":"UEA BasicMotions","source":"aeon.datasets.load_basic_motions",
        "dataset_protocol":protocol,
        "participant_disjoint_claim":False,
        "assumed_gravity_or_other_sensor_identity":False,
        "independent_of_uci_har":True,
        "train_windows":40,"test_windows_evaluated":len(test),
        "train_channels":6,"timepoints":100,"classes":class_labels,
        "model_type":args.model,"fitted_model_sha256":calls["fitted_model_sha256"],
        "preprocessed_data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "native_baseline_upstream_sha":UPSTREAM_SHA,
        "native_baseline_budget_adapted_not_original_settings":True,
        "same_frozen_classifier_source_target_and_donor_per_window":True,
        "methods_have_different_intervention_operators":True,
        "component_fraction_per_group":1/len(components),
        "beacon_standard_max_groups":args.max_changed,
        "beacon_extended_max_groups":args.max_extended,
        "offline_prefix_oracle_only_descriptive":args.offline_prefix,
        "offline_prefix_oracle_model_evaluations":offline_prefix_queries,
        "offline_prefix_does_not_prove_global_minimality":True,
        "max_additional_blackbox_Q":qs,
        "cached_source_model_evaluations":len(test),
        "online_blackbox_model_evaluations":online,
        "total_model_evaluations_including_offline_reference":calls["examples"],
        "line_search_operator_changes_whole_signal":True,
        "test_labels_never_used_in_search":True,
        "limitations":[
            "Archive split has no subject IDs; no unseen-participant inference",
            "The 40-window test set yields wide descriptive uncertainty intervals",
            "Training size 40 makes neural classifier highly seed-sensitive",
            "No physically validated intervention operator on BasicMotions",
            "Native MultiSubSpaCE settings adapted to Q8 and Q16",
            "Wilson intervals ignore multiple comparisons and adaptation history",
            "Target-class flips do not prove causal or sufficient explanations",
            "Extended path and prefix-reachability analyses were motivated by exploratory test results",
            "Offline full-prefix reachability cannot be counted as Q8/Q16 online performance",
        ],
    }
    (output/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    print(json.dumps({
        "model_SHA":calls["fitted_model_sha256"],
        "test_accuracy_descriptive":float(np.mean(np.argmax(baseline,axis=1)==ytest[:len(test)])),
        "summary":table,
    },indent=2,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",required=True,type=Path)
    p.add_argument("--upstream",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--model",choices=("raw_mlp","extratrees"),default="extratrees")
    p.add_argument("--budgets",default="8,16")
    p.add_argument("--max-test",type=int,default=40)
    p.add_argument("--max-changed",type=int,default=8)
    p.add_argument("--max-extended",type=int,default=16)
    p.add_argument("--offline-prefix",action="store_true")
    p.add_argument("--donors",type=int,default=8)
    p.add_argument("--taper",type=int,default=2)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--seed",type=int,default=57)
    run(p.parse_args())


if __name__=="__main__":
    main()
