#!/usr/bin/env python3
"""Previously unexamined UEA NATOPS: frozen two-donor robust CF Q16 protocol.

Read artifacts/beacon_natops_sparse_q16_preregistered_2026-11-21.md.
All methods share A, B, two TRAIN donors, ten groups and ONE TOTAL Q16.
Test labels are used only for the descriptive classifier-accuracy endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_donor_robustness import (
    rank_confirmed_donors, exact_two_sided_mcnemar,
)
from beaconxai.cf_path_search import search_backward_counterfactual
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import apply_standardizer, load_npz_dataset
from beaconxai.donor_robust_cf import (
    search_backward_robust_counterfactual,search_robust_counterfactual,
    search_sparse_robust_counterfactual,train_profile_donor_gains,
)
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.verified_counterfactual import splice
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv

Q=16
DONOR_BANK_SEED=67
SEED=57
TREES=120
DONOR_LIMIT=16
SUPPORT=.35
BINS=10
TAPER=2
METHODS=("sparse_maximin","forward_robust","backward_robust_chunks")


def no_donor():
    return {
        "status":"not_two_model_confirmed_train_donors",
        "certified":0,"selected":None,"model_queries":0,
        "minimal_on_both":0,"d1_flip":0,"d2_flip":0,
    }


def measured_search(x,base,donors,groups,rank,model,counter,method,profile_gains=None):
    target=int(np.argsort(-base,kind="stable")[1])
    before=int(counter["examples"])
    selected=None
    minimal=False
    status="not_found_within_budget"
    if method=="sparse_maximin":
        if profile_gains is None:
            raise ValueError("missing TRAIN profile donor gains")
        result=search_sparse_robust_counterfactual(
            x,base,donors,groups,profile_gains,model,
            target_class=target,budget=Q,maximum_modified_fraction=SUPPORT,taper=TAPER,
        )
        selected=result.certified_edit
        minimal=result.local_minimality_checked
        status=result.status
        reported=result.queries
    elif method=="backward_robust_chunks":
        result=search_backward_robust_counterfactual(
            x,base,donors,groups,list(reversed(rank)),model,
            target_class=target,budget=Q,taper=TAPER,
        )
        selected=result.certified_edit
        minimal=result.local_minimality_checked
        status=result.status
        reported=result.queries
    elif method=="forward_robust":
        result=search_robust_counterfactual(
            x,base,donors,groups,rank,model,
            target_class=target,budget=Q,max_changed_components=len(groups),
            seed_singletons=1,taper=TAPER,prune=True,
        )
        selected=result.certified_edit
        minimal=result.local_minimality_checked
        status=result.status
        reported=result.queries
    else:
        raise ValueError("unknown method")

    used=int(counter["examples"])-before
    if not 0<=used<=Q or used!=reported:
        raise AssertionError("unexpected online model-example cost")
    if selected is None:
        return {
            "status":status,"certified":0,"selected":None,
            "model_queries":used,"minimal_on_both":0,
            "d1_flip":0,"d2_flip":0,
        }
    masks=tuple(int(j) for j in selected)
    altered=[
        splice(x,donor,groups,masks,taper=TAPER)
        for donor in donors
    ]
    fractions=[
        float(np.mean(np.abs(y-x)>1e-8))
        for y in altered
    ]
    # No extra black-box call: both donors were verified under this SAME mask.
    return {
        "status":status,"certified":1,"selected":masks,
        "model_queries":used,
        "minimal_on_both":int(bool(minimal)),
        "d1_flip":1,"d2_flip":1,
        "fraction_d1":fractions[0],
        "fraction_d2":fractions[1],
        "joint_fraction":max(fractions),
        "strict_joint":int(max(fractions)<=SUPPORT),
        "both_within_20pct":int(max(fractions)<=.2),
        "both_within_35pct":int(max(fractions)<=.35),
        "both_within_50pct":int(max(fractions)<=.5),
    }


def primary_paired(records):
    lookup={(r["sample"],r["method"]):r for r in records}
    counts={"both":0,"new_only":0,"forward_only":0,"neither":0}
    for i in range(180):
        a=int(lookup[i,"sparse_maximin"].get("strict_joint",0))
        b=int(lookup[i,"forward_robust"].get("strict_joint",0))
        key=("both" if a and b else
             "new_only" if a else
             "forward_only" if b else "neither")
        counts[key]+=1
    if sum(counts.values())!=180:
        raise AssertionError("missing paired test rows")
    counts["N"]=180
    counts["new_joint"]=counts["both"]+counts["new_only"]
    counts["forward_joint"]=counts["both"]+counts["forward_only"]
    counts["effect_new_minus_forward_pp"]=100*(
        counts["new_only"]-counts["forward_only"])/180
    counts["exact_mcnemar_p"]=exact_two_sided_mcnemar(
        counts["new_only"],counts["forward_only"])
    counts["predeclared_alpha"]=.05
    return counts


def summarize(records):
    summary=[]
    for method in METHODS:
        subset=[r for r in records if r["method"]==method]
        success=[r for r in subset if r["certified"]]
        if len(subset)!=180 or len({r["sample"] for r in subset})!=180:
            raise AssertionError("incomplete 180-window comparison")
        summary.append({
            "method":method,"n_test":180,
            "n_certified_two_donors":len(success),
            "n_joint_le20":sum(r["both_within_20pct"] for r in success),
            "n_joint_le35":sum(r["both_within_35pct"] for r in success),
            "n_joint_le50":sum(r["both_within_50pct"] for r in success),
            "n_local_minimal_on_both":sum(
                r["minimal_on_both"] for r in success),
            "mean_online_model_queries":float(np.mean(
                [r["model_queries"] for r in subset])),
            "median_fraction_d1":(
                float(np.median([r["fraction_d1"] for r in success]))
                if success else None),
            "median_fraction_d2":(
                float(np.median([r["fraction_d2"] for r in success]))
                if success else None),
            "abstentions":180-len(success),
        })
    return summary


def run(args):
    if (args.seed, args.trees, args.donors)!=(SEED,TREES,DONOR_LIMIT):
        raise ValueError("frozen study: seed57 trees120 donors16")
    path=Path(args.npz_path)
    raw_train,ytrain,raw_test,ytest=load_npz_dataset(path)
    with np.load(path) as z:
        labels=z["labels"].tolist()
        protocol=str(z["dataset_protocol"])
    if (raw_train.shape!=(180,51,24) or raw_test.shape!=(180,51,24)
        or len(set(ytrain))!=6 or len(set(ytest))!=6
        or "UEA NATOPS official TRAIN 180 / TEST 180" not in protocol):
        raise ValueError("wrong official NATOPS split or geometry")
    mu,sd,profiles,model,counter=fit_predictor(
        raw_train,ytrain,"extratrees",SEED,TREES,
    )
    train=apply_standardizer(raw_train,mu,sd)
    test=apply_standardizer(raw_test,mu,sd)
    comps=make_components((51,24),BINS,"all_channels")
    if len(comps)!=10:
        raise AssertionError("partition mismatch")
    bank=fit_real_donor_bank(
        train,ytrain,comps,max_donors=DONOR_LIMIT,
        seed=DONOR_BANK_SEED,
    )
    base=model(test)
    baseline_calls=int(counter["examples"])
    if baseline_calls!=180:
        raise AssertionError("original outputs not cached correctly")
    screened={}
    for cls in sorted(bank.donors):
        screened[cls]=np.argmax(model(bank.donors[cls]),axis=1)
    offline_calls=int(counter["examples"])-baseline_calls
    if offline_calls!=sum(len(v) for v in bank.donors.values()):
        raise AssertionError("uncounted TRAIN-only donor screening")
    records=[]
    online=0
    missing=0
    for i,x in enumerate(test):
        source=int(np.argmax(base[i]))
        target=int(np.argsort(-base[i],kind="stable")[1])
        rank=np.argsort(
            -np.abs(contrastive_prior(x,source,target,profiles,comps)),
            kind="stable",
        ).astype(int).tolist()
        pool=bank.donors[target]
        eligible=rank_confirmed_donors(
            x,pool,screened[target],target,
            bank.selected_train_indices[target],
        )
        if len(eligible)<2:
            missing+=1
        chosen=eligible[:2]
        selected_donors=tuple(pool[j] for j in chosen)
        gains=(train_profile_donor_gains(
            x,selected_donors,comps,profiles,source,target,taper=TAPER)
            if len(chosen)==2 else None)
        for method in METHODS:
            if len(chosen)<2:
                result=no_donor()
            else:
                result=measured_search(
                    x,base[i],selected_donors,comps,rank,
                    model,counter,method,gains,
                )
            online+=result["model_queries"]
            records.append({
                "sample":i,"method":method,
                "source_pred":source,"target_pred":target,
                "available_train_donors":len(eligible),
                "train_donor_1":(
                    int(bank.selected_train_indices[target][chosen[0]])
                    if len(chosen)==2 else None),
                "train_donor_2":(
                    int(bank.selected_train_indices[target][chosen[1]])
                    if len(chosen)==2 else None),
                **result,
            })
    if int(counter["examples"])!=baseline_calls+offline_calls+online:
        raise AssertionError("total model evaluation ledger does not reconcile")
    primary=primary_paired(records)
    comparison=summarize(records)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",[
        {**r,"selected":(
            ",".join(map(str,r["selected"])) if r["selected"] is not None else "")
        } for r in records
    ])
    write_csv(out/"summary.csv",comparison)
    manifest={
        "preregistered_protocol":"artifacts/beacon_natops_robust_q16_preregistered_2026-11-21.md",
        "dataset":"UEA NATOPS official split; aviation command gestures",
        "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "trained_classifier_sha256":counter["fitted_model_sha256"],
        "scaler_sha256":counter["training_standardizer_sha256"],
        "model":"ExtraTrees","seed":SEED,"trees":TREES,
        "source_class_is_model_argmax":True,
        "target_is_model_second_best_not_test_label":True,
        "participant_id_overlap_not_directly_verifiable":True,
        "documented_participants_split":"unknown; participant IDs unavailable",
        "test_recordings":180,"train_recordings":180,
        "classes":labels,"timepoints":51,"channels":24,
        "model_accuracy":float(np.mean(np.argmax(base,axis=1)==ytest)),
        "adequacy_accuracy_threshold":.7,
        "donor_limit_per_class":DONOR_LIMIT,
        "donor_seed":DONOR_BANK_SEED,
        "two_nearest_model_confirmed_train_donors":True,
        "time_blocks":BINS,"taper":TAPER,
        "joint_support_threshold":SUPPORT,
        "total_Q_per_search_across_donors":Q,
        "cached_original_model_evaluations":baseline_calls,
        "offline_train_donor_screening_evaluations":offline_calls,
        "online_model_evaluations_all_methods":online,
        "total_model_evaluations":counter["examples"],
        "windows_without_two_donors":missing,
        "no_posthoc_queries_excluded_from_Q":True,
        "test_labels_used_only_for_final_accuracy":True,
        "no_physical_plausibility_or_causality_claim":True,
        "new_sparse_method_returns_only_compact_two_donor_certificates":True,
        "sparse_prior_uses_TRAIN_profiles_only":True,
        "primary":primary,
        "summary":comparison,
    }
    (out/"manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "model_accuracy":manifest["model_accuracy"],
        "adequacy":manifest["model_accuracy"]>=.7,
        "primary":primary,"summary":comparison,
        "windows_without_donors":missing,
    },ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--trees",type=int,default=TREES)
    p.add_argument("--donors",type=int,default=DONOR_LIMIT)
    run(p.parse_args())


if __name__=="__main__":
    main()
