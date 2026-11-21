#!/usr/bin/env python3
"""Preregistered UEA SpokenArabicDigits: K2 vs K3 donor robust sparse CF.

The **same existing** compact sparse search core is used for both policies.
K3 verifies a third, distant TRAIN donor *inside the shared Q16*.
Only D4/D5 are unseen for the primary transfer outcome and their posthoc
classifier evaluations are logged separately, not hidden inside Q16.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_donor_robustness import (
    rank_confirmed_donors, exact_two_sided_mcnemar,
)
from beaconxai.contrastive import make_components
from beaconxai.datasets import apply_standardizer,load_npz_dataset
from beaconxai.donor_robust_cf import (
    search_sparse_robust_counterfactual,train_profile_donor_gains,
)
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.verified_counterfactual import splice
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv


SEED=57
TREES=120
BANK_SEED=67
BANK_SIZE=16
TAPER=2
BINS=10
SUPPORT=.35
Q=16
SELECT_SEED=5722
N_SELECTED=320
METHODS=("K2_nearest","K3_stress")


def assign_five_donor_roles(source,bank_windows,model_labels,target,train_indices):
    """Return actual bank indices (D1,D2,furthest D3,nearest remaining D4,D5).

    Only TRAIN bank values, TRAIN IDs and offline bank model outputs are used.
    Never sees current x model interventions or TEST ground truth.
    """
    ranked=rank_confirmed_donors(
        source,bank_windows,model_labels,target,train_indices,
    )
    if len(ranked)<5:
        return None
    first=tuple(ranked[:2])
    distances=np.mean((np.asarray(bank_windows,dtype=float)-source[None])**2,axis=(1,2))
    rest=ranked[2:]
    third=min(rest,key=lambda j:(-float(distances[j]),int(train_indices[j])))
    remaining=tuple(j for j in rest if j!=third)
    roles=first+(third,)+tuple(remaining[:2])
    if len(roles)!=5 or len(set(roles))!=5:
        raise AssertionError("TRAIN donor roles not unique")
    return roles


def no_donor_result():
    return {
        "status":"insufficient_five_model_confirmed_train_donors",
        "online_verified":0,"compact_online":0,"selected":None,
        "model_queries":0,"minimal_on_online_donors":0,
        "changed_fraction_max_online":None,
        "transfer_D4_D5":0,"posthoc_queries":0,
        "new_donor_changed_fraction_max":None,
        "search_wall_seconds":0.,
        "examination_wall_seconds":0.,
    }


def evaluate_method(x,original,donors,unseen,groups,profiles,model,counter,method):
    if method not in METHODS:
        raise ValueError("unrecognized preregistered donor policy")
    pool=donors[:2] if method=="K2_nearest" else donors[:3]
    target=int(np.argsort(-original,kind="stable")[1])
    source=int(np.argmax(original))
    prior=train_profile_donor_gains(
        x,pool,groups,profiles,source,target,taper=TAPER,
    )
    previous=int(counter["examples"])
    start=time.perf_counter()
    result=search_sparse_robust_counterfactual(
        x,original,pool,groups,prior,model,
        target_class=target,budget=Q,
        maximum_modified_fraction=SUPPORT,taper=TAPER,
    )
    elapsed=time.perf_counter()-start
    used=int(counter["examples"])-previous
    if used!=result.queries or not 0<=used<=Q:
        raise AssertionError("online Q16 must count all evaluated model inputs")
    if not result.is_certified:
        return {
            "status":result.status,
            "online_verified":0,"compact_online":0,"selected":None,
            "model_queries":used,"minimal_on_online_donors":0,
            "changed_fraction_max_online":None,
            "transfer_D4_D5":0,"posthoc_queries":0,
            "new_donor_changed_fraction_max":None,
            "search_wall_seconds":float(elapsed),
            "examination_wall_seconds":0.,
        }
    mask=tuple(map(int,result.certified_edit))
    if not mask:
        raise AssertionError("A cannot switch to B under an empty mask")
    for d in range(len(pool)):
        if not any(p.components==mask and p.donor_index==d and p.targeted_flip
                   for p in result.probes):
            raise AssertionError("certificate must contain direct per-donor model evidence")
    fractions=[
        float(np.mean(np.abs(splice(x,d,groups,mask,taper=TAPER)-x)>1e-8))
        for d in pool
    ]
    if max(fractions)>SUPPORT+1e-12:
        raise AssertionError("noncompact online certificate must not escape")
    # Only now, after the mask is frozen, evaluate D4 and D5. There is NO
    # dependence of search logic on these model outputs.
    edits=np.stack([
        splice(x,donor,groups,mask,taper=TAPER)
        for donor in unseen
    ])
    prior_audit=int(counter["examples"])
    t0=time.perf_counter()
    scores=np.asarray(model(edits),dtype=float)
    audit_seconds=time.perf_counter()-t0
    posthoc=int(counter["examples"])-prior_audit
    if scores.shape!=(len(unseen),len(original)) or posthoc!=len(unseen):
        raise AssertionError("external donor audit must count exactly two examples")
    unseen_frac=[
        float(np.mean(np.abs(z-x)>1e-8)) for z in edits
    ]
    good=bool(all(
        int(np.argmax(scores[j]))==target and unseen_frac[j]<=SUPPORT
        for j in range(len(unseen))
    ))
    return {
        "status":result.status,
        "online_verified":1,"compact_online":1,
        "selected":mask,"model_queries":used,
        "minimal_on_online_donors":int(result.local_minimality_checked),
        "changed_fraction_max_online":max(fractions),
        "transfer_D4_D5":int(good),"posthoc_queries":posthoc,
        "new_donor_changed_fraction_max":max(unseen_frac),
        "search_wall_seconds":float(elapsed),
        "examination_wall_seconds":float(audit_seconds),
    }


def paired_primary(records):
    lookup={(r["sample"],r["method"]):r for r in records}
    if len(lookup)!=N_SELECTED*len(METHODS):
        raise AssertionError("missing paired comparison records")
    counts={"both":0,"K3_only":0,"K2_only":0,"neither":0}
    for i in range(N_SELECTED):
        a=int(lookup[i,"K3_stress"]["transfer_D4_D5"])
        b=int(lookup[i,"K2_nearest"]["transfer_D4_D5"])
        key="both" if a and b else "K3_only" if a else "K2_only" if b else "neither"
        counts[key]+=1
    counts["n_test"]=N_SELECTED
    counts["K3_fresh"]=counts["both"]+counts["K3_only"]
    counts["K2_fresh"]=counts["both"]+counts["K2_only"]
    counts["delta_pp"]=100*(counts["K3_only"]-counts["K2_only"])/N_SELECTED
    counts["mcnemar_p_exact_two_sided"]=exact_two_sided_mcnemar(
        counts["K3_only"],counts["K2_only"]
    )
    counts["alpha_predeclared"]=.05
    return counts


def summarize(records):
    out=[]
    for method in METHODS:
        rows=[r for r in records if r["method"]==method]
        if len(rows)!=N_SELECTED or len(set(r["sample"] for r in rows))!=N_SELECTED:
            raise AssertionError("incomplete policy summary")
        certified=[r for r in rows if r["online_verified"]]
        transferable=[r for r in rows if r["transfer_D4_D5"]]
        out.append({
            "method":method,"N":N_SELECTED,
            "online_certified":len(certified),
            "fresh_D4_D5_success":len(transferable),
            "conditional_fresh_transfer_rate":(
                float(len(transferable)/len(certified)) if certified else None
            ),
            "unconditional_fresh_rate":float(len(transferable)/N_SELECTED),
            "verified_local_1_deletion":sum(
                r["minimal_on_online_donors"] for r in rows),
            "mean_online_Q":float(np.mean([r["model_queries"] for r in rows])),
            "max_online_Q":max(r["model_queries"] for r in rows),
            "total_posthoc_calls":sum(r["posthoc_queries"] for r in rows),
            "mean_search_cpu_wall_s":float(np.mean([
                r["search_wall_seconds"] for r in rows])),
            "n_abstain":N_SELECTED-len(certified),
        })
    return out


def run(args):
    if (args.seed,args.trees,args.donors)!=(SEED,TREES,BANK_SIZE):
        raise ValueError("this preregistered study requires fixed seed57 trees120 bank16")
    path=Path(args.npz_path)
    raw_train,ytrain,raw_test,ytest=load_npz_dataset(path)
    with np.load(path) as z:
        labels=z["labels"].tolist()
        protocol=str(z["dataset_protocol"])
    if (raw_train.shape!=(6599,93,13) or raw_test.shape!=(2199,93,13)
        or len(np.unique(ytrain))!=10 or len(np.unique(ytest))!=10
        or "UEA SpokenArabicDigits official aeon 1.5.0 TRAIN 6599 TEST 2199"
           not in protocol):
        raise ValueError("requires the exact official SpokenArabicDigits 6599/2199 corpus")
    mu,sd,profiles,model,counter=fit_predictor(
        raw_train,ytrain,"extratrees",SEED,TREES
    )
    train=apply_standardizer(raw_train,mu,sd)
    test=apply_standardizer(raw_test,mu,sd)
    groups=make_components((93,13),BINS,"all_channels")
    if len(groups)!=BINS:
        raise AssertionError("component definition mismatch")
    bank=fit_real_donor_bank(
        train,ytrain,groups,max_donors=BANK_SIZE,seed=BANK_SEED
    )
    before=int(counter["examples"])
    original=model(test)
    baseline_calls=int(counter["examples"])-before
    if baseline_calls!=2199:
        raise AssertionError("TEST f(x) should be exactly one per TEST observation")
    screened={}
    before=int(counter["examples"])
    for label in sorted(bank.donors):
        screened[label]=np.argmax(model(bank.donors[label]),axis=1)
    screening_calls=int(counter["examples"])-before
    if screening_calls!=sum(len(x) for x in bank.donors.values()):
        raise AssertionError("TRAIN donor screening ledger mismatch")

    subset=np.sort(
        np.random.default_rng(SELECT_SEED).permutation(len(test))[:N_SELECTED]
    )
    records=[]
    online=0
    posthoc=0
    missing=0
    for sample, test_index in enumerate(subset):
        x=test[int(test_index)]
        baseline=original[int(test_index)]
        source=int(np.argmax(baseline))
        target=int(np.argsort(-baseline,kind="stable")[1])
        roles=assign_five_donor_roles(
            x,bank.donors[target],screened[target],target,
            bank.selected_train_indices[target],
        )
        if roles is None:
            missing+=1
        for method in METHODS:
            if roles is None:
                record=no_donor_result()
            else:
                online_donors=tuple(
                    bank.donors[target][int(k)] for k in roles[:3]
                )
                unseen=tuple(
                    bank.donors[target][int(k)] for k in roles[3:]
                )
                record=evaluate_method(
                    x,baseline,online_donors,unseen,
                    groups,profiles,model,counter,method
                )
            online+=record["model_queries"]
            posthoc+=record["posthoc_queries"]
            records.append({
                "sample":sample,"test_index":int(test_index),
                "method":method,"source_pred":source,"target_pred":target,
                "n_donors_online":2 if method=="K2_nearest" else 3,
                "has_five_eligible_train_donors":int(roles is not None),
                "train_donor_role_indices":(
                    ",".join(str(int(bank.selected_train_indices[target][idx]))
                             for idx in roles) if roles is not None else ""
                ),
                **record,
            })
    accounted=baseline_calls+screening_calls+online+posthoc
    if int(counter["examples"])!=accounted:
        raise AssertionError("model input counts do not reconcile")
    primary=paired_primary(records)
    table=summarize(records)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",[
        {**r,"selected":(
            ",".join(map(str,r["selected"])) if r["selected"] is not None else "")
        } for r in records
    ])
    write_csv(out/"summary.csv",table)
    manifest={
        "preregistered":"artifacts/beacon_spokenarabic_k3_donor_transfer_preregistered_2026-11-22.md",
        "dataset":"UEA SpokenArabicDigits official aeon 1.5.0",
        "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_sha256":counter["fitted_model_sha256"],
        "scaler_sha256":counter["training_standardizer_sha256"],
        "full_TEST_records":2199,"evaluated_TEST_subset":N_SELECTED,
        "TEST_subset_seed":SELECT_SEED,
        "TEST_indices":subset.tolist(),
        "train_records":6599,
        "test_classifier_accuracy":float(
            np.mean(np.argmax(original,axis=1)==ytest)
        ),
        "predeclared_model_accuracy_gate":.7,
        "test_labels_only_used_in_accuracy":True,
        "unknown_speaker_overlap":True,
        "classes":labels,"model":"extratrees",
        "model_tree_count":TREES,"model_seed":SEED,
        "donor_seed":BANK_SEED,"donor_limit_per_class":BANK_SIZE,
        "donor_roles":"D1,D2 nearest; D3 farthest; D4,D5 nearest remaining",
        "D4_D5_unseen_during_online_mask_search":True,
        "time_bins":BINS,"taper":TAPER,
        "modified_fraction_limit":SUPPORT,
        "shared_online_Q_per_policy":Q,
        "original_predictions_cached":baseline_calls,
        "offline_train_donor_screening_inputs":screening_calls,
        "online_model_inputs":online,
        "independent_D4_D5_posthoc_inputs":posthoc,
        "all_model_inputs":accounted,
        "cases_without_five_eligible_train_donors":missing,
        "primary":primary,"summary":table,
        "limitations":[
            "Online certificate only for exact selected donors",
            "D3 observed in K3 search; NOT independent held-out donor",
            "D4,D5 unseen by both policies but drawn from same TRAIN pool",
            "All claims conditioned on fixed splice and source model",
            "No causal, physical, speaker-level or population guarantees",
            "Offline donor screening not free computationally",
        ],
    }
    (out/"manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),
        encoding="utf8",
    )
    print(json.dumps({
        "accuracy":manifest["test_classifier_accuracy"],
        "adequacy":manifest["test_classifier_accuracy"]>=.7,
        "primary":primary,"summary":table,
        "online_model_inputs":online,
        "independent_posthoc_inputs":posthoc,
        "missing_donors":missing,
    },ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--seed",type=int,default=SEED)
    p.add_argument("--trees",type=int,default=TREES)
    p.add_argument("--donors",type=int,default=BANK_SIZE)
    run(p.parse_args())


if __name__=="__main__":
    main()
