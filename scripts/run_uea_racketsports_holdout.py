#!/usr/bin/env python3
"""Pre-registered RacketSports/Q16 donor-stability holdout.

Read artifacts/beacon_racketsports_preregistered_2026-11-21.md.
Never use TEST labels or alternate donor responses in source/target choice,
donor ranking, Q16 online decisions, or parameter calibration.
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
    rank_confirmed_donors, joint_quality, component_jaccard,
    exact_two_sided_mcnemar,
)
from beaconxai.cf_path_search import (
    search_counterfactual_path, search_backward_counterfactual,
)
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import apply_standardizer, load_npz_dataset
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.verified_counterfactual import splice
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv

BUDGET=16
N_BINS=10
SUPPORT=.35
SEED=57
TREES=120
DONORS=16
TAPER=2
METHODS=("forward","backward")
DONOR_IDS=("D1","D2")


def measure_one(x,base,donor,components,order,model,method):
    before=measure_one.model_counter["examples"]
    if method=="forward":
        result=search_counterfactual_path(
            x,base,donor,components,order,model,
            target_class=int(np.argsort(-base,kind="stable")[1]),
            budget=BUDGET, max_changed_components=len(components),
            seed_singletons=3, taper=TAPER,
        )
    elif method=="backward":
        result=search_backward_counterfactual(
            x,base,donor,components,list(reversed(order)),model,
            target_class=int(np.argsort(-base,kind="stable")[1]),
            budget=BUDGET,taper=TAPER,
        )
    else:
        raise ValueError("unregistered method")
    used=int(measure_one.model_counter["examples"])-before
    if used!=result.calls or not 0<=used<=BUDGET:
        raise AssertionError("online model evaluation and Q16 counters disagree")
    selected=result.verified_edit if result.is_valid else None
    if selected is not None:
        if not any(
            p.targeted_flip and tuple(p.components)==tuple(sorted(selected))
            for p in result.probes
        ):
            raise AssertionError("unmeasured/unverified candidate was returned")
        candidate=splice(x,donor,components,selected,taper=TAPER)
        fraction=float(np.mean(np.abs(candidate-x)>1e-8))
        l2=float(np.linalg.norm(candidate-x)/
                 max(np.linalg.norm(x),np.linalg.norm(donor),1e-8))
    else:
        fraction=None
        l2=None
    return {
        "status":result.status,"verified":int(result.is_valid),
        "model_queries":used,
        "selected":tuple(selected) if selected is not None else None,
        "modified_fraction":fraction,
        "relative_l2":l2,
        "local_minimality_checked":int(result.local_minimality_checked and result.is_valid),
        "joint_primary":int(joint_quality(
            result.is_valid,fraction,result.local_minimality_checked,SUPPORT
        )),
    }


def no_donors_record():
    return {
        "status":"less_than_two_model_confirmed_train_donors",
        "verified":0,"model_queries":0,"selected":None,
        "modified_fraction":None,"relative_l2":None,
        "local_minimality_checked":0,"joint_primary":0,
    }


def summarize(records):
    rows=[]
    for donor_name in DONOR_IDS:
        for method in METHODS:
            sample=[r for r in records if r["donor"]==donor_name and r["method"]==method]
            if len(sample)!=152 or len({r["sample"] for r in sample})!=152:
                raise AssertionError("missing a registered test window")
            success=[r for r in sample if r["verified"]]
            rows.append({
                "donor":donor_name,"method":method,
                "n_test":len(sample),
                "target_flips":len(success),
                "with_modified_fraction_le_20":sum(
                    r["modified_fraction"]<=.20 for r in success),
                "with_modified_fraction_le_35":sum(
                    r["modified_fraction"]<=.35 for r in success),
                "with_modified_fraction_le_50":sum(
                    r["modified_fraction"]<=.50 for r in success),
                "verified_local_minimal":sum(
                    r["local_minimality_checked"] for r in success),
                "joint_verified_compact_local_minimal":sum(
                    r["joint_primary"] for r in sample),
                "mean_online_model_queries":float(np.mean(
                    [r["model_queries"] for r in sample])),
                "median_fraction_valid_only":(
                    float(np.median([r["modified_fraction"] for r in success]))
                    if success else None),
            })
    return rows


def paired_primary(records):
    mapped={(r["sample"],r["method"],r["donor"]):r for r in records}
    backward_only=forward_only=both=neither=0
    for i in range(152):
        a=mapped[i,"forward","D1"]["joint_primary"]
        b=mapped[i,"backward","D1"]["joint_primary"]
        if a and b:
            both+=1
        elif b:
            backward_only+=1
        elif a:
            forward_only+=1
        else:
            neither+=1
    if both+backward_only+forward_only+neither!=152:
        raise AssertionError("incomplete paired set")
    return {
        "n":152,
        "both_joint_success":both,
        "backward_only_joint_success":backward_only,
        "forward_only_joint_success":forward_only,
        "neither_joint_success":neither,
        "forward_joint":both+forward_only,
        "backward_joint":both+backward_only,
        "effect_backward_minus_forward_pp":100.*(
            backward_only-forward_only)/152,
        "mcnemar_exact_twosided_p":exact_two_sided_mcnemar(
            backward_only,forward_only),
        "fixed_hypothesis":"D1 target B + fraction <= 35% + final 1-deletion check",
        "alpha_preregistered":.05,
    }


def run(args):
    if args.seed!=SEED or args.trees!=TREES or args.donors!=DONORS:
        raise ValueError("holdout parameters frozen: seed=57 trees=120 donors=16")
    path=Path(args.npz_path)
    train_raw,ytrain,test_raw,ytest=load_npz_dataset(path)
    with np.load(path) as z:
        labels=z["labels"].tolist()
        protocol=str(z["dataset_protocol"])
    if (train_raw.shape!=(151,30,6) or test_raw.shape!=(152,30,6)
        or len(set(ytrain))!=4 or len(set(ytest))!=4
        or "RacketSports original TRAIN 151 / TEST 152" not in protocol):
        raise ValueError("official untouched RacketSports split required")

    mu,sd,profiles,predict,counted=fit_predictor(
        train_raw,ytrain,"extratrees",SEED,TREES
    )
    measure_one.model_counter=counted
    train=apply_standardizer(train_raw,mu,sd)
    test=apply_standardizer(test_raw,mu,sd)
    components=make_components((30,6),N_BINS,"all_channels")
    if len(components)!=10:
        raise AssertionError("preregistered partition mismatch")
    bank=fit_real_donor_bank(
        train,ytrain,components,max_donors=DONORS,seed=67
    )
    # One cached source evaluation per TEST window, not added to each Q16.
    baseline=predict(test)
    baseline_calls=int(counted["examples"])
    if baseline_calls!=152:
        raise AssertionError("baseline model calls undercounted")

    screened={}
    for lab in sorted(bank.donors):
        screened[lab]=np.argmax(predict(bank.donors[lab]),axis=1)
    train_screening_calls=int(counted["examples"])-baseline_calls
    if train_screening_calls!=sum(len(v) for v in bank.donors.values()):
        raise AssertionError("offline train donor screenings undercounted")

    records=[]
    transfers=[]
    cross=[]
    missing=0
    query_sum=0
    audit_sum=0

    for i,x in enumerate(test):
        source=int(np.argmax(baseline[i]))
        target=int(np.argsort(-baseline[i],kind="stable")[1])
        rank=np.argsort(
            -np.abs(contrastive_prior(x,source,target,profiles,components)),
            kind="stable"
        ).astype(int).tolist()
        donor_pool=bank.donors[target]
        ranked=rank_confirmed_donors(
            x,donor_pool,screened[target],target,
            bank.selected_train_indices[target]
        )
        has_two=len(ranked)>=2
        if not has_two:
            missing+=1
        per={}
        for donor_name,ordinal in (("D1",0),("D2",1)):
            idx=ranked[ordinal] if has_two else None
            for method in METHODS:
                result=(
                    measure_one(x,baseline[i],donor_pool[idx],
                                components,rank,predict,method)
                    if has_two else no_donors_record()
                )
                query_sum+=result["model_queries"]
                row={
                    "sample":i,"method":method,"donor":donor_name,
                    "source_pred":source,"target_pred":target,
                    "eligible_donors":len(ranked),
                    "selected_train_id":(
                        int(bank.selected_train_indices[target][idx])
                        if idx is not None else None
                    ),
                    **result,
                }
                records.append(row)
                per[donor_name,method]=row

        for method in METHODS:
            d1=per["D1",method]
            d2=per["D2",method]
            stable=None
            compact=None
            audit_calls=0
            if has_two and d1["verified"]:
                # Only an OUT-OF-BUDGET, never-fed-back audit.
                # D2 is chosen before D1 search, not tuned on success.
                transferred=splice(
                    x,donor_pool[ranked[1]],components,d1["selected"],
                    taper=TAPER,
                )
                before=int(counted["examples"])
                prediction=predict(transferred[None])
                audit_calls=int(counted["examples"])-before
                if audit_calls!=1:
                    raise AssertionError("donor transfer is one *extra* query")
                stable=int(np.argmax(prediction[0])==target)
                compact=int(np.mean(np.abs(transferred-x)>1e-8)<=SUPPORT)
            audit_sum+=audit_calls
            transfers.append({
                "sample":i,"method":method,
                "d1_verified":d1["verified"],
                "d2_tested":int(audit_calls==1),
                "transfer_target_flip":stable,
                "transfer_fraction_le35":compact,
                "transfer_joint_target_and_compact":(
                    int(bool(stable and compact)) if stable is not None else None
                ),
                "offline_transfer_queries":audit_calls,
                "offline_not_in_online_Q16":True,
            })
            both=bool(d1["verified"] and d2["verified"])
            cross.append({
                "sample":i,"method":method,
                "d1_verified":d1["verified"],
                "d2_verified":d2["verified"],
                "d1_joint":d1["joint_primary"],
                "d2_joint":d2["joint_primary"],
                "both_verified":int(both),
                "component_jaccard":(
                    component_jaccard(d1["selected"],d2["selected"])
                    if both else None
                ),
            })

    if len(records)!=152*2*2 or len(transfers)!=152*2:
        raise AssertionError("missing registered rows")
    expected=baseline_calls+train_screening_calls+query_sum+audit_sum
    if int(counted["examples"])!=expected:
        raise AssertionError("uncounted online or audit model evaluations")

    summary=summarize(records)
    primary=paired_primary(records)
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    # Avoid dict tuples in CSV; preserving the precise selected group IDs.
    serial=[{
        **r,"selected":(
            ",".join(map(str,r["selected"])) if r["selected"] is not None else ""
        )
    } for r in records]
    write_csv(output/"per_sample.csv",serial)
    write_csv(output/"summary.csv",summary)
    write_csv(output/"offline_donor_transfer.csv",transfers)
    write_csv(output/"paired_donor_runs.csv",cross)

    stability={}
    for method in METHODS:
        cc=[r for r in cross if r["method"]==method]
        tt=[r for r in transfers if r["method"]==method]
        both=[r["component_jaccard"] for r in cc if r["both_verified"]]
        audits=[r for r in tt if r["d2_tested"]]
        stability[method]={
            "n_all_test":152,
            "n_both_donors_target_verified":len(both),
            "n_d1_target_verified_and_transfer_audited":len(audits),
            "n_transferred_target_flip":sum(r["transfer_target_flip"] for r in audits),
            "n_transferred_target_flip_and_le35":sum(
                r["transfer_joint_target_and_compact"] for r in audits),
            "median_block_jaccard_given_both_valid":(
                float(np.median(both)) if both else None),
            "n_joint_d1_and_d2":sum(
                r["d1_joint"] and r["d2_joint"] for r in cc),
            "n_joint_d1_not_d2":sum(
                r["d1_joint"] and not r["d2_joint"] for r in cc),
            "n_joint_d2_not_d1":sum(
                r["d2_joint"] and not r["d1_joint"] for r in cc),
        }

    accuracy=float(np.mean(np.argmax(baseline,axis=1)==ytest))
    manifest={
        "protocol_path":"artifacts/beacon_racketsports_preregistered_2026-11-21.md",
        "protocol_sha_commit":"40e57e9cf5d638f2e710c0aa4f90fb04ed7ab29e",
        "dataset":"UEA RacketSports, original split 151/152",
        "dataset_source":"aeon.datasets.load_classification",
        "official_participant_disjoint_verified":False,
        "train_shape":list(train.shape),"test_shape":list(test.shape),
        "train_labels":labels,
        "model":"extratrees","seed":SEED,"trees":TREES,
        "model_sha256":counted["fitted_model_sha256"],
        "scaler_sha256":counted["training_standardizer_sha256"],
        "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "donor_reservoir_seed":67,"donors_max_per_class":DONORS,
        "donor_select_rule":"two closest distinct model-predicted B TRAIN donors",
        "n_without_two_donors":missing,
        "blocks":N_BINS,"taper":TAPER,"Q_per_run":BUDGET,
        "max_support":SUPPORT,
        "cached_source_model_evaluations":baseline_calls,
        "offline_train_donor_screening_evaluations":train_screening_calls,
        "online_search_evaluations_all_donors_methods":query_sum,
        "offline_transfer_evaluations_excluded_from_each_Q16":audit_sum,
        "total_model_evaluations":int(counted["examples"]),
        "test_labels_only_used_for_descriptive_accuracy":True,
        "base_accuracy":accuracy,
        "predeclared_minimum_accuracy_for_interpretability":.70,
        "model_adequacy_gate_passed":accuracy>=.70,
        "statistical_scope":"one predeclared paired D1 joint outcome; D2 descriptive",
        "important_limits":[
            "TRAIN and TEST may contain the same people; participant IDs unavailable",
            "Two donors are distinct TRAIN observations but not statistically independent cohorts",
            "Counterfactual validity is a frozen classifier output, NOT causal truth",
            "Single-block removal minimality is donor/operator relative, NOT global",
            "Donor transfer replay uses an additional OFFLINE model query",
            "Search algorithms see only D1 or D2 in their own independent Q16 run",
            "No tuning using RacketSports TEST, including after a negative result",
        ],
        "primary":primary,
        "donor_stability":stability,
    }
    (output/"manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({
        "base_accuracy":accuracy,"adequacy_gate":accuracy>=.70,
        "primary":primary,"summary":summary,
        "donor_stability":stability,"missing_two_donors":missing,
    },indent=2,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=57)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--donors",type=int,default=16)
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
