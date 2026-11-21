#!/usr/bin/env python3
"""Probe actual classifier-query cost of AUTHENTIC pinned FastPACE HCEM.

The H-CEM implementation and source hyperparameters come from upstream
hcem_final.json. No pretrained policy is claimed: FastPACECF.fit raises
NotImplementedError. Q limits are enforced on all evaluated input examples,
including resets, vectorized planner simulations and final scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_path_search import search_counterfactual_path
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import apply_standardizer, load_npz_dataset
from beaconxai.native_cf_budget import QueryBudgetExceeded
from beaconxai.native_fastpace_meter import MeteredFastPACEModel, FastPACEOutlier
from beaconxai.real_donor import fit_real_donor_bank, nearest_context_index
from beaconxai.subject_sampling import balanced_subject_sample
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv

UPSTREAM_SHA = "4bbeae86bbc6ad02f7ee79fb550c9866682f6deb"

CEM_PARAMS = {
    "elite_frac": .1, "alpha": .75,
    "warm_alpha": 0., "hierarchy_warm_alpha": .25,
    "use_conditional_reuse": False, "tabu_mode": True,
    "train_num_simulations_ratio": 2., "inf_num_simulations_ratio": 2.,
    "train_planning_steps": 3, "inf_planning_steps": 3,
    "train_cem_iters": 3, "inf_cem_iters": 3,
    "train_plan_every": 3, "inf_plan_every": 3,
    "search_target": "fitness_weighted",
    "train_target": None, "train_target_type": None,
    "train_pisoft_func": None, "temp": 1.,
    "planning_credit": False, "block_non_adj_mask": True,
}


def instantiate_fastpace(native_class, wrapper, outliers, train_nct,
                         donor_nct, plan_calls: int, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    dummy_nuns=np.repeat(donor_nct[None,None],len(train_nct),axis=0)
    explainer = native_class(
        model_wrapper=wrapper, outlier_calculator=outliers,
        X_train=train_nct, nuns=dummy_nuns,
        latent_ts_block_pcts=[.2,.2,.05],
        latent_ts_block_train_pct=None,
        ch_block_pcts=[.2,.05,.05],
        ch_block_train_pct=None, pruning_ch_block_pct=None,
        channel_groups="HierarchicalClusteringFixedK",
        standardize=False, ch_similarity="pearson",
        reward_type="incremental", non_valid_penalization=-1,
        weight_losses=[.10,.32,.48,0.], mask_init="ones",
        max_steps=50, algorithm_params=CEM_PARAMS,
        device="cpu", tensorboard_path=None,
        cf_max_plan_calls=plan_calls,
    )
    # Upstream RandomState() is otherwise unseeded; seed it explicitly
    # without replacing any search, fitness or evolution method.
    for idx, cem in explainer.agent.level_CEMNNs.items():
        cem.rng.seed(seed+int(idx))
    return explainer


def run_fastpace(native_class, model_func, train_nct, outliers,
                 x, donor, target, cap, plan_calls, seed):
    model=MeteredFastPACEModel(
        model_func, x.shape[1], x.shape[0], 6, max_evaluations=cap
    )
    status="uninitialized"
    error=""
    result=None
    rejected_batch=None
    try:
        explainer=instantiate_fastpace(
            native_class,model,outliers,train_nct,
            donor.T.astype(np.float32),plan_calls,seed
        )
        answer=explainer.generate_counterfactual_specific(
            x_orig=x.T.astype(np.float32),
            desired_target=target,
            nun_example=donor.T.astype(np.float32)[None]
        )
        candidate=np.asarray(answer["cfs"],dtype=np.float32)
        if model.was_measured_target(candidate,target):
            result=candidate.T
            status="target_measured"
        else:
            status="unconfirmed_result"
    except QueryBudgetExceeded as e:
        status="hard_Q_exceeded"
        error=str(e)
        rejected_batch=e.requested
    except Exception as e:
        status="upstream_exception"
        error=f"{type(e).__name__}: {str(e)[:350]}"
    return {
        "status":status, "exception":error,
        "Q_max":cap,"planning_call_cap":plan_calls,
        "online_model_evaluations":model.evaluations,
        "model_batches":model.batches,
        "largest_model_batch":model.largest_batch,
        "rejected_batch":rejected_batch,
        "first_batches":",".join(map(str,model.batch_sizes[:40])),
        "target_measured":int(result is not None),
        "modified_fraction":(
            float(np.mean(np.abs(result-x)>1e-8))
            if result is not None else None
        ),
        "full_training_donor":(
            int(np.allclose(result,donor,rtol=1e-6,atol=1e-6))
            if result is not None else 0
        ),
    }


def run(args):
    upstream=Path(args.upstream).resolve()
    if not (upstream/"methods/FastPACECF.py").is_file():
        raise FileNotFoundError("native FastPACE source missing")
    sha=subprocess.check_output(
        ["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    if sha!=UPSTREAM_SHA:
        raise ValueError("upstream SHA mismatch")
    sys.path.insert(0,str(upstream))
    from methods.FastPACECF import FastPACECF

    path=Path(args.npz_path)
    raw_train,y_train,raw_test,y_test=load_npz_dataset(path)
    with np.load(path) as z:
        sid_train=np.asarray(z["subject_train"],dtype=int)
        sid_test=np.asarray(z["subject_test"],dtype=int)
        protocol=(str(z["dataset_protocol"]) if
                  "dataset_protocol" in z.files else "UCI HAR official")
    if set(sid_train)&set(sid_test) or raw_train.shape[1:]!=(128,9):
        raise ValueError("subject overlap or wrong dataset dimensions")
    mu,sd,profiles,model,calls=fit_predictor(
        raw_train,y_train,args.model,args.seed,args.trees
    )
    train=apply_standardizer(raw_train,mu,sd)
    train_nct=np.ascontiguousarray(train.transpose(0,2,1),dtype=np.float32)
    outliers=FastPACEOutlier(train)
    comps=make_components(train.shape[1:],12,"pamap3")
    bank=fit_real_donor_bank(
        train,y_train,comps,max_donors=args.donors,seed=args.seed+10
    )
    chosen=balanced_subject_sample(sid_test,args.max_test,args.seed+301)
    test=apply_standardizer(raw_test[chosen],mu,sd)
    base=model(test)
    # Offline train-only check of donor *model predictions*. FastPACE uses
    # the predicted NUN class, which must equal our predeclared target B.
    donor_classes={}
    for cls in sorted(bank.donors):
        donor_classes[cls]=np.argmax(model(np.stack(bank.donors[cls])),axis=1)
    n_train_donor_checks=sum(len(bank.donors[k]) for k in bank.donors)
    caps=sorted({int(z) for z in args.query_caps.split(",")})
    if not caps or min(caps)<1 or args.max_plan_calls<1:
        raise ValueError("caps and planning steps must be positive")
    rows=[]
    no_eligible=0
    for i,x in enumerate(test):
        source=int(np.argmax(base[i]))
        target=int(np.argsort(-base[i],kind="stable")[1])
        eligible=np.flatnonzero(donor_classes[target]==target)
        if not len(eligible):
            no_eligible+=1
            continue
        anchor=nearest_context_index(x,0,bank,target,comps)
        idx=(anchor if anchor in eligible else int(eligible[
            np.argmin(np.abs(eligible-anchor))
        ]))
        donor=np.asarray(bank.donors[target][idx],dtype=float)
        for cap in caps:
            before=int(calls["examples"])
            measured=run_fastpace(
                FastPACECF,model,train_nct,outliers,
                x,donor,target,cap,args.max_plan_calls,
                seed=args.seed*100003+int(chosen[i])
            )
            used=int(calls["examples"])-before
            if used!=measured["online_model_evaluations"] or used>cap:
                raise AssertionError("model evaluation counter mismatch")
            rows.append({
                "sample":i,"test_index":int(chosen[i]),
                "subject":int(sid_test[chosen[i]]),
                "method":"native_fastpace_hcem",
                "source":source,"target":target,
                "donor_bank_index":idx,
                "donor_train_index":int(bank.selected_train_indices[target][idx]),
                **measured,
            })
        if args.beacon_reference:
            prior=contrastive_prior(x,source,target,profiles,comps)
            rank=np.argsort(-np.abs(prior),kind="stable").tolist()
            before=int(calls["examples"])
            cf=search_counterfactual_path(
                x,base[i],donor,comps,rank,model,target_class=target,
                budget=16,max_changed_components=8,
                seed_singletons=3,taper=2
            )
            used=int(calls["examples"])-before
            if used!=cf.calls or used>16:
                raise AssertionError("BEACON Q16 count mismatch")
            rows.append({
                "sample":i,"test_index":int(chosen[i]),
                "subject":int(sid_test[chosen[i]]),
                "method":"beacon_q16", "source":source,"target":target,
                "Q_max":16,"planning_call_cap":None,
                "online_model_evaluations":used,
                "target_measured":int(cf.is_valid),"status":cf.status,
                "full_training_donor":0,
            })
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",rows)
    summary=[]
    for cap in caps:
        group=[r for r in rows if r["method"]=="native_fastpace_hcem"
               and r["Q_max"]==cap]
        summary.append({
            "Q_max":cap, "windows":len(group),
            "confirmed_target_flips":sum(r["target_measured"] for r in group),
            "full_training_donor":sum(r["full_training_donor"] for r in group),
            "hard_budget_exceeded":sum(r["status"]=="hard_Q_exceeded" for r in group),
            "upstream_exceptions":sum(r["status"]=="upstream_exception" for r in group),
            "mean_online_model_evaluations":(
                float(np.mean([r["online_model_evaluations"] for r in group]))
                if group else None),
            "maximum_native_batch":max((r["largest_model_batch"] for r in group),default=0),
        })
    write_csv(out/"summary.csv",summary)
    manifest={
        "upstream":"MarioRefoyo/FastPACE/methods/FastPACECF.py",
        "upstream_SHA":UPSTREAM_SHA,
        "config":"experiments/params_cf/hcem_final.json",
        "no_plausibility_weight":True,
        "CEM_algorithm_source_unchanged":True,
        "FastPACECF_fit_is_NotImplementedError":True,
        "pretrained_agent_claim":False,
        "plan_call_cap":args.max_plan_calls,
        "per_input_model_evaluation_caps":caps,
        "trained_classifier_sha":calls["fitted_model_sha256"],
        "dataset_sha":hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset_protocol":protocol,
        "n_test_windows_sampled":len(chosen),
        "n_windows_without_model_predicted_target_donor":no_eligible,
        "train_only_donor_class_checks_offline":n_train_donor_checks,
        "cached_original_window_calls":len(test),
        "all_model_evaluations_including_offline":calls["examples"],
        "CPU_execution":True,
        "native_RNG_seeded_for_reproducibility":True,
        "limits":[
            "Upstream planner is not adapted for Q8/Q16; oversize native batches are rejected atomically",
            "Published black-box classifier and outlier model not reproduced",
            "Only selected UCI HAR windows; corpus studied repeatedly",
            "Planning-call cap may be below original setting 50",
            "Initial mask 'ones' can return entire training donor; explicitly flagged",
            "No guarantees of physical realism or counterfactual minimality",
        ],
    }
    (out/"manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({"summary":summary,"no_eligible":no_eligible},
                     ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",required=True,type=Path)
    p.add_argument("--upstream",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--model",choices=("raw_mlp","extratrees"),default="raw_mlp")
    p.add_argument("--query-caps",default="16,2048")
    p.add_argument("--max-plan-calls",type=int,default=1)
    p.add_argument("--max-test",type=int,default=3)
    p.add_argument("--donors",type=int,default=8)
    p.add_argument("--seed",type=int,default=57)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--beacon-reference",action="store_true")
    run(p.parse_args())


if __name__=="__main__":
    main()
