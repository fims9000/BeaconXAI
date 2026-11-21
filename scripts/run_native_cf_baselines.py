#!/usr/bin/env python3
"""Pilot using *native*, pinned MultiSubSpaCE with exact black-box Q8/Q16.

Upstream implementation is unmodified; populations and anomaly scoring are
budget-adapted and therefore do NOT reproduce the publication's experiments.
All policies use one frozen classifier, original windows, target and donor.
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
sys.path.insert(0, str(ROOT))

from beaconxai.cf_path_search import search_counterfactual_path
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import apply_standardizer, load_npz_dataset
from beaconxai.native_cf_budget import (
    CountedCFModel, QueryBudgetExceeded, NoFeatureImportance, TrainingGaussianOutlier
)
from beaconxai.physical_groups import coupled_priority, uci_har_coupled_groups
from beaconxai.native_cf_projection import ProjectedCFModel, ProjectedTrainingOutlier
from beaconxai.real_donor import fit_real_donor_bank, nearest_context_index
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.verified_counterfactual import splice
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv
from scripts.run_verified_counterfactual_benchmark import gravity_residual_roughness

UPSTREAM_SHA = "4bbeae86bbc6ad02f7ee79fb550c9866682f6deb"


def run_native(x, donor, target, model, scorer, q, seed, native_type,
               *, gravity_normalizer=None):
    if gravity_normalizer is None:
        wrapper = CountedCFModel(model, x.shape, n_classes=6, budget=q)
        objective_scorer = scorer
    else:
        mu, sd = gravity_normalizer
        wrapper = ProjectedCFModel(
            model, x, mu, sd, n_classes=6, budget=q)
        objective_scorer = ProjectedTrainingOutlier(
            scorer, wrapper.projector)
    pop = {8: 2, 16: 4}[q]
    np.random.seed(seed)
    random.seed(seed)
    upstream = native_type(
        model_wrapper=wrapper, outlier_calculator=objective_scorer,
        fi_method=NoFeatureImportance(), grouped_channels_iter=0,
        individual_channels_iter=1, pruning_iter=0,
        population_size=pop, init_pct=.4, reinit=False,
        invalid_penalization=20, plausibility_objective="ios",
    )
    status = "not_found"
    result = None
    err = ""
    try:
        native = upstream.generate_counterfactual_specific(
            x_orig=x, desired_target=target, nun_example=donor
        )
        cf = wrapper.verified_result(native["cfs"], target, source=x)
        if cf is not None:
            result = cf.x
            status = "verified_target_flip"
        else:
            status = "not_found_or_unqueried_output"
    except QueryBudgetExceeded as e:
        status, err = "native_budget_exceeded", str(e)
    except (ValueError, IndexError, FloatingPointError,
            TypeError, ZeroDivisionError) as e:
        status, err = "native_algorithm_error", f"{type(e).__name__}: {e}"
    return result, status, wrapper.evaluations, wrapper.batches, pop, err


def summarize(rows):
    groups = sorted({(r["method"], r["Q"]) for r in rows})
    answer = []
    for name,q in groups:
        a = [r for r in rows if r["method"]==name and r["Q"]==q]
        subjects=sorted({r["subject"] for r in a})
        if len({(r["sample"],r["subject"]) for r in a})!=len(a):
            raise AssertionError("duplicated tested windows")
        correct=[r for r in a if r["verified"]]
        answer.append({
            "method":name, "Q":q, "windows":len(a), "subjects":len(subjects),
            "target_flips":len(correct),
            "subject_macro_flip":float(np.mean([
                np.mean([r["verified"] for r in a if r["subject"]==sub])
                for sub in subjects
            ])),
            "mean_actual_calls":float(np.mean([r["online_calls"] for r in a])),
            "native_budget_abort":sum(r["status"]=="native_budget_exceeded" for r in a),
            "native_errors":sum(r["status"]=="native_algorithm_error" for r in a),
            "median_relative_l2_if_flip":(
                float(np.median([r["relative_l2"] for r in correct]))
                if correct else None),
            "median_modified_fraction_if_flip":(
                float(np.median([r["modified_fraction"] for r in correct]))
                if correct else None),
            "median_gravity_roughness_if_flip":(
                float(np.median([r["gravity_roughness"] for r in correct]))
                if correct else None),
        })
    return answer


def run(args):
    checkout=Path(args.upstream).resolve()
    if not (checkout/"methods/MultiSubSpaCECF.py").is_file():
        raise FileNotFoundError("genuine upstream MultiSubSpaCECF.py required")
    sha=subprocess.check_output(
        ["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
    if sha!=UPSTREAM_SHA:
        raise ValueError(f"unapproved upstream revision: {sha}")
    sys.path.insert(0,str(checkout))
    from methods.MultiSubSpaCECF import MultiSubSpaCECF

    qs=sorted(set(int(z) for z in args.budgets.split(",")))
    if not qs or not set(qs).issubset({8,16}):
        raise ValueError("pilot supports exact Q8 and Q16")
    data=Path(args.npz_path)
    xtr,ytr,xte,yte=load_npz_dataset(data)
    with np.load(data) as z:
        st=np.asarray(z["subject_train"],dtype=int)
        sv=np.asarray(z["subject_test"],dtype=int)
    if set(st)&set(sv) or xtr.shape[1:]!=(128,9):
        raise ValueError("wrong UCI HAR shape or participant leakage")
    mu,sd,profiles,model,counter=fit_predictor(
        xtr,ytr,args.model,args.seed,args.trees
    )
    train=apply_standardizer(xtr,mu,sd)
    outliers=TrainingGaussianOutlier(train)
    comps=make_components(train.shape[1:],12,"pamap3")
    linked=uci_har_coupled_groups(comps,gravity_preserving=True,mean=mu,scale=sd)
    bank=fit_real_donor_bank(
        train,ytr,comps,max_donors=args.donors,seed=args.seed+10
    )
    inds=balanced_subject_sample(sv,args.max_test,args.seed+301)
    test=apply_standardizer(xte[inds],mu,sd)
    bases=model(test)
    records=[]
    for i,x in enumerate(test):
        base=bases[i]
        source=int(np.argmax(base))
        target=int(np.argsort(-base,kind="stable")[1])
        di=nearest_context_index(x,0,bank,target,comps)
        donor=bank.donors[target][di]
        prior=contrastive_prior(x,source,target,profiles,comps)
        ranked=np.argsort(-np.abs(prior),kind="stable").astype(int).tolist()
        linked_rank=np.argsort(
            -coupled_priority(prior,comps),kind="stable").astype(int).tolist()
        old_gravity=gravity_residual_roughness(x,mu,sd)
        for q in qs:
            for method in ("beacon_independent","beacon_gravity","native_multisubspace"):
                start=int(counter["examples"])
                population=None
                err=""
                if method=="native_multisubspace":
                    edited,status,count,batches,population,err=run_native(
                        x,donor,target,model,outliers,q,
                        args.seed*100003+int(inds[i]),MultiSubSpaCECF
                    )
                else:
                    atoms=linked if method=="beacon_gravity" else comps
                    ranking=linked_rank if method=="beacon_gravity" else ranked
                    result=search_counterfactual_path(
                        x,base,donor,atoms,ranking,model,
                        target_class=target,budget=q,max_changed_components=8,
                        seed_singletons=3,taper=args.taper
                    )
                    edited=(splice(x,donor,atoms,result.verified_edit,taper=args.taper)
                            if result.is_valid else None)
                    status=result.status
                    count=result.calls
                    batches=count
                actual=int(counter["examples"])-start
                if actual!=count or actual>q:
                    raise AssertionError("black-box query count did not match Q")
                relative=(float(np.linalg.norm(edited-x)/
                                max(np.linalg.norm(x),np.linalg.norm(donor),1e-8))
                          if edited is not None else None)
                changed=(float(np.mean(np.abs(edited-x)>1e-8))
                         if edited is not None else None)
                rough=(float(gravity_residual_roughness(edited,mu,sd)/
                             max(old_gravity,1e-6))
                       if edited is not None else None)
                records.append({
                    "sample":i, "official_test_index":int(inds[i]),
                    "subject":int(sv[inds[i]]), "source":source,
                    "target":target, "method":method, "Q":q,
                    "verified":int(edited is not None), "status":status,
                    "online_calls":actual, "model_batches":batches,
                    "native_population":population, "native_error":err,
                    "relative_l2":relative, "modified_fraction":changed,
                    "gravity_roughness":rough,
                    "original_correct_offline":int(source==int(yte[inds[i]])),
                    "donor_train_index":int(bank.selected_train_indices[target][di]),
                })
    online=sum(r["online_calls"] for r in records)
    if counter["examples"]!=len(test)+online:
        raise AssertionError("unaccounted model evaluations")
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    summary=summarize(records)
    write_csv(out/"per_sample.csv",records)
    write_csv(out/"summary.csv",summary)
    manifest={
        "upstream":"MarioRefoyo/FastPACE", "sha":UPSTREAM_SHA,
        "native_source_file":"methods/MultiSubSpaCECF.py",
        "native_algorithm_code_unmodified":True,
        "reproduction_of_paper_settings":False,
        "configuration":{
            "feature_importance":"NoneFI (zeros), no learned gradients",
            "outlier_score":"train-only squared standardized deviation, NOT original autoencoder",
            "native_stage":"individual channels only, one iteration",
            "Q8_population_size":2,"Q16_population_size":4,
            "reinit":False,"invalid_penalty":20,
        },
        "population_default_is_100_and_exceeds_Q16":True,
        "same_frozen_classifier_x_donor_target_for_all_methods":True,
        "frozen_model_sha256":counter["fitted_model_sha256"],
        "npz_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),
        "precomputed_baseline_calls":len(test),
        "additional_online_queries":online,
        "total_model_evaluations":counter["examples"],
        "Q_maximum":qs,
        "native_candidate_requires_actual_model_evaluation":True,
        "physical_constraints_different_between_methods":True,
        "baseline_is_budget_adapted_pilot_not_full_SOTA_comparison":True,
        "limitations":[
            "Native MultiSubSpaCE with original population 100 cannot run at Q8/Q16",
            "Changed FI, outlier and evolutionary hyperparameters preclude paper-level SOTA claim",
            "Unrealistic independent-channel edits may break physical acceleration consistency",
            "No multi-donor audit in this pilot; fixed-target validity only",
            "Single dataset and model seed",
        ],
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"model_sha":counter["fitted_model_sha256"],
                      "summary":summary},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",required=True,type=Path)
    p.add_argument("--upstream",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--model",choices=("raw_mlp","extratrees"),default="raw_mlp")
    p.add_argument("--budgets",default="8,16")
    p.add_argument("--max-test",type=int,default=48)
    p.add_argument("--donors",type=int,default=8)
    p.add_argument("--taper",type=int,default=2)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--seed",type=int,default=57)
    run(p.parse_args())


if __name__=="__main__":
    main()
