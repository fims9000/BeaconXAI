#!/usr/bin/env python3
"""Matched UCI HAR falsification: native MultiSubSpaCE vs gravity projection.

Native upstream produces actual model-confirmed counterfactuals. For each
such candidate, project it to the closest (raw-unit L2) edit with exactly
preserved source total_acc-body_acc; verify the target class with one more
black-box evaluation INSIDE the same Q8/Q16 budget.

The upstream algorithm is unchanged, but trained outlier scorer and search
hyperparameters are budget-adapted. The projection is an additional
postprocessing *ablation*, not part of the original MultiSubSpaCE method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.cf_path_search import search_counterfactual_path
from beaconxai.cf_physical_projection import project_uci_har_preserve_gravity
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import apply_standardizer, load_npz_dataset
from beaconxai.native_cf_budget import TrainingGaussianOutlier
from beaconxai.physical_groups import coupled_priority, uci_har_coupled_groups
from beaconxai.real_donor import fit_real_donor_bank, nearest_context_index
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.verified_counterfactual import splice
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_native_cf_baselines import UPSTREAM_SHA, run_native
from scripts.run_q16_matched_ablation import write_csv
from scripts.run_verified_counterfactual_benchmark import gravity_residual_roughness


def summary(rows):
    out=[]
    for method,q in sorted({(r["method"],r["Q"]) for r in rows}):
        g=[r for r in rows if r["method"]==method and r["Q"]==q]
        successful=[r for r in g if r["verified"]]
        subjects=sorted({r["subject"] for r in g})
        out.append({
            "method":method, "Q":q, "windows":len(g),
            "success":len(successful), "n_subjects":len(subjects),
            "subject_macro_success":float(np.mean([
                np.mean([r["verified"] for r in g if r["subject"]==sub])
                for sub in subjects
            ])),
            "mean_online_calls_if_run_alone":float(np.mean([
                r["online_calls_if_run_alone"] for r in g])),
            "median_gravity_ratio_valid_only":(
                float(np.median([r["gravity_ratio"] for r in successful]))
                if successful else None),
            "median_edited_fraction_valid_only":(
                float(np.median([r["edited_fraction"] for r in successful]))
                if successful else None),
            "median_relative_l2_valid_only":(
                float(np.median([r["relative_l2"] for r in successful]))
                if successful else None),
        })
    return out


def run(args):
    upstream=Path(args.upstream).resolve()
    if not (upstream/"methods/MultiSubSpaCECF.py").is_file():
        raise FileNotFoundError("pinned upstream native source missing")
    sha=subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],
                                text=True).strip()
    if sha!=UPSTREAM_SHA:
        raise ValueError("upstream source SHA mismatch; refuse unverified comparison")
    sys.path.insert(0,str(upstream))
    from methods.MultiSubSpaCECF import MultiSubSpaCECF

    q_values=sorted({int(s) for s in args.budgets.split(",")})
    if not q_values or not set(q_values).issubset({8,16}):
        raise ValueError("only Q8/Q16 audited")
    data=Path(args.npz_path)
    xtr,ytr,xte,yte=load_npz_dataset(data)
    with np.load(data) as z:
        strn=np.asarray(z["subject_train"],dtype=int)
        ste=np.asarray(z["subject_test"],dtype=int)
        protocol=(str(z["dataset_protocol"]) if "dataset_protocol" in z.files
                  else "UCI HAR official subject-disjoint test split")
    if set(strn)&set(ste) or xtr.shape[1:]!=(128,9):
        raise ValueError("subject leakage or non-UCI input")
    mu,sd,profiles,predict,calls=fit_predictor(
        xtr,ytr,args.model,args.seed,args.trees)
    train=apply_standardizer(xtr,mu,sd)
    scorer=TrainingGaussianOutlier(train)
    components=make_components(train.shape[1:],12,"pamap3")
    linked=uci_har_coupled_groups(components,gravity_preserving=True,
                                  mean=mu,scale=sd)
    bank=fit_real_donor_bank(train,ytr,components,
                             max_donors=args.donors,seed=args.seed+10)
    selected=balanced_subject_sample(ste,args.max_test,args.seed+301)
    test=apply_standardizer(xte[selected],mu,sd)
    baselines=predict(test)
    records=[]
    actual_online=0
    for i,x in enumerate(test):
        base=baselines[i]
        source=int(np.argmax(base))
        target=int(np.argsort(-base,kind="stable")[1])
        donor_i=nearest_context_index(x,0,bank,target,components)
        donor=bank.donors[target][donor_i]
        prior=contrastive_prior(x,source,target,profiles,components)
        ranking=np.argsort(-coupled_priority(prior,components),
                           kind="stable").astype(int).tolist()
        original_roughness=gravity_residual_roughness(x,mu,sd)
        for q in q_values:
            before=int(calls["examples"])
            native,status,native_calls,batches,pop,error=run_native(
                x,donor,target,predict,scorer,q,
                args.seed*100003+int(selected[i]),MultiSubSpaCECF)
            if int(calls["examples"])-before!=native_calls:
                raise AssertionError("native model query mismatch")
            actual_online+=native_calls
            projected=None
            projected_status="native_not_valid"
            extra=0
            if native is not None:
                projected=project_uci_har_preserve_gravity(x,native,mu,sd)
                if native_calls+1>q:
                    raise AssertionError("projection must be within Q")
                projected_score=predict(projected[None])
                extra=1
                actual_online+=1
                projected_status=("verified_target_flip" if
                                  int(np.argmax(projected_score[0]))==target
                                  else "projection_lost_target_flip")
            native_valid=bool(native is not None)
            projected_valid=projected is not None and projected_status=="verified_target_flip"

            beacon_before=int(calls["examples"])
            b=search_counterfactual_path(
                x,base,donor,linked,ranking,predict,target_class=target,
                budget=q,max_changed_components=8,seed_singletons=3,taper=args.taper)
            actual_beacon=int(calls["examples"])-beacon_before
            if actual_beacon!=b.calls or actual_beacon>q:
                raise AssertionError("BEACON model query mismatch")
            actual_online+=actual_beacon
            beacon=(splice(x,donor,linked,b.verified_edit,taper=args.taper)
                    if b.is_valid else None)

            values=[
                ("native_original",native,native_valid,status,native_calls),
                ("native_projected",projected,projected_valid,projected_status,native_calls+extra),
                ("beacon_gravity",beacon,bool(b.is_valid),b.status,actual_beacon),
            ]
            for name,attempt,valid,stage,cost in values:
                # Physical diagnostics on valid evidence only; failures
                # are recorded explicitly as false, not discarded.
                seen=attempt if valid else None
                ratio=(float(gravity_residual_roughness(seen,mu,sd)/
                             max(original_roughness,1e-6))
                       if seen is not None else None)
                distance=(float(np.linalg.norm(seen-x)/
                                max(np.linalg.norm(x),np.linalg.norm(donor),1e-8))
                          if seen is not None else None)
                edited=(float(np.mean(np.abs(seen-x)>1e-8))
                        if seen is not None else None)
                records.append({
                    "method":name,"Q":q,"sample":i,
                    "official_test_index":int(selected[i]),
                    "subject":int(ste[selected[i]]),
                    "source":source,"target":target,
                    "verified":int(valid),"status":stage,
                    "online_calls_if_run_alone":cost,
                    "native_queries_before_projection":native_calls,
                    "projection_query_used":extra,
                    "relative_l2":distance,
                    "edited_fraction":edited,
                    "gravity_ratio":ratio,
                    "source_prediction_correct_offline":int(source==int(yte[selected[i]])),
                    "native_upstream_error":error,
                    "native_population":pop,
                    "same_original_native_output_projected":True,
                })
    if int(calls["examples"])!=len(test)+actual_online:
        raise AssertionError("shared experiment actual calls not accounted")
    output=Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    report=summary(records)
    write_csv(output/"per_sample.csv",records)
    write_csv(output/"summary.csv",report)
    manifest={
        "upstream":"MarioRefoyo/FastPACE, native MultiSubSpaCECF",
        "immutable_source_SHA":UPSTREAM_SHA,
        "native_pilot_configuration":"no FI, train-only Gaussian outlier, population 2/4, one individual-channel iteration",
        "native_baseline_is_not_paper_configuration":True,
        "projection_is_not_part_of_upstream_method":True,
        "projection_formula_raw_units":"b'=(b_native+t_native-(t_source-b_source))/2; t'=b'+(t_source-b_source)",
        "projection_minimizes_raw_l2_to_native_under_residual_constraint":True,
        "physical_validity_beyond_single_invariant_unproven":True,
        "same_classifier_donor_target_x_across_methods":True,
        "dataset_protocol":protocol,
        "training_subjects":sorted(map(int,np.unique(strn))),
        "evaluation_subjects":sorted(map(int,np.unique(ste[selected]))),
        "source_model_SHA":calls["fitted_model_sha256"],
        "npz_SHA":hashlib.sha256(data.read_bytes()).hexdigest(),
        "Q_additional_includes_projected_verification":q_values,
        "baseline_model_calls":len(test),
        "actual_online_model_calls_including_shared_native_prefix":actual_online,
        "total_model_evaluations":calls["examples"],
        "native_projection_counts_share_source_prefix":True,
        "no_donor_robustness_claim":True,
    }
    (output/"manifest.json").write_text(
        json.dumps(manifest,indent=2),encoding="utf8")
    print(json.dumps({"model_sha":calls["fitted_model_sha256"],
                      "summary":report},indent=2),flush=True)


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
