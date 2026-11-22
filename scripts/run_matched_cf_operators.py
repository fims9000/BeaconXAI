#!/usr/bin/env python3
"""One trained black-box classifier, three operators, exact matched Q budgets.

All operators share original windows, frozen model weights, chosen A/B classes
and target-class TRAIN donor. Test labels and full-donor offline references
are read only after completing each sample's budgeted inference decisions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import load_npz_dataset, apply_standardizer
from beaconxai.real_donor import fit_real_donor_bank, nearest_context_index
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.verified_counterfactual import (
    find_verified_counterfactual, donor_stability_audit, splice
)
from beaconxai.cf_path_search import search_counterfactual_path
from beaconxai.physical_groups import uci_har_coupled_groups, coupled_priority
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv
from scripts.run_verified_counterfactual_benchmark import gravity_residual_roughness


def macro_summary(rows):
    """Subject-level macro metrics; conditional validity kept separate."""
    keys = sorted({(r["operator"],r["method"],int(r["budget"])) for r in rows})
    summary=[]
    for op,method,q in keys:
        group=[r for r in rows if (r["operator"],r["method"],int(r["budget"]))==(op,method,q)]
        subjects=sorted({int(r["subject"]) for r in group})
        if not group or not subjects:
            raise ValueError("empty operator/method/subject group")
        blocks=[[r for r in group if int(r["subject"])==sid] for sid in subjects]
        valid=[r for r in group if r["verified"]]
        summary.append({
            "operator":op,"method":method,"budget":q,
            "n_windows":len(group),"n_subjects":len(subjects),
            "n_verified":len(valid),
            "subject_macro_success":float(np.mean([np.mean([r["verified"] for r in b]) for b in blocks])),
            "subject_macro_queries":float(np.mean([np.mean([r["additional_calls"] for r in b]) for b in blocks])),
            "n_verified_local_minimality":sum(r["locally_minimal"] for r in valid),
            "n_robust_two_donors":sum(r["donor_checks"]==2 and r["donor_flips"]==2 for r in valid),
            "mean_logical_components_if_valid":(
                float(np.mean([r["logical_components"] for r in valid])) if valid else None),
            "mean_physical_regions_if_valid":(
                float(np.mean([r["three_axis_regions"] for r in valid])) if valid else None),
            "median_gravity_roughness_ratio_if_valid":(
                float(np.median([r["gravity_roughness_ratio"] for r in valid])) if valid else None),
        })
    return summary


def run(args):
    path=Path(args.npz_path)
    x_train,y_train,x_test,y_test=load_npz_dataset(path)
    with np.load(path) as z:
        train_subjects=np.asarray(z["subject_train"],dtype=int)
        test_subjects=np.asarray(z["subject_test"],dtype=int)
    if len(x_train)!=len(y_train) or len(x_test)!=len(y_test) or set(train_subjects)&set(test_subjects):
        raise ValueError("subject leakage or dimension mismatch")

    # EXACTLY one model and one normalizer fitted for all three operators.
    mu,sd,profiles,model,calls=fit_predictor(
        x_train,y_train,args.model,args.seed,args.trees)
    train=apply_standardizer(x_train,mu,sd)
    components=make_components(train.shape[1:],12,"pamap3")
    bank=fit_real_donor_bank(
        train,y_train,components,max_donors=args.donors,seed=args.seed+10)
    indices=balanced_subject_sample(test_subjects,args.max_test,args.seed+301)
    test=apply_standardizer(x_test[indices],mu,sd)
    baseline=model(test)  # one cached output for each test window
    frozen_sha=calls["fitted_model_sha256"]

    operators={
        "independent":components,
        "joint_donor_acc":uci_har_coupled_groups(components),
        "gravity_preserved":uci_har_coupled_groups(
            components,gravity_preserving=True,mean=mu,scale=sd),
    }
    q_grid=sorted({int(v) for v in args.budgets.split(",")})
    if (not q_grid or min(q_grid)<4 or max(q_grid)>24 or
        args.reserve<0 or args.reserve>=min(q_grid)):
        raise ValueError("invalid Q grid or donor budget reserve")
    records=[]
    offline_calls=0
    for i,x in enumerate(test):
        z=baseline[i]
        source=int(np.argmax(z))
        target=int(np.argsort(-z,kind="stable")[1])
        if source not in profiles or target not in bank.donors:
            raise ValueError("unknown class")
        # Same pre-existing donor policy as initial studies, fixed across modes.
        nearest=nearest_context_index(x,0,bank,target,components)
        donor=bank.donors[target][nearest]
        other=[bank.donors[target][j] for j in range(len(bank.donors[target]))
               if j!=nearest][:args.reserve]
        prior=contrastive_prior(x,source,target,profiles,components)
        for operator,candidates in operators.items():
            score=(np.abs(prior) if operator=="independent"
                   else coupled_priority(prior,components))
            directed=np.argsort(-score,kind="stable").astype(int).tolist()
            uniform=np.random.default_rng(args.seed*100003+int(indices[i])).permutation(
                len(candidates)).astype(int).tolist()
            for q in q_grid:
                for method,order in (("pair",directed),("path",directed),("uniform_path",uniform)):
                    start=int(calls["examples"])
                    if method=="pair":
                        result=find_verified_counterfactual(
                            x,z,donor,candidates,order,model,target_class=target,
                            budget=q-args.reserve,max_changed_components=2,
                            single_fraction=args.single_fraction,taper=args.taper)
                        n_search=result.evaluations
                        minimal=result.is_valid  # both singleton deletions measured
                    else:
                        result=search_counterfactual_path(
                            x,z,donor,candidates,order,model,target_class=target,
                            budget=q-args.reserve,max_changed_components=8,
                            seed_singletons=min(3,q-args.reserve),taper=args.taper)
                        n_search=result.calls
                        minimal=result.local_minimality_checked
                    audit=donor_stability_audit(
                        x,z,target,result.verified_edit or (),other,candidates,
                        model,budget_remaining=min(args.reserve,q-n_search),taper=args.taper)
                    n_calls=int(calls["examples"])-start
                    if n_calls!=n_search+audit["tested"] or n_calls>q:
                        raise AssertionError("query budget mismatch")
                    changed=(splice(x,donor,candidates,result.verified_edit,taper=args.taper)
                             if result.is_valid else None)
                    physical=(sum(len(getattr(candidates[j],"regions",(candidates[j],)))
                                  for j in result.verified_edit) if result.is_valid else None)
                    rough=(gravity_residual_roughness(changed,mu,sd)/
                           max(gravity_residual_roughness(x,mu,sd),1e-6)
                           if changed is not None else None)
                    records.append({
                        "sample":i,"original_window_index":int(indices[i]),
                        "subject":int(test_subjects[indices[i]]),
                        "operator":operator,"method":method,"budget":q,
                        "source":source,"target":target,"verified":int(result.is_valid),
                        "logical_components":len(result.verified_edit) if result.is_valid else None,
                        "three_axis_regions":physical,
                        "locally_minimal":int(result.is_valid and minimal),
                        "additional_calls":n_calls,
                        "donor_checks":audit["tested"],"donor_flips":audit["target_flips"],
                        "gravity_roughness_ratio":rough,
                        "donor_train_index":int(bank.selected_train_indices[target][nearest]),
                    })
        # After inference is finished: diagnostic references only, not selection.
        full=[splice(x,donor,parts,tuple(range(len(parts))),taper=args.taper)
              for parts in operators.values()]
        reference=model(np.stack([donor]+full))
        offline_calls+=len(reference)
        data={
            "donor_pred_target_offline":int(np.argmax(reference[0])==target),
            "original_correct_offline":int(source==int(y_test[indices[i]])),
        }
        for j,name in enumerate(operators):
            data["full_"+name+"_flip_offline"]=int(np.argmax(reference[j+1])==target)
        for row in records:
            if row["sample"]==i:
                row.update(data)
    total=len(test)+sum(r["additional_calls"] for r in records)+offline_calls
    if total!=calls["examples"]:
        raise AssertionError("model calls not fully accounted")
    summ=macro_summary(records)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",records)
    write_csv(out/"summary.csv",summ)
    manifest={
        "model_sha256_one_frozen_classifier_for_all_operators":frozen_sha,
        "same_classifier_across_all_operators":True,
        "train_standardizer_sha256":calls["training_standardizer_sha256"],
        "npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed":args.seed,"dataset":"UCI HAR official subject-disjoint split",
        "train_subjects":sorted(map(int,np.unique(train_subjects))),
        "test_subjects":sorted(map(int,np.unique(test_subjects[indices]))),
        "candidate_count":{name:len(c) for name,c in operators.items()},
        "baseline_calls":len(test),
        "additional_budgeted_calls_all_methods":sum(r["additional_calls"] for r in records),
        "offline_reference_calls":offline_calls,
        "total_model_evaluations":calls["examples"],
        "Q_max_per_method_and_window":q_grid,
        "Q_reserve_for_two_donor_audits":args.reserve,
        "selection_never_reads_test_labels_or_full_references":True,
        "limitations":[
            "A preserved gravity residual is necessary under this representation, not sufficient for physical realism",
            "Coupled edit candidates cover more 3-axis regions than independent candidates",
            "Training donor selection excludes the first component in context matching",
            "One model / one subject split; no broad population generalization",
            "Verified local minimality is not global minimality or causal explanation",
            "Offline full-edit feasibility is measured separately from online Q budget",
        ]}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print(json.dumps({"frozen_model_sha256":frozen_sha,"summary":summ},
                     indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",required=True,type=Path)
    p.add_argument("--out-dir",required=True,type=Path)
    p.add_argument("--model",choices=["raw_mlp","extratrees"],default="raw_mlp")
    p.add_argument("--max-test",type=int,default=180)
    p.add_argument("--donors",type=int,default=8)
    p.add_argument("--reserve",type=int,default=2)
    p.add_argument("--budgets",default="16")
    p.add_argument("--single-fraction",type=float,default=.5)
    p.add_argument("--taper",type=int,default=2)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--seed",type=int,default=57)
    run(p.parse_args())

if __name__=="__main__":
    main()
