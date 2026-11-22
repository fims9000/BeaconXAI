#!/usr/bin/env python3
"""BEACON sign-aware counter-evidence, 0/16 model calls, original UCI HAR.

The old absolute moving-top2 margin need not change on class reversal.
Here the source class identity is fixed before interventions. We learn
positive counter-evidence C=max(0,m_a(T_g x)-m_a(x)) from cheap features,
and evaluate separately from positive support S=max(0,m_a(x)-m_a(T_g x)).

A "top3 contradictions" comparison is meaningful only on windows with at
least 3 genuinely positive counter-responses in the OFFLINE Q36 scan.
Every inference-time candidate selection uses no Q36 reference, test label,
or future intervention. Reference Q36 is assessed only afterwards.
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
from beaconxai.contrastive import make_components,neutralize_interp
from beaconxai.datasets import load_npz_dataset,apply_standardizer
from beaconxai.conformal_evidence import (
    cheap_feature_matrix,fit_response_surrogate,fit_pairwise_top3_surrogate
)
from beaconxai.directed_evidence import (
    signed_fragment_evidence,positive_reference_ranking
)
from beaconxai.active_residual_evidence import (
    fit_residual_conditional,active_counterevidence_from_features
)
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_cheap_conformal_explanations import (
    separate_subject_roles,subset_by_subject,subject_macro
)
from scripts.run_q16_matched_ablation import class_cheap,best_q,write_csv


def full_signed_scan(x,base,components,model):
    changed=np.stack([neutralize_interp(x,c) for c in components])
    replies=np.asarray(model(changed),dtype=float)
    if replies.shape!=(len(components),len(base)):
        raise ValueError("invalid directional Q36 model response")
    return signed_fragment_evidence(base,replies)


def selected_signed_scan(x,base,components,selected,model):
    if len(selected)!=len(set(selected)):
        raise ValueError("duplicate directional model probes")
    changed=np.stack([neutralize_interp(x,components[j]) for j in selected])
    outputs=np.asarray(model(changed),dtype=float)
    c,opp,sup=signed_fragment_evidence(base,outputs)
    ranking=[selected[j] for j in np.argsort(-opp,kind="stable")]
    return ranking,opp,sup


def aggregate_eligible(rows,methods):
    report=[]
    subjects=sorted({int(r["subject"]) for r in rows})
    for method in methods:
        blocks=[]
        for subject in subjects:
            group=[r for r in rows if r["method"]==method and r["subject"]==subject]
            if not group:
                raise AssertionError("missing method for test subject")
            eligible=[r for r in group if r["has_three_counterexamples"]]
            blocks.append({
                "subject":subject,"method":method,
                "n_total":len(group),"n_eligible":len(eligible),
                "macro_eligible_recall3":(
                    float(np.mean([r["counter_top3_overlap"] for r in eligible]))
                    if eligible else None),
                "positive_recall3_shortlist":(
                    float(np.mean([r["top3_in_shortlist"] for r in eligible]))
                    if eligible and group[0]["extra_model_evaluations"] else None),
                "mean_queries":float(np.mean([r["extra_model_evaluations"] for r in group]))
            })
        valid=[b for b in blocks if b["n_eligible"]]
        report.append({
            "method":method,
            "test_subjects":len(blocks),
            "eligible_subjects":len(valid),
            "eligible_windows":sum(b["n_eligible"] for b in blocks),
            "total_windows":sum(b["n_total"] for b in blocks),
            "subject_macro_recall3":(
                float(np.mean([b["macro_eligible_recall3"] for b in valid]))
                if valid else None),
            "mean_extra_model_evaluations":float(np.mean(
                [b["mean_queries"] for b in blocks])),
        })
    return report


def run(args):
    path=Path(args.npz_path)
    xtrain,ytrain,xtest,_=load_npz_dataset(path)
    with np.load(path) as z:
        st=z["subject_train"].astype(int)
        sv=z["subject_test"].astype(int)
    if set(map(int,np.unique(st))) & set(map(int,np.unique(sv))):
        raise ValueError("official train/test subject leakage")
    roles=separate_subject_roles(st,args.seed)
    fit=np.isin(st,roles["model_fit"])
    mu,sigma,profiles,model,calls=fit_predictor(
        xtrain[fit],ytrain[fit],args.model,args.seed,args.trees)
    comps=make_components(xtrain.shape[1:],12,"pamap3")
    if len(comps)!=36 or args.q!=16:
        raise ValueError("matched directional study requires M36,Q16")

    # Fit only on unseen-by-classifier TRAIN subjects, never outer test.
    raw,subjects,_=subset_by_subject(
        xtrain,st,roles["surrogate_fit"],args.windows_per_subject,args.seed+103)
    sample=apply_standardizer(raw,mu,sigma)
    bases=model(sample)
    F=[];Y=[];fit_usable=0
    for i,x in enumerate(sample):
        original=int(np.argmax(bases[i]))
        F.append(cheap_feature_matrix(x,profiles[original][0],bases[i],comps))
        _,opposing,_=full_signed_scan(x,bases[i],comps,model)
        Y.append(opposing)
        fit_usable+=int(np.count_nonzero(opposing>1e-8)>=3)
    ridge=fit_response_surrogate(F,Y,comps,ridge_alpha=args.ridge_alpha)
    ranker=fit_pairwise_top3_surrogate(F,Y,topk=3,cost=args.pairwise_c)
    residual=fit_residual_conditional(
        F,Y,comps,ridge_alpha=args.ridge_alpha,
        temporal_width=2.,empirical_weight=.65,ridge_noise=.05)
    training_queries=calls["examples"]

    outer_sampling_seed=getattr(args,"test_sampling_seed",None)
    if outer_sampling_seed is None:
        outer_sampling_seed=args.seed+301
    raw_test,test_subj,indices=subset_by_subject(
        xtest,sv,np.unique(sv),args.windows_per_subject,outer_sampling_seed)
    test=apply_standardizer(raw_test,mu,sigma)
    baselines=model(test)
    methods=[
        "cheap_original_q0","cheap_ridge_q0","cheap_pairwise_q0",
        "original_beacon_q16","signed_ridge_q16","signed_pairwise_q16",
        "active_residual_q16","uniform_q16"
    ]
    rows=[]
    rng=np.random.default_rng(args.seed+91)
    for i,x in enumerate(test):
        baseline=baselines[i]
        a=int(np.argmax(baseline))
        phi=cheap_feature_matrix(x,profiles[a][0],baseline,comps)
        estimated,_=ridge.predict(phi,comps)
        paired=ranker.score(phi)
        cheap=class_cheap(x,profiles[a][0],comps)[2]
        ranks={
            "cheap_original_q0":np.argsort(-cheap,kind="stable").tolist(),
            "cheap_ridge_q0":np.argsort(-estimated,kind="stable").tolist(),
            "cheap_pairwise_q0":np.argsort(-paired,kind="stable").tolist(),
        }
        proposed={
            "original_beacon_q16":best_q(cheap,args.q),
            "signed_ridge_q16":ranks["cheap_ridge_q0"][:args.q],
            "signed_pairwise_q16":ranks["cheap_pairwise_q0"][:args.q],
            "uniform_q16":rng.choice(36,size=args.q,replace=False).tolist()
        }
        measured={}
        for name,selected in proposed.items():
            ranking,opp,sup=selected_signed_scan(
                x,baseline,comps,selected,model)
            measured[name]=(ranking,opp,sup,selected)
        # A new ONLINE sequential policy: 8 genuine model probes, update
        # unqueried counter-response estimates using TRAIN-only residual
        # covariance, then 8 DIFFERENT probes selected by posterior UCB.
        # It never receives Q36, external labels or future responses.
        active=active_counterevidence_from_features(
            x,baseline,comps,phi,residual,model,paired,
            q_first=8,q_second=8,uncertainty_bonus=.5)
        if active.extra_model_evaluations!=args.q or active.forward_batches!=2:
            raise AssertionError("real adaptive Q8+8 model calls missing")
        measured["active_residual_q16"]=(
            active.conditional_ranking_all_components,
            np.asarray(active.model_measured_positive),
            np.zeros(args.q),active.candidate_ids)
        # Full-Q36 may be used only for OFFLINE comparison here.
        signed,opposing,supporting=full_signed_scan(x,baseline,comps,model)
        reference,positive_count=positive_reference_ranking(opposing)
        eligible=positive_count>=3
        for method in methods:
            if method in measured:
                ranking,observed_opposing,observed_support,selected=measured[method]
                q=args.q
                actual_positive=int(np.count_nonzero(observed_opposing>1e-8))
            else:
                ranking=ranks[method]
                selected=list(range(36))
                q=0
                actual_positive=None
            top3=set(reference[:3])
            overlap=(len(top3 & set(ranking[:3]))/3 if eligible else None)
            measured_only_overlap=(
                len(top3 & set(active.measured_only_ranking[:3]))/3
                if method=="active_residual_q16" and eligible else None)
            inferred_unqueried_top3=(
                sum(v not in set(active.candidate_ids)
                    for v in ranking[:3])
                if method=="active_residual_q16" else None)
            rows.append({
                "subject":int(test_subj[i]),"sample":i,
                "original_window_index":int(indices[i]),
                "method":method,"source_class":a,
                "extra_model_evaluations":q,
                "positive_counterevidence_components_Q36":positive_count,
                "has_three_counterexamples":int(eligible),
                "counter_top3_overlap":overlap,
                "top3_in_shortlist":(
                    len(top3 & set(selected))/3 if eligible and q else None),
                "model_measured_positive_fragments":actual_positive,
                "active_measured_only_top3_overlap":measured_only_overlap,
                "active_inferred_unqueried_top3_count":inferred_unqueried_top3,
                "active_inferred_top3_not_all_directly_probed":bool(
                    inferred_unqueried_top3) if inferred_unqueried_top3 is not None else None,
                "counterevidence_peak":float(np.max(opposing)),
                "support_peak":float(np.max(supporting)),
                "signed_min":float(np.min(signed)),
                "signed_max":float(np.max(signed)),
                "estimated_top3":",".join(map(str,ranking[:3])),
                "reference_top3":",".join(map(str,reference[:3]))
            })
    summary=aggregate_eligible(rows,methods)
    total_evaluated=len(test)
    n_eligible=sum(rows[i]["has_three_counterexamples"]
                   for i in range(0,len(rows),len(methods)))
    total_live_queries=sum(r["extra_model_evaluations"] for r in rows)
    if calls["examples"]!=training_queries+total_evaluated*(1+36)+total_live_queries:
        raise AssertionError("directional model query accounting mismatch")
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",rows)
    write_csv(out/"summary.csv",summary)
    manifest={
        "task":"separate mathematically signed support vs counterevidence of original prediction",
        "source_class_fixed_before_interventions":True,
        "not_moving_top2":True,
        "operator":"identical interpolation neutralization for all comparisons",
        "model":args.model,
        "dataset":"UCI HAR official 21 train / 9 heldout subjects",
        "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_subject_groups":roles,
        "test_subjects":sorted(map(int,np.unique(sv))),
        "fit_counterexample_eligible_windows":fit_usable,
        "fitting_windows":len(sample),
        "fit_pairwise_valid_windows":ranker.fitting_windows,
        "fit_pairwise_discarded_ties":ranker.discarded_tied_windows,
        "active_residual_training_windows":residual.fitted_windows,
        "active_residual_covariance_float32_bytes":int(
            residual.covariance.astype("float32").nbytes),
        "active_residual_kernel":"65pct empirical PSD residual covariance + 35pct smooth sensor/time PSD kernel",
        "active_residual_online_queried_stages":[8,8],
        "active_residual_unqueried_inference_is_not_verified":True,
        "active_residual_uncertainty_bonus":.5,
        "test_windows":total_evaluated,
        "outer_test_window_sampling_seed":int(outer_sampling_seed),
        "test_counterexample_eligible_windows":n_eligible,
        "eligible_fraction":n_eligible/total_evaluated,
        "M":36,"Q":args.q,
        "fit_training_offline_Q36_calls":len(sample)*36,
        "test_offline_Q36_reference_calls":total_evaluated*36,
        "test_actual_budgeted_model_calls":total_live_queries,
        "test_cached_baseline_model_calls":total_evaluated,
        "model_examples_counted":calls["examples"],
        "test_labels_never_read":True,
        "not_causal_or_class_transition_proof":True,
        "limitation":[
            "Removing a fragment with interpolation can itself create nonphysical signals.",
            "A positive signed response denotes evidence against the original predicted class only under the specified intervention.",
            "If fewer than three truly positive changes exist, top3-counterevidence localization is undefined and is excluded instead of inventing contradictory fragments.",
            "Few independent surrogate-fit subjects, no TinyML energy/latency measurement.",
            "Conditional covariance adjustments are uncalibrated and may rank UNMEASURED effects incorrectly; never call them verified evidence."
        ]
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("DIRECTED_EVIDENCE eligible=",
          n_eligible,"of",total_evaluated,"fit_eligible=",fit_usable,
          "train_probes=",len(sample)*36,flush=True)
    for r in summary:
        print("DIRECTED_RESULT",r["method"],"valid_windows=",r["eligible_windows"],
              "macro_overlap=",r["subject_macro_recall3"],
              "Q=",r["mean_extra_model_evaluations"],flush=True)
    return manifest,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--model",choices=("raw_mlp","extratrees"),default="raw_mlp")
    p.add_argument("--seed",type=int,default=57)
    p.add_argument("--test-sampling-seed",type=int,
                   help="Freeze outer-test windows across train-role seeds")
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--windows-per-subject",type=int,default=6)
    p.add_argument("--q",type=int,default=16)
    p.add_argument("--ridge-alpha",type=float,default=100.)
    p.add_argument("--pairwise-c",type=float,default=.1)
    run(p.parse_args())


if __name__=="__main__":
    main()
