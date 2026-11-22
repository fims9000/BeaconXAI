#!/usr/bin/env python3
"""Train-only-calibrated BEACON Q8->Q16->Q32 on natural UCI HAR windows.

Official UCI HAR test participants are never used for policy calibration.
Inner validation subjects come only from the official training split,
excluded from the inner model's fit and class-mean profiles.

Outer online policy makes *actual* 8/8/16 model calls and stops when its
fixed train-selected rule allows it; Q36 full reference and fixed controls
are separate OFFLINE evaluations, not observations given to deployment.
This is model sensitivity under interpolation, NOT causal attribution.
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

from beaconxai.contrastive import fit_class_profiles, make_components, neutralize_interp
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.sensitivity_reference import moving_margin_effects, rank_effects, reference_metrics
from beaconxai.staged_budget import (
    StagedPolicy,fit_staged_policy,fit_constrained_upgrade,
    run_staged_audit,exact_cost_matched_random_upgrade,
)
from beaconxai.subject_sampling import balanced_subject_sample
from scripts.run_component_conflict_benchmark import _anfis_features,_train_extratrees_local
from scripts.run_q16_matched_ablation import class_cheap,write_csv

METHODS=("statistical_q0","fixed_q8","fixed_q16","fixed_q32",
         "calibrated_policy","constrained_upgrade16","random_at_matched_budget")


def fit_predictor(train_raw: np.ndarray, train_y: np.ndarray,
                  model_type: str, seed: int, trees: int) -> tuple:
    """One model and scaler fitted entirely on the supplied train subjects."""
    mu,sigma=fit_channel_standardizer(train_raw)
    train=apply_standardizer(train_raw,mu,sigma)
    if model_type=="extratrees":
        fitted=_train_extratrees_local(train,train_y,trees,.7,1)
        clf=fitted.model
        def prob(x):
            return clf.predict_proba(_anfis_features(np.asarray(x)))
    elif model_type=="raw_mlp":
        from sklearn.neural_network import MLPClassifier
        clf=MLPClassifier(hidden_layer_sizes=(32,),alpha=.01,
                          batch_size=128,max_iter=30,early_stopping=True,
                          n_iter_no_change=6,random_state=seed)
        clf.fit(train.reshape(len(train),-1),train_y)
        def prob(x):
            z=np.asarray(x)
            return clf.predict_proba(z.reshape(len(z),-1))
    else:
        raise ValueError("Unknown model")
    labels=np.asarray(clf.classes_,dtype=int)
    if (len(labels)<2 or np.any(labels<0)
        or len(labels)!=len(set(labels.tolist()))
        or not set(np.unique(train_y)).issubset(set(labels))):
        raise ValueError("invalid class mapping")
    n_classes=int(max(labels.max(),np.max(train_y)))+1
    # Hash learned numeric parameters instead of pickle metadata:
    # stable across processes, independent of Python object addresses.
    model_hasher=hashlib.sha256()
    model_hasher.update(np.ascontiguousarray(labels).tobytes())
    model_hasher.update(type(clf).__qualname__.encode("utf8"))
    learned_tensors=[]
    if hasattr(clf,"coefs_") and hasattr(clf,"intercepts_"):
        learned_tensors=list(clf.coefs_)+list(clf.intercepts_)
    elif hasattr(clf,"estimators_"):
        for tree_est in clf.estimators_:
            tree=tree_est.tree_
            learned_tensors.extend((tree.children_left,tree.children_right,
                                    tree.feature,tree.threshold,tree.value))
    for tensor in learned_tensors:
        a=np.ascontiguousarray(tensor)
        model_hasher.update(str(a.dtype).encode("ascii"))
        model_hasher.update(np.asarray(a.shape,dtype="<i8").tobytes())
        model_hasher.update(a.tobytes())
    counted={
        "examples":0,"batches":0,
        "fitted_model_sha256":model_hasher.hexdigest(),
        "training_standardizer_sha256":hashlib.sha256(
            np.ascontiguousarray(mu).tobytes()+
            np.ascontiguousarray(sigma).tobytes()).hexdigest(),
        "model_fingerprint_covers_learned_numeric_tensors":bool(learned_tensors),
    }
    def model(xs):
        z=np.asarray(xs)
        if z.ndim!=3 or z.shape[1:]!=train.shape[1:]:
            raise ValueError("model expects [batch,time,channels]")
        counted["examples"]+=len(z)
        counted["batches"]+=1
        p=np.full((len(z),n_classes),1e-12,dtype=float)
        raw=np.asarray(prob(z),dtype=float)
        if raw.shape!=(len(z),len(labels)):
            raise ValueError("invalid classifier probabilities")
        p[:,labels]=raw
        return np.log(np.clip(p,1e-12,1))
    return (mu,sigma,fit_class_profiles(train,train_y),model,counted)


def full_reference(x,baseline,comps,predict_many):
    perturbed=np.stack([neutralize_interp(x,c) for c in comps])
    effects=moving_margin_effects(baseline,predict_many(perturbed))
    order=rank_effects(effects,list(range(len(comps))))
    s=np.sort(effects)
    return order,float(s[-3]-s[-4]),float(s[-1])


def _sample_rows(raw, subjects, maximum: int, seed: int):
    if maximum<2:
        raise ValueError("need at least two sampled windows")
    idx=balanced_subject_sample(subjects,maximum,seed)
    return raw[idx],subjects[idx].astype(int),idx


def calibrate_inner(
    xtrain: np.ndarray, ytrain: np.ndarray, subjects: np.ndarray,
    model_type: str, seed: int, trees: int, max_calib: int,
    penalty: float, components,
) -> tuple[StagedPolicy,dict,list[dict]]:
    unique=sorted(map(int,np.unique(subjects)))
    inner_val_subjects=unique[::4]
    if len(unique)<8 or len(inner_val_subjects)<2:
        raise ValueError("need multiple independent inner-validation subjects")
    fit_mask=~np.isin(subjects,inner_val_subjects)
    val_mask=~fit_mask
    if (len(set(ytrain[fit_mask]))<2
        or not len(np.unique(subjects[fit_mask]))>=2):
        raise ValueError("insufficient inner training set")
    mu,sigma,profiles,model,calls=fit_predictor(
        xtrain[fit_mask],ytrain[fit_mask],model_type,seed,trees)
    sampled,sampled_subj,_=_sample_rows(
        xtrain[val_mask],subjects[val_mask],max_calib,seed+19)
    test=apply_standardizer(sampled,mu,sigma)
    baseline=model(test)
    rows=[]
    for i,x in enumerate(test):
        a=int(np.argmax(baseline[i]))
        prior=class_cheap(x,profiles[a][0],components)[2]
        measured=run_staged_audit(
            x,components,prior,baseline[i],model,StagedPolicy("fixed32"))
        if (measured.queries!=32 or
            set(measured.stage_rankings)!={8,16,32}):
            raise AssertionError("calibration Q32 failed")
        ref,gap,max_effect=full_reference(x,baseline[i],components,model)
        metrics={q:reference_metrics(ref,measured.stage_rankings[q],len(components),q)
                 for q in (8,16,32)}
        rows.append({
            "subject":int(sampled_subj[i]),"sample":i,
            "gap8":measured.stage_features[8],
            "gap16":measured.stage_features[16],
            "overlap8":metrics[8]["top3_overlap"],
            "overlap16":metrics[16]["top3_overlap"],
            "overlap32":metrics[32]["top3_overlap"],
            "full_reference_top3_gap":gap,
            "full_reference_max_effect":max_effect,
        })
    policy,selection=fit_staged_policy(rows,penalty=penalty)
    selection.update({
        "inner_fit_subjects":sorted(map(int,np.unique(subjects[fit_mask]))),
        "inner_validation_subjects":inner_val_subjects,
        "inner_reference_calls_offline":len(test)*len(components),
        "inner_Q32_observation_calls_for_calibration":len(test)*32,
        "inner_model_calls_total_including_baseline":calls["examples"],
        "inner_fitted_model_sha256":calls["fitted_model_sha256"],
        "inner_training_standardizer_sha256":
            calls["training_standardizer_sha256"],
        "reference_proxy_is_not_causal_truth":True,
    })
    assert calls["examples"]==len(test)*(1+32+len(components))
    return policy,selection,rows


def summarize_subject_macro(rows:list[dict],penalty:float):
    per_subject=[]
    for method in METHODS:
        for subject in sorted({int(r["subject"]) for r in rows}):
            group=[r for r in rows if r["method"]==method and int(r["subject"])==subject]
            if not group:
                raise AssertionError("method omitted for a test participant")
            if len({int(r["sample"]) for r in group})!=len(group):
                raise AssertionError("duplicated window across one policy")
            mean_overlap=float(np.mean([r["top3_overlap"] for r in group]))
            mean_q=float(np.mean([r["extra_model_evaluations"] for r in group]))
            per_subject.append({
                "method":method,"subject":subject,
                "n_windows":len(group),"top3_overlap":mean_overlap,
                "mean_extra_model_evaluations":mean_q,
                "utility":mean_overlap-penalty*mean_q/32.0,
                "stable_reference_windows":sum(
                    float(r["reference_top3_gap"])>1e-6 for r in group),
            })
    summary=[]
    for method in METHODS:
        group=[r for r in per_subject if r["method"]==method]
        summary.append({
            "method":method,"subjects":len(group),
            "windows":sum(r["n_windows"] for r in group),
            "subject_macro_top3_overlap":float(np.mean(
                [r["top3_overlap"] for r in group])),
            "subject_macro_mean_queries":float(np.mean(
                [r["mean_extra_model_evaluations"] for r in group])),
            "subject_macro_utility":float(np.mean(
                [r["utility"] for r in group])),
        })
    return summary,per_subject


def run(args):
    path=Path(args.npz_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    xtrain,ytrain,xtest,_=load_npz_dataset(path)
    with np.load(path) as z:
        st=z["subject_train"].astype(int)
        sv=z["subject_test"].astype(int)
    if set(map(int,np.unique(st)))&set(map(int,np.unique(sv))):
        raise ValueError("participant train/test leakage")
    if (len(xtrain)!=len(st) or len(xtest)!=len(sv)
        or xtrain.shape[1:]!=xtest.shape[1:]):
        raise ValueError("input/participant geometry mismatch")
    components=make_components(xtrain.shape[1:],12,"pamap3")
    if len(components)!=36:
        raise AssertionError("expected 36 components in three sensor groups")
    # Inner validation never touches xtest or its activity labels.
    policy,selection,calib_rows=calibrate_inner(
        xtrain,ytrain,st,args.model,args.seed,args.trees,
        args.max_calib,args.penalty,components)
    # Independent, prespecified safety-first analysis: never below Q16,
    # permit up to 25% Q32 upgrades on inner validation only.
    upgrade_policy,upgrade_fit=fit_constrained_upgrade(
        calib_rows,max_mean_q=20.,min_gain=.01,
        min_excess_over_random=.01)
    random_rate=upgrade_fit["random_upgrade_probability_calibrated"]
    random_source=np.random.default_rng(args.seed+2027)
    t0=time.perf_counter()
    mu,sigma,profiles,model,counts=fit_predictor(
        xtrain,ytrain,args.model,args.seed,args.trees)
    outer_model_fit_seconds=time.perf_counter()-t0
    samples,subjects,original_indices=_sample_rows(
        xtest,sv,args.max_test,args.seed+101)
    test=apply_standardizer(samples,mu,sigma)
    baseline=model(test)
    rows=[]
    explicit_query_total=0
    for i,x in enumerate(test):
        source=int(np.argmax(baseline[i]))
        prior=class_cheap(x,profiles[source][0],components)[2]
        before=counts["examples"]
        # Important: method NEVER receives full Q36 responses.
        deployed=run_staged_audit(x,components,prior,baseline[i],model,policy)
        if deployed.queries not in (8,16,32):
            raise AssertionError("invalid selected online Q")
        upgrade=run_staged_audit(
            x,components,prior,baseline[i],model,upgrade_policy)
        # Random budget comparator is prospective, independent of the
        # full-Q36 reference; actual Q16/Q32 calls are executed.
        random_policy=StagedPolicy(
            "fixed32" if random_source.random()<random_rate else "fixed16")
        random_control=run_staged_audit(
            x,components,prior,baseline[i],model,random_policy)
        reference,reference_gap,reference_max=full_reference(
            x,baseline[i],components,model)
        controls={}
        for q in (8,16,32):
            controls[q]=run_staged_audit(
                x,components,prior,baseline[i],model,
                StagedPolicy("fixed"+str(q)))
            if controls[q].queries!=q:
                raise AssertionError("control budget not executed")
        true_calls=counts["examples"]-before
        expected_calls=(
            deployed.queries+upgrade.queries+random_control.queries+36+8+16+32)
        if true_calls!=expected_calls:
            raise AssertionError(
                f"model-call accounting: actual={true_calls}, planned={expected_calls}")
        explicit_query_total+=deployed.queries
        cheap=rank_effects(prior,list(range(len(components))))
        methods=[
            ("statistical_q0",cheap,0,0),
            ("fixed_q8",controls[8].ranking,8,controls[8].actual_forward_batches),
            ("fixed_q16",controls[16].ranking,16,controls[16].actual_forward_batches),
            ("fixed_q32",controls[32].ranking,32,controls[32].actual_forward_batches),
            ("calibrated_policy",deployed.ranking,deployed.queries,
             deployed.actual_forward_batches),
            ("constrained_upgrade16",upgrade.ranking,upgrade.queries,
             upgrade.actual_forward_batches),
            ("random_at_matched_budget",random_control.ranking,
             random_control.queries,random_control.actual_forward_batches),
        ]
        for name,rank,q,batches in methods:
            metrics=reference_metrics(reference,rank,len(components),q)
            rows.append({
                "method":name,"subject":int(subjects[i]),
                "sample":i,"original_window_index":int(original_indices[i]),
                "extra_model_evaluations":q,"forward_batches":batches,
                "reference_top3_gap":reference_gap,
                "reference_max_effect":reference_max,
                "stage8_measured_gap":deployed.stage_features.get(8,""),
                "stage16_measured_gap":deployed.stage_features.get(16,""),
                "selected_policy":policy.mode,
                "constrained_upgrade_policy":upgrade_policy.mode,
                "upgrade16_stage16_observed_gap":upgrade.stage_features.get(16,""),
                "random_expected_upgrade_probability":random_rate,
                "estimated_top3":",".join(map(str,rank[:3])),
                "offline_full_top3":",".join(map(str,reference[:3])),
                **metrics,
            })
        if (i+1)%30==0:
            print(f"staged budget evaluated {i+1}/{len(test)} held-out windows",flush=True)
    if counts["examples"]!=len(test)+sum(
        r["extra_model_evaluations"] for r in rows if r["method"]!="statistical_q0"
    )+36*len(test):
        raise AssertionError("global actual query accounting differs from manifest")
    summary,per_subject=summarize_subject_macro(rows,args.penalty)
    random_exact_match=exact_cost_matched_random_upgrade(
        rows,seed=args.seed+311,permutations=1000)
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",rows)
    write_csv(out/"per_subject.csv",per_subject)
    write_csv(out/"summary.csv",summary)
    write_csv(out/"inner_calibration_windows.csv",calib_rows)
    (out/"random_exact_cost_match.json").write_text(
        json.dumps(random_exact_match,indent=2),encoding="utf8")
    (out/"inner_calibration.json").write_text(
        json.dumps({
            "policy":policy.__dict__,"fit":selection,
            "constrained_upgrade_policy":upgrade_policy.__dict__,
            "constrained_upgrade_fit":upgrade_fit,
        },indent=2),encoding="utf8")
    m={
        "dataset":"UCI HAR official train/test subject partition",
        "npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "model":args.model,
        "train_subjects":sorted(map(int,np.unique(st))),
        "test_subjects":sorted(map(int,np.unique(sv))),
        "calibration_subjects_separate_from_inner_model_training":True,
        "outer_test_participants_never_used_to_fit_policy":True,
        "outer_test_labels_never_used_to_select_or_stop":True,
        "operator":"interpolation neutralization; original moving top-two margin",
        "full_Q36_only_offline":True,
        "full_Q36_not_a_causal_ground_truth":True,
        "uncertain_top3_gap_diagnostic":1e-6,
        "n_outer_windows":len(test),
        "per_subject_windows":{
            str(t):int(np.sum(subjects==t)) for t in np.unique(subjects)},
        "selected_policy":policy.__dict__,
        "constrained_upgrade_policy":upgrade_policy.__dict__,
        "constrained_upgrade_calibration":upgrade_fit,
        "random_upgrade_rate_frozen_from_inner_validation":random_rate,
        "random_budget_comparator_executes_real_model_probes":True,
        "random_budget_expected_not_forced_equal_per_outer_sample":True,
        "calibration_penalty_per_32_queries":args.penalty,
        "outer_model_training_seconds":outer_model_fit_seconds,
        "outer_fitted_model_sha256":counts["fitted_model_sha256"],
        "outer_training_standardizer_sha256":
            counts["training_standardizer_sha256"],
        "deployed_staged_queries_actual":explicit_query_total,
        "constrained_upgrade_queries_actual":sum(
            r["extra_model_evaluations"] for r in rows
            if r["method"]=="constrained_upgrade16"),
        "random_budget_queries_actual":sum(
            r["extra_model_evaluations"] for r in rows
            if r["method"]=="random_at_matched_budget"),
        "random_exact_cost_matched_offline_reference":random_exact_match,
        "full_Q36_offline_queries":len(test)*36,
        "fixed_Q8_Q16_Q32_offline_control_queries":len(test)*(8+16+32),
        "outer_cached_baseline_queries":len(test),
        "outer_total_measured_model_examples_including_controls":counts["examples"],
        "calibration":selection,
        "limitations":[
            "The rule is tuned on only inner training subjects, not independently calibrated.",
            "The method's fidelity target is same-operator full Q36; no causal ground truth.",
            "Overlapping motion windows are dependent; subject macro averages are descriptive.",
            "The selected policy may be fixed-Q if adaptive threshold has insufficient gain.",
            "All CPU times exclude MCU runtime, model quantization and energy."
        ],
    }
    (out/"manifest.json").write_text(json.dumps(m,indent=2),encoding="utf8")
    print("POLICY_FIT",json.dumps(selection,sort_keys=True),flush=True)
    print("EXACT_COST_RANDOM_COMPARE",json.dumps({
        "selected_online_q":random_exact_match["actual_online_policy_subject_macro_mean_q"],
        "selected_top3":random_exact_match["observed_subject_macro_overlap"],
        "random_equal_q_top3":random_exact_match["random_matched_overlap_mean"],
        "selected_minus_random":random_exact_match["observed_minus_random_matched_mean"],
        "permutations":random_exact_match["n_random_permutations"],
        "not_population_significance":True,
    },sort_keys=True),flush=True)
    for r in summary:
        print(f"BUDGET_RESULT model={args.model} method={r['method']} "
              f"top3={r['subject_macro_top3_overlap']:.6f} "
              f"Q={r['subject_macro_mean_queries']:.3f} "
              f"utility={r['subject_macro_utility']:.6f}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--model",choices=("extratrees","raw_mlp"),default="raw_mlp")
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--max-calib",type=int,default=72)
    p.add_argument("--max-test",type=int,default=90)
    p.add_argument("--penalty",type=float,default=0.75)
    p.add_argument("--seed",type=int,default=57)
    run(p.parse_args())


if __name__=="__main__":
    main()
