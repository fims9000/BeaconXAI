#!/usr/bin/env python3
"""Can mathematical cheap-to-response prediction replace some BEACON Q36 calls?

The classifier is trained on 8 UCI HAR TRAIN subjects, the cheap ridge
surrogate on 4 OTHER TRAIN subjects, and the simultaneous one-sided
conformal envelope on 9 THIRD-GROUP TRAIN subjects. 9 official TEST
subjects never enter training, tuning or the certificate.

At most 16 actual model interventions online; Q36 is measured independently
only AFTER all candidate explanations are selected. A certified top-3 is
an interpolation-response top-3, NOT a class-causal explanation.

We compare fixed Q16 original BEACON, same-budget ridge mean/UCB policies,
a random Q16 control, Q0 statistical/ridge predictions, and the explicit
certify/abstain online path. We report full off-line training/probe cost too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.contrastive import make_components,neutralize_interp
from beaconxai.datasets import apply_standardizer,load_npz_dataset
from beaconxai.conformal_evidence import (
    cheap_feature_matrix,fit_response_surrogate,
    calibrate_subject_bound,full_model_responses,
    run_certified_explanation,
)
from beaconxai.sensitivity_reference import (
    moving_margin_effects,rank_effects,reference_metrics,
)
from scripts.run_q16_matched_ablation import class_cheap,best_q,write_csv
from scripts.run_clean_calibrated_staged_budget import fit_predictor


def separate_subject_roles(train_subjects,seed):
    ids=sorted(map(int,np.unique(train_subjects)))
    if len(ids)!=21:
        raise ValueError("requires exactly 21 official UCI HAR train subjects")
    permutation=np.random.default_rng(seed).permutation(ids).tolist()
    roles={"model_fit":sorted(permutation[:8]),
           "surrogate_fit":sorted(permutation[8:12]),
           "conformal_calibrate":sorted(permutation[12:])}
    blocks=[set(x) for x in roles.values()]
    if any(blocks[i]&blocks[j] for i in range(3) for j in range(i+1,3)):
        raise AssertionError("subject role leakage")
    return roles


def subset_by_subject(raw,subject_ids,selected,per_subject,seed):
    if per_subject<1:
        raise ValueError("per_subject must be positive")
    rng=np.random.default_rng(seed)
    out=[]
    for subject in sorted(map(int,selected)):
        pool=np.flatnonzero(np.asarray(subject_ids)==subject)
        if len(pool)<per_subject:
            raise ValueError("insufficient subject windows for fixed K")
        out.extend(rng.choice(pool,size=per_subject,replace=False).tolist())
    idx=np.array(out,dtype=int)
    return raw[idx],np.asarray(subject_ids)[idx].astype(int),idx


def probe_selected(x,base,components,selected,model):
    if len(selected)!=len(set(selected)):
        raise ValueError("duplicate model query components")
    queries=np.stack([neutralize_interp(x,components[j]) for j in selected])
    scores=np.asarray(model(queries))
    changes=moving_margin_effects(base,scores)
    order=[selected[i] for i in np.argsort(-changes,kind="stable")]
    return order,changes


def subject_macro(rows,methods):
    individual=[]
    for method in methods:
        for subject in sorted({int(row["subject"]) for row in rows}):
            group=[r for r in rows if r["method"]==method and r["subject"]==subject]
            if not group:
                raise ValueError("missing subject-method comparison")
            individual.append({
                "method":method,"subject":subject,"n":len(group),
                "top3_overlap":float(np.mean([r["top3_overlap"] for r in group])),
                "top3_in_qshortlist":(
                    float(np.mean([r["top3_in_shortlist"] for r in group]))
                    if group[0]["top3_in_shortlist"] is not None else None),
                "oracle_top1_in_top3":float(np.mean(
                    [r["oracle_top1_in_top3"] for r in group])),
                "mean_queries":float(np.mean(
                    [r["extra_model_evaluations"] for r in group])),
                "certified_fraction":float(np.mean(
                    [r["certified"] for r in group])),
                "wrong_certified_count":sum(
                    int(r["certified"] and not r["oracle_top3_set_equal"])
                    for r in group),
            })
    summary=[]
    for method in methods:
        group=[r for r in individual if r["method"]==method]
        summary.append({
            "method":method,
            "subject_count":len(group),
            "n_windows":sum(r["n"] for r in group),
            "subject_macro_top3_overlap":float(np.mean(
                [r["top3_overlap"] for r in group])),
            "subject_macro_top3_in_qshortlist":(
                float(np.mean([r["top3_in_qshortlist"] for r in group]))
                if all(r["top3_in_qshortlist"] is not None for r in group)
                else None),
            "subject_macro_oracle_top1_in_top3":float(np.mean(
                [r["oracle_top1_in_top3"] for r in group])),
            "subject_macro_mean_queries":float(np.mean(
                [r["mean_queries"] for r in group])),
            "subject_macro_certified_fraction":float(np.mean(
                [r["certified_fraction"] for r in group])),
            "wrong_certified_count":sum(r["wrong_certified_count"] for r in group),
        })
    return summary,individual


def run(args):
    path=Path(args.npz_path)
    if not path.exists():
        raise FileNotFoundError(path)
    train_raw,ytrain,test_raw,_=load_npz_dataset(path)
    with np.load(path) as z:
        st=np.asarray(z["subject_train"],dtype=int)
        sv=np.asarray(z["subject_test"],dtype=int)
    if (len(train_raw)!=len(st) or len(test_raw)!=len(sv)
        or set(map(int,np.unique(st)))&set(map(int,np.unique(sv)))):
        raise ValueError("subject data shape or leakage")
    roles=separate_subject_roles(st,args.seed)
    all_train=set(map(int,np.unique(st)))
    assert set().union(*map(set,roles.values()))==all_train
    assert not all_train.intersection(map(int,np.unique(sv)))

    fit_mask=np.isin(st,roles["model_fit"])
    mu,sigma,profiles,model,calls=fit_predictor(
        train_raw[fit_mask],ytrain[fit_mask],
        args.model,args.seed,args.trees)
    components=make_components(train_raw.shape[1:],12,"pamap3")
    M=len(components)
    if M!=36 or args.max_queries>=M:
        raise ValueError("expected M=36 and Q<M")

    # Inner surrogate training: these windows were not used to train the
    # classifier or class profiles. Labels of these windows are NOT read.
    proxy_raw,proxy_s,proxy_ids=subset_by_subject(
        train_raw,st,roles["surrogate_fit"],args.windows_per_subject,
        args.seed+103)
    proxy=apply_standardizer(proxy_raw,mu,sigma)
    cheap_train=[];real_train=[]
    proxy_base=model(proxy)
    for i,x in enumerate(proxy):
        source=int(np.argmax(proxy_base[i]))
        features=cheap_feature_matrix(
            x,profiles[source][0],proxy_base[i],components)
        effects=full_model_responses(x,proxy_base[i],components,model)
        cheap_train.append(features)
        real_train.append(effects)
    surrogate=fit_response_surrogate(
        cheap_train,real_train,components,ridge_alpha=args.ridge_alpha)
    fit_model_examples=calls["examples"]

    # Calibration: one held-out subject is ONE exchangeability unit,
    # including a maximum over ALL six sampled windows and 36 components.
    calib_raw,calib_s,calib_ids=subset_by_subject(
        train_raw,st,roles["conformal_calibrate"],
        args.windows_per_subject,args.seed+201)
    calib=apply_standardizer(calib_raw,mu,sigma)
    calib_records=[]
    calibration_baselines=model(calib)
    for i,x in enumerate(calib):
        source=int(np.argmax(calibration_baselines[i]))
        features=cheap_feature_matrix(
            x,profiles[source][0],calibration_baselines[i],components)
        estimate,scale=surrogate.predict(features,components)
        effects=full_model_responses(
            x,calibration_baselines[i],components,model)
        calib_records.append({
            "subject":int(calib_s[i]),"effects":effects,
            "predicted":estimate,"scale":scale,
        })
    bound,calib_receipts=calibrate_subject_bound(
        calib_records,alpha=args.alpha,
        expected_windows_per_subject=args.windows_per_subject)
    calibration_model_examples=calls["examples"]-fit_model_examples
    # The online methods do not see calibration effects or oracle outcomes.
    test_raw_selected,test_subjects,test_indices=subset_by_subject(
        test_raw,sv,np.unique(sv),args.windows_per_subject,args.seed+301)
    test=apply_standardizer(test_raw_selected,mu,sigma)
    test_bases=model(test)
    seed_rng=np.random.default_rng(args.seed+91)
    methods=[
        "cheap_prior_q0","cheap_surrogate_q0",
        "original_adaptive_q16","surrogate_predicted_q16",
        "surrogate_upper_q16","uniform_q16","certify_or_abstain_q16",
    ]
    rows=[]
    empirical_bound_coverage={}
    certified_total=0;wrong_certified_total=0
    total_online_certificate_queries=0
    for i,x in enumerate(test):
        subject=int(test_subjects[i])
        base=test_bases[i]
        source=int(np.argmax(base))
        phi=cheap_feature_matrix(x,profiles[source][0],base,components)
        pred,scale=surrogate.predict(phi,components)
        upper=bound.upper(pred,scale)
        prior=class_cheap(x,profiles[source][0],components)[2]
        ordered_prior=np.argsort(-prior,kind="stable").astype(int).tolist()
        ordered_pred=np.argsort(-pred,kind="stable").astype(int).tolist()
        ordered_upper=np.argsort(-upper,kind="stable").astype(int).tolist()
        uniform=seed_rng.choice(M,size=args.max_queries,replace=False).tolist()

        # Genuine budgeted model responses, separately from Q36 reference.
        online={}
        for method,selection in (
            ("original_adaptive_q16",best_q(prior,args.max_queries)),
            ("surrogate_predicted_q16",ordered_pred[:args.max_queries]),
            ("surrogate_upper_q16",ordered_upper[:args.max_queries]),
            ("uniform_q16",uniform),
        ):
            ranking,_=probe_selected(x,base,components,selection,model)
            online[method]=(ranking,args.max_queries,selection)
        cert=run_certified_explanation(
            x,base,components,pred,bound,scale,model,
            max_queries=args.max_queries,step=args.batch_size)
        total_online_certificate_queries+=cert.extra_model_evaluations
        online["certify_or_abstain_q16"]=(
            cert.ranked,cert.extra_model_evaluations,cert.selected)

        # This offline full Q36 is computed only AFTER all online choices.
        full=full_model_responses(x,base,components,model)
        reference=rank_effects(full,list(range(M)))
        full_set=set(reference[:3])
        empirical_bound_coverage.setdefault(subject,[]).append(
            bool(np.all(full<=upper+1e-8)))
        ranks={
            "cheap_prior_q0":(ordered_prior,0,list(range(M))),
            "cheap_surrogate_q0":(ordered_pred,0,list(range(M))),
            **online,
        }
        for method,(ranking,q,chosen) in ranks.items():
            result=reference_metrics(reference,ranking,M,q)
            detected=(method=="certify_or_abstain_q16" and cert.certified)
            exact=set(ranking[:3])==full_set
            if detected:
                certified_total+=1
                wrong_certified_total+=int(not exact)
            rows.append({
                "sample":i,"subject":subject,
                "original_window_index":int(test_indices[i]),
                "method":method,
                "extra_model_evaluations":q,
                "certified":int(detected),
                "output_status":(
                    "certified" if detected else
                    "abstain" if method=="certify_or_abstain_q16"
                    else "point_estimate"),
                "oracle_top3_set_equal":int(exact),
                "max_full_response":float(np.max(full)),
                "reference_top3_gap":float(np.sort(full)[-3]-np.sort(full)[-4]),
                "simultaneous_upper_bound_holds_on_window":
                    int(np.all(full<=upper+1e-8)),
                "estimated_top3":",".join(map(str,ranking[:3])),
                "reference_top3":",".join(map(str,reference[:3])),
                **result,
            })
    summary,per_subject=subject_macro(rows,methods)
    cal_counts={int(s):len(v) for s,v in empirical_bound_coverage.items()}
    if any(v!=args.windows_per_subject for v in cal_counts.values()):
        raise AssertionError("outer block size differs from calibration")
    heldout_bound_blocks={
        s:all(v) for s,v in empirical_bound_coverage.items()}
    actual_extra_test=sum(
        r["extra_model_evaluations"] for r in rows)
    expected_test=len(test)+actual_extra_test+M*len(test)
    if calls["examples"]-fit_model_examples-calibration_model_examples!=expected_test:
        raise AssertionError("model actual call accounting differs from receipts")
    if total_online_certificate_queries!=sum(
            r["extra_model_evaluations"] for r in rows
            if r["method"]=="certify_or_abstain_q16"):
        raise AssertionError("certificate online query accounting mismatch")
    assert all(r["output_status"]!="certified" or r["certified"]
               for r in rows)

    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"per_sample.csv",rows)
    write_csv(out/"per_subject.csv",per_subject)
    write_csv(out/"summary.csv",summary)
    write_csv(out/"calibration_subject_scores.csv",calib_receipts)
    (out/"conformal_model.json").write_text(json.dumps({
        "features":surrogate.features,
        "ridge_alpha":float(args.ridge_alpha),
        "feature_means":surrogate.feature_scaler.mean_.tolist(),
        "feature_scales":surrogate.feature_scaler.scale_.tolist(),
        "ridge_coefficients":surrogate.ridge.coef_.tolist(),
        "ridge_intercept":float(surrogate.ridge.intercept_),
        "group_residual_scale":surrogate.group_scales.tolist(),
        "conformal_subject_count":bound.calibration_subjects,
        "alpha":bound.alpha,
        "cutoff_rank":bound.cutoff_rank,
        "cutoff":bound.quantile,
    },indent=2),encoding="utf8")
    manifest={
        "experiment":"mathematical zero-query regression plus subject-block conformal certificate",
        "dataset":"UCI HAR official fixed train/test subject partition",
        "dataset_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed":args.seed,"model_type":args.model,
        "model_fit_subjects":roles["model_fit"],
        "surrogate_fit_subjects":roles["surrogate_fit"],
        "calibration_subjects":roles["conformal_calibrate"],
        "outer_test_subjects":sorted(map(int,np.unique(sv))),
        "subject_roles_mutually_disjoint":True,
        "n_test_windows":len(test),
        "same_windows_per_subject_in_calibration_and_test":
            args.windows_per_subject,
        "classifier_fitted_model_sha256":calls["fitted_model_sha256"],
        "training_channel_standardizer_sha256":
            calls["training_standardizer_sha256"],
        "n_sensitivities_fit_offline":len(proxy)*M,
        "n_sensitivities_calibrate_offline":len(calib)*M,
        "additional_training_sensitivities_model_calls":
            (len(proxy)+len(calib))*M,
        "training_surrogate_model_examples_including_baseline":
            fit_model_examples,
        "calibration_model_examples_including_baseline":
            calibration_model_examples,
        "cached_test_baseline_model_examples":len(test),
        "offline_Q36_reference_model_examples":len(test)*M,
        "test_budgeted_model_examples_actual":actual_extra_test,
        "certificate_budgeted_model_examples_actual":
            total_online_certificate_queries,
        "test_model_examples_all_methods_and_Q36_reference":expected_test,
        "all_model_examples_measured":calls["examples"],
        "M":M,"Qmax":args.max_queries,"batch_size":args.batch_size,
        "subject_block_conformal":{
            "alpha":bound.alpha,"number_of_subjects":bound.calibration_subjects,
            "order_statistic_rank":bound.cutoff_rank,
            "one_sided_max_residual_quantile":bound.quantile,
            "confidence_statement":"P(all sampled windows and components covered for a new exchangeable subject) >= 1-alpha; exchangeability is an assumption, not verified by this benchmark",
            "model_and_surrogate_fit_before_calibration":True,
        },
        "heldout_simultaneous_bound_coverage":{
            "subject_blocks_covered":sum(heldout_bound_blocks.values()),
            "subject_blocks_total":len(heldout_bound_blocks),
            "per_subject":heldout_bound_blocks,
        },
        "online_certificate":{
            "certified_windows":certified_total,
            "abstained_windows":len(test)-certified_total,
            "wrong_certified_top3":wrong_certified_total,
            "no_full_reference_access_during_query_selection":True,
        },
        "memory_estimate_surrogate_coefficients_scaler_and_scales_bytes":int(
            surrogate.ridge.coef_.astype("float32").nbytes+
            surrogate.feature_scaler.mean_.astype("float32").nbytes+
            surrogate.feature_scaler.scale_.astype("float32").nbytes+
            surrogate.group_scales.astype("float32").nbytes+4),
        "limitations":[
            "This explains only sensitivity to one fixed interpolation intervention, not causal meaning.",
            "Train/test UCI HAR subject blocks may not be exchangeable; nominal conformal guarantee is CONDITIONAL.",
            "Few calibration subjects make the envelope wide; low certification yield is a valid negative result.",
            "Initial offline full Q=M probes to fit/calibrate cheap surrogate are nonzero and reported.",
            "Subject windows may overlap; subject, not window, is the statistical sampling unit.",
            "Raw-input MLP or ExtraTrees remains fixed throughout all analyses; no on-device hardware runtime.",
        ],
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("CONFORMAL_SCORE",
          "model=",args.model,"S=",bound.calibration_subjects,
          "alpha=",bound.alpha,"k=",bound.cutoff_rank,
          "qhat=",round(bound.quantile,6),flush=True)
    print("CONFORMAL_COVERAGE",len(heldout_bound_blocks),
          "covered=",sum(heldout_bound_blocks.values()),
          "certified=",certified_total,"wrong_certified=",wrong_certified_total,
          "cert_Q_total=",total_online_certificate_queries,flush=True)
    for row in summary:
        print(f"CHEAP_EXPLANATION method={row['method']} "
              f"overlap3={row['subject_macro_top3_overlap']:.6f} "
              f"mean_q={row['subject_macro_mean_queries']:.3f} "
              f"certified_fraction={row['subject_macro_certified_fraction']:.4f}",
              flush=True)
    return manifest,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--model",choices=("extratrees","raw_mlp"),default="raw_mlp")
    p.add_argument("--seed",type=int,default=57)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--windows-per-subject",type=int,default=6)
    p.add_argument("--alpha",type=float,default=.2)
    p.add_argument("--ridge-alpha",type=float,default=100.)
    p.add_argument("--max-queries",type=int,default=16)
    p.add_argument("--batch-size",type=int,default=4)
    run(p.parse_args())


if __name__=="__main__":
    main()
