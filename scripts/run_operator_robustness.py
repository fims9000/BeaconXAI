#!/usr/bin/env python3
"""Intervention-operator robustness of BEACON on naturally observed UCI HAR/PAMAP2.

Compare full M=36 moving-margin intervention rankings from:
(1) interpolation neutralization and (2) replacing each component with a
real *training* class-rival medoid, tapered at borders. This is a diagnostic
of operator dependence, NOT proof of physical plausibility or causal truth.

The test subject activity label is NEVER read by the model or explanation.
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
from beaconxai.datasets import load_npz_dataset, fit_channel_standardizer, apply_standardizer
from beaconxai.contrastive import fit_class_profiles, make_components, neutralize_interp
from beaconxai.real_donor import fit_real_donor_bank, make_target_replacement
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.sensitivity_reference import (
    moving_margin_effects, rank_effects, reference_metrics, subject_macro_summary
)
from scripts.run_component_conflict_benchmark import _anfis_features, _train_extratrees_local
from scripts.run_q16_matched_ablation import best_q, class_cheap, evaluate_selected, write_csv


def study(args):
    path=Path(args.npz_path)
    xt,yt,xv,_=load_npz_dataset(path)
    with np.load(path) as z:
        st,sv=z["subject_train"],z["subject_test"]
    if set(map(int,np.unique(st)))&set(map(int,np.unique(sv))):
        raise ValueError("train/test subject overlap")
    mu,std=fit_channel_standardizer(xt)
    train=apply_standardizer(xt,mu,std)
    chosen=balanced_subject_sample(sv,args.max_test,args.seed)
    test=apply_standardizer(xv[chosen],mu,std)
    sid=sv[chosen]
    comps=make_components(test.shape[1:],args.time_bins,"pamap3")
    M=len(comps)
    if args.budget < 3 or args.budget >= M:
        raise ValueError("Need 3<=Q<M")
    profiles=fit_class_profiles(train,yt)
    bank=fit_real_donor_bank(train,yt,comps,max_donors=1,seed=args.seed+51)
    mode=args.model
    if mode=="extratrees":
        fitted=_train_extratrees_local(train,yt,args.trees,.7,1)
        clf=fitted.model
        def proba(xs):return clf.predict_proba(_anfis_features(np.asarray(xs)))
    elif mode=="raw_mlp":
        from sklearn.neural_network import MLPClassifier
        clf=MLPClassifier(hidden_layer_sizes=(32,),alpha=.01,batch_size=128,
                          max_iter=30,early_stopping=True,n_iter_no_change=6,
                          random_state=args.seed)
        clf.fit(train.reshape(len(train),-1),yt)
        def proba(xs):
            a=np.asarray(xs)
            return clf.predict_proba(a.reshape(len(a),-1))
    else:
        raise ValueError("unknown model")
    classes=np.asarray(clf.classes_,dtype=int)
    if (len(classes)<2 or not np.isin(classes,list(bank.means)).all()
        or len(set(classes.tolist()))!=len(classes)):
        raise ValueError("training class-bank mapping invalid")
    nclasses=int(max(classes.max(),np.max(yt)))+1
    def model(xs):
        p=proba(np.asarray(xs))
        out=np.full((len(p),nclasses),1e-12,dtype=float)
        out[:,classes]=p
        return np.log(np.clip(out,1e-12,1))

    baselines=model(test)
    rows=[]
    breakdown=[]
    for i,x in enumerate(test):
        original=baselines[i]
        a=int(np.argmax(original))
        b=int(np.argsort(-original,kind="stable")[1])
        if b not in bank.donors:
            raise ValueError("predicted rival has no training medoid")
        # Intervention references are for *offline evaluation* only.
        interpolated=np.stack([neutralize_interp(x,c) for c in comps])
        medoid=np.stack([make_target_replacement(
            x,c,j,b,"medoid",bank,comps,taper=2)[0] for j,c in enumerate(comps)])
        full_interp=rank_effects(moving_margin_effects(original,model(interpolated)),list(range(M)))
        medoid_effects=moving_margin_effects(original,model(medoid))
        full_medoid=rank_effects(medoid_effects,list(range(M)))
        cheap=class_cheap(x,profiles[a][0],comps)[2]
        selected=best_q(cheap,args.budget)
        qmed=rank_effects(moving_margin_effects(original,model(medoid[selected])),selected)
        qint=evaluate_selected(x,comps,selected,model,original,a,b,scoring="moving")
        random_selected=np.random.default_rng(args.seed+i*37).choice(
            M,args.budget,replace=False).astype(int).tolist()
        qrand=rank_effects(moving_margin_effects(original,model(medoid[random_selected])),
                           random_selected)
        s=np.sort(medoid_effects)
        refmax=float(s[-1]);gap=float(s[-3]-s[-4])
        methods=(
            ("medoid_q16",qmed,args.budget),
            ("interpolation_q16",qint,args.budget),
            ("uniform_medoid_q16",qrand,args.budget),
            ("full_interpolation_q36",full_interp,M),
        )
        for method,order,cost in methods:
            metrics=reference_metrics(full_medoid,order,M,cost)
            rows.append({
                "method":method,"sample":i,"subject":int(sid[i]),
                "original_window_index":int(chosen[i]),
                "extra_model_evaluations":cost,
                "reference_max_shift":refmax,
                "reference_top3_gap":gap,
                "source_class_predicted":a,
                "rival_class_predicted":b,
                **metrics,
            })
        breakdown.append({
            "subject":int(sid[i]),"sample":i,"original_window_index":int(chosen[i]),
            "reference_top3_interpolation":",".join(map(str,full_interp[:3])),
            "reference_top3_real_medoid":",".join(map(str,full_medoid[:3])),
            "full_reference_top3_agreement":len(set(full_interp[:3])&set(full_medoid[:3]))/3,
            "medoid_reference_top3_cutoff_gap":gap,
            "medoid_reference_max_margin_shift":refmax,
            "medoid_reference_ambiguous":int(gap<=1e-6),
            "rival_class_predicted":b,
        })
    summ,subject=subject_macro_summary(rows)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"summary.csv",summ)
    write_csv(out/"per_subject.csv",subject)
    write_csv(out/"per_sample.csv",rows)
    write_csv(out/"operator_reference_comparison.csv",breakdown)
    manifest={
        "dataset":args.dataset,"npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_subjects":sorted(map(int,np.unique(st))),
        "test_subjects":sorted(map(int,np.unique(sv))),
        "n_samples":len(test),"Q":args.budget,"M":M,
        "model_type":mode,
        "replacement_operator":"train-only real rival-class medoid; component taper=2",
        "reference_operator":"same train-only rival medoid full Q=M",
        "comparison_operator":"interpolation neutralization full Q=M",
        "reference_is_causal_truth":False,
        "no_test_activity_labels_read_for_inference":True,
        "donor_bank_memory_bytes":bank.bank_bytes,
        "offline_full_reference_evaluations_per_window":2*M,
        "extra_model_queries_medoid_q16":args.budget,
        "extra_model_queries_interpolation_q16":args.budget,
        "extra_model_queries_random_medoid_q16":args.budget,
        "notes":[
            "Medoid from a rival activity class is a real training window but pasted segments may be unrealistic",
            "Full Q=M of different intervention operators can disagree; don't treat either as causal ground truth",
            "Taper alters seams; operator dependence is measured, not controlled away",
            "Participant windows overlap; report per-subject descriptive results only",
            "Donor bank RAM may be substantial; no physical TinyML RAM/energy/latency measured"
        ]
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("=== full interpolation vs full real-medoid operator on natural signals ===")
    for r in summ:
        print(f"{r['method']:28} Q={r['extra_model_evaluations']:2d} "
              f"overlap3_with_medoid_Q36={r['subject_macro_top3_overlap']:.4f}",flush=True)
    print("full_operator_reference_agreement=",
          float(np.mean([x["full_reference_top3_agreement"] for x in breakdown])),flush=True)
    print("medoid_reference_ambiguous=",sum(x["medoid_reference_ambiguous"] for x in breakdown),
          "/",len(breakdown),flush=True)


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--dataset",choices=("PAMAP2","UCI_HAR"),default="UCI_HAR")
    p.add_argument("--model",choices=("extratrees","raw_mlp"),default="extratrees")
    p.add_argument("--budget",type=int,default=16)
    p.add_argument("--time-bins",type=int,default=12)
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--max-test",type=int,default=64)
    p.add_argument("--seed",type=int,default=57)
    study(p.parse_args())


if __name__=="__main__":
    cli()
