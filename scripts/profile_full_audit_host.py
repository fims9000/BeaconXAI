#!/usr/bin/env python3
"""Reference CPU host profiling of a full BEACON dual-operator MLP audit.

Not an MCU benchmark: no compiled firmware, board flash map, tensor arena,
clock/voltage/current, energy, or real peak RAM measured. Reports projected
float32 tensor bytes and measured host-only latency at the actual model/Q.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from beaconxai.contrastive import fit_class_profiles, make_components, neutralize_interp
from beaconxai.class_counterfactual import replace_component
from beaconxai.datasets import fit_channel_standardizer, apply_standardizer, load_npz_dataset
from beaconxai.dual_operator_audit import dual_operator_audit
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.resource_accounting import mlp_resource_projection
from beaconxai.subject_sampling import balanced_subject_sample
from scripts.run_q16_matched_ablation import class_cheap


def _median_ms(fn, warmups: int, repetitions: int) -> float:
    if repetitions < 1 or warmups < 0:
        raise ValueError("invalid timer repetitions")
    for _ in range(warmups):
        fn()
    times=[]
    for _ in range(repetitions):
        t=time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns()-t)/1e6)
    return float(np.median(times))


def run(args) -> None:
    path=Path(args.npz_path)
    xt,yt,xv,yv=load_npz_dataset(path)
    with np.load(path) as data:
        a=set(map(int,np.unique(data["subject_train"])))
        b=set(map(int,np.unique(data["subject_test"])))
    if a&b:
        raise ValueError("subject overlap in input dataset")
    idx=balanced_subject_sample(np.load(path)["subject_test"],args.max_test,args.seed)
    mu,sigma=fit_channel_standardizer(xt)
    xtrain=apply_standardizer(xt,mu,sigma)
    test=apply_standardizer(xv[idx],mu,sigma)
    if args.budget < 6 or args.budget % 2:
        raise ValueError("dual operator requires even Q >=6")
    from sklearn.neural_network import MLPClassifier
    mlp=MLPClassifier(hidden_layer_sizes=(32,),alpha=.01,batch_size=128,
                      max_iter=30,early_stopping=True,n_iter_no_change=6,
                      random_state=args.seed)
    start=time.perf_counter()
    mlp.fit(xtrain.reshape(len(xtrain),-1),yt)
    fit_seconds=time.perf_counter()-start
    labels=mlp.classes_.astype(int)
    if len(labels)<2 or np.any(labels<0):
        raise ValueError("invalid classification labels")
    nclasses=int(max(np.max(labels),np.max(yt)))+1
    def model(x):
        z=np.asarray(x)
        ps=mlp.predict_proba(z.reshape(len(z),-1))
        full=np.full((len(z),nclasses),1e-12,dtype=float)
        full[:,labels]=ps
        return np.log(np.clip(full,1e-12,1))
    baseline=model(test)
    predicted=np.argmax(baseline,axis=1)
    measured_accuracy=float(np.mean(predicted==yv[idx]))
    components=make_components(test.shape[1:],args.time_bins,"pamap3")
    profiles=fit_class_profiles(xtrain,yt)
    bank=fit_real_donor_bank(xtrain,yt,components,max_donors=1,seed=args.seed+51)
    if len(bank.medoids) != len(labels):
        raise ValueError("training donor medoid coverage differs from model output")
    mem=mlp_resource_projection(
        mlp.coefs_,mlp.intercepts_,sample_shape=test.shape[1:],
        n_classes=nclasses,n_components=len(components),
        query_budget=args.budget,stored_profiles=2)
    if args.budget//2>len(components):
        raise ValueError("Q/2 exceeds components")
    rows=[]
    for i,x in enumerate(test):
        source=int(np.argmax(baseline[i]))
        rival=int(np.argsort(-baseline[i],kind="stable")[1])
        prior=class_cheap(x,profiles[source][0],components)[2]
        template=bank.medoids[rival]
        # The timer includes constructing/stacking perturbed samples and the
        # actual Q classifier predictions; excludes training and prior fitting.
        def audit():
            return dual_operator_audit(
                x,components,prior,model,baseline[i],template,budget=args.budget)
        result=audit()
        audit_ms=_median_ms(audit,args.warmups,args.repeats)
        selected=result.selected_components
        if result.extra_model_evaluations != args.budget:
            raise AssertionError("incorrect model-evaluation accounting")
        probes=np.concatenate((
            np.stack([neutralize_interp(x,components[j]) for j in selected]),
            np.stack([replace_component(x,components[j],template,taper=2)
                      for j in selected])
        ),axis=0)
        if len(probes)!=args.budget:
            raise AssertionError("generated probe count differs from Q")
        batch_ms=_median_ms(lambda:model(probes),args.warmups,args.repeats)
        serial_ms=_median_ms(
            lambda:[model(sample[None]) for sample in probes],
            args.warmups,args.repeats)
        rows.append({
            "sample":i,"subject":int(np.load(path)["subject_test"][idx[i]]),
            "model_predicted_class":source,
            "extra_model_examples":args.budget,
            "distinct_components":len(selected),
            "operators":2,
            "reported_by_uncalibrated_gate":int(result.reported),
            "audit_median_ms_host":audit_ms,
            "batch_predict_median_ms_host":batch_ms,
            "serial_predict_median_ms_host":serial_ms,
            "explicit_host_probe_batch_nbytes":int(probes.nbytes),
        })
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"per_window.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary={
        "source_dataset":"UCI HAR official subject-disjoint split",
        "npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_subjects":sorted(a),
        "test_subjects":sorted(b),
        "test_windows_sampled":len(rows),
        "model":"sklearn.MLPClassifier(hidden=32) raw input",
        "classification_accuracy_on_sampled_test_windows_diagnostic_only":measured_accuracy,
        "training_seconds_cpu_host":fit_seconds,
        "sklearn_pickle_size_bytes_reference_only":len(pickle.dumps(mlp,protocol=pickle.HIGHEST_PROTOCOL)),
        "median_per_window_audit_ms_host":float(np.median([r["audit_median_ms_host"] for r in rows])),
        "median_per_window_batched_predict_ms_host":float(np.median([r["batch_predict_median_ms_host"] for r in rows])),
        "median_per_window_serial_predict_ms_host":float(np.median([r["serial_predict_median_ms_host"] for r in rows])),
        "repetitions_per_window":args.repeats,
        "warmups_per_window":args.warmups,
        "host_python":sys.version,
        "host_platform":platform.platform(),
        "host_only_not_mcu_benchmark":True,
        "measured_ram_peak":None,
        "measured_energy_joules":None,
        "measured_board_flash_bytes":None,
        "f32_and_int8_only_projections_not_validated_deployment":True,
        "resources":mem,
    }
    (out/"host_resource_profile.json").write_text(
        json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps({
        "model":"raw-window MLP", "n":len(rows),
        "Q":args.budget,"accuracy_diagnostic":measured_accuracy,
        "median_audit_ms_HOST":summary["median_per_window_audit_ms_host"],
        "median_batched_forward_ms_HOST":summary["median_per_window_batched_predict_ms_host"],
        "median_serial_forward_ms_HOST":summary["median_per_window_serial_predict_ms_host"],
        "model_F32_projected_bytes":mem["model_parameters_projected_float32_bytes"],
        "profile_F32_projected_bytes":mem["training_profiles_float32_bytes"],
        "streaming_explicit_buffer_floor":mem["algorithmic_streaming_explicit_buffer_floor_bytes"],
        "batch_explicit_buffer_floor":mem["algorithmic_q_batch_explicit_buffer_floor_bytes"],
        "not_MCU_approved":True,
    },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--max-test",type=int,default=12)
    p.add_argument("--budget",type=int,default=16)
    p.add_argument("--time-bins",type=int,default=12)
    p.add_argument("--warmups",type=int,default=2)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--seed",type=int,default=57)
    run(p.parse_args())


if __name__=="__main__":
    main()
