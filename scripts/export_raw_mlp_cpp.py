#!/usr/bin/env python3
"""Train raw-window MLP, export FP32 C++ arrays, and write parity fixtures.

Train/test subjects must be disjoint. No outer-test labels are passed to
the model selection or explanation. Output is a research artifact, not a
firmware image or proof of device feasibility.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from beaconxai.cpp_mlp_export import render_raw_mlp_header,save_parity_fixture
from beaconxai.datasets import fit_channel_standardizer,apply_standardizer,load_npz_dataset
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.resource_accounting import mlp_resource_projection


def run(args):
    fn=Path(args.npz_path)
    xt,yt,xv,_=load_npz_dataset(fn)
    with np.load(fn) as z:
        train_subjects=set(map(int,np.unique(z["subject_train"])))
        test_subjects=set(map(int,np.unique(z["subject_test"])))
        subject_test=z["subject_test"].astype(int)
    if train_subjects&test_subjects:
        raise ValueError("train/test participant overlap")
    mu,std=fit_channel_standardizer(xt)
    train=apply_standardizer(xt,mu,std)
    idx=balanced_subject_sample(subject_test,args.parity_samples,args.seed)
    parity_x=apply_standardizer(xv[idx],mu,std)
    model=MLPClassifier(hidden_layer_sizes=(32,),alpha=.01,
                        batch_size=128,max_iter=30,early_stopping=True,
                        n_iter_no_change=6,random_state=args.seed)
    t=time.perf_counter()
    model.fit(train.reshape(len(train),-1),yt)
    fit_seconds=time.perf_counter()-t
    classes=model.classes_.astype(int)
    if len(classes)<2 or not np.array_equal(classes,np.arange(len(classes))):
        raise ValueError("Exporter requires consecutive zero-indexed activity classes")
    header=render_raw_mlp_header(
        model.coefs_,model.intercepts_,mu,std,sample_shape=parity_x.shape[1:])

    flat=parity_x.reshape(len(parity_x),-1)
    z1=np.maximum(flat@model.coefs_[0]+model.intercepts_[0],0)
    logits=z1@model.coefs_[1]+model.intercepts_[1]
    logits_sorted=np.sort(logits,axis=1)
    logit_gap=logits_sorted[:,-1]-logits_sorted[:,-2]
    p=model.predict_proba(flat)
    lp=np.log(np.clip(p,1e-12,1.0))
    score_sorted=np.sort(lp,axis=1)
    logp_gap=score_sorted[:,-1]-score_sorted[:,-2]

    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    (out/"beacon_mlp_model.hpp").write_text(header,encoding="utf8")
    save_parity_fixture(out/"model_parity_f32.bin",parity_x,logit_gap)
    projected=mlp_resource_projection(
        model.coefs_,model.intercepts_,sample_shape=parity_x.shape[1:],
        n_classes=len(classes),n_components=36,query_budget=16)
    discrepancy=np.abs(logit_gap-logp_gap)
    manifest={
        "data_npz_sha256":hashlib.sha256(fn.read_bytes()).hexdigest(),
        "dataset":"UCI HAR official subject-disjoint train/test split",
        "train_subjects":sorted(train_subjects),
        "test_subjects":sorted(test_subjects),
        "n_parity_windows":len(parity_x),
        "parity_window_subjects":subject_test[idx].tolist(),
        "model":"MLPClassifier hidden=32, relu, six-class softmax",
        "training_seconds_host":fit_seconds,
        "float32_model_parameter_bytes":projected["model_parameters_projected_float32_bytes"],
        "output_classes":classes.tolist(),
        "input_shape":list(parity_x.shape[1:]),
        "max_abs_difference_logit_gap_vs_clipped_log_probability_gap":float(
            np.max(discrepancy)),
        "fraction_logit_gap_vs_logprob_gap_gt_1e3":float(
            np.mean(discrepancy>1e-3)),
        "export_float32_not_quantized":True,
        "training_label_used_for_fitting_only":True,
        "test_labels_not_read_for_export_or_parity":True,
        "not_firmware_or_MCU_validated":True,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print(json.dumps({
        "parity_n":len(parity_x),
        "f32_model_bytes":manifest["float32_model_parameter_bytes"],
        "max_clipping_semantic_difference":manifest[
            "max_abs_difference_logit_gap_vs_clipped_log_probability_gap"],
        "header_bytes":len(header.encode()),
        "not_hardware_validated":True,
    },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=57)
    p.add_argument("--parity-samples",type=int,default=12)
    run(p.parse_args())


if __name__=="__main__":
    main()
