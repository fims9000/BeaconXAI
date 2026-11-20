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
# Running "python scripts/export_raw_mlp_cpp.py" sets sys.path[0] to scripts/.
# Add the repository root before importing the original reference script.
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from sklearn.neural_network import MLPClassifier
from beaconxai.cpp_mlp_export import render_raw_mlp_header,save_parity_fixture
from beaconxai.datasets import fit_channel_standardizer,apply_standardizer,load_npz_dataset
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.resource_accounting import mlp_resource_projection
from beaconxai.cpp_adaptive_v2_export import (
    render_class_means_header, save_baseline_q16_fixture,
    render_board_reference_window_header,
)
from beaconxai.contrastive import fit_class_profiles, make_components, neutralize_interp
from scripts.run_q16_matched_ablation import class_cheap, best_q


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
    # Store RAW sensor inputs: C++ must now independently apply exported
    # TRAIN-derived channel normalization before its forward pass.
    save_parity_fixture(out/"model_parity_f32.bin",xv[idx],logit_gap)
    projected=mlp_resource_projection(
        model.coefs_,model.intercepts_,sample_shape=parity_x.shape[1:],
        n_classes=len(classes),n_components=36,query_budget=16)
    discrepancy=np.abs(logit_gap-logp_gap)
    adaptive_info=None
    if getattr(args, "adaptive_parity", False):
        q=16
        components=make_components(parity_x.shape[1:],12,"pamap3")
        class_profiles=fit_class_profiles(train,yt)
        class_means=np.stack([class_profiles[int(c)][0] for c in classes])
        means_exported=class_means.astype(np.float32)
        profile_header=render_class_means_header(
            class_means,sample_shape=parity_x.shape[1:])
        (out/"beacon_adaptive_profiles.hpp").write_text(
            profile_header,encoding="utf8")
        n_classes=len(classes)
        cheap_values=[]
        source_classes=[]
        selected_sets=[]
        reference_ranks=[]
        cutoff_gaps=[]
        min_effect_separations=[]
        quantized_profile_set_changes=0
        max_prior_quantization_diff=0.0
        for i,x in enumerate(parity_x):
            a=int(np.argmax(lp[i]))
            cheap=class_cheap(x,class_means[a],components)[2]
            quantized=class_cheap(x,means_exported[a],components)[2]
            if len(cheap)!=len(components) or not np.isfinite(cheap).all():
                raise AssertionError("bad original adaptive_v2 prior")
            selected=best_q(cheap,q)
            if set(selected)!=set(best_q(quantized,q)):
                quantized_profile_set_changes+=1
            max_prior_quantization_diff=max(max_prior_quantization_diff,
                float(np.max(np.abs(cheap-quantized))))
            altered=np.stack([neutralize_interp(x,components[j]) for j in selected])
            lp1=np.log(np.clip(model.predict_proba(
                altered.reshape(q,-1)),1e-12,1.0))
            moving=np.sort(lp1,axis=1)
            original_margin=np.sort(lp[i])[-1]-np.sort(lp[i])[-2]
            effects=np.abs(original_margin-(moving[:,-1]-moving[:,-2]))
            ordering=[selected[j] for j in np.argsort(-effects,kind="stable")]
            cheap_sorted=np.sort(cheap)
            cutoff=float(cheap_sorted[-q]-cheap_sorted[-q-1])
            effect_gap=float(min(
                abs(float(effects[u]-effects[v]))
                for u in range(q) for v in range(u+1,q)))
            source_classes.append(a)
            cheap_values.append(cheap)
            selected_sets.append(sorted(selected))
            reference_ranks.append(ordering)
            cutoff_gaps.append(cutoff)
            min_effect_separations.append(effect_gap)
        save_baseline_q16_fixture(
            out/"adaptive_v2_q16_parity.bin",xv[idx],
            np.asarray(source_classes),np.stack(cheap_values),
            np.asarray(selected_sets),np.asarray(reference_ranks),
            np.asarray(cutoff_gaps),np.asarray(min_effect_separations))
        # One frozen raw test window for eventual board self-check. The
        # expected answer NEVER enters the adaptive_v2 model/candidate path.
        reference_header=render_board_reference_window_header(
            xv[idx[0]],source_classes[0],
            np.asarray(selected_sets[0]),np.asarray(reference_ranks[0]))
        (out/"beacon_board_reference.hpp").write_text(
            reference_header,encoding="utf8")
        adaptive_info={
            "reference":"Original Python adaptive_v2 _component_cheap_scores"
                " + np.argpartition(-prior) + moving top-two log-prob margin",
            "n_components":len(components),
            "query_budget":q,
            "selected_ids_compared_as_set":True,
            "argpartition_candidate_order_not_stable":True,
            "n_profile_fp32_quantization_changes_to_candidate_set":
                quantized_profile_set_changes,
            "max_abs_prior_difference_from_fp32_class_mean":
                max_prior_quantization_diff,
            "n_ambiguous_prior_cutoffs_at_1e6":int(np.sum(
                np.asarray(cutoff_gaps)<=1e-6)),
            "n_ambiguous_measured_effect_orders_at_1e6":int(np.sum(
                np.asarray(min_effect_separations)<=1e-6)),
            "class_profile_fp32_bytes":int(means_exported.nbytes),
            "exported_class_profile_bytes":len(profile_header.encode()),
            "raw_input_reference_labels_never_read":True
        }
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
        "parity_fixture_contains_raw_unnormalized_sensor_input":True,
        "cpp_training_mu_sigma_parity_required":True,
        "not_firmware_or_MCU_validated":True,
        "adaptive_v2_baseline_parity":adaptive_info,
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print(json.dumps({
        "parity_n":len(parity_x),
        "f32_model_bytes":manifest["float32_model_parameter_bytes"],
        "max_clipping_semantic_difference":manifest[
            "max_abs_difference_logit_gap_vs_clipped_log_probability_gap"],
        "header_bytes":len(header.encode()),
        "not_hardware_validated":True,
        "baseline_adaptive_q16_parity":adaptive_info,
    },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=57)
    p.add_argument("--parity-samples",type=int,default=12)
    p.add_argument("--adaptive-parity",action="store_true",
                   help="Export training class means plus original Q16 set/rank fixtures")
    run(p.parse_args())


if __name__=="__main__":
    main()
