#!/usr/bin/env python3
"""Full-Q perturbation-sensitivity reference on unmodified PAMAP2 windows.

This answers: how closely do cheap/Q16 methods reproduce a Q=M model-response
ranking when they use the same interpolation intervention? It does NOT
establish ground-truth causal attribution. The Q=M reference is computed
strictly offline and never passed to the candidate-selection methods.

Train/test split is by participant. Subject-balanced sampling is seed-fixed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beaconxai.contrastive import fit_class_profiles, make_components, neutralize_interp
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.sensitivity_reference import (
    moving_margin_effects, rank_effects, reference_metrics, subject_macro_summary
)
from scripts.run_component_conflict_benchmark import _anfis_features, _train_extratrees_local
from scripts.run_q16_matched_ablation import evaluate_one_method, write_csv

METHODS = (
    "adaptive_v2_cheap_q0",
    "adaptive_v2_original_q16",
    "adaptive_v2_time_balanced_q16",
    "adaptive_v2_group_balanced_q16",
    "uniform_q16",
)


def run(args) -> None:
    path = Path(args.npz_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    xt, yt, xv, _ = load_npz_dataset(path)
    with np.load(path) as f:
        if not {"subject_train", "subject_test"}.issubset(f.files):
            raise ValueError("NPZ must contain subject_train and subject_test")
        st, sv = f["subject_train"], f["subject_test"]
    if set(map(int,np.unique(st))) & set(map(int,np.unique(sv))):
        raise ValueError("participant leakage between training and testing")
    idx = balanced_subject_sample(sv, args.max_test, args.seed)
    subj = sv[idx].astype(int)
    mu, std = fit_channel_standardizer(xt)
    train = apply_standardizer(xt, mu, std)
    test = apply_standardizer(xv[idx], mu, std)
    comps = make_components(test.shape[1:], args.time_bins, "pamap3")
    M = len(comps)
    grid_spec = getattr(args, "q_grid", "")
    if grid_spec:
        budgets = [int(part.strip()) for part in grid_spec.split(",")]
        if not budgets or len(budgets) != len(set(budgets)):
            raise ValueError("Budget grid must contain unique integers")
        plans = [("adaptive_v2_cheap_q0", "adaptive_v2_cheap_q0", 0)]
        for q in budgets:
            for method in METHODS:
                if method.endswith("_q0"):
                    continue
                plans.append((method.replace("_q16", f"_q{q}"), method, q))
    else:
        budgets = [args.budget]
        plans = [(method, method, 0 if method.endswith("_q0") else args.budget)
                 for method in METHODS]
    if any(q < 3 or q >= M for q in budgets):
        raise ValueError("Need 3 <= Q < M for full-budget reference comparisons")
    t0 = time.perf_counter()
    model_type = getattr(args, "model", "extratrees")
    if model_type == "extratrees":
        classifier = _train_extratrees_local(train, yt, args.trees, .7, 1)
        classes = np.asarray(classifier.model.classes_, dtype=int)

        def raw_probabilities(batch):
            return classifier.model.predict_proba(_anfis_features(np.asarray(batch)))
        model_description = "ExtraTrees on engineered ANFIS features"
    elif model_type == "raw_mlp":
        from sklearn.neural_network import MLPClassifier
        classifier = MLPClassifier(hidden_layer_sizes=(32,),alpha=.01,
                                   batch_size=128,max_iter=30,early_stopping=True,
                                   n_iter_no_change=6,random_state=args.seed)
        classifier.fit(train.reshape(len(train), -1), yt)
        classes = np.asarray(classifier.classes_,dtype=int)

        def raw_probabilities(batch):
            array = np.asarray(batch)
            return classifier.predict_proba(array.reshape(len(array), -1))
        model_description = "Raw-window MLPClassifier(hidden=32, alpha=.01, max_iter=30)"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    train_time_sec = time.perf_counter() - t0
    if (classes.ndim != 1 or not len(classes) or np.any(classes < 0)
        or len(set(classes.tolist())) != len(classes)):
        raise ValueError("Unexpected classifier class index mapping")
    n_classes = int(max(classes.max(),np.max(yt))) + 1
    profiles = fit_class_profiles(train, yt)

    def predict_many(batch):
        raw = raw_probabilities(np.asarray(batch))
        p = np.full((len(raw), n_classes), 1e-12, dtype=float)
        p[:,classes] = raw
        return np.log(np.clip(p, 1e-12, 1))

    baseline = predict_many(test)
    component_ids = list(range(M))
    rows = []
    reference_rows = []
    for i, x in enumerate(test):
        ref_batch = np.stack([neutralize_interp(x, c) for c in comps])
        ref_out = predict_many(ref_batch)  # M offline model evaluations
        scores = moving_margin_effects(baseline[i], ref_out)
        reference = rank_effects(scores, component_ids)
        ordered_scores = np.sort(scores)
        max_change = float(ordered_scores[-1])
        cutoff_gap = float(ordered_scores[-3] - ordered_scores[-4])
        reference_rows.append({
            "sample": i, "subject": int(subj[i]),
            "original_window_index": int(idx[i]),
            "reference_max_shift": max_change,
            "reference_top3_gap": cutoff_gap,
            "reference_top3": ",".join(str(j) for j in reference[:3]),
            "reference_changed_by_ge_001": int(max_change >= .01),
            "reference_top3_tie": int(cutoff_gap <= 1e-6),
        })
        for label, method, q in plans:
            # Every strategy/budget is evaluated directly, not inferred from
            # the complete Q=M responses used as the offline reference.
            order, extra, duration = evaluate_one_method(
                method, x, comps, profiles, predict_many, baseline[i],
                q if q else budgets[0], seed=args.seed + 100003*i
            )
            if extra != q:
                raise AssertionError("Incorrect Q accounting")
            metrics = reference_metrics(reference, order, M, extra)
            rows.append({
                "sample": i, "subject": int(subj[i]),
                "original_window_index": int(idx[i]),
                "method": label, "extra_model_evaluations": extra,
                "runtime_ms": 1000*duration,
                "reference_max_shift": max_change,
                "reference_top3_gap": cutoff_gap,
                "reference_top3": ",".join(str(j) for j in reference[:3]),
                "estimated_top3": ",".join(str(j) for j in order[:3]),
                **metrics,
            })
        if (i+1) % 32 == 0:
            print(f"completed {i+1}/{len(test)} clean windows", flush=True)

    summary, per_subject = subject_macro_summary(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_sample.csv", rows)
    write_csv(out / "per_subject.csv", per_subject)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "reference_windows.csv", reference_rows)

    manifest = {
        "dataset": "PAMAP2 (original unmodified test signals)",
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reference": "full Q=M model margin sensitivity, same neutralization operator",
        "reference_is_causal_ground_truth": False,
        "no_injected_synthetic_patch": True,
        "model": model_description,
        "model_type": model_type,
        "model_class_indices": classes.tolist(),
        "model_training_seconds": train_time_sec,
        "model_fit_on_training_subjects_only": True,
        "train_subjects": sorted(map(int,np.unique(st))),
        "test_subjects": sorted(map(int,np.unique(sv))),
        "per_subject_windows": {str(s):int((subj==s).sum()) for s in np.unique(subj)},
        "n_windows": len(test), "M": M, "Q": args.budget,
        "budget_grid": budgets,
        "budget_grid_evaluated_directly_not_simulated": True,
        "additional_reference_queries_per_window_offline": M,
        "cached_baseline_queries_per_window": 1,
        "additional_model_queries_by_method": {
            label: q for label, _, q in plans
        },
        "no_test_labels_or_reference_ranks_passed_to_methods": True,
        "reference_stability_exploratory_thresholds": {
            "max_margin_shift": .01, "top3_cutoff_gap": 1e-6
        },
        "limitations": [
            "Full perturbation sensitivity is not causal or expert-validated attribution",
            "Interpolating signal components may yield out-of-distribution patterns",
            "Overlapping windows are dependent; per-subject aggregates are descriptive",
            "Q=M reference model calls are charged to OFFLINE measurement, not deployment",
            "No MCU RAM/latency/energy measurement; the classifier uses engineered features",
        ],
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("=== Full-budget Q=M sensitivity reference (descriptive only) ===",flush=True)
    for row in summary:
        print(f"{row['method']:34s} Q={row['extra_model_evaluations']:2d} "
              f"overlap3={row['subject_macro_top3_overlap']:.4f} "
              f"reference_top1_in3={row['subject_macro_oracle_top1_in_top3']:.4f}",
              flush=True)
    for row in per_subject:
        if row["method"] in ("adaptive_v2_original_q16","adaptive_v2_time_balanced_q16","adaptive_v2_cheap_q0"):
            print(f"SUBJECT {row['subject']} {row['method']} "
                  f"n={row['n_windows']} overlap3={row['top3_overlap']:.4f} "
                  f"n_strong_ref={row['n_reference_effect_ge_001']} "
                  f"n_stable_ref={row['n_stable_reference_top3']}",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,default=Path("outputs_composite/full_sensitivity_reference"))
    p.add_argument("--max-test",type=int,default=128)
    p.add_argument("--time-bins",type=int,default=12)
    p.add_argument("--budget",type=int,default=16)
    p.add_argument("--q-grid",default="",help="Optional comma-separated Q, e.g. 8,16,32")
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--model",choices=("extratrees","raw_mlp"),default="extratrees")
    p.add_argument("--seed",type=int,default=57)
    run(p.parse_args())


if __name__ == "__main__":
    main()
