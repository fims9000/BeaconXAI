#!/usr/bin/env python3
"""Matched Q=16 PAMAP2 ablation: published adaptive_v2 vs contrastive prototype.

Single model, same injected windows, same components and interpolation, same cached
baseline for every method. No test label/true injected component enters selection.

The original adaptive_v2 scoring changes the top-1 class after intervention; a
second version holds the source/runner-up pair fixed for a paired comparison.
All Q16 methods evaluate Q=16 perturbed *examples*, with identical BATCHED
model inference (the Q model evaluations are never treated as one evaluation).

Outputs per-sample/per-subject metrics; no population inference from two subjects.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beaconxai.contrastive import (
    _order_from_prior, contrastive_prior, fit_class_profiles,
    make_components, neutralize_interp,
)
from scripts.run_adaptive_v2_localization import _component_cheap_scores
from scripts.run_contrastive_q16 import mask_ranking, ranking_metrics


METHODS = (
    "adaptive_v2_original_q16",
    "adaptive_v2_fixed_pair_q16",
    "contrastive_q16",
    "uniform_q16",
    "rise_like_q16",
    "kernelshap_like_q16",
    "adaptive_v2_cheap_q0",
    "contrastive_cheap_q0",
    "energy_q0",
    "variance_q0",
)


def class_cheap(x: np.ndarray, mean: np.ndarray,
                components) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use EXACT scores from the repository's original adaptive_v2."""
    ng = len({c.group for c in components})
    nt = len({c.time_bin for c in components})
    groups = [(components[g * nt].c0, components[g * nt].c1) for g in range(ng)]
    times = [(components[b].t0, components[b].t1) for b in range(nt)]
    return _component_cheap_scores(x, mean, groups, times)


def basic_statistics(x: np.ndarray, components) -> tuple[np.ndarray, np.ndarray]:
    energy, variance = [], []
    for c in components:
        p = x[c.t0:c.t1, c.c0:c.c1]
        energy.append(float(np.mean(p * p)))
        variance.append(float(np.var(p)))
    return np.array(energy), np.array(variance)


def best_q(scores: np.ndarray, q: int) -> list[int]:
    """Replicate original np.argpartition candidate selection (ties included)."""
    if q < 1 or q > len(scores):
        raise ValueError("Invalid Q")
    return (np.argpartition(-scores, kth=q-1)[:q] if q < len(scores)
            else np.arange(len(scores))).astype(int).tolist()


def evaluate_selected(x: np.ndarray, components, component_ids: list[int],
                      predict_many, baseline: np.ndarray,
                      a: int, b: int, scoring: str = "fixed") -> list[int]:
    """Q batched model evaluations; no unqueried component gets an invented score."""
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("Repeated component would consume budget twice")
    patches = np.stack([neutralize_interp(x, components[j]) for j in component_ids])
    outputs = np.asarray(predict_many(patches))
    if len(outputs) != len(component_ids):
        raise ValueError("model changed number of evaluations")
    if scoring == "fixed":
        m0 = float(baseline[a] - baseline[b])
        m1 = outputs[:, a] - outputs[:, b]
    elif scoring == "moving":
        sorted_outputs = np.sort(outputs, axis=1)
        m1 = sorted_outputs[:, -1] - sorted_outputs[:, -2]
        sorted_base = np.sort(baseline)
        m0 = float(sorted_base[-1] - sorted_base[-2])
    else:
        raise ValueError(scoring)
    strengths = np.abs(m0 - m1)
    # Stable order preserves deterministic tie handling. All outputs are measured.
    return [component_ids[j] for j in np.argsort(-strengths, kind="stable")]


def evaluate_one_method(method: str, x: np.ndarray, components, profiles,
                        predict_many, baseline: np.ndarray, q: int, seed: int):
    """Returns (ranked component IDs, billed additional model examples, latency)."""
    a = int(np.argmax(baseline))
    temp = baseline.copy()
    temp[a] = -np.inf
    b = int(np.argmax(temp))
    t0 = time.perf_counter()
    if method.startswith("adaptive_v2"):
        _e, _p, combined = class_cheap(x, profiles[a][0], components)
        if method == "adaptive_v2_cheap_q0":
            ranking = np.argsort(-combined, kind="stable").astype(int).tolist()
            return ranking, 0, time.perf_counter() - t0
        selected = best_q(combined, q)
        scoring = "moving" if method == "adaptive_v2_original_q16" else "fixed"
        ranking = evaluate_selected(x, components, selected, predict_many, baseline, a, b, scoring)
        return ranking, q, time.perf_counter() - t0

    if method.startswith("contrastive"):
        prior = contrastive_prior(x, a, b, profiles, components)
        if method == "contrastive_cheap_q0":
            ranking = np.argsort(-np.abs(prior), kind="stable").astype(int).tolist()
            return ranking, 0, time.perf_counter() - t0
        selected = _order_from_prior(prior, q, seed, 0.25)
        ranking = evaluate_selected(x, components, selected, predict_many, baseline, a, b)
        return ranking, q, time.perf_counter() - t0

    if method == "uniform_q16":
        selected = np.random.default_rng(seed).choice(len(components), size=q, replace=False).astype(int).tolist()
        ranking = evaluate_selected(x, components, selected, predict_many, baseline, a, b)
        return ranking, q, time.perf_counter() - t0

    if method in ("energy_q0", "variance_q0"):
        energy, variance = basic_statistics(x, components)
        values = energy if method == "energy_q0" else variance
        ranking = np.argsort(-values, kind="stable").astype(int).tolist()
        return ranking, 0, time.perf_counter() - t0

    if method in ("rise_like_q16", "kernelshap_like_q16"):
        g0 = float(baseline[a] - baseline[b])
        name = "rise_like" if method.startswith("rise") else "kernelshap_like"
        ranking = mask_ranking(x, predict_many, components, a, b, g0, name, q, seed)
        return ranking, q, time.perf_counter() - t0

    raise ValueError(f"Unknown method: {method}")


def summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Aggregate seeds within window, then present per-subject descriptives.

    Mean, not CI: only two held-out subjects and overlapping windows invalidate
    naive independent-window inferential confidence intervals.
    """
    by_method = []
    by_subject = []
    for method in METHODS:
        m_rows = [r for r in rows if r["method"] == method]
        if not m_rows:
            continue
        for subject in sorted({r["subject"] for r in m_rows}):
            sr = [r for r in m_rows if r["subject"] == subject]
            n = len(set(r["sample"] for r in sr))
            by_subject.append({
                "method": method, "subject": int(subject), "windows": n,
                "hit1": float(np.mean([r["hit1"] for r in sr])),
                "hit3": float(np.mean([r["hit3"] for r in sr])),
                "hit5": float(np.mean([r["hit5"] for r in sr])),
                "mrr": float(np.mean([r["mrr"] for r in sr])),
                "runtime_ms_median": float(np.median([r["runtime_ms"] for r in sr])),
            })
        subs = [r for r in by_subject if r["method"] == method]
        by_method.append({
            "method": method, "extra_model_evaluations": int(m_rows[0]["extra_model_evaluations"]),
            "windows": len(set(r["sample"] for r in m_rows)),
            "subjects": len(subs),
            "repeats": len(set(r["trial"] for r in m_rows)),
            "hit1": float(np.mean([r["hit1"] for r in m_rows])),
            "hit3": float(np.mean([r["hit3"] for r in m_rows])),
            "hit5": float(np.mean([r["hit5"] for r in m_rows])),
            "mrr": float(np.mean([r["mrr"] for r in m_rows])),
            "subject_macro_hit3": float(np.mean([r["hit3"] for r in subs])),
            "runtime_ms_median": float(np.median([r["runtime_ms"] for r in m_rows])),
            "runtime_ms_mean": float(np.mean([r["runtime_ms"] for r in m_rows])),
        })
    return by_method, by_subject


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def run(args):
    from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
    from scripts.run_component_conflict_benchmark import (
        _anfis_features, _inject_component_conflicts, _train_extratrees_local,
    )
    path = Path(args.npz_path)
    if not path.exists():
        raise FileNotFoundError(path)
    x_train, y_train, x_test, y_test = load_npz_dataset(path)
    with np.load(path) as f:
        if "subject_test" not in f:
            raise ValueError("NPZ has no subject_test. Recreate it with preprocess_pamap2.py")
        subject_test = f["subject_test"].astype(int)

    mu, sigma = fit_channel_standardizer(x_train)
    x_train, x_test = (apply_standardizer(x_train, mu, sigma),
                       apply_standardizer(x_test, mu, sigma))
    if args.max_test > 0 and args.max_test < len(x_test):
        chosen = np.random.default_rng(args.seed).choice(len(x_test), args.max_test, replace=False)
        x_test, y_test, subject_test = x_test[chosen], y_test[chosen], subject_test[chosen]
    if len(np.unique(y_test)) < 2:
        raise ValueError("Need at least two different test activities for injected contrasts")

    t_train = time.perf_counter()
    model = _train_extratrees_local(x_train, y_train, args.trees, .7, 1)
    train_sec = time.perf_counter() - t_train
    profiles = fit_class_profiles(x_train, y_train)
    components = make_components(x_test.shape[1:], args.time_bins, "pamap3")
    if args.budget > len(components):
        raise ValueError("Q exceeds component count")
    groups = [(c.c0, c.c1) for c in components if c.time_bin == 0]
    times = [(c.t0, c.t1) for c in components if c.group == 0]

    x_eval, is_injected, true_components = _inject_component_conflicts(
        x_test, y_test, groups, times, args.conflict_ratio, args.seed+101
    )
    injected_idx = np.flatnonzero(is_injected)
    if not len(injected_idx):
        raise ValueError("No injected windows produced")
    x_eval = x_eval[injected_idx]
    y_truth = true_components[injected_idx]
    subject_eval = subject_test[injected_idx]
    def predict_many(batch):
        features = _anfis_features(np.asarray(batch))
        probs = model.model.predict_proba(features)
        return np.log(np.clip(probs, 1e-12, 1))
    # Original prediction: exactly one shared cached model evaluation per window.
    original = predict_many(x_eval)
    rows = []
    for trial in range(args.repeats):
        for i, x in enumerate(x_eval):
            seed = args.seed+trial*1000003+i*271
            for method in METHODS:
                ordering, q_used, elapsed = evaluate_one_method(
                    method, x, components, profiles, predict_many,
                    original[i], args.budget, seed
                )
                if q_used not in (0, args.budget):
                    raise AssertionError("Incorrect model-evaluation accounting")
                met = ranking_metrics(ordering, int(y_truth[i]), len(components))
                rows.append({
                    "trial": trial, "sample": i, "subject": int(subject_eval[i]),
                    "method": method, "true_component": int(y_truth[i]),
                    "extra_model_evaluations": q_used,
                    "runtime_ms": 1000.0 * elapsed,
                    **met,
                })
        print(f"completed {trial+1}/{args.repeats}, {len(x_eval)} injected windows", flush=True)

    per_method, per_subject = summarize(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"per_sample.csv", rows)
    write_csv(out/"summary.csv", per_method)
    write_csv(out/"per_subject.csv", per_subject)
    info = {
        "data": "UCI PAMAP2", "source_npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": "ExtraTrees(120) on _anfis_features; source script's seed=42",
        "train_time_sec": train_sec, "test_subjects": sorted(set(subject_eval.tolist())),
        "n_injected_windows": len(x_eval), "n_test_sampled": len(x_test),
        "M": len(components), "time_bins": args.time_bins,
        "q_additional": args.budget, "baseline_cached_per_window": 1,
        "subject_dependence_warning": "Per-subject descriptive only; two subjects do not support generalization inference",
        "model_evaluation": "Q examples processed in one batch per method; one cached baseline per window",
        "neutralizer": "Original repository's per-component interpolation for all Q16 methods",
        "prior_memory_bytes": {
            "adaptive_v2_class_means_only": int(sum(a.nbytes for a, b in profiles.values())),
            "contrastive_class_mean_and_var": int(sum(a.nbytes+b.nbytes for a, b in profiles.values())),
        },
        "scoring": {
            "adaptive_v2_original_q16": "abs(margin_original - recomputed_top2_margin_after)",
            "adaptive_v2_fixed_pair_q16": "abs((score_a-score_b)_original - (score_a-score_b)_after)",
            "contrastive_q16": "fixed original class pair as previous experiment",
            "other_mask_methods": "regularized masking controls, not official SHAP / RISE",
        },
        "runtime_notice": "Single GH runner, model evaluations batched; timing exploratory, not edge/MCU benchmark",
        "number_of_repeats": args.repeats,
    }
    (out/"manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== matched Q16 results (descriptive) ===")
    for r in per_method:
        print(f"{r['method']:27s} Q={r['extra_model_evaluations']:2d} "
              f"hit3={r['hit3']:.4f} subj_macro={r['subject_macro_hit3']:.4f} "
              f"median_ms={r['runtime_ms_median']:.2f}", flush=True)
    for s in per_subject:
        if s["method"] in ("adaptive_v2_original_q16", "adaptive_v2_fixed_pair_q16",
                           "contrastive_q16", "uniform_q16"):
            print(f"SUBJECT {s['subject']} {s['method']} n={s['windows']} hit3={s['hit3']:.4f}", flush=True)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path", type=Path, default=Path("data/pamap2_acc9_w200s100_p095.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs_composite/q16_matched_ablation"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trees", type=int, default=120)
    p.add_argument("--max-test", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-bins", type=int, default=12)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--conflict-ratio", type=float, default=.5)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    cli()
