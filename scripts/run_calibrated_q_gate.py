#!/usr/bin/env python3
"""Held-out PAMAP2 evaluation of a TRAIN-only calibrated Q0/Q16 gate.

Each outer test pair is untouched during calibration. Inner validation subjects
are selected from the outer training subjects, the inner model is fitted only
on the remaining training subjects, and threshold selection uses inner injected
windows. The outer model then refits on all outer-train subjects. Outcomes
are synthetic patch hit@3, NOT natural/causal explanation accuracy.
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
from beaconxai.contrastive import make_components, fit_class_profiles
from beaconxai.query_gate import cheap_gate_features, fit_query_gate, evaluate_gate
from scripts.run_q16_matched_ablation import class_cheap, best_q, evaluate_selected, write_csv
from scripts.run_component_conflict_benchmark import (
    _anfis_features, _train_extratrees_local, _inject_component_conflicts
)
from beaconxai.datasets import fit_channel_standardizer, apply_standardizer


def evaluate_split(x_train, y_train, x_eval, y_eval, sid_eval, *, seed, trees,
                   q, time_bins, max_test):
    """Train on x_train only; select/query with no evaluation labels."""
    if len(np.unique(y_eval)) < 2:
        raise ValueError("cannot make different-class injected controls")
    mu, std = fit_channel_standardizer(x_train)
    xt = apply_standardizer(x_train, mu, std)
    xv = apply_standardizer(x_eval, mu, std)
    model = _train_extratrees_local(xt, y_train, trees, .7, 1)
    profiles = fit_class_profiles(xt, y_train)
    if max_test > 0 and max_test < len(xv):
        chosen = np.random.default_rng(seed).choice(len(xv), max_test, replace=False)
        xv, y_eval, sid_eval = xv[chosen], y_eval[chosen], sid_eval[chosen]
    comps = make_components(xv.shape[1:], time_bins, "pamap3")
    if q > len(comps) or q < 3:
        raise ValueError("invalid Q for hit@3")
    groups = [(c.c0, c.c1) for c in comps if c.time_bin == 0]
    times = [(c.t0, c.t1) for c in comps if c.group == 0]
    injected, present, target = _inject_component_conflicts(
        xv, y_eval, groups, times, .5, seed+101
    )
    ix = np.flatnonzero(present)
    if not len(ix):
        raise ValueError("no injected examples")
    injected, targets, subjects = injected[ix], target[ix], sid_eval[ix]

    # Inner training may omit an activity found in validation. Align model
    # scores to the global activity IDs before selecting top-1/top-2.
    n_classes = int(max(np.max(y_train), np.max(y_eval))) + 1
    trained_classes = np.asarray(model.model.classes_, dtype=int)

    def predict_many(batch):
        raw = model.model.predict_proba(_anfis_features(np.asarray(batch)))
        full = np.full((len(raw), n_classes), 1e-12, dtype=float)
        full[:, trained_classes] = raw
        return np.log(np.clip(full, 1e-12, 1))

    base = predict_many(injected)  # one shared cached baseline per example
    rows = []
    for i, x in enumerate(injected):
        a = int(np.argmax(base[i]))
        cheap = class_cheap(x, profiles[a][0], comps)[2]
        prior_order = np.argsort(-cheap, kind="stable").astype(int).tolist()
        selected = best_q(cheap, q)
        # The original BEACON adaptive_v2, with moving top-two margin.
        queried_order = evaluate_selected(x, comps, selected, predict_many,
                                          base[i], a, 0, scoring="moving")
        gold = int(targets[i])
        features = cheap_gate_features(base[i], cheap)
        rows.append({"sample": i, "subject": int(subjects[i]),
                     **features, "hit3_q0": int(gold in prior_order[:3]),
                     "hit3_q16": int(gold in queried_order[:3]),
                     "true_component_for_offline_evaluation": gold})
    return rows


def run(args):
    path = Path(args.npz_path)
    with np.load(path) as z:
        required = ("x_train", "y_train", "x_test", "y_test", "subject_train", "subject_test")
        if any(k not in z for k in required):
            raise ValueError("NPZ lacks required train/test subject fields")
        xt, yt, xv, yv, st, sv = (z[k] for k in required)
    outer_subjects = set(int(s) for s in np.unique(sv))
    train_subjects = sorted(int(s) for s in np.unique(st))
    if outer_subjects.intersection(train_subjects):
        raise ValueError("outer train/test subject leakage")
    if len(train_subjects) < 4:
        raise ValueError("need >=4 outer training subjects for inner held-out subjects")
    # Chosen without reference to the outer-test examples or outcomes.
    inner_subjects = train_subjects[:2]
    inner_validation = np.isin(st, inner_subjects)
    if np.unique(yt[~inner_validation]).size < 2:
        raise ValueError("not enough inner-training classes")
    inner_rows = evaluate_split(
        xt[~inner_validation], yt[~inner_validation],
        xt[inner_validation], yt[inner_validation], st[inner_validation],
        seed=args.seed, trees=args.trees, q=args.budget,
        time_bins=args.time_bins, max_test=args.max_test
    )
    gate, calibration = fit_query_gate(
        inner_rows, budget=args.budget, cost_per_full_budget=args.lambda_cost
    )
    outer_rows = evaluate_split(
        xt, yt, xv, yv, sv, seed=args.seed+7919,
        trees=args.trees, q=args.budget, time_bins=args.time_bins,
        max_test=args.max_test
    )
    outer_summary, per_subject = evaluate_gate(
        outer_rows, gate, cost_per_full_budget=args.lambda_cost
    )
    for row in outer_rows:
        row["gate_queries"] = args.budget if gate.should_query(row) else 0
        row["gate_hit3"] = row["hit3_q16"] if row["gate_queries"] else row["hit3_q0"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"outer_per_sample.csv", outer_rows)
    write_csv(out/"outer_per_subject.csv", per_subject)
    write_csv(out/"inner_calibration.csv", inner_rows)
    manifest = {"dataset": "PAMAP2", "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "outer_subjects": sorted(outer_subjects), "inner_validation_subjects": inner_subjects,
                "inner_training_subjects": [s for s in train_subjects if s not in inner_subjects],
                "chosen_gate": gate.to_dict(), "calibration": calibration,
                "outer": outer_summary, "q_additional": args.budget,
                "label_and_injected_location_not_passed_to_gate": True,
                "calibration_independent_of_outer_subjects": True,
                "offline_full_Q_for_all_test_cases": True,
                "quality_endpoint": "hit@3 of artificially injected donor-class component",
                "limitations": [
                    "The gate estimates synthetic patch localization; not expert explanation correctness",
                    "Outer test outcomes must never be used to alter the gate",
                    "Windows overlap within subject; these are descriptive per-subject statistics",
                    "No latency, energy, or MCU RAM measured",
                    "Query savings are simulated after offline paired full-Q evaluation",
                    "Inner validation has only two subjects and may not generalize"
                ]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps({"gate": gate.to_dict(), "outer": outer_summary,
                      "inner_subjects": inner_subjects,
                      "outer_subjects": sorted(outer_subjects)}, indent=2), flush=True)


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs_composite/calibrated_q_gate"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trees", type=int, default=120)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--time-bins", type=int, default=12)
    p.add_argument("--max-test", type=int, default=128)
    p.add_argument("--lambda-cost", type=float, default=.05)
    run(p.parse_args())


if __name__ == "__main__":
    cli()
