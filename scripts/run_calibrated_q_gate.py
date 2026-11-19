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
from beaconxai.subject_sampling import balanced_subject_sample, inject_subject_balanced
from beaconxai.query_gate import cheap_gate_features, fit_query_gate, evaluate_gate
from beaconxai.selective_audit import run_selective_audit
from scripts.run_q16_matched_ablation import class_cheap, best_q, evaluate_selected, write_csv
from scripts.run_component_conflict_benchmark import (
    _anfis_features, _train_extratrees_local
)
from beaconxai.datasets import fit_channel_standardizer, apply_standardizer


def evaluate_split(x_train, y_train, x_eval, y_eval, sid_eval, *, seed, trees,
                   q, time_bins, max_test, gate=None):
    """Train on x_train only; select/query with no evaluation labels."""
    if len(np.unique(y_eval)) < 2:
        raise ValueError("cannot make different-class injected controls")
    mu, std = fit_channel_standardizer(x_train)
    xt = apply_standardizer(x_train, mu, std)
    xv = apply_standardizer(x_eval, mu, std)
    model = _train_extratrees_local(xt, y_train, trees, .7, 1)
    profiles = fit_class_profiles(xt, y_train)
    candidate_subjects = set(map(int, np.unique(sid_eval)))
    chosen = balanced_subject_sample(sid_eval, max_test, seed)
    xv, y_eval, sid_eval = xv[chosen], y_eval[chosen], sid_eval[chosen]
    if set(map(int, np.unique(sid_eval))) != candidate_subjects:
        raise AssertionError("subject disappeared in balanced sampling")
    comps = make_components(xv.shape[1:], time_bins, "pamap3")
    if q > len(comps) or q < 3:
        raise ValueError("invalid Q for hit@3")
    injected, present, target = inject_subject_balanced(
        xv, y_eval, sid_eval, comps, .5, seed+101
    )
    ix = np.flatnonzero(present)
    if not len(ix):
        raise ValueError("no injected examples")
    if set(map(int, np.unique(sid_eval[ix]))) != candidate_subjects:
        raise AssertionError("subject disappeared in synthetic injection")
    injected, targets, subjects = injected[ix], target[ix], sid_eval[ix]

    # Inner training may omit an activity found in validation. Align model
    # scores to the global activity IDs before selecting top-1/top-2.
    # Class-score dimensions depend only on the fitted classifier.
    # Held-out true labels may construct OFFLINE injections, but must never
    # determine the output class space used by the gate.
    trained_classes = np.asarray(model.model.classes_, dtype=int)
    n_classes = int(trained_classes.max()) + 1

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
        gold = int(targets[i])
        features = cheap_gate_features(base[i], cheap)
        # The inference path runs BEFORE the held-out oracle is inspected.
        # A Q0 choice sends NO perturbed examples to the classifier.
        audit = (run_selective_audit(x, comps, cheap, predict_many, gate, base[i])
                 if gate is not None else None)
        if audit is None:
            selected = best_q(cheap, q)
            queried_order = evaluate_selected(x, comps, selected, predict_many,
                                              base[i], a, 0, scoring="moving")
            gate_order = queried_order
            actual_gate_queries = q
            offline_control_queries = 0
            audit_wall_ms = None
        elif audit.extra_model_evaluations:
            queried_order = audit.ranking
            gate_order = audit.ranking
            actual_gate_queries = audit.extra_model_evaluations
            offline_control_queries = 0
            audit_wall_ms = audit.audit_wall_ms
        else:
            # OFFLINE matched Q16 control, evaluated only AFTER the Q0 gate
            # decision. These are NOT part of the deployment Q0 budget.
            gate_order = audit.ranking
            selected = best_q(cheap, q)
            queried_order = evaluate_selected(x, comps, selected, predict_many,
                                              base[i], a, 0, scoring="moving")
            actual_gate_queries = 0
            offline_control_queries = q
            audit_wall_ms = 0.0
        rows.append({"sample": i, "original_window_index": int(chosen[ix[i]]),
                     "subject": int(subjects[i]), **features,
                     "hit3_q0": int(gold in prior_order[:3]),
                     "hit3_q16": int(gold in queried_order[:3]),
                     "gate_hit3": int(gold in gate_order[:3]),
                     "actual_gate_queries": actual_gate_queries,
                     "offline_control_queries": offline_control_queries,
                     "audit_model_wall_ms": audit_wall_ms,
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
        max_test=args.max_test, gate=gate
    )
    outer_summary, per_subject = evaluate_gate(
        outer_rows, gate, cost_per_full_budget=args.lambda_cost
    )
    for row in outer_rows:
        row["gate_queries"] = args.budget if gate.should_query(row) else 0
        if row["actual_gate_queries"] != row["gate_queries"]:
            raise AssertionError("inference budget diverged from prospective gate")
        expected = row["hit3_q16"] if row["gate_queries"] else row["hit3_q0"]
        if row["gate_hit3"] != expected:
            raise AssertionError("prospective gate changed ranking semantics")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"outer_per_sample.csv", outer_rows)
    write_csv(out/"outer_per_subject.csv", per_subject)
    write_csv(out/"inner_calibration.csv", inner_rows)
    outer_counts = {str(s): sum(int(r["subject"]) == s for r in outer_rows)
                    for s in sorted(outer_subjects)}
    manifest = {"dataset": "PAMAP2", "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sampling": "round-robin subject sampling followed by stratified synthetic donor injections",
                "outer_injected_windows_per_subject": outer_counts,
                "single_example_subjects": [s for s, count in outer_counts.items() if count < 5],
                "outer_subjects": sorted(outer_subjects), "inner_validation_subjects": inner_subjects,
                "inner_training_subjects": [s for s in train_subjects if s not in inner_subjects],
                "chosen_gate": gate.to_dict(), "calibration": calibration,
                "outer": outer_summary, "q_additional": args.budget,
                "prospective_extra_model_evaluations": int(sum(r["actual_gate_queries"] for r in outer_rows)),
                "offline_control_model_evaluations": int(sum(r["offline_control_queries"] for r in outer_rows)),
                "prospective_queries_confirmed_against_gate": True,
                "label_and_injected_location_not_passed_to_gate": True,
                "calibration_independent_of_outer_subjects": True,
                "offline_full_Q_for_all_test_cases": True,
                "prospective_execution": "Q0 cases physically skip all perturbation evaluations; matched Q16 controls execute separately after the decision",
                "quality_endpoint": "hit@3 of artificially injected donor-class component",
                "limitations": [
                    "The gate estimates synthetic patch localization; not expert explanation correctness",
                    "Outer test outcomes must never be used to alter the gate",
                    "Windows overlap within subject; these are descriptive per-subject statistics",
                    "No latency, energy, or MCU RAM measured",
                    "Selective-stage query counts are executed, but offline Q16 controls are also evaluated for honest comparison",
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
