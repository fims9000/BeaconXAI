#!/usr/bin/env python3
"""Held-out UCI HAR counterfactual search: valid target flips, not Q36 top-k.

For each original window: frozen classifier predicts class A, its runner-up B
is target. A TRAIN-only B donor supplies real segments. Compare single/pair search to cumulative multi-component edit paths under
the same hard model-evaluation budget (Q8/Q16). No test
activity label is used in donor choice, search, or model calls. Every model
evaluation, including post-search donor robustness checks, is counted.
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
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import load_npz_dataset, apply_standardizer
from beaconxai.real_donor import fit_real_donor_bank, nearest_context_index
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.verified_counterfactual import (
    find_verified_counterfactual, donor_stability_audit, splice,
)
from beaconxai.cf_path_search import search_counterfactual_path
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv


def run(args):
    path = Path(args.npz_path)
    raw_train, y_train, raw_test, _ = load_npz_dataset(path)
    with np.load(path) as z:
        train_ids = np.asarray(z["subject_train"], dtype=int)
        test_ids = np.asarray(z["subject_test"], dtype=int)
    if set(train_ids) & set(test_ids):
        raise ValueError("subject leakage")
    mu, std, profiles, model, calls = fit_predictor(
        raw_train, y_train, args.model, args.seed, args.trees
    )
    train = apply_standardizer(raw_train, mu, std)
    comps = make_components(train.shape[1:], 12, "pamap3")
    bank = fit_real_donor_bank(
        train, y_train, comps, max_donors=args.donors, seed=args.seed + 10
    )
    selected = balanced_subject_sample(test_ids, args.max_test, args.seed + 301)
    test = apply_standardizer(raw_test[selected], mu, std)
    base = model(test)
    budgets = sorted(set(int(b) for b in args.budgets.split(",")))
    if not budgets or any(b < 2 or b > len(comps) for b in budgets):
        raise ValueError("invalid budgets")
    rows = []
    for i, x in enumerate(test):
        scores = base[i]
        a = int(np.argmax(scores))
        b = int(np.argsort(-scores, kind="stable")[1])
        if a not in profiles or b not in bank.donors:
            raise ValueError("predicted class absent from train")
        rank = np.argsort(-np.abs(contrastive_prior(x, a, b, profiles, comps)),
                          kind="stable").astype(int).tolist()
        rng = np.random.default_rng(args.seed * 100003 + int(selected[i]))
        uniform = rng.permutation(len(comps)).astype(int).tolist()
        nearest = nearest_context_index(x, 0, bank, b, comps)
        donor = bank.donors[b][nearest]
        audit_donors = [bank.donors[b][j] for j in range(len(bank.donors[b]))
                        if j != nearest][:args.reserve]
        for q in budgets:
            methods = (
                ("contrastive", rank, False),
                ("uniform", uniform, False),
                ("contrastive_path", rank, True),
                ("uniform_path", uniform, True),
            )
            for name, order, use_path in methods:
                before = int(calls["examples"])
                if use_path:
                    search = search_counterfactual_path(
                        x, scores, donor, comps, order, model, target_class=b,
                        budget=q-args.reserve, max_changed_components=8,
                        seed_singletons=min(3, q-args.reserve), taper=args.taper,
                    )
                    query_count = search.calls
                    minimal_checked = search.local_minimality_checked
                else:
                    search = find_verified_counterfactual(
                        x, scores, donor, comps, order, model, target_class=b,
                        budget=q-args.reserve, max_changed_components=2,
                        single_fraction=args.single_fraction, taper=args.taper
                    )
                    query_count = search.evaluations
                    # For a confirmed pair both singleton edits were measured
                    # in the same first stage. This proves only 1-deletion
                    # minimality for that donor, not global minimality.
                    minimal_checked = bool(search.is_valid)
                remaining = q-query_count
                audit = donor_stability_audit(
                    x, scores, b, search.verified_edit or (),
                    audit_donors, comps, model,
                    budget_remaining=min(args.reserve, remaining),
                    taper=args.taper
                )
                actual_calls = int(calls["examples"]) - before
                if actual_calls != query_count+audit["tested"] or actual_calls > q:
                    raise AssertionError("counterfactual model budget accounting failed")
                changed = (splice(x, donor, comps, search.verified_edit,
                                  taper=args.taper) if search.is_valid else None)
                edit_distance = (float(np.linalg.norm(changed-x) /
                                       (max(np.linalg.norm(x), np.linalg.norm(donor)) + 1e-8))
                                 if changed is not None else None)
                rows.append({
                    "sample": i, "original_window_index": int(selected[i]),
                    "subject": int(test_ids[selected[i]]),
                    "model": args.model, "source": a, "target": b,
                    "method": name, "budget": q,
                    "verified": int(search.is_valid),
                    "status": search.status,
                    "n_components_changed": (len(search.verified_edit)
                                             if search.is_valid else None),
                    "components": (",".join(map(str, search.verified_edit))
                                   if search.is_valid else ""),
                    "relative_l2": edit_distance,
                    "local_minimality_checked": int(bool(minimal_checked)),
                    "search_calls": query_count,
                    "donor_audit_calls": audit["tested"],
                    "total_additional_calls": actual_calls,
                    "donor_target_flip_fraction": audit["flip_fraction"],
                    "donor_target_flip_count": audit["target_flips"],
                    "fixed_target_pair": True,
                    "selected_donor_train_index":
                        int(bank.selected_train_indices[b][nearest]),
                })
    summaries = []
    for q in budgets:
        for name in ("contrastive", "uniform", "contrastive_path", "uniform_path"):
            group = [r for r in rows if r["budget"] == q and r["method"] == name]
            subjects = sorted({r["subject"] for r in group})
            per_subject = []
            for sid in subjects:
                sg = [r for r in group if r["subject"] == sid]
                valid = [r for r in sg if r["verified"]]
                per_subject.append({
                    "subject": sid,
                    "validity": float(np.mean([r["verified"] for r in sg])),
                    "calls": float(np.mean([r["total_additional_calls"] for r in sg])),
                    "changed_components_if_valid": (
                        float(np.mean([r["n_components_changed"] for r in valid]))
                        if valid else None
                    ),
                })
            summaries.append({
                "method": name, "budget": q, "n_windows": len(group),
                "n_subjects": len(subjects),
                "subject_macro_validity": float(np.mean([r["validity"] for r in per_subject])),
                "subject_macro_mean_queries": float(np.mean([r["calls"] for r in per_subject])),
                "n_verified": sum(r["verified"] for r in group),
                "n_with_donor_audit": sum(r["donor_audit_calls"] > 0 for r in group),
                "n_verified_local_minimality": sum(r["verified"] and r["local_minimality_checked"]
                                                   for r in group),
                "n_verified_at_most_two": sum(r["verified"] and r["n_components_changed"] <= 2
                                              for r in group),
                "mean_changed_if_valid": (
                    float(np.mean([r["n_components_changed"] for r in group if r["verified"]]))
                    if any(r["verified"] for r in group) else None),
                "n_robust_on_both_extra_donors": sum(
                    r["verified"] and r["donor_audit_calls"] == args.reserve and
                    r["donor_target_flip_fraction"] == 1.0 for r in group),
            })
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_sample.csv", rows)
    write_csv(out / "summary.csv", summaries)
    manifest = {
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dataset": "UCI HAR, original held-out windows; no synthetic insertion",
        "train_subjects": sorted(map(int, np.unique(train_ids))),
        "test_subjects": sorted(map(int, np.unique(test_ids))),
        "model": args.model, "classifier_sha256": calls["fitted_model_sha256"],
        "train_standardizer_sha256": calls["training_standardizer_sha256"],
        "donor_bank_bytes": bank.bank_bytes,
        "budgets_additional_including_optional_donor_robustness": budgets,
        "reserve_donor_audits": args.reserve,
        "operator": "TRAIN class B actual window patches; seam taper",
        "target": "runner-up class in original model logits",
        "candidate_policies": ["class_contrastive_prior", "uniform"],
        "search_policies": ["singleton_then_pair", "multi_component_path_with_local_pruning"],
        "max_edited_components": {"pair_search": 2, "path_search": 8}, "taper": args.taper,
        "model_evaluations_total_including_initial_baselines": calls["examples"],
        "no_test_label_used_in_search": True,
        "paired_reference": "same x, classifier, target and donor for all policies within sample",
        "local_minimality_is_not_global": True,
        "limitations": [
            "Success means verified switch to preselected target B under this donor and operator only",
            "A spliced real donor fragment does not imply a physically plausible whole series",
            "First verified edit is not a global minimal counterfactual; subset minimality is flagged only when tested",
            "Donor audit is descriptive, not population coverage guarantee",
            "No model-verified target-class donor feasibility or sensor-physics realism guarantee",
            "Search may fail because of budget, donor, group limit, candidate restriction or model behavior",
            "No real MCU latency, energy, RAM, or causal attribution measurements",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", choices=["raw_mlp", "extratrees"], default="raw_mlp")
    p.add_argument("--budgets", default="8,16")
    p.add_argument("--max-test", type=int, default=36)
    p.add_argument("--donors", type=int, default=8)
    p.add_argument("--reserve", type=int, default=2)
    p.add_argument("--single-fraction", type=float, default=.5)
    p.add_argument("--taper", type=int, default=2)
    p.add_argument("--trees", type=int, default=120)
    p.add_argument("--seed", type=int, default=57)
    args = p.parse_args()
    budgets = [int(q) for q in args.budgets.split(",")]
    if args.reserve < 0 or args.donors < 2 or max(budgets) <= args.reserve:
        raise ValueError("invalid donor audit reserve")
    run(args)


if __name__ == "__main__":
    main()
