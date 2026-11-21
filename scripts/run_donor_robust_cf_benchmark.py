#!/usr/bin/env python3
"""Matched finite-donor counterfactual study on UCI HAR, one frozen classifier.

Compare first-donor path+post-hoc donor audit against an adaptive path that
continues searching after a donor fails. The output is a VERIFIED 3-of-3
fixed-TRAIN-donor certificate or an explicit abstention, never a population
coverage guarantee. All queried inputs count against each window's Q budget.
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

from beaconxai.cf_path_search import search_counterfactual_path
from beaconxai.contrastive import make_components, contrastive_prior
from beaconxai.datasets import apply_standardizer, load_npz_dataset
from beaconxai.donor_robust_cf import search_robust_counterfactual
from beaconxai.physical_groups import coupled_priority, uci_har_coupled_groups
from beaconxai.real_donor import fit_real_donor_bank, nearest_context_index
from beaconxai.subject_sampling import balanced_subject_sample
from beaconxai.verified_counterfactual import donor_stability_audit, splice
from scripts.run_clean_calibrated_staged_budget import fit_predictor
from scripts.run_q16_matched_ablation import write_csv
from scripts.run_verified_counterfactual_benchmark import gravity_residual_roughness


def aggregate(rows: list[dict]) -> list[dict]:
    summary = []
    for method, budget in sorted({(r["method"], r["budget"]) for r in rows}):
        group = [r for r in rows if r["method"] == method and r["budget"] == budget]
        subjects = sorted({r["subject"] for r in group})
        if not subjects:
            raise ValueError("missing subjects")
        per_subject = [
            [r for r in group if r["subject"] == sid] for sid in subjects
        ]
        certified = [r for r in group if r["certified_three_donors"]]
        local_minimal = [r for r in certified if r["locally_minimal"]]
        summary.append({
            "method": method,
            "Q": budget,
            "windows": len(group),
            "subjects": len(subjects),
            "single_donor_flips": sum(r["original_donor_flipped"] for r in group),
            "certified_three_donors": len(certified),
            "locally_minimal_and_three_donors": len(local_minimal),
            "abstentions": len(group) - len(certified),
            "subject_macro_certified": float(np.mean([
                np.mean([r["certified_three_donors"] for r in g])
                for g in per_subject
            ])),
            "subject_macro_calls": float(np.mean([
                np.mean([r["online_model_calls"] for r in g])
                for g in per_subject
            ])),
            "mean_regions_if_certified": (
                float(np.mean([r["n_logical_regions"] for r in certified]))
                if certified else None
            ),
            "median_gravity_ratio_if_certified": (
                float(np.median([r["gravity_residual_roughness_ratio"]
                                 for r in certified]))
                if certified else None
            ),
        })
    return summary


def run(args):
    if args.reserve != 2 or args.donors < 3:
        raise ValueError("3-donor study requires reserve=2, donor pool >=3")
    budgets = sorted({int(q) for q in args.budgets.split(",")})
    if not budgets or min(budgets) < 4 or max(budgets) > 24:
        raise ValueError("budget must lie in [4,24]")
    path = Path(args.npz_path)
    raw_train, y_train, raw_test, y_test = load_npz_dataset(path)
    with np.load(path) as z:
        subject_train = np.asarray(z["subject_train"], dtype=int)
        subject_test = np.asarray(z["subject_test"], dtype=int)
    if (set(subject_train) & set(subject_test) or
            len(raw_train) != len(y_train) or len(raw_test) != len(y_test)):
        raise ValueError("subject leakage or dimension mismatch")
    if raw_train.shape[2] != 9:
        raise ValueError("this protocol requires UCI HAR nine-channel signals")

    mu, std, profiles, model, calls = fit_predictor(
        raw_train, y_train, args.model, args.seed, args.trees
    )
    train = apply_standardizer(raw_train, mu, std)
    originals = make_components(train.shape[1:], 12, "pamap3")
    groups = uci_har_coupled_groups(
        originals, gravity_preserving=True, mean=mu, scale=std
    )
    bank = fit_real_donor_bank(
        train, y_train, originals, max_donors=args.donors, seed=args.seed + 10
    )
    chosen = balanced_subject_sample(subject_test, args.max_test, args.seed + 301)
    test = apply_standardizer(raw_test[chosen], mu, std)
    original_scores = model(test)
    qrows = []
    offline_reference_calls = 0

    for i, x in enumerate(test):
        base = original_scores[i]
        source = int(np.argmax(base))
        target = int(np.argsort(-base, kind="stable")[1])
        if source not in profiles or target not in bank.donors:
            raise ValueError("model class absent from training")
        nearest = nearest_context_index(x, 0, bank, target, originals)
        donor_indices = [nearest] + [
            j for j in range(len(bank.donors[target])) if j != nearest
        ][:args.reserve]
        if len(donor_indices) != 3:
            raise ValueError("expected three donors for the target")
        donor_pool = tuple(bank.donors[target][j] for j in donor_indices)
        cheap = coupled_priority(
            contrastive_prior(x, source, target, profiles, originals), originals
        )
        rank = np.argsort(-cheap, kind="stable").astype(int).tolist()
        uniform = np.random.default_rng(
            args.seed * 100003 + int(chosen[i])
        ).permutation(len(groups)).astype(int).tolist()
        original_gravity = gravity_residual_roughness(x, mu, std)

        for q in budgets:
            policies = (
                ("reference_directed", rank, False),
                ("reference_uniform", uniform, False),
                ("repair_directed", rank, True),
                ("repair_uniform", uniform, True),
            )
            for name, order, repair in policies:
                start = int(calls["examples"])
                if repair:
                    result = search_robust_counterfactual(
                        x, base, donor_pool, groups, order, model,
                        target_class=target, budget=q,
                        max_changed_components=args.max_changed,
                        seed_singletons=min(args.singletons, q),
                        taper=args.taper, prune=True,
                    )
                    edit = result.certified_edit
                    direct_flip = int(result.ever_found_single_donor_flip)
                    certified = int(result.is_certified)
                    locally_minimal = int(
                        result.is_certified and result.local_minimality_checked
                    )
                    status = result.status
                    queries = result.queries
                else:
                    old = search_counterfactual_path(
                        x, base, donor_pool[0], groups, order, model,
                        target_class=target, budget=q-args.reserve,
                        max_changed_components=args.max_changed,
                        seed_singletons=min(args.singletons, q-args.reserve),
                        taper=args.taper,
                    )
                    audit = donor_stability_audit(
                        x, base, target, old.verified_edit or (),
                        donor_pool[1:], groups, model,
                        budget_remaining=min(args.reserve, q-old.calls),
                        taper=args.taper,
                    )
                    direct_flip = int(old.is_valid)
                    certified = int(old.is_valid and audit["tested"] == 2
                                    and audit["target_flips"] == 2)
                    locally_minimal = int(certified and old.local_minimality_checked)
                    edit = old.verified_edit if certified else None
                    status = ("verified_all_selected_donors" if certified
                              else "not_certified_within_budget")
                    queries = old.probes
                actual = int(calls["examples"]) - start
                if actual > q or actual < 0:
                    raise AssertionError("model-call budget violated")
                if repair and actual != result.queries:
                    raise AssertionError("repair count does not match true model calls")
                if not repair and actual != old.calls+audit["tested"]:
                    raise AssertionError("reference count does not match true calls")
                transformed = (
                    splice(x, donor_pool[0], groups, edit, taper=args.taper)
                    if edit is not None else None
                )
                rough = (
                    gravity_residual_roughness(transformed, mu, std) /
                    max(original_gravity, 1e-6)
                    if transformed is not None else None
                )
                qrows.append({
                    "sample": i,
                    "original_test_index": int(chosen[i]),
                    "subject": int(subject_test[chosen[i]]),
                    "source_class": source,
                    "target_class": target,
                    "method": name,
                    "budget": q,
                    "original_donor_flipped": direct_flip,
                    "certified_three_donors": certified,
                    "status": status,
                    "locally_minimal": locally_minimal,
                    "online_model_calls": actual,
                    "n_logical_regions": len(edit) if edit is not None else None,
                    "components": ",".join(map(str, edit)) if edit else "",
                    "gravity_residual_roughness_ratio": rough,
                    "train_donor_indices": ",".join(str(bank.selected_train_indices[target][j])
                                                   for j in donor_indices),
                    "no_test_labels_used": True,
                })
        # This separate, explicitly counted offline diagnostic never affects
        # the Q budget or any decision for the current test sample.
        reference = model(np.stack([
            donor_pool[0],
            splice(x, donor_pool[0], groups, tuple(range(len(groups))),
                   taper=args.taper),
        ]))
        offline_reference_calls += len(reference)
        for row in qrows:
            if row["sample"] == i:
                row["original_prediction_correct_offline"] = int(
                    source == int(y_test[chosen[i]])
                )
                row["donor_recognized_as_target_offline"] = int(
                    np.argmax(reference[0]) == target
                )
                row["full_patch_target_flip_offline"] = int(
                    np.argmax(reference[1]) == target
                )

    online_total = sum(r["online_model_calls"] for r in qrows)
    if calls["examples"] != len(test)+online_total+offline_reference_calls:
        raise AssertionError("some model evaluations were not accounted")
    summary = aggregate(qrows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_sample.csv", qrows)
    write_csv(out / "summary.csv", summary)
    manifest = {
        "dataset": "UCI HAR official subject-disjoint test windows",
        "frozen_model_sha256": calls["fitted_model_sha256"],
        "train_standardizer_sha256": calls["training_standardizer_sha256"],
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model": args.model,
        "seed": args.seed,
        "train_subjects": sorted(map(int, np.unique(subject_train))),
        "test_subjects": sorted(map(int, np.unique(subject_test[chosen]))),
        "candidate_operator": "gravity-preserved UCI HAR body/total + gyro",
        "same_frozen_model_windows_target_and_donors_for_all_methods": True,
        "donors_are_training_examples_of_label_B_not_population_samples": True,
        "required_donors_for_certificate": 3,
        "budget_max_additional_queries_including_donor_checks": budgets,
        "baseline_model_evaluations": len(test),
        "all_online_model_evaluations": online_total,
        "offline_diagnostic_evaluations": offline_reference_calls,
        "total_evaluations": calls["examples"],
        "n_windows": len(test),
        "model_evaluation_means_one_evaluated_input_not_one_batch": True,
        "limitations": [
            "3-of-3 donor certificate is valid only for selected donors, this classifier and operator",
            "No statistical population coverage or physical causality is implied",
            "Prioritization, donor selection and model training costs are offline",
            "Local minimality is relative to single component deletions and selected donors",
            "Test labels and full-donor diagnostic references are not available to search",
            "Single classifier seed and UCI HAR test split do not prove generalization",
            "Search policies may query different sets; all are hard-capped by Q",
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8"
    )
    print(json.dumps({"frozen_model_sha256": calls["fitted_model_sha256"],
                      "summary": summary}, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--model", choices=["raw_mlp","extratrees"], default="raw_mlp")
    p.add_argument("--budgets", default="8,16")
    p.add_argument("--max-test", type=int, default=180)
    p.add_argument("--max-changed", type=int, default=8)
    p.add_argument("--singletons", type=int, default=3)
    p.add_argument("--donors", type=int, default=8)
    p.add_argument("--reserve", type=int, default=2)
    p.add_argument("--taper", type=int, default=2)
    p.add_argument("--trees", type=int, default=120)
    p.add_argument("--seed", type=int, default=57)
    run(p.parse_args())


if __name__ == "__main__":
    main()
