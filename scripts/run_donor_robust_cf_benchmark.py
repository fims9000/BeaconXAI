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
from beaconxai.donor_selection import farthest_first_donors, cover_radius
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
            "n_certified_surviving_all_unseen_training_donors": sum(
                r["withheld_all_flipped"] for r in certified),
            "mean_withheld_flip_fraction_given_certified": (
                float(np.mean([r["withheld_flip_fraction"] for r in certified]))
                if certified else None),
            "mean_withheld_donor_checks_given_certified": (
                float(np.mean([r["withheld_n_checked"] for r in certified]))
                if certified else None),
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
            "mean_train_bank_cover_radius": float(np.mean([
                r["selected_donors_cover_radius"] for r in group
            ])),
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
        dataset_protocol = (str(z["dataset_protocol"])
                            if "dataset_protocol" in z.files
                            else "UCI HAR official train/test split")
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
    offline_withheld_donor_calls = 0

    for i, x in enumerate(test):
        base = original_scores[i]
        source = int(np.argmax(base))
        target = int(np.argsort(-base, kind="stable")[1])
        if source not in profiles or target not in bank.donors:
            raise ValueError("model class absent from training")
        nearest = nearest_context_index(x, 0, bank, target, originals)
        reference_indices = [nearest] + [
            j for j in range(len(bank.donors[target])) if j != nearest
        ][:args.reserve]
        diverse_indices = list(farthest_first_donors(
            bank.donors[target], nearest, count=3
        ))
        if len(reference_indices) != 3 or len(diverse_indices) != 3:
            raise ValueError("expected three donors for the target")
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
                ("reference_directed", rank, False, False),
                ("reference_uniform", uniform, False, False),
                ("repair_directed", rank, True, False),
                ("repair_uniform", uniform, True, False),
                ("reference_diverse", rank, False, True),
                ("repair_diverse", rank, True, True),
            )
            for name, order, repair, diverse in policies:
                donor_indices = diverse_indices if diverse else reference_indices
                donor_pool = tuple(bank.donors[target][j] for j in donor_indices)
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
                    direct_flip = int(any(p.donor_index == 0 and p.targeted_flip
                                          for p in result.probes))
                    certified = int(result.is_certified)
                    locally_minimal = int(
                        result.is_certified and result.local_minimality_checked
                    )
                    status = result.status
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
                    "selected_bank_indices": ",".join(map(str, donor_indices)),
                    "selected_donors_cover_radius": cover_radius(
                        bank.donors[target], donor_indices
                    ),
                    "no_test_labels_used": True,
                })
        # A separate post-hoc challenge set of training donors was not used
        # to select components or certify the 3-donor result. This cannot
        # guarantee generalization to unseen people or new distributions.
        sample_rows = [r for r in qrows if r["sample"] == i]
        for row in sample_rows:
            selected_bank_indices = {
                int(v) for v in row["selected_bank_indices"].split(",")
            }
            heldout = tuple(
                bank.donors[target][j]
                for j in range(len(bank.donors[target]))
                if j not in selected_bank_indices
            )
            if not heldout:
                raise ValueError("No withheld TRAIN donors available for validation")
            row["withheld_n_checked"] = 0
            row["withheld_n_flipped"] = 0
            row["withheld_flip_fraction"] = None
            row["withheld_all_flipped"] = 0
            if not row["certified_three_donors"]:
                continue
            key = tuple(int(v) for v in row["components"].split(","))
            start_check = int(calls["examples"])
            withheld_audit = donor_stability_audit(
                x, base, target, key, heldout, groups, model,
                budget_remaining=len(heldout), taper=args.taper
            )
            spent = int(calls["examples"]) - start_check
            if spent != withheld_audit["tested"]:
                raise AssertionError("withheld donor evaluations not accounted")
            offline_withheld_donor_calls += spent
            row["withheld_n_checked"] = withheld_audit["tested"]
            row["withheld_n_flipped"] = withheld_audit["target_flips"]
            row["withheld_flip_fraction"] = withheld_audit["flip_fraction"]
            row["withheld_all_flipped"] = int(
                withheld_audit["tested"] == len(heldout) and
                withheld_audit["target_flips"] == len(heldout)
            )
        # This separate, explicitly counted offline diagnostic never affects
        # the Q budget or any decision for the current test sample.
        original_donor = bank.donors[target][nearest]
        reference = model(np.stack([
            original_donor,
            splice(x, original_donor, groups, tuple(range(len(groups))),
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
    if calls["examples"] != (
        len(test) + online_total + offline_reference_calls +
        offline_withheld_donor_calls
    ):
        raise AssertionError("some model evaluations were not accounted")
    summary = aggregate(qrows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "per_sample.csv", qrows)
    write_csv(out / "summary.csv", summary)
    manifest = {
        "dataset": dataset_protocol,
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
        "donor_selection_policies": [
            "nearest_anchor_plus_first_two_bank_entries",
            "nearest_anchor_plus_farthest_first_two_by_train_normalized_L2",
        ],
        "donor_selection_uses_no_model_calls": True,
        "farther_selection_is_heuristic_not_coverage_certificate": True,
        "budget_max_additional_queries_including_donor_checks": budgets,
        "baseline_model_evaluations": len(test),
        "all_online_model_evaluations": online_total,
        "offline_reference_evaluations": offline_reference_calls,
        "offline_withheld_donor_checks": offline_withheld_donor_calls,
        "withheld_donors_per_certified_candidate": len(bank.donors[target])-3,
        "withheld_audit_is_posthoc_and_not_part_of_online_Q": True,
        "total_evaluations": calls["examples"],
        "n_windows": len(test),
        "model_evaluation_means_one_evaluated_input_not_one_batch": True,
        "limitations": [
            "3-of-3 donor certificate is valid only for selected donors, this classifier and operator",
            "Withheld donors are unseen in search but drawn from the same TRAIN bank, not independent study subjects",
            "The withheld subset differs between donor selection policies; all-eight pass is comparable, conditional pass fractions are not directly paired",
            "No statistical population coverage or physical causality is implied",
            "Prioritization, donor selection and model training costs are offline",
            "Local minimality is relative to single component deletions and selected donors",
            "Test labels and full-donor diagnostic references are not available to search",
            "Single classifier seed and UCI HAR split do not prove generalization",
            "Internal resplits from official TRAIN reuse material previously seen during method development",
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
