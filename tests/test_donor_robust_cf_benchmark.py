"""Aggregate multi-donor certificates without confusing flips and explanations."""
import pytest

from scripts.run_donor_robust_cf_benchmark import aggregate


def row(subject, method, certified, direct, calls=8, minimal=False):
    return {
        "subject": subject,
        "method": method,
        "budget": 16,
        "original_donor_flipped": int(direct),
        "certified_three_donors": int(certified),
        "locally_minimal": int(minimal),
        "online_model_calls": calls,
        "n_logical_regions": 2 if certified else None,
        "gravity_residual_roughness_ratio": 1.0 if certified else None,
        "withheld_n_checked": 5 if certified else 0,
        "withheld_n_flipped": 5 if certified else 0,
        "withheld_flip_fraction": 1.0 if certified else None,
        "withheld_all_flipped": int(certified),
    }


def test_subject_macro_is_not_window_weighted():
    rows = ([row(1, "repair_directed", True, True, minimal=True)] * 9
            + [row(2, "repair_directed", False, True)])
    s = aggregate(rows)[0]
    assert s["windows"] == 10
    assert s["certified_three_donors"] == 9
    assert s["single_donor_flips"] == 10
    assert s["abstentions"] == 1
    assert s["subject_macro_certified"] == pytest.approx(.5)
    assert s["locally_minimal_and_three_donors"] == 9
    assert s["n_certified_surviving_all_unseen_training_donors"] == 9
    assert s["mean_withheld_flip_fraction_given_certified"] == 1.0


def test_zero_certificates_are_explicit_abstentions():
    s = aggregate([row(4, "reference_uniform", False, True, 13)])[0]
    assert s["certified_three_donors"] == 0
    assert s["single_donor_flips"] == 1
    assert s["mean_regions_if_certified"] is None
    assert s["mean_withheld_flip_fraction_given_certified"] is None
    assert s["median_gravity_ratio_if_certified"] is None
    assert s["subject_macro_calls"] == 13


def test_independent_policies_get_separate_groups():
    s = aggregate([row(4, "repair_directed", False, True),
                   row(4, "reference_directed", True, True)])
    assert {r["method"] for r in s} == {"repair_directed", "reference_directed"}
    assert aggregate([]) == []
