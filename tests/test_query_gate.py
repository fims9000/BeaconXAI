"""Leakage, query-accounting and subject-balanced calibration tests."""
import numpy as np
import pytest
from beaconxai.query_gate import (
    QueryGate, cheap_gate_features, evaluate_gate, fit_query_gate
)


def row(subject, margin, gap, q0, q16):
    return dict(subject=subject, margin=margin, prior_gap=gap,
                hit3_q0=q0, hit3_q16=q16)


def test_cheap_features_need_no_test_label_or_perturbed_output():
    f = cheap_gate_features(np.array([-1., -3., -5.]), np.array([10., 4., 0., 1.]))
    assert f["margin"] == pytest.approx(2)
    assert f["prior_gap"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        cheap_gate_features(np.array([0., np.nan]), np.array([1., 2.]))


def test_threshold_counts_exact_deployment_queries_and_offline_cost():
    rows = [row(2, 0.1, .2, 0, 1), row(2, 0.9, .8, 1, 0),
            row(3, 0.2, .3, 0, 1), row(3, 0.8, .7, 1, 0)]
    g = QueryGate("threshold", "margin", "le", .5, 16)
    summary, per_subject = evaluate_gate(rows, g, cost_per_full_budget=.1)
    assert [r["query_rate"] for r in per_subject] == [.5, .5]
    assert summary["observed_query_count_simulated"] == 32
    assert summary["offline_evaluations_for_paired_q16"] == 64
    assert summary["subject_macro_hit3_gate"] == 1
    assert summary["subject_macro_mean_queries"] == 8
    assert summary["subject_macro_utility"] == pytest.approx(.95)


def test_gate_selected_using_only_inner_data():
    inner = [
        row(2, .1, .1, 0, 1), row(2, .9, .9, 1, 0),
        row(3, .2, .2, 0, 1), row(3, .8, .8, 1, 0),
    ]
    gate, info = fit_query_gate(inner, cost_per_full_budget=.05)
    assert gate.mode == "threshold"
    assert gate.should_query(inner[0])
    assert not gate.should_query(inner[1])
    assert info["calibration"]["subject_macro_hit3_gate"] == 1
    outer = [row(6, .1, .1, 1, 0), row(7, .9, .9, 0, 1)]
    before = gate.to_dict()
    evaluated, _ = evaluate_gate(outer, gate)
    assert gate.to_dict() == before
    assert evaluated["subject_macro_hit3_gate"] == 0


def test_subject_macro_prevents_large_subject_dominance():
    # Large subject prefers full Q, small subject gains more from Q0.
    rows = [row(2, .1, .1, 0, 1)] * 100 + [row(3, .1, .1, 1, 0)]
    always, _ = evaluate_gate(rows, QueryGate("q16"))
    assert always["subject_macro_hit3_gate"] == .5
    assert always["n_subjects"] == 2


def test_validation_and_no_implicit_ground_truth():
    with pytest.raises(ValueError):
        fit_query_gate([row(2, .1, .1, 1, 0)])
    with pytest.raises(ValueError):
        evaluate_gate([row(2, .1, .1, 1, 2)], QueryGate("q0"))
    with pytest.raises(ValueError):
        QueryGate("threshold", "unknown", "le", 0)
    with pytest.raises(ValueError):
        QueryGate("q0", budget=0)
    with pytest.raises(ValueError):
        evaluate_gate([row(2, .1, .1, 0, 1)], QueryGate("q16"), -1)


def test_gate_q0_q16_fixed_baselines():
    rows = [row(2, .1, .1, 0, 1), row(3, .1, .1, 1, 0)]
    q0, _ = evaluate_gate(rows, QueryGate("q0"))
    q16, _ = evaluate_gate(rows, QueryGate("q16"))
    assert q0["subject_macro_mean_queries"] == 0
    assert q16["subject_macro_mean_queries"] == 16
    assert q0["subject_macro_hit3_gate"] == q0["subject_macro_hit3_q0"]
    assert q16["subject_macro_hit3_gate"] == q16["subject_macro_hit3_q16"]
