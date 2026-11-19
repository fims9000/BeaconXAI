"""Scientific protocol tests for the matched Q16 PAMAP2 ablation."""
from __future__ import annotations

from collections import Counter
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beaconxai.contrastive import fit_class_profiles, make_components
from scripts.run_adaptive_v2_localization import _component_cheap_scores
from scripts.run_q16_matched_ablation import (
    METHODS, basic_statistics, best_q, class_cheap,
    evaluate_one_method, evaluate_selected, run, summarize,
)


def example():
    rng = np.random.default_rng(124)
    x_train = rng.normal(0.0, .12, size=(50, 96, 9))
    y_train = np.array([0]*25 + [1]*25)
    x_train[:25, 16:40, :3] += .8
    x_train[25:, 16:40, :3] -= .8
    sample = x_train[2].copy()
    comp = make_components(sample.shape, 12)
    profiles = fit_class_profiles(x_train, y_train)
    return sample, comp, profiles


def predict_many(xs):
    signal = np.asarray(xs)[:, 16:40, :3].mean(axis=(1, 2))
    return np.column_stack((signal, -signal))


def test_original_cheap_score_identical_to_source_formula():
    x, comps, profiles = example()
    val = class_cheap(x, profiles[0][0], comps)
    original = _component_cheap_scores(
        x, profiles[0][0], [(0, 3), (3, 6), (6, 9)],
        [(c.t0, c.t1) for c in comps[:12]],
    )
    for new, old in zip(val, original):
        np.testing.assert_allclose(new, old)
    assert best_q(val[2], 16) == np.argpartition(-val[2], 15)[:16].tolist()


def test_cheap_baselines_are_nonempty_and_finite():
    x, comps, profiles = example()
    e, v = basic_statistics(x, comps)
    assert e.shape == v.shape == (36,)
    assert np.isfinite(e).all() and np.isfinite(v).all()
    assert len(set(best_q(e, 16))) == 16


def test_model_call_count_exact_for_each_method():
    x, comps, profiles = example()
    baseline = predict_many(x[None])[0]
    for method in METHODS:
        calls = [0]
        def counting(batch):
            calls[0] += len(batch)
            return predict_many(batch)
        rank, billed, _ = evaluate_one_method(
            method, x, comps, profiles, counting, baseline, 16, seed=51
        )
        assert billed == calls[0] == (0 if method.endswith("_q0") else 16)
        assert len(rank) == (36 if billed == 0 else 16)
        assert len(set(rank)) == len(rank)


def test_unqueried_components_do_not_enter_q16_ranking():
    x, comps, _ = example()
    baseline = predict_many(x[None])[0]
    selected = list(range(16))
    ranked = evaluate_selected(x, comps, selected, predict_many, baseline, 0, 1)
    assert set(ranked) == set(selected)
    assert len(ranked) == 16
    with pytest.raises(ValueError, match="Repeated"):
        evaluate_selected(x, comps, [0, 0], predict_many, baseline, 0, 1)


def test_moving_and_pinned_score_semantics_distinct():
    x, comps, _ = example()
    baseline = np.array([2., 1., 0.])
    def flip_model(batch):
        logits = np.repeat(np.array([[0., 4., 0.]]), len(batch), axis=0)
        return logits
    selected = [0, 1, 2, 3]
    moving = evaluate_selected(x, comps, selected, flip_model, baseline, 0, 1, "moving")
    pinned = evaluate_selected(x, comps, selected, flip_model, baseline, 0, 1, "fixed")
    assert len(moving) == len(pinned) == 4
    with pytest.raises(ValueError):
        evaluate_selected(x, comps, selected, flip_model, baseline, 0, 1, "unknown")


def test_reproducibility_and_subject_aggregation():
    x, comps, profiles = example()
    base = predict_many(x[None])[0]
    r1 = evaluate_one_method("contrastive_q16", x, comps, profiles, predict_many, base, 16, 42)
    r2 = evaluate_one_method("contrastive_q16", x, comps, profiles, predict_many, base, 16, 42)
    assert r1[0] == r2[0]
    rows = []
    for sid in (8, 9):
        for i in range(2):
            for method in METHODS:
                rows.append({
                    "method":method, "subject":sid, "sample":sid*10+i,
                    "trial":0, "extra_model_evaluations": 0 if method.endswith("_q0") else 16,
                    "hit1":int(i == 0), "hit3":int(i == 0),
                    "hit5":int(i == 0), "mrr": 1.0 if i==0 else 1/37,
                    "runtime_ms": 1.0
                })
    summary, per_subject = summarize(rows)
    assert len(summary) == len(METHODS)
    assert len(per_subject) == 2*len(METHODS)
    assert summary[0]["subjects"] == 2
    assert summary[0]["hit3"] == .5
    assert summary[0]["subject_macro_hit3"] == .5


def test_full_runner_fake_npz_two_subjects(monkeypatch, tmp_path):
    """Integration test with synthetic input; not a scientific PAMAP2 result."""
    import scripts.run_component_conflict_benchmark as base_script
    rng = np.random.default_rng(7)
    train = rng.normal(0, .15, size=(40, 96, 9))
    train[:20, 16:40, :3] += 1
    train[20:, 16:40, :3] -= 1
    lab_train = np.array([0]*20 + [1]*20)
    xt = train[[0,1,2,3,4,20,21,22,23,24,5,25]].copy()
    yt = lab_train[[0,1,2,3,4,20,21,22,23,24,5,25]]
    subject_test = np.array([8]*6 + [9]*6)
    path = tmp_path/"synthetic.npz"
    np.savez_compressed(path, x_train=train, y_train=lab_train,
                        x_test=xt, y_test=yt, subject_test=subject_test)

    class FakeModel:
        def predict_proba(self, features):
            z = np.tanh(features[:, 0])
            p = (z + 1) / 2
            return np.column_stack((np.clip(p, .01, .99),
                                    np.clip(1-p, .01, .99)))

    class Adapter:
        model = FakeModel()

    monkeypatch.setattr(base_script, "_train_extratrees_local",
                        lambda *a: Adapter())
    out = tmp_path/"out"
    args = SimpleNamespace(npz_path=path, out_dir=out, max_test=12, seed=42,
                           trees=120, time_bins=12, budget=16,
                           repeats=2, conflict_ratio=.5)
    run(args)
    with (out/"summary.csv").open(encoding="utf-8") as f:
        summary = list(csv.DictReader(f))
    with (out/"per_subject.csv").open(encoding="utf-8") as f:
        subject_rows = list(csv.DictReader(f))
    data = json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    assert len(summary) == len(METHODS)
    assert {x["method"] for x in summary} == set(METHODS)
    assert data["M"] == 36 and data["q_additional"] == 16
    assert len(subject_rows) > 0
    assert all(int(x["extra_model_evaluations"]) in (0,16) for x in summary)
    assert data["test_subjects"]
