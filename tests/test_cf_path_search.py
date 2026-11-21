"""Pair synergy, target identity, measured minimality, and hard query budget."""
import numpy as np
import pytest
from beaconxai.cf_path_search import search_counterfactual_path

class C:
    def __init__(self, i):
        self.t0 = i
        self.t1 = i + 1
        self.c0 = 0
        self.c1 = 1

COMP = [C(i) for i in range(5)]
X = np.zeros((5, 1))
D = np.ones((5, 1))

def predicts(batch):
    z = np.asarray(batch)
    margin = z[:, 0, 0] + z[:, 1, 0] + z[:, 2, 0] - 2.5
    return np.stack((np.zeros(len(z)), margin), axis=1)

def test_three_patch_success_with_fixed_source_target_and_count():
    calls = [0]
    def wrapped(x):
        calls[0] += len(x)
        return predicts(x)
    r = search_counterfactual_path(
        X, predicts(X[None])[0], D, COMP, list(range(5)), wrapped,
        target_class=1, budget=8, seed_singletons=3, taper=0)
    assert r.is_valid
    assert r.verified_edit == (0, 1, 2)
    assert r.local_minimality_checked
    assert calls[0] == r.calls
    assert r.calls <= 8
    assert all(x.stage in {"singleton", "grow", "prune"} for x in r.probes)

def test_already_singleton_flip_and_zero_uncounted_baseline():
    def one(x):
        return np.column_stack((np.zeros(len(x)), x[:, 0, 0] - .5))
    r = search_counterfactual_path(
        X, one(X[None])[0], D, COMP, list(range(5)), one,
        target_class=1, budget=8, taper=0)
    assert r.verified_edit == (0,)
    assert r.calls == 1 and r.local_minimality_checked

def test_abstention_if_budget_only_singletons():
    r = search_counterfactual_path(
        X, predicts(X[None])[0], D, COMP, list(range(5)), predicts,
        target_class=1, budget=2, seed_singletons=2, taper=0)
    assert not r.is_valid and r.verified_edit is None and r.calls == 2

def test_3_class_third_class_flip_does_not_count():
    def third(x):
        return np.column_stack((np.ones(len(x)), np.zeros(len(x)),
                                np.sum(x[:, :, 0], axis=1) * 2))
    r = search_counterfactual_path(
        X, third(X[None])[0], D, COMP, list(range(5)), third,
        target_class=1, budget=5, taper=0)
    assert not r.is_valid
    assert any(p.predicted_class == 2 for p in r.probes)
    assert all(not p.targeted_flip for p in r.probes)

def test_partial_minimality_is_not_falsely_certified():
    def two(x):
        return np.column_stack((np.zeros(len(x)), x[:, 0, 0] + x[:, 1, 0] - .5))
    r = search_counterfactual_path(
        X, two(X[None])[0], D, COMP, [2, 3, 0, 1, 4], two,
        target_class=1, budget=5, seed_singletons=2, taper=0)
    assert r.is_valid
    assert not r.local_minimality_checked

def test_validation():
    with pytest.raises(ValueError):
        search_counterfactual_path(
            X, predicts(X[None])[0], D, COMP, [0, 0, 1, 2, 3], predicts,
            target_class=1, budget=8)
    with pytest.raises(ValueError):
        search_counterfactual_path(
            X, predicts(X[None])[0], D, COMP, list(range(5)), predicts,
            target_class=0, budget=8)
