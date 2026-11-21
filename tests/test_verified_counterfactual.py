"""Target-flip evidence and budget accounting, independent of any model library."""
import numpy as np
import pytest
from beaconxai.contrastive import Component
from beaconxai.verified_counterfactual import (
    splice, target_margin, find_verified_counterfactual, donor_stability_audit,
)

COMP = [Component(0, j, j, j + 1, 0, 1) for j in range(4)]
X = np.zeros((4, 1))
DONOR = np.ones_like(X)

def model(batch):
    # Target 1 wins only if BOTH patches 0 and 1 are present.
    z = np.asarray(batch)
    s = z[:, 0, 0] + z[:, 1, 0] - 1.5
    return np.column_stack((np.zeros(len(z)), s))

def test_pair_interaction_detected_and_verified():
    baseline = model(X[None])[0]
    count = [0]
    def wrapped(batch):
        count[0] += len(batch)
        return model(batch)
    r = find_verified_counterfactual(X, baseline, DONOR, COMP,
                                     [0, 1, 2, 3], wrapped,
                                     target_class=1, budget=6, taper=0)
    assert r.status == "verified_target_flip"
    assert r.verified_edit == (0, 1)
    assert r.evaluations == count[0] == 5  # 3 singleton, then (0,1) pair
    assert r.queries[-1].interaction_pair == pytest.approx(0.0)
    assert all(not q.flip_to_target for q in r.queries[:-1])

def test_single_patch_is_preferred_to_pair():
    def one(batch):
        z = np.asarray(batch)
        return np.column_stack((np.zeros(len(z)), z[:, 0, 0] - .5))
    r = find_verified_counterfactual(X, one(X[None])[0], DONOR, COMP,
                                     [0, 1, 2, 3], one,
                                     target_class=1, budget=4, taper=0)
    assert r.verified_edit == (0,)
    assert r.evaluations == 1

def test_unverified_is_explicit_abstention_and_budget_hard():
    base = model(X[None])[0]
    r = find_verified_counterfactual(X, base, DONOR, COMP,
                                     [2, 3, 0, 1], model,
                                     target_class=1, budget=2, taper=0)
    assert not r.is_valid
    assert r.verified_edit is None
    assert r.status == "not_found_within_budget"
    assert r.evaluations == 2

def test_no_target_label_or_ground_truth_parameter_and_third_class_check():
    def other(batch):
        z = np.asarray(batch)
        return np.column_stack((np.ones(len(z)), np.zeros(len(z)),
                                z[:, 0, 0] * 3.))
    base = other(X[None])[0]
    r = find_verified_counterfactual(X, base, DONOR, COMP, [0, 1, 2, 3],
                                     other, target_class=1, budget=3, taper=0)
    assert not r.is_valid
    assert r.queries[0].predicted_class == 2
    assert not r.queries[0].flip_to_target

def test_reject_overlap_nonfinite_and_bad_rank():
    with pytest.raises(ValueError):
        splice(X, DONOR, COMP, [0, 0])
    with pytest.raises(ValueError):
        find_verified_counterfactual(X, model(X[None])[0], DONOR, COMP, [0, 0, 1, 2],
                                     model, target_class=1, budget=4)
    with pytest.raises(ValueError):
        target_margin(np.array([1., np.nan]), 1)

def test_donor_robustness_requires_extra_model_calls():
    base = model(X[None])[0]
    donors = [DONOR, DONOR * .1, DONOR]
    calls = [0]
    def wrapped(batch):
        calls[0] += len(batch)
        return model(batch)
    report = donor_stability_audit(X, base, 1, (0, 1), donors, COMP,
                                   wrapped, budget_remaining=2, taper=0)
    assert report == {"tested": 2, "target_flips": 1, "flip_fraction": .5}
    assert calls[0] == 2
    assert donor_stability_audit(X, base, 1, (0, 1), donors, COMP,
                                 wrapped, budget_remaining=0)["flip_fraction"] is None
