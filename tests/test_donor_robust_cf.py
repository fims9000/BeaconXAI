"""Target-specific finite-donor counterfactual certificates."""
import numpy as np
import pytest

from beaconxai.donor_robust_cf import search_robust_counterfactual


class C:
    def __init__(self, i):
        self.t0, self.t1, self.c0, self.c1 = i, i + 1, 0, 1


X = np.zeros((6, 1))
REGIONS = [C(i) for i in range(6)]
DONORS = [np.ones_like(X), np.ones_like(X) * .7, np.ones_like(X) * .8]


def model(batch):
    """Donor0 flips at 2 regions, donors1/2 need 3 regions."""
    z = np.asarray(batch)
    margin = np.sum(z[:, :3, 0], axis=1) - 1.8
    return np.column_stack([np.zeros(len(z)), margin])


def run(*, budget=10, prune=True, predictor=model):
    return search_robust_counterfactual(
        X, model(X[None])[0], DONORS, REGIONS,
        list(range(len(REGIONS))), predictor, target_class=1,
        budget=budget, max_changed_components=5,
        seed_singletons=1, taper=0, prune=prune,
    )


def test_nonrobust_pair_is_repaired_to_three_donor_certificate():
    count = [0]
    def wrapped(batch):
        count[0] += len(batch)
        return model(batch)
    result = run(budget=10, predictor=wrapped)
    assert result.is_certified
    assert result.certified_edit == (0, 1, 2)
    assert result.n_donors == 3
    assert result.queries == count[0] == len(result.probes) == 10
    assert result.local_minimality_checked
    assert any(p.components == (0, 1) and
               p.donor_index == 1 and not p.targeted_flip for p in result.probes)
    for donor_index in range(3):
        assert any(p.components == (0, 1, 2) and
                   p.donor_index == donor_index and p.targeted_flip
                   for p in result.probes)


def test_certificate_does_not_claim_minimality_without_pruning_budget():
    result = run(budget=6)
    assert result.is_certified
    assert result.certified_edit == (0, 1, 2)
    assert result.queries == 6
    assert not result.local_minimality_checked


def test_abstain_even_after_a_single_donor_flip_if_budget_missing():
    result = run(budget=2)
    assert not result.is_certified
    assert result.certified_edit is None
    assert result.status == "budget_exhausted_without_certificate"
    assert result.queries == 2
    assert result.ever_found_single_donor_flip


def test_failing_first_donor_or_disconnected_other_class_never_certifies():
    def other(batch):
        z = np.asarray(batch)
        return np.column_stack([
            np.full(len(z), 1.),
            np.zeros(len(z)),
            np.sum(z[:, :3, 0], axis=1) * 2
        ])
    result = search_robust_counterfactual(
        X, other(X[None])[0], DONORS, REGIONS,
        list(range(len(REGIONS))), other, target_class=1,
        budget=10, max_changed_components=3,
        seed_singletons=1, taper=0,
    )
    assert not result.is_certified
    assert any(p.predicted_class == 2 for p in result.probes)
    assert all(not p.targeted_flip for p in result.probes)


def test_exact_count_and_cache_still_work_without_minimize():
    r = run(budget=10, prune=False)
    assert r.is_certified and r.certified_edit == (0, 1, 2)
    assert r.queries == 6
    assert not r.local_minimality_checked
    keys = [(p.components, p.donor_index) for p in r.probes]
    assert len(keys) == len(set(keys))


def test_reject_invalid_training_pool_rank_and_target():
    with pytest.raises(ValueError, match="at least two"):
        search_robust_counterfactual(X, model(X[None])[0], DONORS[:1],
                                     REGIONS, list(range(6)), model,
                                     target_class=1, budget=8)
    with pytest.raises(ValueError, match="ranking"):
        search_robust_counterfactual(X, model(X[None])[0], DONORS,
                                     REGIONS, [0, 0, 1, 2, 3, 4], model,
                                     target_class=1, budget=8)
    with pytest.raises(ValueError, match="target"):
        search_robust_counterfactual(X, model(X[None])[0], DONORS,
                                     REGIONS, list(range(6)), model,
                                     target_class=0, budget=8)
    with pytest.raises(ValueError, match="training donors"):
        search_robust_counterfactual(X, model(X[None])[0], DONORS[:2] + [np.ones((7, 1))],
                                     REGIONS, list(range(6)), model,
                                     target_class=1, budget=8)


def test_malformed_model_scores_are_not_certificates():
    def wrong(batch):
        return np.zeros((len(batch), 3))
    with pytest.raises(ValueError, match="model must return"):
        run(predictor=wrong)
