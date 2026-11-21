"""Native FastPACE instrumentation must account for batched planner calls."""
import numpy as np
import pytest

from beaconxai.native_cf_budget import QueryBudgetExceeded
from beaconxai.native_fastpace_meter import (
    MeteredFastPACEModel, FastPACEOutlier, cf_fingerprint
)


def make(n, c, t, limit):
    history = []
    def predictor(ntc):
        history.append(len(ntc))
        result = np.column_stack((
            np.zeros(len(ntc)),
            ntc[:, :2, 0].mean(axis=1) + 0.5
        ))
        return result
    return MeteredFastPACEModel(predictor, c, t, 2, limit), history


def test_exact_batches_and_transposition():
    model, history = make(4, 3, 8, 5)
    inp = np.zeros((3, 3, 8), dtype=np.float32)
    probs = model.predict(inp)
    assert probs.shape == (3, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert model.predict_class(inp[0]) == [1]
    assert model.evaluations == 4 and model.batches == 2
    assert model.largest_batch == 3 and history == [3,1]
    with pytest.raises(QueryBudgetExceeded) as error:
        model.predict(inp[:2])
    assert (error.value.requested, error.value.remaining) == (2, 1)
    assert history == [3,1]
    assert model.was_measured_target(inp[0], 1)


def test_hash_only_candidate_verification_not_predictive_claim():
    model, _ = make(4, 3, 8, 3)
    x = np.zeros((3,8),dtype=np.float32)
    z = np.ones((3,8),dtype=np.float32)
    assert not model.was_measured_target(x,1)
    model.predict(x)
    assert model.was_measured_target(x,1)
    assert not model.was_measured_target(z,1)
    assert cf_fingerprint(x) != cf_fingerprint(z)


def test_exact_budget_when_a_planning_batch_is_too_large():
    model, history = make(4, 3, 8, 16)
    model.predict(np.zeros((3,8)))
    with pytest.raises(QueryBudgetExceeded) as e:
        model.predict(np.zeros((16,3,8)))
    assert e.value.remaining == 15
    assert model.evaluations == 1
    assert history == [1]


def test_outlier_adapter_is_identical_under_transpose():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(20,8,3))
    adapter = FastPACEOutlier(train)
    nct = train[:2].transpose(0,2,1)
    expected = adapter.reference.get_outlier_scores(train[:2])
    np.testing.assert_allclose(adapter.get_outlier_scores(nct), expected)
    assert adapter.get_outlier_scores(nct[0]).shape == (1,)


def test_wrong_geometry_and_logits_rejected():
    model, _ = make(1,3,8,None)
    with pytest.raises(ValueError):
        model.predict(np.zeros((1,4,8)))
    bad = MeteredFastPACEModel(
        lambda x: np.full((len(x),2),float("nan")), 3,8,2
    )
    with pytest.raises(ValueError):
        bad.predict(np.zeros((3,8)))
    assert bad.evaluations == 0
