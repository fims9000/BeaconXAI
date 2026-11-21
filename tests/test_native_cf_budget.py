"""External model calls are counted per *input*, never per Python batch."""
import numpy as np
import pytest

from beaconxai.native_cf_budget import (
    CountedCFModel, QueryBudgetExceeded,
    TrainingGaussianOutlier, NoFeatureImportance,
)


def test_reject_oversized_batch_before_any_model_call():
    tracked = []
    def f(x):
        tracked.append(len(x))
        return np.column_stack((np.zeros(len(x)), np.ones(len(x))))
    model = CountedCFModel(f, (8,3), 2, budget=8)
    p = model.predict(np.zeros((3,8,3)))
    assert p.shape == (3,2) and p.sum(axis=1).tolist() == pytest.approx([1,1,1])
    assert model.evaluations == 3 and model.remaining == 5
    with pytest.raises(QueryBudgetExceeded) as error:
        model.predict(np.zeros((6,8,3)))
    assert (error.value.requested,error.value.remaining) == (6,5)
    assert tracked == [3]  # not silently broken into batches


def test_exact_budget_and_verified_candidate_from_native_return():
    model = CountedCFModel(
        lambda z: np.stack((-np.sum(z,axis=(1,2)),np.sum(z,axis=(1,2))),axis=1),
        (4,2), 2, budget=2
    )
    attempted = np.ones((2,4,2))
    output = model.predict(attempted)
    assert model.evaluations == 2 and model.batches == 1
    result = model.verified_result(attempted[0], 1)
    assert result is not None and result.prediction == 1
    assert model.verified_result(np.zeros((4,2)), 1) is None
    with pytest.raises(QueryBudgetExceeded):
        model.predict(np.zeros((4,2)))
    assert np.isfinite(output).all()


def test_no_validity_without_actual_model_query():
    model = CountedCFModel(
        lambda x: np.column_stack((np.ones(len(x)),np.zeros(len(x)))),
        (8,2),2,3
    )
    model.predict(np.ones((1,8,2)))
    assert model.verified_result(np.ones((8,2)),1) is None
    assert len(model.queried)==1


def test_invalid_logits_do_not_claim_usage():
    model=CountedCFModel(lambda x: np.full((len(x),2),np.nan),(3,2),2,4)
    with pytest.raises(ValueError,match="invalid"):
        model.predict(np.ones((2,3,2)))
    assert model.evaluations==0


def test_train_only_outlier_and_zero_fi():
    rng=np.random.default_rng(9)
    data=rng.normal(size=(20,12,9))
    score=TrainingGaussianOutlier(data)
    assert np.isfinite(score.get_outlier_scores(data)).all()
    assert score.get_outlier_scores(data[0]).shape == (1,)
    assert np.count_nonzero(NoFeatureImportance().calculate_feature_importance(data[0]))==0
