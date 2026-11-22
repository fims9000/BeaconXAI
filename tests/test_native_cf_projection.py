"""The search objective must see precisely the physically corrected signal."""
import numpy as np
from beaconxai.native_cf_budget import TrainingGaussianOutlier
from beaconxai.native_cf_projection import (
    ProjectedCFModel, ProjectedTrainingOutlier, UciGravityProjector,
)


def test_model_is_only_queried_on_projection_and_calls_are_exact():
    x=np.zeros((4,9))
    mu=np.arange(9)*.1
    sd=np.arange(1,10,dtype=float)
    proposal=np.zeros((1,4,9))
    proposal[:,:,6:9]=1.
    actual=[]
    def blackbox(z):
        actual.extend(z.copy())
        margin=z[:,:,:3].mean(axis=(1,2))+.1
        return np.stack((np.zeros(len(z)),margin),axis=1)
    wrapper=ProjectedCFModel(blackbox,x,mu,sd,n_classes=2,budget=2)
    probs=wrapper.predict(proposal)
    assert probs.shape==(1,2) and wrapper.evaluations==1
    out=np.asarray(actual)[0]
    raw=lambda a:a*sd+mu
    np.testing.assert_allclose(
        raw(out)[:,6:9]-raw(out)[:,:3],
        raw(x)[:,6:9]-raw(x)[:,:3],atol=1e-12)
    candidate=wrapper.verified_result(proposal,1)
    assert candidate is not None
    np.testing.assert_allclose(candidate.x,out)
    assert wrapper.evaluations==1  # native result read from tested responses


def test_outlier_objective_uses_same_projected_candidates():
    rng=np.random.default_rng(4)
    train=rng.normal(size=(4,8,9))
    x=train[0]
    wanted=train[1]
    projector=UciGravityProjector(x,np.zeros(9),np.ones(9))
    old=TrainingGaussianOutlier(train)
    scorer=ProjectedTrainingOutlier(old,projector)
    expected=old.get_outlier_scores(projector(wanted))
    np.testing.assert_allclose(
        scorer.get_outlier_scores(wanted),expected,atol=1e-12)
    assert scorer.get_outlier_scores(wanted[None]).shape==(1,)
