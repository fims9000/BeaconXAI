"""Frozen TRAIN-only joint-mask ranking and exact original-core Q16 invariants."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.donor_robust_cf import search_sparse_robust_counterfactual
from beaconxai.transfer_candidate_policy import (
    N_FEATURES, candidate_features, fixed_training_masks,
    fit_candidate_transfer_prior, TransferAwareRanking,
)


def make_simple():
    x=np.zeros((50,2))
    donors=(np.ones_like(x),np.full_like(x,.8))
    groups=make_components((50,2),10,"all_channels")
    gains=np.ones((2,10),dtype=float)
    original=np.array([0.,-.1])
    return x,donors,groups,gains,original


def test_candidate_features_bounded_and_no_unseen_donor_input():
    x,donors,groups,gains,original=make_simple()
    feat,support=candidate_features(
        x,donors,groups,gains,original,1,(1,3),taper=2,
    )
    assert feat.shape==(N_FEATURES,)==(25,)
    assert support.shape==(2,)
    assert 0<support.min()<=support.max()<=.35
    assert feat[1+1]==1. and feat[1+3]==1.
    assert feat[0]==.2
    with pytest.raises(ValueError,match="invalid"):
        candidate_features(
            x,donors+(np.ones_like(x)*.7,),
            groups,gains,original,1,(1,3),taper=2,
        )
    with pytest.raises(ValueError,match="invalid"):
        candidate_features(x,donors,groups,gains,original,1,(3,3))


def test_train_candidate_fixed_rng_and_ridge_not_used():
    x=fixed_training_masks(np.random.default_rng(7714))
    y=fixed_training_masks(np.random.default_rng(7714))
    assert x==y and len(x)==20 and len(set(x))==20
    assert sum(len(k)==1 for k in x)==10
    assert sum(len(k)==2 for k in x)==5
    assert sum(len(k)==3 for k in x)==5


def test_prior_is_supervised_with_train_labels_and_constant_fallback():
    rng=np.random.default_rng(17)
    x=rng.normal(size=(400,N_FEATURES))
    y=(x[:,0]+x[:,8]>0).astype(int)
    model=fit_candidate_transfer_prior(x,y)
    assert model.n_fit==400 and model.joint_successes==int(y.sum())
    assert model.model is not None
    probs=model.predict(x)
    assert 0<probs.min() and probs.max()<1
    assert probs[y==1].mean()>probs[y==0].mean()
    fallback=fit_candidate_transfer_prior(
        np.zeros((8,N_FEATURES)),np.zeros(8,dtype=int)
    )
    assert fallback.model is None
    np.testing.assert_allclose(
        fallback.predict(np.zeros((2,N_FEATURES))),1/10
    )


def test_default_sparse_q16_parity_with_observation_callback():
    x,donors,groups,gains,original=make_simple()
    queried=[0]
    def model(batch):
        queried[0]+=len(batch)
        # This can be true for two or more distinct time regions.
        p=np.mean(batch,axis=(1,2))
        return np.column_stack((np.zeros(len(batch)),p-.15))
    old=search_sparse_robust_counterfactual(
        x,original,donors,groups,gains,model,target_class=1,
        budget=16,maximum_modified_fraction=.35,taper=2,
    )
    first=queried[0]
    seen=[]
    def same_priority(masks,score,support,probes):
        seen.append(len(probes))
        assert len(masks)==len(score)==len(support)
        assert np.all(np.isfinite(score))
        return score
    new=search_sparse_robust_counterfactual(
        x,original,donors,groups,gains,model,target_class=1,
        budget=16,maximum_modified_fraction=.35,taper=2,
        candidate_priority=same_priority,
    )
    assert old.certified_edit==new.certified_edit
    assert old.is_certified==new.is_certified
    assert old.local_minimality_checked==new.local_minimality_checked
    assert old.queries==new.queries==queried[0]-first<=16
    assert [(p.components,p.donor_index,p.targeted_flip) for p in old.probes] == [
        (p.components,p.donor_index,p.targeted_flip) for p in new.probes
    ]
    assert seen and seen[0]==0


@pytest.mark.parametrize("Q",[8,16])
@pytest.mark.parametrize("adaptive",[False,True])
def test_learned_prior_never_returns_unmeasured_certificate(Q,adaptive):
    x,donors,groups,gains,original=make_simple()
    prior=fit_candidate_transfer_prior(
        np.zeros((12,N_FEATURES)),np.ones(12,dtype=int),
    )
    callback=TransferAwareRanking(
        x,donors,groups,gains,original,1,prior,adaptive=adaptive,
    )
    count=[0]
    def impossible(batch):
        count[0]+=len(batch)
        return np.tile(np.array([0.,-.25]),(len(batch),1))
    result=search_sparse_robust_counterfactual(
        x,original,donors,groups,gains,impossible,target_class=1,
        budget=Q,maximum_modified_fraction=.35,taper=2,
        candidate_priority=callback,
    )
    assert not result.is_certified and result.certified_edit is None
    assert 0<count[0]==result.queries<=Q
    assert callback.call_count>=1
    if adaptive:
        assert callback.max_feedback>0
    else:
        assert callback.max_feedback==0


def test_invalid_policy_scores_stop_before_blackbox_call():
    x,donors,groups,gains,original=make_simple()
    count=[0]
    def model(batch):
        count[0]+=len(batch)
        return np.tile(original,(len(batch),1))
    with pytest.raises(ValueError,match="candidate policy"):
        search_sparse_robust_counterfactual(
            x,original,donors,groups,gains,model,target_class=1,
            budget=8,candidate_priority=lambda m,g,s,p:np.full(len(m),np.nan),
        )
    assert count[0]==0


def test_feedback_ranking_changes_only_after_measured_probes():
    x,donors,groups,gains,original=make_simple()
    prior=fit_candidate_transfer_prior(
        np.zeros((10,N_FEATURES)),np.zeros(10,dtype=int)
    )
    ranker=TransferAwareRanking(
        x,donors,groups,gains,original,1,prior,adaptive=True
    )
    masks=((0,), (1,), (0,1))
    gainsarr=np.ones(3)
    supp=np.ones(3)*.1
    before=ranker(masks,gainsarr,supp,())
    from beaconxai.donor_robust_cf import DonorProbe
    measured=(DonorProbe((0,),0,0,-.4,False,"compact_candidate"),)
    after=ranker(masks,gainsarr,supp,measured)
    assert before[0]==before[1]
    assert after[0]<after[1]
    assert ranker.max_feedback==1
    with pytest.raises(ValueError,match="unseen donor"):
        ranker(masks,gainsarr,supp,(
            DonorProbe((0,),2,1,.2,True,"illegal"),
        ))
