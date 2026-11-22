"""Train-only pair effects, sequential posterior and strict shared-Q certificates."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.learned_interactions import (
    feature_vector,fit_interaction_prior,search_learned_adaptive,
)
from beaconxai.verified_counterfactual import splice


def test_pair_model_recovers_true_two_block_synergy():
    rng=np.random.default_rng(41)
    rows=[]
    for _ in range(24):
        g=rng.uniform(.5,1.5,size=5)
        for k in (1,2,3):
            for mask in list(__import__("itertools").combinations(range(5),k)):
                f=feature_vector(g,mask,pairwise=True)
                # Interaction effect cannot be represented by additive-only features.
                y=.25*f[0]+2.0*int(0 in mask and 1 in mask)
                rows.append((g,mask,y))
    learned=fit_interaction_prior(rows,n_components=5,pairwise=True,alpha=10.)
    additive=fit_interaction_prior(rows,n_components=5,pairwise=False,alpha=10.)
    assert learned.training_rows==len(rows)
    assert learned.pair_coefficient_norm>.05
    assert learned.calibration_mae < additive.calibration_mae*.6
    assert learned.vector(np.ones(5),(0,1)).shape==(1+1+5+10,)


def test_pair_training_does_not_call_classifier():
    rows=[(np.ones(4),(i,),float(i)) for i in range(4) for _ in range(3)]
    model=fit_interaction_prior(rows,n_components=4,pairwise=True)
    assert model.pairwise and model.training_rows==12
    assert model.pair_coefficient_norm==0.


def test_shared_q16_two_donor_evidence_and_compactness():
    x=np.zeros((50,2))
    donors=[np.ones_like(x),np.full_like(x,.9)]
    groups=make_components((50,2),10,"all_channels")
    prior=fit_interaction_prior([
        (np.ones(10), (i,), 0.) for i in range(10) for _ in range(4)
    ],n_components=10,pairwise=True)
    gain=np.ones((2,10))
    calls=[0]
    def model(batch):
        calls[0]+=len(batch)
        mu=batch.mean(axis=(1,2))
        # Only two specific local blocks together flip the model.
        first=batch[:,0:5].mean(axis=(1,2))
        second=batch[:,5:10].mean(axis=(1,2))
        delta=(first>.5)&(second>.5)
        return np.column_stack((np.zeros(len(batch)),np.where(delta,.6,-.6)))
    result=search_learned_adaptive(
        x,np.array([0.,-.6]),donors,groups,gain,prior,model,
        target_class=1,budget=16,maximum_modified_fraction=.35,adaptive=True
    )
    assert 0<result.queries==calls[0]<=16
    assert result.posterior_updates==result.queries
    if result.is_certified:
        for donor_i,d in enumerate(donors):
            candidate=splice(x,d,groups,result.certified_edit,taper=2)
            assert np.mean(np.abs(candidate-x)>1e-8)<=.35
            assert any(p.components==result.certified_edit
                       and p.donor_index==donor_i and p.targeted_flip
                       for p in result.certificate.probes)
    else:
        assert result.certified_edit is None


@pytest.mark.parametrize("Q",[8,16])
@pytest.mark.parametrize("adaptive",[False,True])
def test_no_target_flip_never_certified_and_budget_never_exceeded(Q,adaptive):
    x=np.zeros((50,2))
    comps=make_components((50,2),10,"all_channels")
    prior=fit_interaction_prior(
        [(np.ones(10),(j,),0.) for j in range(10) for _ in range(4)],
        n_components=10,pairwise=True
    )
    count=[0]
    def no_flip(batch):
        count[0]+=len(batch)
        return np.column_stack((np.ones(len(batch)),np.zeros(len(batch))))
    result=search_learned_adaptive(
        x,np.array([1.,0.]),[np.ones_like(x),np.ones_like(x)*.8],
        comps,np.ones((2,10)),prior,no_flip,target_class=1,
        budget=Q,adaptive=adaptive
    )
    assert not result.is_certified and result.certified_edit is None
    assert result.queries==count[0]<=Q
    assert result.posterior_updates==(count[0] if adaptive else 0)


def test_invalid_feature_map_and_donor_geometry_rejected():
    with pytest.raises(ValueError,match="mask"):
        feature_vector(np.ones(3),(0,3),pairwise=True)
    with pytest.raises(ValueError,match="insufficient"):
        fit_interaction_prior([(np.ones(3),(0,),.1)],n_components=3,pairwise=True)
    comps=make_components((50,2),10,"all_channels")
    prior=fit_interaction_prior([
        (np.ones(10),(i,),0.) for i in range(10) for _ in range(3)
    ],n_components=10,pairwise=False)
    with pytest.raises(ValueError,match="geometry"):
        search_learned_adaptive(
            np.zeros((50,2)),np.array([1.,0.]),
            [np.zeros((50,2)),np.zeros((49,2))],
            comps,np.ones((2,10)),prior,lambda a:a,target_class=1,
        )
