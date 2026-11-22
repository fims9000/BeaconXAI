"""PSD conditional inference and sequential real-Q8+8 BEACON controls."""
import numpy as np
import pytest

from beaconxai.active_residual_evidence import (
    fit_residual_conditional,active_counterevidence_from_features
)
from beaconxai.conformal_evidence import cheap_feature_matrix
from beaconxai.contrastive import make_components


def fixture():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(24,9)).astype("float32")
    profile=rng.normal(size=(24,9)).astype("float32")
    comps=make_components(x.shape,12,"pamap3")
    base=np.array([2.,1.,0.])
    F=[];Y=[]
    for i in range(10):
        data=x + .1*rng.normal(size=x.shape)
        f=cheap_feature_matrix(data,profile,base,comps)
        response=np.maximum(
            0., .2+.15*f[:,4]+.2*f[:,0]+.1*rng.normal(size=len(comps)))
        F.append(f);Y.append(response)
    return x,profile,comps,base,F,Y


def test_train_only_covariance_is_symmetric_psd_and_conditioned():
    x,profile,comps,base,F,Y=fixture()
    fitted=fit_residual_conditional(F,Y,comps)
    K=fitted.covariance
    assert K.shape==(36,36)
    assert np.allclose(K,K.T)
    assert np.linalg.eigvalsh(K).min()>0.
    pred,unc=fitted.conditional(F[0],comps,[],[])
    assert pred.shape==(36,) and np.all(unc>0)
    ids=[1,2,3,5,7,9,12,15]
    measured=np.maximum(0.,Y[0][ids])
    posterior,variance=fitted.conditional(F[0],comps,ids,measured)
    assert np.allclose(posterior[ids],measured)
    assert np.all(variance[ids]==0.)
    assert np.isfinite(posterior).all()
    assert np.all(posterior>=0)


def test_real_active_8plus8_stages_no_repeat_no_full_oracle():
    x,profile,comps,base,F,Y=fixture()
    fitted=fit_residual_conditional(F,Y,comps)
    prior=np.arange(len(comps),0,-1,dtype=float)
    calls=[]
    def model(batch):
        calls.append(len(batch))
        total=np.sum(np.square(batch),axis=(1,2))
        return np.column_stack((2.-total/300,np.ones(len(batch)),np.zeros(len(batch))))
    current=cheap_feature_matrix(x,profile,base,comps)
    result=active_counterevidence_from_features(
        x,base,comps,current,fitted,model,prior)
    assert calls==[8,8]
    assert result.extra_model_evaluations==16
    assert result.forward_batches==2
    assert len(set(result.candidate_ids))==16
    assert not (set(result.first_stage_ids)&set(result.second_stage_ids))
    assert len(result.measured_only_ranking)==16
    assert len(result.conditional_ranking_all_components)==36
    assert len(set(result.conditional_ranking_all_components))==36
    assert np.isfinite(result.inferred_effects_all_components).all()
    assert np.all(result.uncertainty_all_components>=0)
    assert all(result.uncertainty_all_components[g]==0
               for g in result.candidate_ids)


def test_bad_covariance_observations_cannot_invent_verified_queries():
    x,profile,comps,base,F,Y=fixture()
    fitted=fit_residual_conditional(F,Y,comps)
    with pytest.raises(ValueError):
        fitted.conditional(F[0],comps,[1,1],[.1,.2])
    with pytest.raises(ValueError):
        fitted.conditional(F[0],comps,[1],[-.1])
    with pytest.raises(ValueError):
        active_counterevidence_from_features(
            x,base,comps,F[0][:,:3],fitted,
            lambda xs:np.ones((len(xs),3)),np.ones(36))
    with pytest.raises(ValueError):
        active_counterevidence_from_features(
            x,base,comps,F[0],fitted,
            lambda xs:np.ones((len(xs),2)),np.ones(36))
    with pytest.raises(ValueError):
        fit_residual_conditional(F,Y,comps,empirical_weight=1.1)
