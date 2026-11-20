import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.conformal_evidence import (
    cheap_feature_matrix, full_model_responses,fit_response_surrogate,
    calibrate_subject_bound,SubjectBound,affordable_certificate,
    run_certified_explanation,
)


def _sample():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(24,9))
    mean=rng.normal(size=(24,9))
    components=make_components(x.shape,12,"pamap3")
    baseline=np.array([.9,-.2,.0])
    return x,mean,components,baseline


def test_original_cheap_four_term_statistics_and_no_model_calls():
    x,mean,comps,base=_sample()
    features=cheap_feature_matrix(x,mean,base,comps)
    assert features.shape==(36,17)
    assert np.isfinite(features).all()
    assert np.allclose(features[:,4],np.sum(features[:,:4],axis=1))
    assert features[0,15]==1 and features[12,16]==1
    # Changing class prediction margin changes cheap context but not raw x.
    shifted=cheap_feature_matrix(x,mean,np.array([2.,1.,0.]),comps)
    assert not np.array_equal(features[:,10],shifted[:,10])
    with pytest.raises(ValueError):
        cheap_feature_matrix(x,mean[:2],base,comps)


def test_surrogate_training_and_heteroscedastic_scale():
    x,mean,comps,base=_sample()
    features=[];effects=[]
    for i in range(4):
        f=cheap_feature_matrix(x+i*.1,mean,base,comps)
        features.append(f)
        effects.append(.15+np.maximum(f[:,4],0)*.3+np.arange(36)/200)
    fit=fit_response_surrogate(features,effects,comps,ridge_alpha=1.)
    pred,scale=fit.predict(features[0],comps)
    assert len(pred)==36
    assert np.isfinite(pred).all() and np.all(pred>=0)
    assert np.all(scale>0)
    assert fit.fit_window_count==4
    assert fit.trained_effect_count==144
    assert fit.features==17
    with pytest.raises(ValueError):
        fit_response_surrogate(features,[np.array([-1.])],comps)


def test_subject_block_quantile_and_infinite_when_insufficient_groups():
    records=[]
    for sid in range(10,19):
        for k in range(2):
            records.append(dict(
                subject=sid,
                predicted=np.zeros(4),
                scale=np.ones(4),
                effects=np.array([float(sid-10)+k/10,0,0,0]),
            ))
    b,receipts=calibrate_subject_bound(
        records,alpha=.2,expected_windows_per_subject=2)
    # S=9 alpha=.2 -> ceil(10*.8)=8th score -> subject=17.
    assert b.cutoff_rank==8
    assert b.quantile==pytest.approx(7.1)
    assert len(receipts)==9
    assert b.minimum_windows_per_subject==2
    assert np.allclose(b.upper(np.zeros(4),np.ones(4)),7.1)
    b2,_=calibrate_subject_bound(records[:8],alpha=.1,
                                expected_windows_per_subject=2)
    assert np.isinf(b2.quantile)
    assert np.isinf(b2.upper(np.zeros(4),np.ones(4))).all()
    with pytest.raises(ValueError,match="same window count"):
        calibrate_subject_bound(records[:-1],alpha=.2,expected_windows_per_subject=2)


def test_certificate_inequality_is_sound_when_upper_bound_holds():
    envelope=np.array([.5,.7,.8,1.,1.])
    ok,third,unqueried=affordable_certificate([2,3,4],[.8,1.1,1.2],
                                              envelope,5)
    assert ok and third==.8 and unqueried==.7
    ok,_,_=affordable_certificate([0,1,2],[1.0,.3,.9],envelope,5)
    assert not ok
    with pytest.raises(ValueError):
        affordable_certificate([0,0,1],[.3,.4,.5],envelope,5)


def test_online_certificate_uses_real_probes_not_full_oracle():
    x,mean,comps,base=_sample()
    n=36
    # Shift one highly ranked component to dominate globally.
    special=comps[0]
    x[special.t0:special.t1,special.c0:special.c1]=10.
    calls=[]
    def fake_model(batch):
        calls.append(len(batch))
        score=np.array([np.sum(np.square(z)) for z in batch])
        # The original baseline is only a supplied constant score vector.
        # Fine to use a fake deterministic black box for API tests.
        return np.column_stack((score/100, np.zeros(len(batch)),np.zeros(len(batch))))
    predicted=np.full(n,.01)
    predicted[:4]=10.
    bound=SubjectBound(alpha=.2,calibration_subjects=9,cutoff_rank=8,
                       quantile=0.,minimum_windows_per_subject=2,
                       maximum_windows_per_subject=2)
    result=run_certified_explanation(x,base,comps,predicted,bound,
        np.ones(n),fake_model,max_queries=16,step=4)
    assert len(result.ranked)==result.extra_model_evaluations==sum(calls)
    assert all(0<=g<n for g in result.ranked)
    assert result.forward_batches==len(calls)
    # A certificate is claimed only if actual observations beat all
    # unmeasured upper envelopes; otherwise the API explicitly abstains.
    if result.certified:
        truth=full_model_responses(x,base,comps,fake_model)
        reference=np.argsort(-truth,kind="stable")[:3].tolist()
        assert set(result.ranked[:3])==set(reference)
    else:
        assert result.reason=="abstain_unqueried_ambiguous"


def test_infinite_conformal_envelope_must_abstain_and_spend_full_budget():
    x,mean,comps,base=_sample()
    calls=[]
    def model(batch):
        calls.append(len(batch))
        return np.tile(base,(len(batch),1))
    b=SubjectBound(.1,4,5,np.inf,2,2)
    result=run_certified_explanation(
        x,base,comps,np.zeros(36),b,np.ones(36),
        model,max_queries=16,step=4)
    assert not result.certified
    assert result.extra_model_evaluations==16
    assert calls==[4,4,4,4]
    assert np.isinf(result.maximum_unqueried_upper_bound)
