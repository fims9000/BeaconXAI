"""Tests for train-only donor risk, calibrated refusal and Q-free features."""
import numpy as np
import pytest

from beaconxai.donor_transfer_risk import (
    FEATURE_NAMES, fit_transfer_risk, donor_transfer_features,
    upper_clopper_pearson, calibrate_transfer_gate,
)


def test_features_require_only_two_measured_donors():
    x=np.zeros((12,3))
    a=np.ones_like(x)
    b=np.ones_like(x)*2
    f=donor_transfer_features(
        x,(a,b),(1,2),(.2,.3),(.5,.3),-.6,
        components_count=10,online_queries=11,minimality_checked=True,
    )
    assert f.shape==(len(FEATURE_NAMES),)==(9,)
    np.testing.assert_allclose(f[[0,1,2,3,4,5,6,7,8]],
                               [.3,.3,.2,.2,4.,1.,11/16,-.6,1.])
    with pytest.raises(ValueError,match="invalid"):
        donor_transfer_features(
            x,(a,b),(1,2),(.2,.3),(.5,.3),-.6,
            components_count=10,online_queries=17,
        )


def test_logistic_is_trained_on_only_supplied_rows():
    rng=np.random.default_rng(9)
    x=rng.normal(size=(80,9))
    y=(x[:,0]+x[:,5]>.2).astype(int)
    model=fit_transfer_risk(x,y)
    scores=model.predict(x)
    assert model.fit_n==80
    assert model.model is not None and model.constant is None
    assert scores.shape==(80,)
    assert float(scores[y==1].mean())>float(scores[y==0].mean())
    with pytest.raises(ValueError,match="finite"):
        model.predict(np.full((2,9),np.nan))


def test_single_class_and_empty_training_fall_back_to_laplace():
    a=fit_transfer_risk(np.zeros((5,9)),np.ones(5,dtype=int))
    assert a.model is None
    assert a.predict(np.zeros((2,9)))[0]==pytest.approx(6/7)
    b=fit_transfer_risk(np.empty((0,9)),np.empty(0,dtype=int))
    assert b.model is None and b.constant==.5


def test_one_sided_exact_bounds_and_bonferroni_gate():
    assert upper_clopper_pearson(0,100,.02)<.05
    assert upper_clopper_pearson(100,100,.02)==1.
    assert upper_clopper_pearson(0,5,.02)>.5
    x=np.r_[np.linspace(0.,.3,100),np.linspace(.4,1.,100)]
    s=np.r_[np.linspace(0.,.3,100),np.linspace(.4,1.,100)]
    y=np.r_[np.zeros(100,dtype=int),np.ones(100,dtype=int)]
    support=np.linspace(.02,.3,len(s))
    gate=calibrate_transfer_gate(x,s,y,support)
    assert gate.threshold is not None
    assert gate.calibration_accepted>=5
    assert gate.calibration_upper<=.5
    assert gate.accepts(0.)
    assert not gate.accepts(float("nan"))
    assert gate.support_threshold is not None


def test_gate_refuses_all_if_calibration_cannot_certify():
    x=np.linspace(0,1,40)
    gate=calibrate_transfer_gate(
        x,x,np.ones(40,dtype=int),np.linspace(.01,.3,40),
    )
    assert gate.threshold is None
    assert not gate.accepts(0.)
    assert not gate.accepts_support(.01)


def test_gate_thresholds_unchanged_when_outcomes_not_used():
    fit=np.linspace(0,1,100)
    scores=np.linspace(0,1,70)
    support=np.linspace(.01,.35,70)
    g1=calibrate_transfer_gate(fit,scores,np.zeros(70,dtype=int),support)
    g2=calibrate_transfer_gate(fit,scores,np.ones(70,dtype=int),support)
    assert g1.fit_thresholds==g2.fit_thresholds
    assert g1.threshold is not None and g2.threshold is None
