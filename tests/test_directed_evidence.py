import numpy as np
import pytest

from beaconxai.directed_evidence import (
    fixed_source_margin,signed_fragment_evidence,positive_reference_ranking
)
from beaconxai.sensitivity_reference import moving_margin_effects


def test_moving_top_two_fails_to_notice_class_reversal():
    baseline=np.array([2.,1.,0.])
    altered=np.array([[1.,2.,0.]])
    # Old BEACON: both top2 margins = 1 -> zero effect!
    assert moving_margin_effects(baseline,altered)[0]==0.
    # Signed, fixed predicted original class 0: margin 1 -> -1.
    c,opposes,supports=signed_fragment_evidence(baseline,altered)
    assert c[0]==-2.
    assert opposes[0]==0.
    assert supports[0]==2.


def test_signed_counterevidence_vs_support_is_not_abs_moving():
    base=np.array([2.,1.,0.])
    probes=np.array([
        [3.,1.,0.], # remove feature: class 0 gets stronger (opposed)
        [1.,2.,0.], # remove feature: class 0 weakens (supported)
        [2.,1.,0.], # unchanged
    ])
    change,counter,support=signed_fragment_evidence(base,probes)
    assert np.allclose(change,[1,-2,0])
    assert np.allclose(counter,[1,0,0])
    assert np.allclose(support,[0,2,0])
    assert float(fixed_source_margin(base,0))==1.
    assert np.allclose(fixed_source_margin(probes,0),[2,-1,1])


def test_counterevidence_top3_requires_three_nonzero_responses():
    rank,n=positive_reference_ranking(np.array([.0,.7,.3,.0,.2]))
    assert n==3 and rank[:3]==[1,2,4]
    _,n=positive_reference_ranking(np.zeros(6))
    assert n==0
    with pytest.raises(ValueError):
        positive_reference_ranking(np.array([-1.,0.,0.]))


def test_signed_response_rejects_bad_source_and_outputs():
    with pytest.raises(ValueError):
        fixed_source_margin(np.array([1.,2.,3.]),3)
    with pytest.raises(ValueError):
        signed_fragment_evidence(np.array([1.,0.]),np.zeros((3,3)))
    with pytest.raises(ValueError):
        signed_fragment_evidence(np.array([1.,0.]),
                                 np.array([[np.nan,1.]]))
