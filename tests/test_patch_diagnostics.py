from dataclasses import dataclass
import numpy as np
import pytest
from beaconxai.patch_diagnostics import patch_effects, summarize_patch_effects, summarize_method_effects


@dataclass
class Component:
    group: int = 0
    time_bin: int = 1
    t0: int = 0
    t1: int = 2
    c0: int = 0
    c1: int = 1


def test_zero_effect_and_predicted_switch_stable_softmax():
    x = np.zeros((2,4,2))
    y = x.copy()
    y[1,:2,0] = 3
    ref = np.array([[0.,1.], [500.,0.]])
    edited = np.array([[0.,1.], [0.,500.]])
    records = patch_effects(x,y,ref,edited,np.array([0,0]),[Component()])
    assert records[0]['patch_model_tv'] == pytest.approx(0.0)
    assert records[0]['patch_changed_predicted_class'] == 0
    assert records[1]['patch_model_tv'] == pytest.approx(1.0)
    assert records[1]['patch_model_tv_ge_010'] == 1
    assert records[1]['patch_changed_predicted_class'] == 1
    assert records[1]['patch_rms'] == pytest.approx(3.)
    agg=summarize_patch_effects(records,[9,9])
    assert agg[0]['windows']==2 and agg[0]['prediction_change_rate']==.5


def test_method_summary_conditions_are_separate_from_shortlist():
    effects = [
        dict(method='q16',trial=0,subject=9,sample=0,patch_model_tv_ge_005=1,
             patch_changed_predicted_class=1,candidate_covered=1,hit3=1),
        dict(method='q16',trial=0,subject=9,sample=1,patch_model_tv_ge_005=0,
             patch_changed_predicted_class=0,candidate_covered=0,hit3=0),
        dict(method='q16',trial=1,subject=9,sample=0,patch_model_tv_ge_005=1,
             patch_changed_predicted_class=1,candidate_covered=1,hit3=1),
        dict(method='q16',trial=1,subject=9,sample=1,patch_model_tv_ge_005=0,
             patch_changed_predicted_class=0,candidate_covered=0,hit3=0),
    ]
    report = summarize_method_effects(effects)[0]
    assert report['windows']==2
    assert report['n_tv_ge_005_windows']==1
    assert report['hit3_tv_ge_005']==1.
    assert report['hit3_tv_lt_005']==0.
    assert report['direct_shortlist_coverage']==.5
    assert report['hit3_given_shortlisted']==1.


def test_no_shortlist_for_q0_and_empty_condition_is_null():
    row=dict(method='q0',trial=0,subject=1,sample=0,patch_model_tv_ge_005=0,
             patch_changed_predicted_class=0,candidate_covered='',hit3=1)
    x=summarize_method_effects([row])[0]
    assert x['direct_shortlist_coverage'] is None
    assert x['hit3_tv_ge_005'] is None
    assert x['n_tv_ge_005_windows'] == 0


def test_invalid_geometry_rejected():
    x=np.zeros((2,5,3))
    z=np.zeros((2,2))
    with pytest.raises(ValueError):
        patch_effects(x,x,z,z,np.array([0,1]),[Component()])
    with pytest.raises(ValueError):
        patch_effects(x,x[:1],z,z,np.array([0,0]),[Component()])
