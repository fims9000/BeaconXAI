"""Staged BEACON: real Q budgets, train-only policy selection, negative inputs."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.staged_budget import (
    StagedPolicy, measured_top3_gap, choose_budget,run_staged_audit,
    fit_staged_policy,evaluate_policy_offline,
)


def _toy():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(24,9)).astype("float32")
    comps=make_components(x.shape,12,"pamap3")
    prior=np.arange(len(comps),0,-1,dtype=float)
    baseline=np.array([2.,0.])
    return x,comps,prior,baseline


@pytest.mark.parametrize("mode,wanted_batches,wanted_calls", [
    ("fixed8",[8],8),("fixed16",[8,8],16),("fixed32",[8,8,16],32)
])
def test_only_requested_extra_model_examples_are_executed(mode,wanted_batches,wanted_calls):
    x,comps,prior,baseline=_toy()
    batch_sizes=[]
    def model(batch):
        batch_sizes.append(len(batch))
        m=np.sum(np.square(batch),axis=(1,2))
        return np.column_stack((2.-m/200,np.zeros(len(batch))))
    result=run_staged_audit(
        x,comps,prior,baseline,model,StagedPolicy(mode))
    assert batch_sizes==wanted_batches
    assert result.queries==wanted_calls
    assert result.actual_forward_batches==len(wanted_batches)
    assert len(result.selected)==wanted_calls
    assert len(result.ranking)==wanted_calls
    assert set(result.ranking)==set(result.selected)
    assert set(result.stage_rankings)==set([8,16,32][:len(wanted_batches)])


def test_adaptive_stops_at_q8_q16_or_q32_without_future_queries():
    x,comps,prior,baseline=_toy()
    all_sizes=[]
    def model(batch):
        all_sizes.append(len(batch))
        m=np.sum(np.square(batch),axis=(1,2))
        return np.column_stack((2.-m/200,np.zeros(len(batch))))
    cases=[
        (StagedPolicy("adaptive",0,1.01),8,[8]),
        (StagedPolicy("adaptive",1.01,0),16,[8,8]),
        (StagedPolicy("adaptive",1.01,1.01),32,[8,8,16])
    ]
    for p,expected,sizes in cases:
        all_sizes.clear()
        out=run_staged_audit(x,comps,prior,baseline,model,p)
        assert out.queries==expected
        assert all_sizes==sizes
        if expected==8:
            assert 16 not in out.stage_features and 32 not in out.stage_rankings


def test_observable_gap_and_threshold_contract():
    assert measured_top3_gap(np.array([.1,.3,.5,.7,.9]))==pytest.approx(.2/.9)
    assert measured_top3_gap(np.zeros(8))==0.
    assert choose_budget(StagedPolicy("adaptive",.4,.3),.5,0.)==8
    assert choose_budget(StagedPolicy("adaptive",.4,.3),.1,.3)==16
    assert choose_budget(StagedPolicy("adaptive",.4,.3),.1,.2)==32


@pytest.mark.parametrize("mode",["fixed8","fixed16","fixed32","adaptive"])
def test_no_q36_or_ytest_passed_to_online_api(mode):
    x,comps,prior,baseline=_toy()
    def model(batch):
        return np.zeros((len(batch),2))
    result=run_staged_audit(
        x,comps,prior,baseline,model,StagedPolicy(mode))
    assert result.queries in (8,16,32)
    assert len(result.stage_rankings)<=3


@pytest.mark.parametrize("bad",[
    StagedPolicy("adaptive",1.01,1.01),
    StagedPolicy("fixed32")
])
def test_malformed_model_output_fails_instead_of_claiming_cost(bad):
    x,comps,prior,baseline=_toy()
    with pytest.raises(ValueError,match="predict_many"):
        run_staged_audit(x,comps,prior,baseline,
                         lambda batch:np.zeros((len(batch),1)),bad)


def test_nonfinite_input_or_features_rejected():
    x,comps,prior,baseline=_toy()
    prior[0]=np.nan
    with pytest.raises(ValueError):
        run_staged_audit(x,comps,prior,baseline,lambda batch:batch,StagedPolicy("fixed8"))
    with pytest.raises(ValueError):
        StagedPolicy("adaptive",float("nan"),.1)
    with pytest.raises(ValueError):
        measured_top3_gap(np.array([1,1,np.nan,0]))


def _calibration():
    rows=[]
    for subject in (1,2,4):
        for i in range(4):
            easy=i%2==0
            rows.append({
                "subject":subject,
                "gap8":.5 if easy else 0.,
                "gap16":.5 if easy else 0.,
                "overlap8":1. if easy else 0.,
                "overlap16":1. if easy else 2/3,
                "overlap32":1.,
            })
    return rows


def test_train_thresholds_can_promote_adaptive_when_genuine_utility_gain():
    p,report=fit_staged_policy(_calibration(),penalty=.5,required_gain=.01)
    assert p.mode=="adaptive"
    assert report["adaptive_promoted"]
    assert report["best_adaptive_utility"]>report["best_fixed_utility"]+.01
    assert choose_budget(p,.5,.5)==8
    assert choose_budget(p,0.,0.)==32
    assert report["no_outer_test_labels_or_reference_used_to_fit"]


def test_fixed_q_fallback_not_forcing_unreliable_adaptive_gain():
    all_easy=[{**r,"gap8":.2,"gap16":.2,"overlap8":1.,"overlap16":1.,
                "overlap32":1.} for r in _calibration()]
    p,report=fit_staged_policy(all_easy,penalty=.5,required_gain=.01)
    assert p.mode=="fixed8"
    assert not report["adaptive_promoted"]
    assert report["selected_policy"]=="fixed8"
    assert evaluate_policy_offline(all_easy,p,.5)["macro_q"]==8.


def test_calibration_requires_multiple_subjects_and_finite_reference():
    with pytest.raises(ValueError,match="at least two"):
        fit_staged_policy([_calibration()[0]])
    rows=_calibration()
    rows[0]["overlap32"]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        fit_staged_policy(rows)
