"""Q16 operator consensus: exact call count, abstention and negative checks."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.dual_operator_audit import dual_operator_audit


def _inputs():
    x=np.zeros((24,9),dtype=np.float32)
    comps=make_components(x.shape,4,"pamap3")
    cheap=np.linspace(12,1,12)
    template=np.ones_like(x)
    return x,comps,cheap,template


def test_different_operator_ranks_abstain_at_exact_q16():
    x,comps,cheap,template=_inputs()
    calls=[]
    effects_a=np.linspace(.9,.2,8)
    effects_b=np.linspace(.2,.9,8)
    def predict_many(batch):
        calls.append(batch.copy())
        return np.column_stack((
            np.concatenate([2-effects_a,2-effects_b]),np.zeros(16)))
    out=dual_operator_audit(x,comps,cheap,predict_many,np.array([2.,0.]),
                            template,budget=16)
    assert len(calls)==1 and calls[0].shape==(16,*x.shape)
    assert len(out.ranking)==8
    assert out.interpolation_ranking[:3]==[0,1,2]
    assert out.donor_ranking[:3]==[7,6,5]
    assert out.top3_agreement==0
    assert out.reported is False
    assert out.extra_model_evaluations==16
    assert out.baseline_model_evaluations==0
    assert len(set(out.selected_components))==8
    assert set(out.ranking)==set(out.selected_components)


def test_agreement_reports_and_measures_only_selected_candidates():
    x,comps,cheap,template=_inputs()
    impacts=np.linspace(.7,.1,8)
    def predict_many(batch):
        return np.column_stack((
            np.concatenate([2-impacts,2-impacts]),np.zeros(16)))
    out=dual_operator_audit(x,comps,cheap,predict_many,np.array([2.,0.]),
                            template,budget=16)
    assert out.reported is True
    assert out.top3_agreement==1.
    assert out.ranking[:3]==[0,1,2]


@pytest.mark.parametrize("budget", [0,5,7,25,26])
def test_invalid_budget_rejected_without_query(budget):
    x,comps,cheap,template=_inputs()
    def forbid(batch):
        raise AssertionError("called model on malformed budget")
    with pytest.raises(ValueError):
        dual_operator_audit(x,comps,cheap,forbid,np.array([2.,0.]),
                            template,budget=budget)


def test_model_shape_and_nan_response_rejected():
    x,comps,cheap,template=_inputs()
    with pytest.raises(ValueError,match="predict_many"):
        dual_operator_audit(x,comps,cheap,lambda b:np.zeros((15,2)),
                            np.array([2.,0.]),template)
    with pytest.raises(ValueError,match="predict_many"):
        dual_operator_audit(x,comps,cheap,lambda b:np.full((16,2),np.nan),
                            np.array([2.,0.]),template)


def test_rival_template_geometry_and_prior_validation():
    x,comps,cheap,template=_inputs()
    with pytest.raises(ValueError):
        dual_operator_audit(x,comps,cheap,lambda b:np.zeros((16,2)),
                            np.array([2.,0.]),template[:10])
    with pytest.raises(ValueError):
        dual_operator_audit(x,comps,np.full(12,np.nan),lambda b:np.zeros((16,2)),
                            np.array([2.,0.]),template)
