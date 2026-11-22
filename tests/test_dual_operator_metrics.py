"""Post-hoc gate diagnostics must not inflate dependent sample counts."""
import pytest
from beaconxai.dual_operator_metrics import summarize_dual_operator


def _row(method,subject,sample,medoid,interpolation,reported=None,consistent=0):
    return {
        "method":method,"subject":subject,"sample":sample,
        "extra_model_evaluations":16,
        "worst_reference_overlap":min(medoid,interpolation),
        "mean_reference_overlap":(medoid+interpolation)/2,
        "medoid_reference_overlap":medoid,
        "interpolation_reference_overlap":interpolation,
        "reported_not_abstained":reported,
        "full_operator_reference_consistent":consistent
    }


def test_reported_gate_and_subject_macro_not_window_micro():
    rows=[
        _row("dual_operator_q16",2,i,1,1,1,1) for i in range(9)
    ]+[
        _row("dual_operator_q16",9,10,0,0,1,0)
    ]
    summary,per_subject,gate=summarize_dual_operator(rows)
    assert summary[0]["subject_macro_mean_reference_overlap"]==.5
    assert summary[0]["n_subjects"]==2
    assert gate["n_windows"]==10
    assert gate["n_reported"]==10
    assert gate["n_full_reference_consistent"]==9
    assert gate["subject_macro_reference_consistent_given_reported"]==.5
    assert gate["subject_macro_reported_rate"]==1.


def test_no_reported_returns_null_not_nan():
    rows=[_row("dual_operator_q16",9,0,.3,.2,0,0)]
    _,_,gate=summarize_dual_operator(rows)
    assert gate["n_reported"]==0
    assert gate["subject_macro_reference_consistent_given_reported"] is None
    assert gate["subject_macro_reported_given_reference_consistent"] is None


def test_unique_subject_sample_and_budget_required():
    r=_row("dual_operator_q16",9,0,.2,.3,0,0)
    with pytest.raises(ValueError,match="repeated"):
        summarize_dual_operator([r,r])
    with pytest.raises(ValueError,match="inconsistent budget"):
        summarize_dual_operator([r,{**r,"sample":1,"extra_model_evaluations":8}])


def test_missing_explicit_report_rejected():
    with pytest.raises(ValueError,match="report"):
        summarize_dual_operator([_row("dual_operator_q16",9,0,.2,.3,None,0)])


def test_no_dual_row_rejected():
    with pytest.raises(ValueError,match="missing"):
        summarize_dual_operator([_row("medoid_q16",1,0,.2,.3,None,0)])
