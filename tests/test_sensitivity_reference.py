import numpy as np
import pytest
from beaconxai.sensitivity_reference import moving_margin_effects, rank_effects, reference_metrics, subject_macro_summary


def test_moving_margin_matches_explicit_top_two():
    base=np.array([2.,1.,0.])
    probes=np.array([[3.,1.,0.],[1.,2.,5.],[2.,1.,0.]])
    v=moving_margin_effects(base,probes)
    assert np.allclose(v,[1,2,0])
    assert rank_effects(v,[0,1,2])==[1,0,2]
    with pytest.raises(ValueError):
        moving_margin_effects(base,np.array([[float('nan'),0.,1.]]))


def test_equal_budget_top3_metrics_are_honest():
    ref=list(range(7))
    r=reference_metrics(ref,[0,3,2,5],7,4)
    assert r['top3_overlap']==2/3
    assert r['top3_in_shortlist']==2/3
    assert r['oracle_top1_in_top3']==1
    q0=reference_metrics(ref,[4,3,0,1,2,5,6],7,0)
    assert q0['top3_in_shortlist'] is None
    assert q0['top3_overlap']==1/3
    with pytest.raises(ValueError):
        reference_metrics(ref,[0,2],7,4)


def test_participant_macro_not_window_micro():
    base=dict(method='q16',extra_model_evaluations=16,top3_overlap=1.,
              oracle_top1_in_top3=1,top3_in_shortlist=1.,
              reference_max_shift=.4,reference_top3_gap=.2)
    rows=[dict(base,subject=2,sample=i) for i in range(9)]
    rows.append(dict(base,subject=9,sample=10,top3_overlap=0.,
                     oracle_top1_in_top3=0,top3_in_shortlist=0.,reference_max_shift=0.))
    summary,by_subject=subject_macro_summary(rows)
    assert summary[0]['subject_macro_top3_overlap']==.5
    assert summary[0]['n_subjects']==2
    assert by_subject[1]['n_reference_effect_ge_001']==0
    assert by_subject[1]['overlap_on_reference_effect_ge_001'] is None
