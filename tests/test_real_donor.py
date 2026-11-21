"""Train-only donor controls, query budgets and score sign tests."""
import numpy as np
import pytest
from beaconxai.contrastive import make_components
from beaconxai.real_donor import (
    fit_real_donor_bank, component_fingerprints, nearest_context_index,
    boundary_align_lag, make_target_replacement, plausibility_proxies,
)
from scripts.run_real_donor_benchmark import (
    select_cases, evaluate_case, summarize_case, aggregate, METHODS,
)


def fixture_data():
    rng=np.random.default_rng(72)
    x=rng.normal(size=(40,96,9)).astype('float32')
    y=np.array([0]*20+[1]*20)
    x[20:,30:60,:3]+=4
    cs=make_components((96,9),12)
    return x,y,cs,fit_real_donor_bank(x,y,cs,max_donors=8,seed=7)


def test_train_only_and_real_medoid():
    x,y,cs,bank=fixture_data()
    assert bank.labels==[0,1]
    assert bank.counts=={0:20,1:20}
    assert set(bank.selected_train_indices[1]).issubset(set(range(20,40)))
    assert any(np.allclose(bank.medoids[1],v) for v in x[y==1])
    assert bank.bank_bytes>0
    assert bank.fingerprints[0].shape==(8,36,2)


def test_context_search_excludes_selected_bin_and_is_repeatable():
    x,y,cs,b=fixture_data()
    j=2
    i=nearest_context_index(x[0],j,b,1,cs)
    assert 0<=i<8
    assert i==nearest_context_index(x[0],j,b,1,cs)
    assert component_fingerprints(x[0],cs).shape==(1,36,2)


def test_all_interventions_only_modify_selected_region():
    x,y,cs,b=fixture_data()
    p=x[0];j=4;c=cs[j]
    for kind in METHODS:
        edited,meta=make_target_replacement(p,c,j,1,kind,b,cs,seed=3,taper=2)
        mask=np.ones_like(p,bool)
        mask[c.t0:c.t1,c.c0:c.c1]=False
        np.testing.assert_array_equal(edited[mask],p[mask])
        assert np.isfinite(edited).all()
        stats=plausibility_proxies(p,edited,c,b,1,cs)
        assert all(np.isfinite(v) for v in stats.values())
        if kind in ('random_donor','context_donor','aligned_donor'):
            assert meta['donor_index']>=0
            assert 20<=meta['train_index']<40


def test_budget_and_prob_deltas():
    rng=np.random.default_rng(81)
    train=rng.normal(size=(60,96,9)).astype('float32')
    train[30:,32:48,:3]+=1
    y=np.array([0]*30+[1]*30)
    cs=make_components((96,9),12)
    bank=fit_real_donor_bank(train,y,cs,max_donors=16)
    sample=train[3]
    def predict(batch):
        z=np.mean(batch[:,32:48,:3],axis=(1,2))
        p=1/(1+np.exp(-z))
        return np.column_stack((np.log(p),np.log1p(-p)))
    baseline=predict(sample[None])[0]
    for method in METHODS:
        calls=[0]
        def counted(xs):
            calls[0]+=len(xs)
            return predict(xs)
        rows,t=evaluate_case(sample,baseline,bank,cs,list(range(16)),0,1,method,counted,seed=9)
        assert calls[0]==len(rows)==16
        assert all(abs(r['delta_prob_source']+r['delta_prob_rival'])<1e-6 for r in rows)
        one=summarize_case(rows,t)
        assert one['queries']==16
        assert 0<=one['directed_fraction']<=1


def test_stratified_selection_and_aggregation():
    y=np.array([0,0,0,0,1,1,1,1])
    pred=np.array([0,0,1,1,1,1,0,0])
    a=select_cases(y,pred,1,1,8,23)
    assert len(a)==4
    assert np.array_equal(a,select_cases(y,pred,1,1,8,23))
    rows=[{'method':'mean','subject':8,'flip_any':0,'directed_any':1,
           'directed_fraction':.5,'best_gap_reduction':.2,
           'mean_target_fp_nn_dist':1.,'mean_relative_l2':.1,
           'mean_seam_excess':0.,'wall_ms':1.}]
    assert aggregate(rows,('method',))[0]['mean_directed_any']==1


def test_invalid_inputs():
    x,y,cs,b=fixture_data()
    with pytest.raises(ValueError): fit_real_donor_bank(x,y,cs,max_donors=0)
    with pytest.raises(ValueError): make_target_replacement(x[0],cs[0],0,1,'none',b,cs)
    with pytest.raises(ValueError): boundary_align_lag(x[0],x[2],cs[0],max_shift=-1)
