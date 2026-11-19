import numpy as np
import pytest
from dataclasses import dataclass
from beaconxai.query_gate import QueryGate
from beaconxai.selective_audit import run_selective_audit
from beaconxai.contrastive import neutralize_interp

@dataclass
class Component:
    t0: int
    t1: int
    c0: int
    c1: int


def setup():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(96,3)).astype('float32')
    comps=[Component(i*8,(i+1)*8,0,3) for i in range(12)]
    cheap=np.array([2.0,1.7,.1,1.4,1.1,.8,.7,.6,.4,.2,.3,.5])
    return x,comps,cheap


def model(xs):
    xs=np.asarray(xs)
    score=xs[:,8:32,0].mean(axis=1) + xs[:,48:64,2].std(axis=1)
    return np.column_stack((score, -score, .1*score))


def test_q0_makes_no_extra_model_calls():
    x,comps,cheap=setup()
    calls=[]
    def predict(xs):
        calls.append(len(xs))
        return model(xs)
    result=run_selective_audit(x,comps,cheap,predict,QueryGate('q0',budget=8))
    assert calls == [1]
    assert result.extra_model_evaluations == 0
    assert result.baseline_model_evaluations == 1
    assert result.selected_components == []
    assert result.ranking == np.argsort(-cheap,kind='stable').tolist()


def test_q8_full_uses_exact_queries_and_matches_moving_margin_reference():
    x,comps,cheap=setup()
    calls=[]
    def predict(xs):
        calls.append(len(xs))
        return model(xs)
    baseline=model(x[None])[0]
    r=run_selective_audit(x,comps,cheap,predict,QueryGate('q16',budget=8),baseline)
    assert calls==[8]
    assert r.extra_model_evaluations == 8
    assert r.baseline_model_evaluations == 0
    sel=np.argpartition(-cheap,7)[:8].tolist()
    outputs=model(np.stack([neutralize_interp(x,comps[j]) for j in sel]))
    m0=np.diff(np.sort(baseline)[-2:])[0]
    m=np.sort(outputs,axis=1)[:,-1]-np.sort(outputs,axis=1)[:,-2]
    expected=[sel[i] for i in np.argsort(-abs(m0-m),kind='stable')]
    assert r.ranking == expected


def test_threshold_changes_model_call_count_without_label():
    x,comps,cheap=setup()
    base=model(x[None])[0]
    feature_margin=np.diff(np.sort(base)[-2:])[0]
    query=QueryGate('threshold','margin','ge',float(feature_margin)-.01,8)
    noquery=QueryGate('threshold','margin','le',float(feature_margin)-1,8)
    a=run_selective_audit(x,comps,cheap,model,query,base)
    b=run_selective_audit(x,comps,cheap,model,noquery,base)
    assert a.extra_model_evaluations==8
    assert b.extra_model_evaluations==0


def test_malformed_score_and_budget_rejected():
    x,comps,cheap=setup()
    with pytest.raises(ValueError):
        run_selective_audit(x,comps,cheap,model,QueryGate('q16',budget=16))
    with pytest.raises(ValueError):
        run_selective_audit(x,comps,np.zeros(11),model,QueryGate('q0'))
    with pytest.raises(ValueError):
        run_selective_audit(x,comps,cheap,model,QueryGate('q0'),np.array([np.nan,0]))
