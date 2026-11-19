"""Repeated-donor claim generator never sees offline reference outcomes."""
import numpy as np
import pytest
from beaconxai.contrastive import make_components
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.evidence_consensus import run_consensus


def example():
    rng=np.random.default_rng(123)
    train=rng.normal(0,.2,(100,96,9)).astype('float32')
    lab=np.array([0]*50+[1]*50)
    train[50:,16:64,:3]-=.5
    train[:50,16:64,:3]+=.5
    cs=make_components((96,9),12)
    bank=fit_real_donor_bank(train,lab,cs,max_donors=40,seed=4)
    def model(batch):
        z=batch[:,16:64,:3].mean(axis=(1,2))
        return np.stack([z,-z],axis=1)
    x=train[3]
    base=model(x[None])[0]
    return x,base,cs,bank,model


def test_exact_16_calls_per_method_plus_offline_reference():
    x,base,cs,bank,model=example()
    calls=[0]
    def counted(arr):
        calls[0]+=len(arr)
        return model(arr)
    rows,out=run_consensus(x,base,cs,list(range(16)),bank,0,1,counted,seed=3)
    assert calls[0]==16+16+128
    assert out['model_calls_single']==out['model_calls_consensus']==16
    assert out['offline_reference_model_calls']==128
    assert len(rows)==16
    assert sum(r['replicates_evaluated'] for r in rows)==16
    assert all(r['replicates_evaluated']==0 for r in rows[4:])
    assert all(0<=r['reference_positive_fraction']<=1 for r in rows)
    assert out['single']['claims']>=0 and out['consensus']['claims']>=0


def test_repeatable_and_rejects_overlap_or_invalid_donor_count():
    x,base,cs,bank,f=example()
    a=run_consensus(x,base,cs,list(range(16)),bank,0,1,f,seed=10)
    b=run_consensus(x,base,cs,list(range(16)),bank,0,1,f,seed=10)
    assert a==b
    with pytest.raises(ValueError): run_consensus(x,base,cs,[0]*16,bank,0,1,f)
    with pytest.raises(ValueError): run_consensus(x,base,cs,list(range(16)),bank,0,0,f)
    with pytest.raises(ValueError): run_consensus(x,base,cs,list(range(16)),bank,0,1,f,reference_count=100)
