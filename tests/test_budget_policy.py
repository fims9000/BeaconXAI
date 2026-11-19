"""Prospective multi-target allocation: one-budget invariants and no label leakage."""
import numpy as np
import pytest
from beaconxai.contrastive import make_components
from beaconxai.real_donor import fit_real_donor_bank
from beaconxai.budget_policy import run_budget_policy,POLICIES,preplan


def setup_case():
    rng=np.random.default_rng(8)
    x=rng.normal(size=(45,96,9)).astype('float32')*.1
    y=np.repeat([0,1,2],15)
    x[y==0,16:64,:3]+=1
    x[y==1,16:64,:3]-=1
    x[y==2,16:64,:3]+=.2
    cs=make_components((96,9),12)
    bank=fit_real_donor_bank(x,y,cs,max_donors=6,seed=5)
    def predict(sample):
        k=np.mean(sample[16:64,:3])
        return np.array([k,-k,.3*k])
    return x[3],cs,bank,predict


def test_exact_global_query_and_target_selection():
    x,cs,bank,f=setup_case()
    base=f(x)
    for policy in POLICIES:
        calls=[0]
        def model(a):
            calls[0]+=1
            return f(a)
        rows,res=run_budget_policy(x,base,bank,cs,list(range(16)),model,q=16,policy=policy)
        assert 1<=res['queries']<=16
        assert calls[0]==len(rows)==res['queries']
        assert set(r['target'] for r in rows).issubset({res['runnerup'],res['third']})
        assert all(r['query']==i for i,r in enumerate(rows,1))
        assert res['status'] in ('target_flip','directed_only','abstain')
        if policy!='adaptive_stop':
            assert res['queries']==16
        if policy=='equal':
            assert sum(r['target']==res['runnerup'] for r in rows)==8
            assert sum(r['target']==res['third'] for r in rows)==8


def test_no_test_label_and_deterministic_policy():
    x,cs,bank,f=setup_case()
    base=f(x)
    a=run_budget_policy(x,base,bank,cs,list(range(16)),f,q=16,policy='adaptive')
    b=run_budget_policy(x,base,bank,cs,list(range(16)),f,q=16,policy='adaptive')
    assert a[1]==b[1]
    assert [(r['target'],r['component']) for r in a[0]]==[(r['target'],r['component']) for r in b[0]]


def test_reject_wrong_budget_and_policy():
    x,cs,bank,f=setup_case()
    base=f(x)
    for kwargs in (dict(q=15),dict(q=40),dict(policy='wrong'),dict(selected=[0]*16)):
        default=dict(q=16,policy='equal',selected=list(range(16)))
        default.update(kwargs)
        with pytest.raises(ValueError):
            run_budget_policy(x,base,bank,cs,default.pop('selected'),f,**default)
    assert len(preplan((1,2),16,'equal'))==16
