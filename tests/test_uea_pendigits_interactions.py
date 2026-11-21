"""PenDigits raw-MLP protocol: fixed TRAIN/CALIB and Q8/Q16 shared budget."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.learned_interactions import fit_interaction_prior
from scripts.preprocess_uea_pendigits import prepare_pendigits
from scripts.run_uea_pendigits_interactions import (
    METHODS,BUDGETS,calibration_masks,paired_primary,
    stratified_official_train_split,single_result,
)


def test_pendigits_geometry_and_stratified_train_calibration(tmp_path):
    rng=np.random.default_rng(21)
    train=rng.normal(size=(7494,2,8))
    test=rng.normal(size=(3498,2,8))
    labels=np.array([str(i) for i in range(10)])
    ytr=labels[np.arange(len(train))%10]
    yte=labels[np.arange(len(test))%10]
    p=tmp_path/"pen.npz"
    report=prepare_pendigits(train,ytr,test,yte,p)
    assert report["subjects_disjoint_verified"] is False
    with np.load(p) as z:
        assert z["x_train"].shape==(7494,8,2)
        assert z["x_test"].shape==(3498,8,2)
        fit,cal=stratified_official_train_split(z["y_train"])
        assert len(fit)==7394 and len(cal)==100
        assert set(fit).isdisjoint(set(cal))
        for k in range(10):
            assert sum(z["y_train"][cal]==k)==10
            assert sum(z["y_train"][fit]==k)>10


def test_pendigits_shape_and_class_checks(tmp_path):
    x=np.zeros((7494,2,8))
    te=np.zeros((3498,2,8))
    labels=np.arange(len(x))%10
    tl=np.arange(len(te))%10
    with pytest.raises(ValueError,match="7494"):
        prepare_pendigits(x[:-1],labels[:-1],te,tl,tmp_path/"bad.npz")
    with pytest.raises(ValueError,match="10 matching"):
        prepare_pendigits(x,[0]*7494,te,tl,tmp_path/"bad.npz")


def test_pendigits_fixed_28_train_masks():
    a=calibration_masks(np.random.default_rng(571))
    b=calibration_masks(np.random.default_rng(571))
    assert a==b and len(a)==28 and len(set(a))==28
    assert sum(len(v)==1 for v in a)==8
    assert sum(len(v)==2 for v in a)==20


def test_pendigits_each_policy_counts_real_model_inputs():
    x=np.zeros((8,2))
    donors=[np.ones_like(x),np.full_like(x,.8)]
    comps=make_components((8,2),8,"all_channels")
    profiles={0:(np.zeros_like(x),np.ones_like(x)),
              1:(np.ones_like(x),np.ones_like(x))}
    rows=[(np.ones(8),(i,),0.) for i in range(8) for _ in range(4)]
    pair=fit_interaction_prior(rows,n_components=8,pairwise=True)
    additive=fit_interaction_prior(rows,n_components=8,pairwise=False)
    c={"examples":0}
    def model(batch):
        c["examples"]+=len(batch)
        p=batch.mean(axis=(1,2))
        return np.column_stack((np.zeros(len(batch)),p-.12))
    for q in BUDGETS:
        for method in METHODS:
            prev=c["examples"]
            res=single_result(method,q,x,np.array([0.,-.12]),
                              donors,comps,profiles,(pair,additive),model,c)
            assert res["model_queries"]==c["examples"]-prev<=q
            if res["strict_joint"]:
                assert res["fraction_d1"]<=.35
                assert res["fraction_d2"]<=.35


def test_pendigits_primary_is_additive_vs_static_not_pair():
    rows=[]
    for i in range(320):
        for q in BUDGETS:
            for method in METHODS:
                success=int(i<20) if method=="additive_adaptive" else (
                    int(i<15) if method=="static_sparse" else 0)
                rows.append({
                    "sample":i,"method":method,"Q":q,
                    "strict_joint":success,
                })
    result=paired_primary(rows)
    assert result["n_test"]==320
    assert result["adaptive_joint"]==20 and result["static_joint"]==15
    assert result["adaptive_only"]==5 and result["static_only"]==0
    assert result["exact_mcnemar_p"]==.0625
