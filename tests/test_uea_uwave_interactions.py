"""UWave archive/split, prespecified masks, paired Q16 and budget accounting."""
from __future__ import annotations

import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.learned_interactions import fit_interaction_prior
from scripts.preprocess_uea_uwave import prepare_uwave
from scripts.run_uea_uwave_interactions import (
    METHODS,BUDGETS,calibration_masks,paired_primary,
    single_result,stratified_official_train_split,summarize,
)


def test_uwave_exact_archive_and_stratified_train_calibration(tmp_path):
    rng=np.random.default_rng(4)
    xtr=rng.normal(size=(120,3,315))
    xte=rng.normal(size=(320,3,315))
    labels=np.array([str(i) for i in range(8)])
    ytr=labels[np.arange(120)%8]
    yte=labels[np.arange(320)%8]
    dest=tmp_path/"uwave.npz"
    record=prepare_uwave(xtr,ytr,xte,yte,dest)
    assert record["subject_disjoint"] is None
    with np.load(dest) as z:
        assert z["x_train"].shape==(120,315,3)
        assert z["x_test"].shape==(320,315,3)
        fit,cal=stratified_official_train_split(z["y_train"])
        assert (len(fit),len(cal))==(96,24)
        assert set(fit).isdisjoint(set(cal))
        assert sorted(fit.tolist()+cal.tolist())==list(range(120))
        for target in range(8):
            assert sum(z["y_train"][fit]==target)==12
            assert sum(z["y_train"][cal]==target)==3


def test_uwave_wrong_geometry_and_class_counts_rejected(tmp_path):
    x=np.zeros((120,3,315))
    t=np.zeros((320,3,315))
    train_y=np.arange(120)%8
    test_y=np.arange(320)%8
    with pytest.raises(ValueError,match="120/320"):
        prepare_uwave(x[:-1],train_y[:-1],t,test_y,tmp_path/"no.npz")
    with pytest.raises(ValueError,match="15 per class"):
        incorrect=np.arange(120)%8
        incorrect[0]=1  # eight class labels remain, but counts become 14/16
        prepare_uwave(x,incorrect,t,test_y,tmp_path/"no.npz")


def test_calibration_mask_distribution_is_unchanged_and_deterministic():
    a=calibration_masks(np.random.default_rng(571))
    b=calibration_masks(np.random.default_rng(571))
    assert a==b and len(a)==24 and len(set(a))==24
    assert [len(mask) for mask in a].count(1)==10
    assert [len(mask) for mask in a].count(2)==9
    assert [len(mask) for mask in a].count(3)==5


def test_policy_wrapper_does_not_make_hidden_blackbox_calls():
    x=np.zeros((315,3))
    donors=(np.ones_like(x),np.ones_like(x)*.85)
    groups=make_components((315,3),10,"all_channels")
    rows=[(np.ones(10),(j,),.5) for j in range(10) for _ in range(4)]
    priors=(
        fit_interaction_prior(rows,n_components=10,pairwise=True),
        fit_interaction_prior(rows,n_components=10,pairwise=False),
    )
    fitted={0:(np.zeros_like(x),np.ones_like(x)),
            1:(np.ones_like(x),np.ones_like(x))}
    counter={"examples":0}
    def model(batch):
        counter["examples"]+=len(batch)
        signed=batch.mean(axis=(1,2))-.25
        return np.column_stack((np.zeros(len(batch)),signed))
    baseline=np.array([0.,-.25])
    for Q in BUDGETS:
        for method in METHODS:
            previous=counter["examples"]
            res=single_result(
                method,Q,x,baseline,donors,groups,
                fitted,priors,model,counter
            )
            assert res["model_queries"]==counter["examples"]-previous<=Q
            if method in ("pairs_adaptive","additive_adaptive"):
                assert res["posterior_updates"]==res["model_queries"]
            else:
                assert res["posterior_updates"]==0
            if res["strict_joint"]:
                assert res["fraction_d1"]<=.35
                assert res["fraction_d2"]<=.35


def test_primary_paired_is_exact_and_keeps_failed_samples():
    rows=[]
    for i in range(320):
        for q in BUDGETS:
            for method in METHODS:
                ok=int(
                    (i<20) if method=="pairs_adaptive"
                    else (i<15) if method=="static_sparse"
                    else False
                )
                rows.append({
                    "sample":i,"Q":q,"method":method,
                    "strict_joint":ok,"model_queries":8,
                    "both_le20":ok,"minimal_on_both":ok,
                    "search_wall_seconds":.01,"posterior_updates":0,
                    "fraction_d1":.2 if ok else None,
                    "fraction_d2":.2 if ok else None,
                    "audit_extra_two_valid":None,
                })
    result=paired_primary(rows)
    assert result["n_test"]==320
    assert result["adaptive_joint"]==20
    assert result["static_joint"]==15
    assert result["both"]==15
    assert result["adaptive_only"]==5
    assert result["static_only"]==0
    assert result["exact_mcnemar_p"]==.0625
    assert len(summarize(rows))==8
