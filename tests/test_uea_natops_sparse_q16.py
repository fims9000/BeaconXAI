"""Finite-donor sparse Q16, train-only ranking and untouched NATOPS split."""
import numpy as np
import pytest

from beaconxai.contrastive import make_components
from beaconxai.donor_robust_cf import (
    search_sparse_robust_counterfactual,train_profile_donor_gains,
)
from scripts.preprocess_uea_natops import prepare_natops
from scripts.run_uea_natops_sparse_q16 import measured_search,primary_paired,summarize,METHODS


def test_natops_archive_preserves_official_train_test_and_class_labels(tmp_path):
    rng=np.random.default_rng(41)
    a=rng.normal(size=(180,24,51))
    b=rng.normal(size=(180,24,51))
    names=np.array(list("abcdef"))
    tr=names[np.arange(180)%6]
    te=names[np.arange(180)%6]
    p=tmp_path/"natops.npz"
    summary=prepare_natops(a,tr,b,te,p)
    assert summary["participant_ids_verified"] is False
    with np.load(p) as z:
        assert z["x_train"].shape==(180,51,24)
        assert z["x_test"].shape==(180,51,24)
        assert len(set(z["y_train"]))==6
        np.testing.assert_allclose(z["x_train"][0],a[0].T,rtol=1e-6,atol=1e-6)


def test_invalid_natops_geometry_and_label_split_are_rejected(tmp_path):
    a=np.zeros((180,24,51))
    names=np.array(list("abcdef"))
    labels=names[np.arange(180)%6]
    with pytest.raises(ValueError,match="finite"):
        prepare_natops(a[:179],labels[:179],a,labels,tmp_path/"bad.npz")
    with pytest.raises(ValueError,match="six archive classes"):
        prepare_natops(a,labels,a,["a"]*180,tmp_path/"bad.npz")


def test_train_only_donor_gain_calculation_finite_and_targeted():
    x=np.zeros((51,24))
    donors=[np.ones_like(x),np.full_like(x,.9)]
    groups=make_components((51,24),10,"all_channels")
    profiles={0:(np.zeros_like(x),np.ones_like(x)),
              1:(np.ones_like(x),np.ones_like(x))}
    g=train_profile_donor_gains(x,donors,groups,profiles,0,1,taper=2)
    assert g.shape==(2,10)
    assert np.isfinite(g).all()
    assert (g>0).all()


@pytest.mark.parametrize("method",METHODS)
def test_each_method_counts_exact_examples_with_shared_q(method):
    x=np.zeros((51,24))
    donors=[np.ones_like(x),np.full_like(x,.9)]
    groups=make_components((51,24),10,"all_channels")
    counts={"examples":0}
    def classifier(batch):
        counts["examples"]+=len(batch)
        return np.column_stack((np.zeros(len(batch)),batch.mean(axis=(1,2))-.12))
    prior=np.tile(np.arange(10,0,-1,dtype=float),(2,1))
    result=measured_search(
        x,np.array([0.,-.12]),donors,groups,list(range(10)),
        classifier,counts,method,prior,
    )
    assert result["model_queries"]==counts["examples"]
    assert 0<result["model_queries"]<=16
    if result["certified"]:
        assert result["selected"]
        assert result["d1_flip"]==1 and result["d2_flip"]==1
    else:
        assert result["selected"] is None


def test_sparse_returns_only_compact_mask_verified_for_both_donors():
    x=np.zeros((51,24))
    donors=[np.ones_like(x),np.full_like(x,.9)]
    groups=make_components((51,24),10,"all_channels")
    calls=[0]
    def model(batch):
        calls[0]+=len(batch)
        return np.column_stack((np.zeros(len(batch)),batch.mean(axis=(1,2))-.11))
    gain=np.tile(np.array([10,9,8,1,1,1,1,1,1,1]),(2,1))
    result=search_sparse_robust_counterfactual(
        x,np.array([0.,-.11]),donors,groups,gain,model,
        target_class=1,budget=16,maximum_modified_fraction=.35,taper=2,
    )
    assert result.queries==calls[0]<=16
    if result.is_certified:
        for d in donors:
            from beaconxai.verified_counterfactual import splice
            edited=splice(x,d,groups,result.certified_edit,taper=2)
            assert np.mean(np.abs(edited-x)>1e-8)<=.35
            assert np.mean(edited)>.11
        for i in range(2):
            assert any(p.components==result.certified_edit
                       and p.donor_index==i and p.targeted_flip
                       for p in result.probes)


def test_sparse_never_certifies_non_target_or_oversize_mask():
    x=np.zeros((51,24))
    groups=make_components((51,24),10,"all_channels")
    donors=[np.ones_like(x),np.full_like(x,.7)]
    calls=[0]
    def always_source(batch):
        calls[0]+=len(batch)
        return np.column_stack((np.ones(len(batch)),np.zeros(len(batch))))
    result=search_sparse_robust_counterfactual(
        x,np.array([1.,0.]),donors,groups,np.ones((2,10)),
        always_source,target_class=1,budget=8,maximum_modified_fraction=.2,
    )
    assert not result.is_certified and result.certified_edit is None
    assert result.queries==calls[0]<=8


def test_sparse_rejects_invalid_donor_prior_tensor():
    x=np.zeros((51,24))
    groups=make_components((51,24),10,"all_channels")
    with pytest.raises(ValueError,match="TRAIN prior"):
        search_sparse_robust_counterfactual(
            x,np.array([1.,0.]),[x.copy(),x.copy()],groups,
            np.zeros((1,10)),lambda z:z,target_class=1,budget=16,
        )


def test_primary_paired_comparison_keeps_full_denominator():
    records=[]
    for i in range(180):
        for method in METHODS:
            j=int(i<10) if method=="sparse_maximin" else int(i<5) if method=="forward_robust" else 0
            records.append({
                "sample":i,"method":method,"strict_joint":j,"certified":j,
                "model_queries":3,"search_wall_seconds":0.,
                "both_within_20pct":j,"both_within_35pct":j,
                "both_within_50pct":j,"minimal_on_both":0,
                "fraction_d1":.1 if j else None,"fraction_d2":.1 if j else None,
            })
    p=primary_paired(records)
    assert p["N"]==180 and p["new_joint"]==10 and p["forward_joint"]==5
    assert p["new_only"]==5 and p["forward_only"]==0
    assert p["exact_mcnemar_p"]==.0625
    assert len(summarize(records))==3
