"""Regression tests for the preregistered untouched RacketSports protocol."""
import numpy as np
import pytest

from scripts.preprocess_uea_racketsports import prepare_racketsports
from scripts.run_uea_racketsports_holdout import measure_one, paired_primary
from beaconxai.cf_donor_robustness import (
    rank_confirmed_donors, joint_quality, component_jaccard,
    exact_two_sided_mcnemar,
)
from beaconxai.contrastive import make_components


def test_official_split_shapes_and_train_only_label_mapping(tmp_path):
    rng=np.random.default_rng(11)
    tr=rng.normal(size=(151,6,30))
    te=rng.normal(size=(152,6,30))
    labs=np.array(["forehand","backhand","clear","smash"])
    ytr=labs[np.arange(151)%4]
    yte=labs[np.arange(152)%4]
    path=tmp_path/"rs.npz"
    report=prepare_racketsports(tr,ytr,te,yte,path)
    assert report["subject_disjoint_claim"] is False
    assert len(report["npz_sha256"])==64
    with np.load(path) as file:
        assert file["x_train"].shape==(151,30,6)
        assert file["x_test"].shape==(152,30,6)
        assert file["y_train"].min()==0 and file["y_train"].max()==3
        assert "no verified participant IDs" in str(file["dataset_protocol"])
        np.testing.assert_array_equal(file["x_train"][0],tr[0].T)


def test_preprocessor_rejects_mismatched_labels_and_geometry(tmp_path):
    xtr=np.zeros((151,6,30))
    xte=np.zeros((152,6,30))
    labels=["a","b","c","d"]
    train=np.array([labels[j%4] for j in range(151)])
    test=np.array([labels[j%4] for j in range(152)])
    with pytest.raises(ValueError,match="official splits"):
        prepare_racketsports(xtr,train,xte,["a"]*152,tmp_path/"bad.npz")
    with pytest.raises(ValueError,match="RacketSports"):
        prepare_racketsports(xtr[:150],train[:150],xte,test,tmp_path/"bad.npz")


def test_two_distinct_confirmed_train_donors_no_label_leak():
    x=np.zeros((30,6))
    bank=np.stack([np.full_like(x,.1),
                   np.full_like(x,.2),
                   np.full_like(x,.2),
                   np.full_like(x,9.)])
    # target=1, predictions for ids 2/7/9, but not closest id 3
    rank=rank_confirmed_donors(
        x,bank,np.array([0,1,1,1]),1,np.array([3,9,7,2])
    )
    assert rank==(2,1,3)
    assert len(set(rank[:2]))==2
    assert rank_confirmed_donors(
        x,bank,np.array([0,0,1,0]),1,np.array([3,9,7,2])
    )==(2,)


def test_exact_paired_test_and_joint_outcome():
    assert exact_two_sided_mcnemar(5,0)==.0625
    assert exact_two_sided_mcnemar(0,0)==1.
    assert exact_two_sided_mcnemar(3,3)==1.
    assert joint_quality(True,.35,True)
    assert not joint_quality(True,.35,False)
    assert not joint_quality(False,.01,True)
    assert not joint_quality(True,.35001,True)
    assert component_jaccard((1,2),(2,3))==1/3
    with pytest.raises(ValueError):
        exact_two_sided_mcnemar(-1,2)


def test_primary_paired_counts_include_unavailable_donors():
    records=[]
    for i in range(152):
        for method in ("forward","backward"):
            for donor in ("D1","D2"):
                joint=int(
                    donor=="D1" and
                    ((method=="backward" and i<10)
                     or (method=="forward" and i<5))
                )
                records.append({
                    "sample":i,"method":method,"donor":donor,
                    "joint_primary":joint
                })
    p=paired_primary(records)
    assert (p["both_joint_success"],p["backward_only_joint_success"],
            p["forward_only_joint_success"],p["neither_joint_success"]) == (5,5,0,142)
    assert p["mcnemar_exact_twosided_p"]==.0625


@pytest.mark.parametrize("method",["forward","backward"])
def test_search_runs_with_real_Q16_and_explicit_verification(method):
    comps=make_components((30,6),10,"all_channels")
    x=np.zeros((30,6))
    d=np.ones((30,6))
    calls={"examples":0}
    def black_box(batch):
        calls["examples"]+=len(batch)
        # Black box requires at least 40% positive signal for class B.
        z=np.mean(batch,axis=(1,2))
        return np.column_stack([np.zeros(len(z)),z-.4])
    measure_one.model_counter=calls
    result=measure_one(
        x,np.array([0.,-.4]),d,comps,list(range(10)),
        black_box,method
    )
    assert 1<=result["model_queries"]<=16
    assert result["model_queries"]==calls["examples"]
    assert result["verified"]
    assert result["selected"] is not None
    assert result["modified_fraction"] is not None


def test_measure_one_does_not_claim_unchecked_target():
    comps=make_components((30,6),10,"all_channels")
    x=np.zeros((30,6))
    d=np.ones((30,6))
    calls={"examples":0}
    measure_one.model_counter=calls
    def always_source(batch):
        calls["examples"]+=len(batch)
        return np.repeat(np.array([[0.,-1.]]),len(batch),axis=0)
    result=measure_one(
        x,np.array([0.,-1.]),d,comps,list(range(10)),
        always_source,"backward"
    )
    assert result["model_queries"]==1
    assert result["verified"]==result["joint_primary"]==0
