"""Pure and synthetic controls for the frozen UEA Epilepsy joint-Q16 study."""
import numpy as np
import pytest

from scripts.preprocess_uea_epilepsy import prepare_epilepsy
from scripts.run_uea_epilepsy_robust_q16 import (
    measured_search,primary_paired,summarize, METHODS,
)
from beaconxai.contrastive import make_components


def test_epilepsy_archive_shapes_classes_and_no_test_label_leak(tmp_path):
    rng=np.random.default_rng(12)
    a=rng.normal(size=(137,3,206))
    b=rng.normal(size=(138,3,206))
    labels=np.array(["epilepsy","running","sawing","walking"])
    train=labels[np.arange(137)%4]
    test=labels[np.arange(138)%4]
    path=tmp_path/"epilepsy.npz"
    report=prepare_epilepsy(a,train,b,test,path)
    assert report["diagnostic_or_medical_labels"] is False
    with np.load(path) as z:
        assert z["x_train"].shape==(137,206,3)
        assert z["x_test"].shape==(138,206,3)
        np.testing.assert_allclose(z["x_train"][0],a[0].T,rtol=1e-6,atol=1e-6)
        assert len(np.unique(z["y_train"]))==4
        assert len(np.unique(z["y_test"]))==4


def test_preprocessing_rejects_invalid_split_and_labels(tmp_path):
    a=np.zeros((137,3,206))
    b=np.zeros((138,3,206))
    labels=np.array(["epilepsy","running","sawing","walking"])
    train=labels[np.arange(137)%4]
    test=labels[np.arange(138)%4]
    with pytest.raises(ValueError,match="finite"):
        prepare_epilepsy(a[:-1],train[:-1],b,test,tmp_path/"bad.npz")
    with pytest.raises(ValueError,match="four"):
        prepare_epilepsy(a,train,b,["running"]*138,tmp_path/"bad.npz")


@pytest.mark.parametrize("method",METHODS)
def test_joint_budget_two_donors_and_no_partial_certificate(method):
    comps=make_components((206,3),10,"all_channels")
    x=np.zeros((206,3))
    d1=np.ones_like(x)
    d2=np.full_like(x,.9)
    counter={"examples":0}
    def predictor(batch):
        counter["examples"]+=len(batch)
        val=batch.mean(axis=(1,2))
        return np.column_stack((np.zeros(len(batch)),val-.3))
    base=np.array([0.,-.3])
    result=measured_search(
        x,base,[d1,d2],comps,list(range(10)),
        predictor,counter,method,
    )
    assert result["model_queries"]==counter["examples"]
    assert 0<result["model_queries"]<=16
    if result["certified"]:
        assert result["selected"] is not None
        assert result["d1_flip"] and result["d2_flip"]
        assert result["fraction_d1"]>=0
        assert result["fraction_d2"]>=0
    else:
        assert result["selected"] is None
        assert result["minimal_on_both"]==0


def test_posthoc_second_donor_failure_is_not_certificate():
    comps=make_components((206,3),10,"all_channels")
    x=np.zeros((206,3))
    counter={"examples":0}
    def predictor(batch):
        counter["examples"]+=len(batch)
        val=batch.mean(axis=(1,2))
        return np.column_stack((np.zeros(len(batch)),val-.3))
    result=measured_search(
        x,np.array([0.,-.3]),
        [np.ones_like(x),np.full_like(x,.01)],
        comps,list(range(10)),predictor,counter,
        "posthoc_d1_plus_d2",
    )
    assert result["model_queries"]==counter["examples"]
    assert result["certified"]==0
    assert result["selected"] is None


def test_primary_paired_keeps_all_138_and_exact_test():
    records=[]
    for i in range(138):
        for method in METHODS:
            joint=(int(i<5) if method=="posthoc_d1_plus_d2"
                   else int(i<10) if method=="backward_robust_chunks"
                   else 0)
            records.append({
                "sample":i,"method":method,
                "strict_joint":joint,"certified":joint,
                "model_queries":10,
                "both_within_20pct":joint,
                "both_within_35pct":joint,
                "both_within_50pct":joint,
                "minimal_on_both":0,
                "fraction_d1":.1 if joint else None,
                "fraction_d2":.1 if joint else None,
            })
    result=primary_paired(records)
    assert result["new_joint"]==10
    assert result["posthoc_joint"]==5
    assert result["both"]==5
    assert result["new_only"]==5
    assert result["posthoc_only"]==0
    assert result["exact_mcnemar_p"]==.0625
    assert len(summarize(records))==3
