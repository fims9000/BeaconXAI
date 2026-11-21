"""Independent corpus ingestion and real black-box segment search."""
import numpy as np
import pytest

from scripts.preprocess_uea_basicmotions import prepare_basicmotions
from beaconxai.cf_donor_line import search_donor_segment
from beaconxai.contrastive import make_components


def test_basicmotions_preserves_official_split_and_labels(tmp_path):
    rng=np.random.default_rng(3)
    train=rng.normal(size=(40,6,100))
    test=rng.normal(size=(40,6,100))
    labels=np.array(["standing","running","badminton","walking"]*10)
    dest=tmp_path/"basicmotions.npz"
    report=prepare_basicmotions(train,labels,test,labels,dest)
    assert report["subject_disjoint_split_verified"] is False
    assert report["n_test"]==40
    with np.load(dest) as a:
        assert a["x_train"].shape==(40,100,6)
        assert a["x_test"].shape==(40,100,6)
        assert not np.array_equal(a["x_train"],a["x_test"])
        assert len(np.unique(a["y_train"]))==4
        assert "participant ids unavailable" in str(a["dataset_protocol"])


def test_basicmotions_rejects_wrong_geometry_and_label_split(tmp_path):
    x=np.zeros((40,6,100))
    y=np.array(["a","b","c","d"]*10)
    with pytest.raises(ValueError,match="BasicMotions"):
        prepare_basicmotions(np.zeros((40,6,99)),y,x,y,tmp_path/"bad.npz")
    with pytest.raises(ValueError,match="labels"):
        prepare_basicmotions(x,y,x,np.array(["a"]*40),tmp_path/"bad.npz")


def test_line_search_measures_each_alpha_exactly():
    x=np.zeros((10,3))
    donor=np.ones((10,3))
    count=[0]
    def model(batch):
        count[0]+=len(batch)
        return np.column_stack((
            np.zeros(len(batch)),
            batch.mean(axis=(1,2))-.35,
        ))
    original=model(x[None])[0]
    count[0]=0
    result=search_donor_segment(
        x,donor,original,model,target_class=1,budget=8
    )
    assert result.is_verified
    assert result.queries==count[0]==8
    assert .35<result.alpha<.36
    assert int(np.argmax(model(result.edited[None])[0]))==1


def test_failed_donor_abstains_after_one_query():
    x=np.zeros((10,2))
    d=np.ones_like(x)
    def bad(batch):
        return np.column_stack((np.ones(len(batch)),np.zeros(len(batch))))
    a=search_donor_segment(x,d,np.array([1.,0.]),bad,
                            target_class=1,budget=16)
    assert not a.is_verified and a.queries==1
    assert a.edited is None and a.tested_alphas==(1.,)


def test_invalid_target_and_model_outputs_rejected():
    x=np.zeros((10,2))
    d=np.ones_like(x)
    with pytest.raises(ValueError,match="target"):
        search_donor_segment(x,d,np.array([1.,0.]),lambda a:a,
                             target_class=0,budget=8)
    with pytest.raises(ValueError,match="model"):
        search_donor_segment(x,d,np.array([1.,0.]),
                             lambda a:np.full((1,2),np.nan),
                             target_class=1,budget=8)



def test_timeblock_components_are_coherent_without_sensor_order_assumption():
    blocks=make_components((100,6),10,"all_channels")
    assert len(blocks)==10
    assert {(c.c0,c.c1) for c in blocks}=={(0,6)}
    assert all(c.t1-c.t0==10 for c in blocks)
    assert [(c.t0,c.t1) for c in blocks]==[
        (i*10,(i+1)*10) for i in range(10)
    ]
