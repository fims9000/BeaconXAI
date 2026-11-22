"""A second UCI HAR split may only use the official TRAIN subject pool."""
import numpy as np
import pytest
from scripts.resplit_uci_har_train_subjects import resplit_training_subjects


def make_source(tmp_path, train_subjects=(1,1,3,3,5,5,7,7), test_subjects=(2,4)):
    source = tmp_path / "official.npz"
    n = len(train_subjects)
    np.savez_compressed(
        source,
        x_train=np.arange(n*128*9).reshape(n,128,9).astype("float32"),
        y_train=np.tile([0,1], n//2),
        subject_train=np.asarray(train_subjects),
        x_test=np.full((len(test_subjects),128,9),-99999.),
        y_test=np.full(len(test_subjects),5),
        subject_test=np.asarray(test_subjects),
    )
    return source


def test_replicate_heldout_only_from_official_training(tmp_path):
    source=make_source(tmp_path)
    dest=tmp_path/"inner.npz"
    result=resplit_training_subjects(source,dest,(3,5))
    assert result["original_official_test_used"] is False
    assert result["train_subjects"]==[1,7]
    assert result["heldout_subjects"]==[3,5]
    with np.load(dest) as data:
        assert len(data["x_train"])==4 and len(data["x_test"])==4
        assert set(data["subject_train"])=={1,7}
        assert set(data["subject_test"])=={3,5}
        assert not np.any(data["x_train"]==-99999)
        assert not np.any(data["x_test"]==999)
        assert "TRAIN" in str(data["dataset_protocol"])


def test_rejects_old_official_test_and_missing_subject(tmp_path):
    src=make_source(tmp_path)
    for ids in [(2,), (11,), (1,1), (1,3,5,7)]:
        with pytest.raises(ValueError):
            resplit_training_subjects(src,tmp_path/"bad.npz",ids)


def test_heldout_classes_cannot_disappear(tmp_path):
    src=make_source(tmp_path,(1,1,3,3,5,5,7,7))
    with np.load(src) as z:
        x=z["x_train"];ids=z["subject_train"];xte=z["x_test"];yte=z["y_test"];teid=z["subject_test"]
    y=np.asarray([0,1,0,0,0,1,0,1])
    np.savez_compressed(src,x_train=x,y_train=y,subject_train=ids,x_test=xte,y_test=yte,subject_test=teid)
    with pytest.raises(ValueError,match="class labels"):
        resplit_training_subjects(src,tmp_path/"bad.npz",(3,))
