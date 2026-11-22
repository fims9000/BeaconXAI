"""Official UCI HAR subject split tests; never download a dataset in unit tests."""
import numpy as np
import pytest
from scripts import preprocess_uci_har_for_q36 as prep


def mock_uci(monkeypatch, tmp_path, *, overlap=False, bad_shape=False):
    folder=tmp_path/"UCI HAR Dataset"
    for split in ("train","test"):
        (folder/split).mkdir(parents=True)
    train_sid=np.array([1,1,2,2],int)
    test_sid=np.array([2,3,4,4],int) if overlap else np.array([3,3,4,4],int)
    np.savetxt(folder/"train"/"subject_train.txt",train_sid,fmt="%d")
    np.savetxt(folder/"test"/"subject_test.txt",test_sid,fmt="%d")
    rng=np.random.default_rng(1)
    train=rng.normal(size=(4,128,9)).astype("float32")
    test=rng.normal(size=(4,128,9 if not bad_shape else 8)).astype("float32")
    monkeypatch.setattr(prep,"ensure_uci_har",lambda root: folder)
    monkeypatch.setattr(prep,"load_uci_har",lambda root:(
        train,np.array([0,1,0,1]),test,np.array([1,0,1,0])))
    return folder


def test_official_subjects_and_9_channels_preserved(monkeypatch,tmp_path):
    mock_uci(monkeypatch,tmp_path)
    output=tmp_path/"exports"/"subject_split.npz"
    meta=prep.prepare_dataset(tmp_path,output)
    assert meta["subject_leakage"] is False
    assert meta["train_subjects"]==[1,2]
    assert meta["test_subjects"]==[3,4]
    assert meta["channel_groups"]==["body_acc","body_gyro","total_acc"]
    with np.load(output) as z:
        assert z["x_train"].shape==(4,128,9)
        assert z["x_test"].shape==(4,128,9)
        assert set(z["subject_train"]).isdisjoint(z["subject_test"])


def test_reject_subject_leakage_before_writing(monkeypatch,tmp_path):
    mock_uci(monkeypatch,tmp_path,overlap=True)
    out=tmp_path/"reject.npz"
    with pytest.raises(ValueError,match="subject overlap"):
        prep.prepare_dataset(tmp_path,out)
    assert not out.exists()


def test_reject_wrong_channel_count(monkeypatch,tmp_path):
    mock_uci(monkeypatch,tmp_path,bad_shape=True)
    with pytest.raises(ValueError,match="Invalid official UCI HAR"):
        prep.prepare_dataset(tmp_path,tmp_path/"bad.npz")
