"""Small end-to-end clean-signal experiment; no downloads in unit tests."""
from types import SimpleNamespace
import csv
import json

import numpy as np
import pytest

from scripts import run_clean_calibrated_staged_budget as study


def _tiny_npz(tmp_path,subject_overlap=False):
    rng=np.random.default_rng(31)
    xtrain=rng.normal(size=(8*8,24,9)).astype("float32")
    ytrain=np.tile(np.array([0,1,0,1,0,1,0,1]),8)
    st=np.repeat(np.arange(1,9),8)
    xtest=rng.normal(size=(2*5,24,9)).astype("float32")
    ytest=np.tile(np.array([0,1,0,1,0]),2)
    sv=np.repeat([5 if subject_overlap else 10,11],5)
    path=tmp_path/"toy.npz"
    np.savez_compressed(path,x_train=xtrain,y_train=ytrain,
                        x_test=xtest,y_test=ytest,
                        subject_train=st,subject_test=sv)
    return path


def _fake_classifier(monkeypatch):
    class Fake:
        classes_=np.array([0,1],dtype=int)
        def predict_proba(self,features):
            z=np.asarray(features,dtype=float)
            effects=np.tanh(z[:,0]*.3)
            p=np.clip(.5+.3*effects,.08,.92)
            return np.column_stack((p,1-p))
    class Adapter:
        model=Fake()
    monkeypatch.setattr(study,"_train_extratrees_local",lambda *args:Adapter())
    monkeypatch.setattr(study,"_anfis_features",
                        lambda x:x.reshape(len(x),-1))


def test_train_calibration_and_genuine_outer_stopping(monkeypatch,tmp_path):
    _fake_classifier(monkeypatch)
    path=_tiny_npz(tmp_path)
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"out",model="extratrees",
                         max_calib=8,max_test=8,trees=8,penalty=.75,seed=57)
    study.run(args)
    with (args.out_dir/"summary.csv").open() as fp:
        summary=list(csv.DictReader(fp))
    assert {r["method"] for r in summary}==set(study.METHODS)
    assert len(summary)==7
    per_sample=list(csv.DictReader((args.out_dir/"per_sample.csv").open()))
    assert len(per_sample)==56
    assert all(int(r["extra_model_evaluations"]) in (0,8,16,32)
               for r in per_sample)
    manifest=json.loads((args.out_dir/"manifest.json").read_text())
    policy=json.loads((args.out_dir/"inner_calibration.json").read_text())
    assert manifest["n_outer_windows"]==8
    assert manifest["outer_test_participants_never_used_to_fit_policy"]
    assert manifest["outer_test_labels_never_used_to_select_or_stop"]
    assert not set(policy["fit"]["inner_fit_subjects"]) & set(
        policy["fit"]["inner_validation_subjects"])
    assert not set(policy["fit"]["calibration_subjects"]) & {10,11}
    assert manifest["outer_total_measured_model_examples_including_controls"]==(
        8*(1+36) +
        sum(int(r["extra_model_evaluations"]) for r in per_sample))
    assert manifest["constrained_upgrade_queries_actual"]>=16*8
    assert manifest["random_budget_queries_actual"]>=16*8
    assert all(int(r["extra_model_evaluations"])>=16 for r in per_sample
               if r["method"]=="constrained_upgrade16")
    before=(args.out_dir/"per_sample.csv").read_bytes()
    with np.load(path) as fp:
        data={key:fp[key].copy() for key in fp.files}
    data["y_test"][:]=999
    np.savez_compressed(path,**data)
    args2=SimpleNamespace(**{**vars(args),"out_dir":tmp_path/"out2"})
    study.run(args2)
    assert (args2.out_dir/"per_sample.csv").read_bytes()==before


def test_subject_leakage_is_hard_failure(tmp_path):
    path=_tiny_npz(tmp_path,subject_overlap=True)
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"out",model="extratrees",
                         max_calib=8,max_test=8,trees=8,penalty=.75,seed=57)
    with pytest.raises(ValueError,match="participant train/test leakage"):
        study.run(args)
