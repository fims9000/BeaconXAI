"""End-to-end mathematical signed counter-evidence training/test separation."""
from types import SimpleNamespace
import csv
import json

import numpy as np
import pytest

from scripts import run_cheap_directed_counterevidence as directed
from scripts import run_clean_calibrated_staged_budget as preexisting


def sample_file(path,leak=False):
    rng=np.random.default_rng(14)
    xtrain=rng.normal(size=(21*8,24,9)).astype(np.float32)
    ytrain=np.tile(np.array([0,1,0,1,1,0,1,0]),21)
    st=np.repeat(np.arange(1,22),8)
    xtest=rng.normal(size=(9*6,24,9)).astype(np.float32)
    ytest=np.tile([1,0,1,0,1,0],9)
    sv=np.repeat(([1]+list(range(101,109))) if leak else
                 list(range(101,110)),6)
    np.savez_compressed(path,x_train=xtrain,y_train=ytrain,
                        subject_train=st,x_test=xtest,y_test=ytest,
                        subject_test=sv)
    return path


def monkey_model(monkeypatch):
    class Fake:
        classes_=np.array([0,1])
        coefs_=[np.arange(17,dtype=float)]
        intercepts_=[np.array([0.,0.])]
        def predict_proba(self,features):
            x=np.asarray(features)
            weight=np.cos(np.arange(x.shape[1])*.13)
            a=np.tanh(x@weight*.02)
            p=np.clip(.50+.35*a,.05,.95)
            return np.column_stack((p,1-p))
    class Wrapper:
        model=Fake()
    monkeypatch.setattr(preexisting,"_train_extratrees_local",
                        lambda *args:Wrapper())
    monkeypatch.setattr(preexisting,"_anfis_features",
                        lambda x:np.asarray(x).reshape(len(x),-1))


def make_args(path,out):
    return SimpleNamespace(
        npz_path=path,out_dir=out,model="extratrees",seed=57,
        trees=8,windows_per_subject=2,q=16,
        ridge_alpha=100.,pairwise_c=.1)


def test_clean_signed_counterevidence_benchmark_and_ytest_invariance(tmp_path,monkeypatch):
    monkey_model(monkeypatch)
    path=sample_file(tmp_path/"sample.npz")
    args=make_args(path,tmp_path/"first")
    manifest,summary=directed.run(args)
    assert len(summary)==7
    assert manifest["source_class_fixed_before_interventions"]
    assert manifest["not_moving_top2"]
    assert manifest["test_labels_never_read"]
    assert manifest["test_windows"]==18
    rows=list(csv.DictReader((args.out_dir/"per_sample.csv").open()))
    assert len(rows)==18*7
    assert all(int(r["extra_model_evaluations"]) in (0,16) for r in rows)
    assert manifest["test_actual_budgeted_model_calls"]==18*4*16
    assert manifest["model_examples_counted"]>manifest["test_actual_budgeted_model_calls"]
    before=(args.out_dir/"per_sample.csv").read_bytes()
    with np.load(path) as fp:
        data={k:fp[k].copy() for k in fp.files}
    data["y_test"][:]=999
    np.savez_compressed(path,**data)
    other=make_args(path,tmp_path/"again")
    directed.run(other)
    assert before==(other.out_dir/"per_sample.csv").read_bytes()


def test_signed_study_rejects_subject_leak(tmp_path):
    path=sample_file(tmp_path/"bad.npz",leak=True)
    with pytest.raises(ValueError,match="leakage"):
        directed.run(make_args(path,tmp_path/"out"))
