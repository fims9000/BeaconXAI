"""Synthetic end-to-end smoke for full-scan model-sensitivity reference."""
from types import SimpleNamespace
import csv
import json
import numpy as np
import pytest
from scripts import run_full_sensitivity_reference as study


def fake_npz(tmp_path, *, overlap=False):
    rng=np.random.default_rng(7)
    train=rng.normal(size=(40,96,9)).astype('float32')
    test=rng.normal(size=(10,96,9)).astype('float32')
    ytrain=np.array([0]*20+[1]*20)
    path=tmp_path/"toy.npz"
    np.savez_compressed(path,x_train=train,y_train=ytrain,
        x_test=test,y_test=np.arange(10)%2,
        subject_train=np.array([1]*20+[2]*20),
        subject_test=np.array([1 if overlap else 8]*5+[9]*5))
    return path


def test_full_reference_clean_windows_and_accounting(monkeypatch,tmp_path):
    path=fake_npz(tmp_path)
    class Model:
        classes_=np.array([0,1],dtype=int)
        def predict_proba(self,feat):
            v=np.tanh(feat[:,0])
            p=np.clip((v+1)/2,.05,.95)
            return np.column_stack((p,1-p))
    class Adapter:
        model=Model()
    monkeypatch.setattr(study,"_train_extratrees_local",lambda *a: Adapter())
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"results",
                         max_test=10,seed=57,trees=12,time_bins=12,budget=16)
    study.run(args)
    with (args.out_dir/"summary.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert {r['method'] for r in rows}==set(study.METHODS)
    assert all(0<=float(r['subject_macro_top3_overlap'])<=1 for r in rows)
    with (args.out_dir/"per_sample.csv").open() as f:
        sample_rows=list(csv.DictReader(f))
    assert len(sample_rows)==10*len(study.METHODS)
    assert {int(r['extra_model_evaluations']) for r in sample_rows}=={0,16}
    manifest=json.loads((args.out_dir/"manifest.json").read_text())
    assert manifest["M"]==36 and manifest["Q"]==16
    assert manifest["no_injected_synthetic_patch"]
    assert manifest["additional_reference_queries_per_window_offline"]==36
    assert manifest["per_subject_windows"]=={"8":5,"9":5}
    assert (args.out_dir/"reference_windows.csv").exists()


def test_subject_overlap_is_rejected(tmp_path):
    path=fake_npz(tmp_path,overlap=True)
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"results",
                         max_test=10,seed=57,trees=12,time_bins=12,budget=16)
    with pytest.raises(ValueError,match="leakage"):
        study.run(args)


def test_raw_mlp_reference_toy_without_test_label_leakage(tmp_path):
    path=fake_npz(tmp_path)
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"raw_result",
                         max_test=6,seed=57,trees=12,time_bins=12,budget=16,
                         model="raw_mlp")
    study.run(args)
    manifest=json.loads((args.out_dir/"manifest.json").read_text())
    assert manifest["model_type"]=="raw_mlp"
    assert manifest["no_test_labels_or_reference_ranks_passed_to_methods"]
    assert manifest["n_windows"]==6
    with (args.out_dir/"summary.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(study.METHODS)
    assert all(0<=float(r['subject_macro_top3_overlap'])<=1 for r in rows)


def test_invalid_model_type_fails_before_evaluation(tmp_path):
    path=fake_npz(tmp_path)
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"invalid_result",
                         max_test=6,seed=57,trees=12,time_bins=12,budget=16,
                         model="made_up")
    with pytest.raises(ValueError,match="Unknown model type"):
        study.run(args)
