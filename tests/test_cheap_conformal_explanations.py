"""Nested subject-disjoint BEACON conformal explanation end-to-end controls."""
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_cheap_conformal_explanations as r
from scripts import run_clean_calibrated_staged_budget as existing


def data_fixture(path,overlap=False):
    rng=np.random.default_rng(7)
    xtr=rng.normal(size=(21*7,24,9)).astype(np.float32)
    yt=np.tile([0,1,0,1,0,1,0],21)
    st=np.repeat(np.arange(1,22),7)
    xte=rng.normal(size=(9*5,24,9)).astype(np.float32)
    yte=np.tile([0,1,1,0,1],9)
    sv=np.repeat([1 if overlap else 101,102,103,104,105,106,107,108,109],5)
    np.savez_compressed(path,x_train=xtr,y_train=yt,
                        subject_train=st,x_test=xte,y_test=yte,
                        subject_test=sv)
    return path


def fast_model(monkeypatch):
    class Fake:
        classes_=np.array([0,1],dtype=int)
        coefs_=[np.arange(2*9,dtype=float).reshape(9,2)]
        intercepts_=[np.array([.1,-.2])]
        def predict_proba(self,features):
            z=np.asarray(features)
            # Include contributions throughout the window so component
            # response differs, without a real trained black box in tests.
            # Distribute signal dependence across all time/sensor blocks
            # so full Q36 has a meaningful (usually strict) top-3.
            weights=np.sin(np.arange(z.shape[1])*.02)
            v=np.tanh(np.dot(z,weights)*.03)
            p=np.clip(.5+.3*v,.05,.95)
            return np.column_stack((p,1-p))
    class Adapter:
        model=Fake()
    monkeypatch.setattr(existing,"_train_extratrees_local",lambda *args: Adapter())
    monkeypatch.setattr(existing,"_anfis_features",
                        lambda batch:np.asarray(batch).reshape(len(batch),-1))


def args(path,out):
    return SimpleNamespace(
        npz_path=path,out_dir=out,model="extratrees",seed=57,
        trees=8,windows_per_subject=2,alpha=.2,
        ridge_alpha=100.,max_queries=16,batch_size=4)


def test_all_train_roles_independent_of_official_outer_subjects():
    roles=r.separate_subject_roles(np.repeat(np.arange(1,22),4),57)
    assert [len(roles[k]) for k in
            ("model_fit","surrogate_fit","conformal_calibrate")]==[8,4,9]
    groups=[set(v) for v in roles.values()]
    assert len(set.union(*groups))==21
    assert not any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    with pytest.raises(ValueError):
        r.separate_subject_roles(np.arange(20),57)


def test_complete_nested_protocol_and_test_label_invariance(tmp_path,monkeypatch):
    fast_model(monkeypatch)
    path=data_fixture(tmp_path/"data.npz")
    config=args(path,tmp_path/"result1")
    manifest,summary=r.run(config)
    assert len(summary)==9
    assert manifest["subject_roles_mutually_disjoint"]
    assert manifest["n_test_windows"]==18
    assert manifest["subject_block_conformal"]["order_statistic_rank"]==8
    assert manifest["same_windows_per_subject_in_calibration_and_test"]==2
    assert manifest["n_sensitivities_fit_offline"]==8*36
    assert manifest["pairwise_top3_fit_windows"]+(
        manifest["pairwise_top3_skipped_ambiguous_windows"])==8
    assert manifest["n_sensitivities_calibrate_offline"]==18*36
    assert manifest["online_certificate"]["certified_windows"]+(
        manifest["online_certificate"]["abstained_windows"])==18
    rows=list(csv.DictReader((config.out_dir/"per_sample.csv").open()))
    assert len(rows)==18*9
    assert all(int(v["extra_model_evaluations"]) in (0,4,8,12,16)
               for v in rows)
    assert all(v["output_status"]!="certified" or v["method"]=="certify_or_abstain_q16"
               for v in rows)
    assert manifest["test_model_examples_all_methods_and_Q36_reference"]==(
        18 + 36*18 + sum(int(v["extra_model_evaluations"]) for v in rows))
    assert manifest["heldout_simultaneous_bound_coverage"]["subject_blocks_total"]==9
    # y_test is never consumed to train the black box, learn the surrogate,
    # calibrate a certificate, pick components or decide when to abstain.
    before=(config.out_dir/"per_sample.csv").read_bytes()
    with np.load(path) as d:
        altered={k:d[k].copy() for k in d.files}
    altered["y_test"][:]=999
    np.savez_compressed(path,**altered)
    c2=args(path,tmp_path/"result2")
    m2,_=r.run(c2)
    assert before==(c2.out_dir/"per_sample.csv").read_bytes()
    assert manifest["classifier_fitted_model_sha256"]==m2["classifier_fitted_model_sha256"]


def test_official_subject_overlap_is_rejected_before_training(tmp_path):
    path=data_fixture(tmp_path/"bad.npz",overlap=True)
    with pytest.raises(ValueError,match="leakage"):
        r.run(args(path,tmp_path/"out"))


def test_selecting_calibration_windows_cannot_omit_subject(tmp_path):
    x=np.zeros((2,24,9))
    subject=np.array([3,4])
    with pytest.raises(ValueError,match="insufficient"):
        r.subset_by_subject(x,subject,[3,4],3,42)
