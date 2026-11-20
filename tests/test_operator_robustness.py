"""End-to-end operator dependence: offline reference cannot leak into Q16 selection."""
from types import SimpleNamespace
import csv
import json
import numpy as np
import pytest
from scripts import run_operator_robustness as study


def _fixture(tmp_path,test_labels=None,overlap=False):
    rng=np.random.default_rng(17)
    xtrain=rng.normal(size=(40,96,9)).astype("float32")
    ytrain=np.array([0]*20+[1]*20)
    xval=rng.normal(size=(10,96,9)).astype("float32")
    ytest=np.array([1,0]*5) if test_labels is None else test_labels
    path=tmp_path/"experiment.npz"
    np.savez_compressed(path,x_train=xtrain,y_train=ytrain,
                        x_test=xval,y_test=ytest,
                        subject_train=np.array([1]*20+[2]*20),
                        subject_test=np.array([1 if overlap else 8]*5+[9]*5))
    return path


def test_real_medoid_vs_interp_reference_no_test_label_use(monkeypatch,tmp_path):
    class FakeModel:
        classes_=np.array([0,1])
        calls=0
        def predict_proba(self,features):
            FakeModel.calls+=len(features)
            z=np.tanh(features[:,0])
            p=np.clip((z+1)/2,.05,.95)
            return np.column_stack((p,1-p))
    class Adapter:
        model=FakeModel()
    monkeypatch.setattr(study,"_train_extratrees_local",lambda *args:Adapter())
    fn=_fixture(tmp_path)
    out=tmp_path/"out"
    args=SimpleNamespace(npz_path=fn,out_dir=out,dataset="PAMAP2",
                         model="extratrees",max_test=10,budget=16,time_bins=12,
                         trees=15,seed=57)
    study.study(args)
    with (out/"summary.csv").open() as f:
        summary=list(csv.DictReader(f))
    assert len(summary)==4
    assert {x["method"] for x in summary}=={
        "full_interpolation_q36","medoid_q16","interpolation_q16",
        "uniform_medoid_q16"}
    assert all(0<=float(x["subject_macro_top3_overlap"])<=1 for x in summary)
    m=json.loads((out/"manifest.json").read_text())
    assert m["M"]==36 and m["Q"]==16 and m["n_samples"]==10
    assert m["donor_bank_memory_bytes"]>0
    assert m["no_test_activity_labels_read_for_inference"]
    # Original three single-operator controls + one dual Q8+Q8 audit.
    assert FakeModel.calls==10*(1+2*36+4*16)
    with (out/"dual_operator_summary.csv").open() as f:
        dual_summary=list(csv.DictReader(f))
    assert {r["method"] for r in dual_summary}=={
        "dual_operator_q16","interpolation_q16","medoid_q16",
        "uniform_medoid_q16"}
    assert all(int(r["extra_model_evaluations"])==16 for r in dual_summary)
    with (out/"dual_operator_per_sample.csv").open() as f:
        dual_samples=list(csv.DictReader(f))
    assert len(dual_samples)==10*4
    dual_rows=[r for r in dual_samples if r["method"]=="dual_operator_q16"]
    assert all(int(r["distinct_probed_components"])==8 for r in dual_rows)
    assert all(r["reported_not_abstained"] in ("0","1") for r in dual_rows)
    gate=json.loads((out/"dual_operator_gate.json").read_text())
    assert gate["n_windows"]==10
    assert gate["full_q36_references_used_at_inference"] is False
    assert gate["not_a_causal_or_calibrated_coverage_guarantee"] is True
    orig=(out/"per_sample.csv").read_bytes()
    orig_dual=(out/"dual_operator_per_sample.csv").read_bytes()
    # Change ONLY held-out activity labels: ranks cannot change.
    with np.load(fn) as z:
        data={key:z[key].copy() for key in z.files}
    data["y_test"][:]=999
    np.savez_compressed(fn,**data)
    out2=tmp_path/"out2"
    study.study(SimpleNamespace(**{**vars(args),"out_dir":out2}))
    assert orig==(out2/"per_sample.csv").read_bytes()
    assert orig_dual==(out2/"dual_operator_per_sample.csv").read_bytes()


def test_subject_overlap_rejected(tmp_path):
    path=_fixture(tmp_path,overlap=True)
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"out",
        dataset="PAMAP2",model="extratrees",max_test=10,budget=16,time_bins=12,
        trees=15,seed=57)
    with pytest.raises(ValueError,match="overlap"):
        study.study(args)
