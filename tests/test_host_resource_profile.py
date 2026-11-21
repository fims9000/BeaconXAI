"""Toy end-to-end host profiler: clear measured vs projected resource claims."""
from types import SimpleNamespace
import csv
import json

import numpy as np
import pytest

from scripts import profile_full_audit_host as profile


def _npz(tmp_path,overlap=False):
    rng=np.random.default_rng(15)
    xtrain=rng.normal(size=(40,96,9)).astype("float32")
    xtest=rng.normal(size=(8,96,9)).astype("float32")
    fn=tmp_path/"toy.npz"
    np.savez_compressed(
        fn,x_train=xtrain,y_train=np.array([0]*20+[1]*20),
        x_test=xtest,y_test=np.array([0,1]*4),
        subject_train=np.array([1]*20+[2]*20),
        subject_test=np.array([1 if overlap else 8]*4+[9]*4))
    return fn


def test_host_full_q16_audit_distinguishes_projection_from_measurement(tmp_path):
    fn=_npz(tmp_path)
    args=SimpleNamespace(npz_path=fn,out_dir=tmp_path/"resource",
                         max_test=4,seed=57,time_bins=12,budget=16,
                         warmups=0,repeats=1)
    profile.run(args)
    report=json.loads((args.out_dir/"host_resource_profile.json").read_text())
    assert report["test_windows_sampled"]==4
    assert report["host_only_not_mcu_benchmark"] is True
    assert report["measured_energy_joules"] is None
    assert report["measured_ram_peak"] is None
    assert report["resources"]["additional_model_examples"]==16
    with (args.out_dir/"per_window.csv").open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==4
    assert all(int(r["extra_model_examples"])==16 for r in rows)
    assert all(int(r["distinct_components"])==8 for r in rows)
    assert all(float(r["audit_median_ms_host"])>=0 for r in rows)


def test_host_profiler_rejects_split_leakage(tmp_path):
    fn=_npz(tmp_path,overlap=True)
    args=SimpleNamespace(npz_path=fn,out_dir=tmp_path/"resource",
                         max_test=4,seed=57,time_bins=12,budget=16,
                         warmups=0,repeats=1)
    with pytest.raises(ValueError,match="subject overlap"):
        profile.run(args)
