#!/usr/bin/env python3
"""Export the official participant-disjoint UCI HAR split for BEACON Q36.

UCI HAR provides 9 inertial channels, 128 samples per window and subject IDs.
This is a *different dataset* from PAMAP2. Use the published train/test
partition: never select different participants based on held-out outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from beaconxai.datasets import ensure_uci_har, load_uci_har


def prepare_dataset(root: str | Path, out: str | Path) -> dict:
    folder = ensure_uci_har(root)
    train, y_train, test, y_test = load_uci_har(root)
    train_ids = np.loadtxt(folder / "train" / "subject_train.txt", dtype=int)
    test_ids = np.loadtxt(folder / "test" / "subject_test.txt", dtype=int)
    if (train.ndim != 3 or test.ndim != 3
        or train.shape[1:] != (128,9) or test.shape[1:] != (128,9)
        or len(y_train) != len(train) or len(y_test) != len(test)
        or len(train_ids) != len(train) or len(test_ids) != len(test)
        or not len(train_ids) or not len(test_ids)
        or not np.isfinite(train).all() or not np.isfinite(test).all()
        or np.any(y_train < 0) or np.any(y_test < 0)):
        raise ValueError("Invalid official UCI HAR arrays, subjects, or class labels")
    a,b=set(map(int,train_ids)),set(map(int,test_ids))
    if a & b:
        raise ValueError(f"UCI HAR train/test subject overlap: {sorted(a&b)}")
    if len(a) < 2 or len(b) < 2 or len(set(y_train)) < 2:
        raise ValueError("Insufficient independent subjects or train activity classes")
    output=Path(out)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,x_train=train,y_train=y_train,
        x_test=test,y_test=y_test,subject_train=train_ids,subject_test=test_ids)
    receipt={
        "dataset":"UCI HAR (official train/test subject split)",
        "source":str(folder),
        "output":str(output),
        "train_windows":int(len(train)),
        "test_windows":int(len(test)),
        "train_subjects":sorted(a),
        "test_subjects":sorted(b),
        "train_activity_classes":int(len(set(map(int,y_train)))),
        "test_activity_classes":int(len(set(map(int,y_test)))),
        "channels":9, "window_length":128,
        "channel_groups":["body_acc","body_gyro","total_acc"],
        "subject_leakage":False,
        "normalization":"deferred to train-only standardization inside benchmark"
    }
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("data/uci_har"))
    p.add_argument("--out",type=Path,default=Path("data/uci_har_official_subjects.npz"))
    args=p.parse_args()
    prepare_dataset(args.root,args.out)


if __name__=="__main__":
    main()
