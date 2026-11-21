#!/usr/bin/env python3
"""Exact official UEA UWaveGestureLibrary archive, not the larger UCR copies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_uwave(xtr,ytr,xte,yte,path):
    train=np.asarray(xtr,dtype=np.float32)
    test=np.asarray(xte,dtype=np.float32)
    labels_train=np.asarray(ytr).reshape(-1)
    labels_test=np.asarray(yte).reshape(-1)
    if (train.shape!=(120,3,315) or test.shape!=(320,3,315)
        or labels_train.shape!=(120,) or labels_test.shape!=(320,)
        or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("Expected UEA UWave 120/320 recordings of shape [3,315]")
    labels=sorted(map(str,np.unique(labels_train)))
    if (len(labels)!=8 or sorted(map(str,np.unique(labels_test)))!=labels):
        raise ValueError("UEA UWave class labels must cover the same eight classes")
    encoding={v:i for i,v in enumerate(labels)}
    y_train=np.asarray([encoding[str(v)] for v in labels_train],dtype=np.int64)
    y_test=np.asarray([encoding[str(v)] for v in labels_test],dtype=np.int64)
    if np.any(np.bincount(y_train,minlength=8)!=15):
        raise ValueError("Preregistered UWave TRAIN must have 15 per class")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        path,
        x_train=np.ascontiguousarray(train.transpose(0,2,1)),
        y_train=y_train,
        x_test=np.ascontiguousarray(test.transpose(0,2,1)),
        y_test=y_test,
        labels=np.asarray(labels),
        dataset_protocol=np.asarray(
            "UEA UWaveGestureLibrary aeon 1.5.0 official TRAIN 120 TEST 320 "
            "T315 C3 eight classes; no verified person/session IDs"
        ),
    )
    return {
        "name":"UEA UWaveGestureLibrary",
        "train":120,"test":320,"timepoints":315,"channels":3,
        "train_per_class":15,"class_labels":labels,
        "subject_disjoint":None,
        "data_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    from aeon.datasets import load_classification
    xtr,ytr=load_classification("UWaveGestureLibrary",split="train")
    xte,yte=load_classification("UWaveGestureLibrary",split="test")
    print(json.dumps(
        prepare_uwave(xtr,ytr,xte,yte,a.out),
        ensure_ascii=False,indent=2,
    ),flush=True)


if __name__=="__main__":
    main()
