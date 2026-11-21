#!/usr/bin/env python3
"""UEA PenDigits official split, eight 2D pen samples, ten digits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def prepare_pendigits(xtr,ytr,xte,yte,path):
    train=np.asarray(xtr,dtype=np.float32)
    test=np.asarray(xte,dtype=np.float32)
    y_train_source=np.asarray(ytr).reshape(-1)
    y_test_source=np.asarray(yte).reshape(-1)
    if (train.shape!=(7494,2,8) or test.shape!=(3498,2,8)
        or y_train_source.shape!=(7494,) or y_test_source.shape!=(3498,)
        or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("Expected official UEA PenDigits TRAIN 7494 TEST 3498 [2,8]")
    labs=sorted(map(str,np.unique(y_train_source)))
    if len(labs)!=10 or sorted(map(str,np.unique(y_test_source)))!=labs:
        raise ValueError("Expected 10 matching digit classes")
    encoder={name:j for j,name in enumerate(labs)}
    y_train=np.array([encoder[str(v)] for v in y_train_source],dtype=np.int64)
    y_test=np.array([encoder[str(v)] for v in y_test_source],dtype=np.int64)
    if np.min(np.bincount(y_train,minlength=10))<=10:
        raise ValueError("Every TRAIN class must have more than ten examples")
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,
        x_train=np.ascontiguousarray(train.transpose(0,2,1)),
        y_train=y_train,x_test=np.ascontiguousarray(test.transpose(0,2,1)),
        y_test=y_test,labels=np.asarray(labs),
        dataset_protocol=np.asarray(
            "UEA PenDigits official TRAIN 7494 TEST 3498 T8 C2 digits; "
            "person identity not provided by this loader"
        ),
    )
    return {
        "dataset":"UEA PenDigits","train":len(train),"test":len(test),
        "timepoints":8,"channels":2,"classes":labs,
        "subjects_disjoint_verified":False,
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    from aeon.datasets import load_classification
    xtr,ytr=load_classification("PenDigits",split="train")
    xte,yte=load_classification("PenDigits",split="test")
    print(json.dumps(
        prepare_pendigits(xtr,ytr,xte,yte,a.out),
        ensure_ascii=False,indent=2,
    ),flush=True)


if __name__=="__main__":
    main()
