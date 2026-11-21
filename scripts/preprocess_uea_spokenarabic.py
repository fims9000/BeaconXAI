#!/usr/bin/env python3
"""Preserve official UEA SpokenArabicDigits TRAIN/TEST and report provenance.

The files are acoustic feature time-series, not raw audio or language semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_spokenarabic(xtr, ytr, xte, yte, out):
    train=np.asarray(xtr,dtype=np.float32)
    test=np.asarray(xte,dtype=np.float32)
    yt=np.asarray(ytr).reshape(-1)
    yv=np.asarray(yte).reshape(-1)
    if (train.shape!=(6599,13,93) or test.shape!=(2199,13,93)
        or yt.shape!=(6599,) or yv.shape!=(2199,)
        or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("official SpokenArabicDigits must be 6599/2199 [13,93]")
    classes=sorted(map(str,np.unique(yt)))
    if len(classes)!=10 or sorted(map(str,np.unique(yv)))!=classes:
        raise ValueError("official SpokenArabicDigits requires ten matching classes")
    code={name:i for i,name in enumerate(classes)}
    ytrain=np.array([code[str(v)] for v in yt],dtype=np.int64)
    ytest=np.array([code[str(v)] for v in yv],dtype=np.int64)
    if np.min(np.bincount(ytrain,minlength=10))<=16:
        raise ValueError("Insufficient examples in at least one TRAIN class")
    path=Path(out)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        path,
        x_train=np.ascontiguousarray(train.transpose(0,2,1)),
        y_train=ytrain,
        x_test=np.ascontiguousarray(test.transpose(0,2,1)),
        y_test=ytest,
        labels=np.asarray(classes),
        dataset_protocol=np.asarray(
            "UEA SpokenArabicDigits official aeon 1.5.0 TRAIN 6599 "
            "TEST 2199 T93 C13 ten digit classes; no verified speaker IDs"
        ),
    )
    return {
        "dataset":"UEA SpokenArabicDigits", "train":len(train),
        "test":len(test),"shape_time_channels":[93,13],
        "num_classes":10,"speaker_disjoint_verified":False,
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    from aeon.datasets import load_classification
    xtr,ytr=load_classification("SpokenArabicDigits",split="train")
    xte,yte=load_classification("SpokenArabicDigits",split="test")
    print(json.dumps(
        prepare_spokenarabic(xtr,ytr,xte,yte,args.out),
        ensure_ascii=False,indent=2,
    ),flush=True)


if __name__=="__main__":
    main()
