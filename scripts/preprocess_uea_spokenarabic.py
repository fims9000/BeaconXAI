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



def _resample_multichannel(samples, expected_n):
    """Interpolate each channel on [0,1] to 93 frames; no cross-row information.

    Handles aeon's list of variable-length [13, L_i] objects, as well as
    fixed-length numpy arrays; preserve all original per-channel lengths.
    """
    if len(samples)!=expected_n:
        raise ValueError("official SpokenArabicDigits split size mismatch")
    destination=np.empty((expected_n,13,93),dtype=np.float32)
    lengths=np.empty((expected_n,13),dtype=np.int32)
    target_grid=np.linspace(0.,1.,93,dtype=np.float64)
    for i,series in enumerate(samples):
        if len(series)!=13:
            raise ValueError("SpokenArabicDigits expects 13 MFCC channels")
        for c in range(13):
            values=np.asarray(series[c],dtype=np.float64).reshape(-1)
            if (len(values)<2 or not np.isfinite(values).all()):
                raise ValueError("requires >=2 finite samples in each MFCC channel")
            lengths[i,c]=len(values)
            if len(values)==93:
                destination[i,c]=values
            else:
                source_grid=np.linspace(0.,1.,len(values),dtype=np.float64)
                destination[i,c]=np.interp(target_grid,source_grid,values)
    if not np.isfinite(destination).all():
        raise ValueError("nonfinite MFCC values after interpolation")
    return destination,lengths


def prepare_spokenarabic(xtr, ytr, xte, yte, out):
    yt=np.asarray(ytr).reshape(-1)
    yv=np.asarray(yte).reshape(-1)
    if yt.shape!=(6599,) or yv.shape!=(2199,):
        raise ValueError("official SpokenArabicDigits must have 6599/2199 labels")
    train,train_lengths=_resample_multichannel(xtr,6599)
    test,test_lengths=_resample_multichannel(xte,2199)
    if train.shape!=(6599,13,93) or test.shape!=(2199,13,93):
        raise AssertionError("interpolated MFCC output geometry incorrect")
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
        train_original_lengths=train_lengths,
        test_original_lengths=test_lengths,
        resampling_description=np.asarray("fixed length 93 normalized-time linear interpolation, amendment A1"),
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
        "resampling_protocol":"amendment A1; normalized-time linear interpolation to 93",
        "train_length_min_max":[int(train_lengths.min()),int(train_lengths.max())],
        "test_length_min_max":[int(test_lengths.min()),int(test_lengths.max())],
        "train_variable_length":bool(train_lengths.min()!=train_lengths.max()),
        "test_variable_length":bool(test_lengths.min()!=test_lengths.max()),
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
