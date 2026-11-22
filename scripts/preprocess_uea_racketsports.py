#!/usr/bin/env python3
"""Prepare the official, *previously unused* UEA RacketSports train/test split.

151 TRAIN, 152 TEST, 6 sensors, 30 timepoints, 4 classes.
NO subject identifiers, NO invented physical channel relationship.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_racketsports(train_ct, train_labels, test_ct, test_labels, output):
    train=np.asarray(train_ct,dtype=np.float32)
    test=np.asarray(test_ct,dtype=np.float32)
    tr=np.asarray(train_labels).reshape(-1)
    te=np.asarray(test_labels).reshape(-1)
    if (train.shape!=(151,6,30) or test.shape!=(152,6,30)
        or tr.shape!=(151,) or te.shape!=(152,)
        or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("UEA RacketSports must be finite TRAIN [151,6,30] TEST [152,6,30]")
    names=sorted(map(str,np.unique(tr)))
    if len(names)!=4 or sorted(map(str,np.unique(te)))!=names:
        raise ValueError("RacketSports labels must agree between official splits (4 classes)")
    label_index={lab:i for i,lab in enumerate(names)}
    ytr=np.array([label_index[str(v)] for v in tr],dtype=np.int64)
    yte=np.array([label_index[str(v)] for v in te],dtype=np.int64)
    if any(np.count_nonzero(ytr==c)<2 for c in range(4)):
        raise ValueError("insufficient class in TRAIN")
    dest=Path(output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        dest,x_train=np.ascontiguousarray(train.transpose(0,2,1)),
        y_train=ytr,x_test=np.ascontiguousarray(test.transpose(0,2,1)),
        y_test=yte,labels=np.asarray(names),
        dataset_protocol=np.array(
            "UEA RacketSports original TRAIN 151 / TEST 152; no verified participant IDs"
        ),
    )
    return {
        "dataset":"UEA RacketSports",
        "source":"aeon.datasets.load_classification('RacketSports')",
        "train_windows":151,
        "test_windows":152,
        "timepoints":30,
        "channels":6,
        "classes":names,
        "subject_disjoint_claim":False,
        "output":str(dest),
        "npz_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from aeon.datasets import load_classification
    xtr,ytr=load_classification("RacketSports",split="train")
    xte,yte=load_classification("RacketSports",split="test")
    print(json.dumps(
        prepare_racketsports(xtr,ytr,xte,yte,args.out),
        ensure_ascii=False,indent=2
    ),flush=True)


if __name__=="__main__":
    main()
