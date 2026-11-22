#!/usr/bin/env python3
"""Preserve official UEA Epilepsy 137/138 split; simulated activities, not diagnoses."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_epilepsy(train_ct, train_labels, test_ct, test_labels, output):
    train=np.asarray(train_ct,dtype=np.float32)
    test=np.asarray(test_ct,dtype=np.float32)
    tr=np.asarray(train_labels).reshape(-1)
    te=np.asarray(test_labels).reshape(-1)
    if (train.shape!=(137,3,206) or test.shape!=(138,3,206)
        or tr.shape!=(137,) or te.shape!=(138,)
        or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("UEA Epilepsy requires finite [137/138,3,206] arrays")
    names=sorted(map(str,np.unique(tr)))
    if len(names)!=4 or sorted(map(str,np.unique(te)))!=names:
        raise ValueError("all four archive classes must occur in official TRAIN and TEST")
    mapper={label:index for index,label in enumerate(names)}
    ytr=np.array([mapper[str(v)] for v in tr],dtype=np.int64)
    yte=np.array([mapper[str(v)] for v in te],dtype=np.int64)
    if min(np.bincount(ytr,minlength=4))<2:
        raise ValueError("insufficient training examples in a class")
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        path,
        x_train=np.ascontiguousarray(train.transpose(0,2,1)),
        y_train=ytr,
        x_test=np.ascontiguousarray(test.transpose(0,2,1)),
        y_test=yte,
        labels=np.asarray(names),
        dataset_protocol=np.array(
            "UEA Epilepsy official TRAIN 137 / TEST 138; simulated activities; "
            "archive reports 3 participants per split, no per-window IDs"
        ),
    )
    return {
        "dataset":"UEA Epilepsy","n_train":len(tr),"n_test":len(te),
        "T":206,"C":3,"classes":names,
        "source":"aeon.datasets.load_classification('Epilepsy')",
        "participant_split_documented_not_per_window_verified":True,
        "diagnostic_or_medical_labels":False,
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from aeon.datasets import load_classification
    xtr,ytr=load_classification("Epilepsy",split="train")
    xte,yte=load_classification("Epilepsy",split="test")
    print(json.dumps(
        prepare_epilepsy(xtr,ytr,xte,yte,args.out),
        ensure_ascii=False,indent=2,
    ),flush=True)


if __name__=="__main__":
    main()
