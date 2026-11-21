#!/usr/bin/env python3
"""Prepare the independently sourced UEA BasicMotions official train/test split.

This dataset is NOT UCI HAR. It has 6 channels (watch acceleration/gyro),
100 timepoints, 4 classes, 40 train / 40 test. The UEA archive has no
reliable participant identifiers, so never call this a subject-held-out test.
Archive channel ordering is left unchanged; no gravity invariant is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_basicmotions(train_ct, train_labels, test_ct, test_labels, out):
    xtr=np.asarray(train_ct,dtype=np.float32)
    xte=np.asarray(test_ct,dtype=np.float32)
    tr=np.asarray(train_labels).reshape(-1)
    te=np.asarray(test_labels).reshape(-1)
    if (xtr.ndim!=3 or xte.ndim!=3 or
        xtr.shape!=(40,6,100) or xte.shape!=(40,6,100) or
        len(tr)!=40 or len(te)!=40 or
        not np.isfinite(xtr).all() or not np.isfinite(xte).all()):
        raise ValueError("expected finite BasicMotions [40,6,100] per split")
    train_names=sorted(map(str,np.unique(tr)))
    test_names=sorted(map(str,np.unique(te)))
    if train_names!=test_names or len(train_names)!=4:
        raise ValueError("inconsistent four-class labels")
    mapping={name:k for k,name in enumerate(train_names)}
    ytr=np.asarray([mapping[str(v)] for v in tr],dtype=np.int64)
    yte=np.asarray([mapping[str(v)] for v in te],dtype=np.int64)
    if any(np.count_nonzero(ytr==k)<2 for k in mapping.values()):
        raise ValueError("insufficient training examples")
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        out,x_train=np.ascontiguousarray(xtr.transpose(0,2,1)),
        y_train=ytr,x_test=np.ascontiguousarray(xte.transpose(0,2,1)),
        y_test=yte,labels=np.array(train_names),
        dataset_protocol=np.array(
            "UEA BasicMotions official train/test; participant ids unavailable"
        )
    )
    return {
        "dataset":"UEA BasicMotions",
        "n_train":len(xtr),"n_test":len(xte),
        "n_timepoints":100,"n_channels":6,
        "labels":train_names,"source":"aeon.datasets.load_basic_motions",
        "subject_disjoint_split_verified":False,
        "archive_channel_physics_invariant_assumed":False,
        "output":str(out),
        "sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    from aeon.datasets import load_basic_motions
    Xtr,Ytr=load_basic_motions(split="train",return_type="numpy3d")
    Xte,Yte=load_basic_motions(split="test",return_type="numpy3d")
    result=prepare_basicmotions(Xtr,Ytr,Xte,Yte,args.out)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
