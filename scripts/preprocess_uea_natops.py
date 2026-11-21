#!/usr/bin/env python3
"""Prepare UEA NATOPS official 180/180 split (24 channels, 51 time steps)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def prepare_natops(xtrain,ytrain,xtest,ytest,out):
    a=np.asarray(xtrain,dtype=np.float32)
    b=np.asarray(xtest,dtype=np.float32)
    yt=np.asarray(ytrain).reshape(-1)
    yv=np.asarray(ytest).reshape(-1)
    if (a.shape!=(180,24,51) or b.shape!=(180,24,51) or
        yt.shape!=(180,) or yv.shape!=(180,) or
        not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("NATOPS requires finite 180/180 [24,51] arrays")
    labels=sorted(map(str,np.unique(yt)))
    if len(labels)!=6 or labels!=sorted(map(str,np.unique(yv))):
        raise ValueError("six archive classes must match train and test")
    ix={v:i for i,v in enumerate(labels)}
    t=np.array([ix[str(v)] for v in yt],dtype=np.int64)
    v=np.array([ix[str(v)] for v in yv],dtype=np.int64)
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,
        x_train=np.ascontiguousarray(a.transpose(0,2,1)),
        y_train=t,x_test=np.ascontiguousarray(b.transpose(0,2,1)),
        y_test=v,labels=np.asarray(labels),
        dataset_protocol=np.asarray(
            "UEA NATOPS official TRAIN 180 / TEST 180; no confirmed participant IDs"
        ),
    )
    return {
        "dataset":"UEA NATOPS","train":180,"test":180,
        "n_channels":24,"n_timepoints":51,
        "source":"aeon.datasets.load_classification('NATOPS')",
        "participant_ids_verified":False,
        "sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    from aeon.datasets import load_classification
    a,yt=load_classification("NATOPS",split="train")
    b,yv=load_classification("NATOPS",split="test")
    print(json.dumps(prepare_natops(a,yt,b,yv,args.out),indent=2),flush=True)


if __name__=="__main__":
    main()
