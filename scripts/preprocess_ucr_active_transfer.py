#!/usr/bin/env python3
"""Pin untouched official UCR Wafer and TwoPatterns classification splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SPECS={
    "Wafer":(1000,6164,152,2),
    "TwoPatterns":(1000,4000,128,4),
}


def prepare_ucr_transfer(name,xtr,ytr,xte,yte,out):
    if name not in SPECS:
        raise ValueError("only prespecified Wafer and TwoPatterns are allowed")
    ntr,nte,length,k=SPECS[name]
    train=np.asarray(xtr,dtype=np.float32)
    test=np.asarray(xte,dtype=np.float32)
    yt=np.asarray(ytr).reshape(-1)
    yv=np.asarray(yte).reshape(-1)
    if (train.shape!=(ntr,1,length) or test.shape!=(nte,1,length)
        or yt.shape!=(ntr,) or yv.shape!=(nte,)
        or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError(f"{name}: expected official {ntr}/{nte} [1,{length}]")
    labs=sorted(map(str,np.unique(yt)))
    if len(labs)!=k or sorted(map(str,np.unique(yv)))!=labs:
        raise ValueError(f"{name}: expected {k} matching TRAIN/TEST labels")
    remap={label:i for i,label in enumerate(labs)}
    ytrain=np.asarray([remap[str(v)] for v in yt],dtype=np.int64)
    ytest=np.asarray([remap[str(v)] for v in yv],dtype=np.int64)
    if min(np.bincount(ytrain,minlength=k))<25:
        raise ValueError(f"{name}: insufficient TRAIN examples for one class")
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        out,
        x_train=np.ascontiguousarray(train.transpose(0,2,1)),
        y_train=ytrain,
        x_test=np.ascontiguousarray(test.transpose(0,2,1)),
        y_test=ytest,
        labels=np.asarray(labs),
        dataset_protocol=np.asarray(
            f"UCR {name} official aeon 1.5.0 TRAIN {ntr} TEST {nte} "
            f"T{length} C1 K{k}; source device/session IDs not verified"
        ),
    )
    return {
        "name":name,"TRAIN":ntr,"TEST":nte,"timepoints":length,
        "channels":1,"classes":labs,
        "original_device_or_subject_disjoint_verified":False,
        "data_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=sorted(SPECS),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    from aeon.datasets import load_classification
    xtr,ytr=load_classification(args.dataset,split="train")
    xte,yte=load_classification(args.dataset,split="test")
    print(json.dumps(
        prepare_ucr_transfer(args.dataset,xtr,ytr,xte,yte,args.out),
        ensure_ascii=False,indent=2,
    ),flush=True)


if __name__=="__main__":
    main()
