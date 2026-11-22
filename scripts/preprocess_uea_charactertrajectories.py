#!/usr/bin/env python3
"""CharacterTrajectories fixed-length _eq archive, not the original unequal series."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare_character_trajectories(xtr,ytr,xte,yte,out):
    train=np.asarray(xtr,dtype=np.float32)
    test=np.asarray(xte,dtype=np.float32)
    yt=np.asarray(ytr).reshape(-1)
    yv=np.asarray(yte).reshape(-1)
    print("ARCHIVE_GEOMETRY",train.shape,test.shape,yt.shape,yv.shape,flush=True)
    if (train.shape!=(1422,3,182) or test.shape!=(1436,3,182)
        or yt.shape!=(1422,) or yv.shape!=(1436,)
        or not np.isfinite(train).all() or not np.isfinite(test).all()):
        raise ValueError("CharacterTrajectories _eq requires finite 1422/1436 [3,182]")
    labels=sorted(map(str,np.unique(yt)))
    if len(labels)!=20 or sorted(map(str,np.unique(yv)))!=labels:
        raise ValueError("twenty same TRAIN/TEST classes required")
    mapping={v:i for i,v in enumerate(labels)}
    fit_y=np.array([mapping[str(a)] for a in yt],dtype=np.int64)
    test_y=np.array([mapping[str(a)] for a in yv],dtype=np.int64)
    if np.min(np.bincount(fit_y,minlength=20))<25:
        raise ValueError("too few per-class TRAIN windows")
    dest=Path(out)
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,
        x_train=np.ascontiguousarray(train.transpose(0,2,1)),
        y_train=fit_y,
        x_test=np.ascontiguousarray(test.transpose(0,2,1)),
        y_test=test_y,
        labels=np.asarray(labels),
        dataset_protocol=np.asarray(
            "UEA CharacterTrajectories _eq aeon 1.5.0 official "
            "TRAIN 1422 TEST 1436 T182 C3; source unequal length"
        ),
    )
    return {
        "dataset":"UEA CharacterTrajectories _eq",
        "train":1422,"test":1436,"T":182,"C":3,
        "classes":20,"source_unequal_length":True,
        "participant_ids_verified":False,
        "sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    from aeon.datasets import load_classification
    a,ya=load_classification(
        "CharacterTrajectories",split="train",load_equal_length=True
    )
    b,yb=load_classification(
        "CharacterTrajectories",split="test",load_equal_length=True
    )
    print(json.dumps(
        prepare_character_trajectories(a,ya,b,yb,args.out),
        indent=2,ensure_ascii=False,
    ),flush=True)


if __name__=="__main__":
    main()
