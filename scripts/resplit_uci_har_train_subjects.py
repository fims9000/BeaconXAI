#!/usr/bin/env python3
"""Hold out previously untested UCI HAR *official training* subjects.

This experiment NEVER uses official test windows for training or evaluation.
Training-window subject IDs are predeclared, not chosen by inspecting scores.
The result is a new subject-disjoint internal replication, NOT a second
independent external UCI HAR test set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def resplit_training_subjects(source_npz: Path, out: Path,
                              heldout_subjects: tuple[int, ...]) -> dict:
    if not heldout_subjects or len(set(heldout_subjects)) != len(heldout_subjects):
        raise ValueError("held-out subject IDs must be distinct and nonempty")
    with np.load(source_npz) as z:
        x = np.asarray(z["x_train"])
        y = np.asarray(z["y_train"])
        sid = np.asarray(z["subject_train"], dtype=int)
        forbidden = np.asarray(z["subject_test"], dtype=int)
    train_subjects = set(int(s) for s in sid)
    requested = set(heldout_subjects)
    if not requested.issubset(train_subjects):
        raise ValueError("held-out IDs must belong exclusively to official TRAIN")
    if requested.intersection(set(forbidden)):
        raise ValueError("official test subjects must never be reused")
    mask = np.isin(sid, heldout_subjects)
    if (not mask.any() or mask.all() or x.ndim != 3 or x.shape[1:] != (128, 9)
        or len(x) != len(y) or len(x) != len(sid)):
        raise ValueError("invalid official training arrays or subject split")
    if (len(set(y[mask])) != len(set(y[~mask])) or
            len(set(y[mask])) < 2):
        raise ValueError("heldout split must contain all fitted class labels")
    train_set, test_set = set(sid[~mask]), set(sid[mask])
    if train_set & test_set or (train_set | test_set) != train_subjects:
        raise AssertionError("subject leakage or dropped original TRAIN subjects")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, x_train=x[~mask], y_train=y[~mask],
        subject_train=sid[~mask], x_test=x[mask], y_test=y[mask],
        subject_test=sid[mask],
    )
    return {
        "provenance": "official UCI HAR TRAIN only",
        "original_official_test_used": False,
        "train_subjects": sorted(map(int, train_set)),
        "heldout_subjects": sorted(map(int, test_set)),
        "train_windows": int(np.count_nonzero(~mask)),
        "test_windows": int(np.count_nonzero(mask)),
        "out": str(out),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--heldout", required=True,
                   help="Comma-separated predeclared subject IDs from official TRAIN")
    args = p.parse_args()
    ids = tuple(int(s) for s in args.heldout.split(","))
    report = resplit_training_subjects(args.source, args.out, ids)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
