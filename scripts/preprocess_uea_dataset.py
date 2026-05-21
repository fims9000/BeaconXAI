#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from aeon.datasets import load_classification


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert UEA dataset to BEACON npz format")
    p.add_argument("--name", required=True, help="UEA dataset name, e.g. Heartbeat")
    p.add_argument("--out", required=True, help="Output .npz path")
    return p.parse_args()


def _encode_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(np.concatenate([y_train, y_test], axis=0))
    mapping = {c: i for i, c in enumerate(classes.tolist())}
    yt = np.array([mapping[v] for v in y_train], dtype=np.int64)
    yv = np.array([mapping[v] for v in y_test], dtype=np.int64)
    return yt, yv


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_classification(name=args.name, split="train")
    x_test, y_test = load_classification(name=args.name, split="test")

    # aeon -> [N, C, T], repo expects [N, T, C]
    x_train = np.transpose(np.asarray(x_train, dtype=np.float32), (0, 2, 1))
    x_test = np.transpose(np.asarray(x_test, dtype=np.float32), (0, 2, 1))
    y_train, y_test = _encode_labels(np.asarray(y_train), np.asarray(y_test))

    np.savez_compressed(
        out,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"saved: {out}")
    print(
        f"x_train={x_train.shape}, x_test={x_test.shape}, "
        f"n_classes={len(np.unique(y_train))}"
    )


if __name__ == "__main__":
    main()
