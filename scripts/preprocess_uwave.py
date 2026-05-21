#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
from pathlib import Path
from urllib.request import urlopen

import numpy as np


UWAVE_URL = "https://zenodo.org/records/10852667/files/UWAVE.npz?download=1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download/convert UWaveGestureLibrary to BEACON npz format")
    p.add_argument("--out", default="data/uwave_gesture_library.npz")
    p.add_argument("--raw-out", default="data/UWAVE.npz")
    p.add_argument("--url", default=UWAVE_URL)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    raw_out = Path(args.raw_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.parent.mkdir(parents=True, exist_ok=True)

    if not raw_out.exists():
        print(f"downloading: {args.url}")
        blob = urlopen(args.url, timeout=180).read()
        raw_out.write_bytes(blob)
    else:
        blob = raw_out.read_bytes()

    data = np.load(io.BytesIO(blob))
    x_train = data["Xtr"].astype(np.float32)
    y_train = np.asarray(data["Ytr"]).reshape(-1).astype(np.int64)
    x_test = data["Xte"].astype(np.float32)
    y_test = np.asarray(data["Yte"]).reshape(-1).astype(np.int64)

    # Labels in the Zenodo bundle are 1-based; the rest of the repo expects 0-based labels.
    if y_train.min() == 1:
        y_train = y_train - 1
        y_test = y_test - 1

    np.savez_compressed(out, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"saved: {out}")
    print(f"x_train={x_train.shape}, x_test={x_test.shape}, n_classes={len(np.unique(y_train))}")


if __name__ == "__main__":
    main()

