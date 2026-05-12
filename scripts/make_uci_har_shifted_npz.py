#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import load_uci_har


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create shifted UCI HAR NPZ for robustness/error-risk eval")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--out", default="./data/uci_har_shifted.npz")
    p.add_argument("--noise-sigma", type=float, default=0.35)
    p.add_argument("--drop-prob", type=float, default=0.35)
    p.add_argument("--mask-prob", type=float, default=0.50)
    p.add_argument("--mask-len", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_train, y_train, x_test, y_test = load_uci_har(args.dataset_root)

    rng = np.random.default_rng(args.seed)
    x_shift = x_test.copy().astype(np.float32)

    # channel-wise noise scale from train statistics
    ch_std = x_train.std(axis=(0, 1), keepdims=True).astype(np.float32)
    noise = rng.normal(0.0, 1.0, size=x_shift.shape).astype(np.float32)
    x_shift += args.noise_sigma * ch_std * noise

    n, t, d = x_shift.shape

    # random per-sample channel dropout
    for i in range(n):
        if rng.random() < args.drop_prob:
            c = int(rng.integers(0, d))
            x_shift[i, :, c] = 0.0

    # random temporal masking segments
    L = max(1, min(args.mask_len, t))
    for i in range(n):
        if rng.random() < args.mask_prob:
            s = int(rng.integers(0, max(1, t - L + 1)))
            e = s + L
            x_shift[i, s:e, :] = 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x_train=x_train, y_train=y_train, x_test=x_shift, y_test=y_test)

    print("Saved:")
    print(out)
    print("Shapes:", x_train.shape, y_train.shape, x_shift.shape, y_test.shape)


if __name__ == "__main__":
    main()
