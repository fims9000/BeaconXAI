#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset, load_uci_har
from beaconxai.models import train_1dcnn, train_extratrees_stats, train_logreg, train_minirocket_if_available
from beaconxai.neutralization import Neutralizer
from beaconxai.rho_sanity import run_rho_sanity
from beaconxai.types import BeaconConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rho_B sanity-check vs rho_exact/beam")
    p.add_argument("--dataset", choices=["uci_har", "npz"], default="uci_har")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--npz-path", default="")
    p.add_argument("--model", choices=["extratrees", "minirocket", "cnn1d", "logreg"], default="extratrees")
    p.add_argument("--neutralization", choices=["zero", "mean", "interp"], default="zero")
    p.add_argument("--k0", type=int, default=8)
    p.add_argument("--q-max", type=int, default=32)
    p.add_argument("--l-min", type=int, default=4)
    p.add_argument("--q-frag-ratio", type=float, default=0.25)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--cnn-epochs", type=int, default=20)
    p.add_argument("--cnn-batch-size", type=int, default=128)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--cnn-label-smoothing", type=float, default=0.0)
    p.add_argument("--cnn-no-class-weights", action="store_true")
    p.add_argument("--cnn-tta-shifts", default="0")
    p.add_argument("--out", default="./outputs_validation/rho_sanity.json")
    return p.parse_args()


def _load_data(args):
    if args.dataset == "uci_har":
        return load_uci_har(args.dataset_root)
    if not args.npz_path:
        raise ValueError("--npz-path required when --dataset npz")
    return load_npz_dataset(args.npz_path)


def main() -> None:
    args = parse_args()
    x_train, y_train, x_test, _ = _load_data(args)

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "extratrees":
        clf = train_extratrees_stats(x_train, y_train)
    elif args.model == "logreg":
        clf = train_logreg(x_train, y_train)
    elif args.model == "cnn1d":
        tta_shifts = tuple(int(v) for v in args.cnn_tta_shifts.split(",") if v.strip())
        if not tta_shifts:
            tta_shifts = (0,)
        clf = train_1dcnn(
            x_train,
            y_train,
            epochs=args.cnn_epochs,
            batch_size=args.cnn_batch_size,
            lr=args.cnn_lr,
            label_smoothing=args.cnn_label_smoothing,
            use_class_weights=not args.cnn_no_class_weights,
            tta_shifts=tta_shifts,
        )
    else:
        clf = train_minirocket_if_available(x_train, y_train)

    channel_means = np.zeros(x_train.shape[-1], dtype=np.float32)
    if args.neutralization == "mean":
        channel_means = x_train.mean(axis=(0, 1)).astype(np.float32)
    neutralizer = Neutralizer(mode=args.neutralization, channel_means=channel_means)

    cfg = BeaconConfig(
        q_max=args.q_max,
        k0=args.k0,
        l_min=args.l_min,
        q_frag_ratio=args.q_frag_ratio,
    )

    out = run_rho_sanity(
        x_test=x_test,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        cfg=cfg,
        n_samples=args.n_samples,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:")
    print(out_path)


if __name__ == "__main__":
    main()
