#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset, load_uci_har
from beaconxai.models import (
    train_1dcnn,
    train_anfis_stats,
    train_extratrees_stats,
    train_logreg,
    train_minirocket_if_available,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick model comparison for HAR datasets")
    p.add_argument("--dataset", choices=["uci_har", "npz"], default="npz")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--npz-path", default="./data/wisdm_phone_accel_gyro_w200s100.npz")
    p.add_argument("--models", default="anfis,extratrees,cnn1d")
    p.add_argument("--anfis-rules", type=int, default=10)
    p.add_argument("--anfis-max-samples", type=int, default=4000)
    p.add_argument("--train-size", type=int, default=8000)
    p.add_argument("--test-size", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cnn-epochs", type=int, default=12)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--cnn-tta-shifts", default="0,50")
    p.add_argument("--out", default="./outputs_composite/quick_model_benchmark.csv")
    return p.parse_args()


def _sample_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_size: int,
    test_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    ntr = min(len(x_train), max(1, train_size))
    nte = min(len(x_test), max(1, test_size))
    tr_idx = rng.choice(len(x_train), size=ntr, replace=False)
    te_idx = rng.choice(len(x_test), size=nte, replace=False)
    return x_train[tr_idx], y_train[tr_idx], x_test[te_idx], y_test[te_idx]


def _accuracy(clf, x_test: np.ndarray, y_test: np.ndarray) -> float:
    pred = np.array([clf.predict(x) for x in x_test], dtype=np.int64)
    return float((pred == y_test).mean())


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.dataset == "uci_har":
        x_train, y_train, x_test, y_test = load_uci_har(args.dataset_root)
    else:
        x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)
    x_train, y_train, x_test, y_test = _sample_subset(
        x_train, y_train, x_test, y_test, args.train_size, args.test_size, args.seed
    )

    tta_shifts = tuple(int(v) for v in args.cnn_tta_shifts.split(",") if v.strip())
    if not tta_shifts:
        tta_shifts = (0,)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows: list[dict] = []

    for model_name in models:
        t0 = time.time()
        if model_name == "logreg":
            clf = train_logreg(x_train, y_train)
        elif model_name == "anfis":
            clf = train_anfis_stats(
                x_train,
                y_train,
                n_rules=args.anfis_rules,
                max_fit_samples=args.anfis_max_samples,
            )
        elif model_name == "extratrees":
            clf = train_extratrees_stats(x_train, y_train)
        elif model_name == "minirocket":
            clf = train_minirocket_if_available(x_train, y_train)
        elif model_name == "cnn1d":
            clf = train_1dcnn(
                x_train,
                y_train,
                epochs=args.cnn_epochs,
                batch_size=args.cnn_batch_size,
                lr=args.cnn_lr,
                label_smoothing=0.0,
                use_class_weights=True,
                tta_shifts=tta_shifts,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        train_sec = time.time() - t0

        t1 = time.time()
        acc = _accuracy(clf, x_test, y_test)
        eval_sec = time.time() - t1

        row = {
            "model": model_name,
            "accuracy": round(acc, 6),
            "train_sec": round(train_sec, 3),
            "eval_sec": round(eval_sec, 3),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "n_classes_train": int(len(np.unique(y_train))),
            "n_classes_test": int(len(np.unique(y_test))),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out_path = Path(args.out)
    _write_csv(out_path, rows)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
