#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.models import (
    train_1dcnn,
    train_extratrees_stats,
    train_histgbt_stats,
    train_minirocket_if_available,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAMAP2 baseline benchmark (accuracy + macro-F1)")
    p.add_argument("--npz-path", default="./data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--models", default="cnn1d,histgbt,extratrees")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cnn-epochs", type=int, default=20)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--cnn-tta-shifts", default="0,50")
    p.add_argument("--out", default="./outputs_composite/pamap2_baseline.csv")
    return p.parse_args()


def _predict_all(clf, x_test: np.ndarray) -> np.ndarray:
    return np.array([clf.predict(x) for x in x_test], dtype=np.int64)


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
    x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)
    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    rows: list[dict] = []
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tta_shifts = tuple(int(v) for v in args.cnn_tta_shifts.split(",") if v.strip())
    if not tta_shifts:
        tta_shifts = (0,)

    for model_name in models:
        t0 = time.time()
        if model_name == "cnn1d":
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
        elif model_name == "extratrees":
            clf = train_extratrees_stats(x_train, y_train, n_estimators=1000, max_features=0.7, min_samples_leaf=1)
        elif model_name == "histgbt":
            clf = train_histgbt_stats(
                x_train, y_train, max_iter=220, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=20
            )
        elif model_name == "minirocket":
            clf = train_minirocket_if_available(x_train, y_train)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        train_sec = time.time() - t0

        t1 = time.time()
        pred = _predict_all(clf, x_test)
        eval_sec = time.time() - t1
        acc = float(np.mean(pred == y_test))
        f1m = float(f1_score(y_test, pred, average="macro"))

        row = {
            "model": model_name,
            "accuracy": round(acc, 6),
            "macro_f1": round(f1m, 6),
            "train_sec": round(train_sec, 3),
            "eval_sec": round(eval_sec, 3),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "n_classes": int(len(np.unique(y_train))),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out = Path(args.out)
    _write_csv(out, rows)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

