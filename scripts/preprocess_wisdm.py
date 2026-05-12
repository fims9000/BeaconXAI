#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess WISDM CSV into windowed NPZ")
    p.add_argument("--csv", required=True, help="Path to WISDM raw CSV")
    p.add_argument("--out", default="./data/wisdm_preprocessed.npz")
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--train-user-frac", type=float, default=0.7)
    return p.parse_args()


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise ValueError(f"Missing columns. Need one of: {candidates}")


def _windowize(arr: np.ndarray, labels: np.ndarray, win: int, stride: int):
    xs = []
    ys = []
    n = arr.shape[0]
    for s in range(0, n - win + 1, stride):
        e = s + win
        w = arr[s:e]
        lw = labels[s:e]
        y = np.bincount(lw).argmax()
        xs.append(w)
        ys.append(y)
    if not xs:
        return np.empty((0, win, arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    user_col = _find_col(df, ["user", "subject", "user_id"])
    act_col = _find_col(df, ["activity", "label"])
    ts_col = _find_col(df, ["timestamp", "time", "unix_time"])
    x_col = _find_col(df, ["x", "acc_x", "accel_x"])
    y_col = _find_col(df, ["y", "acc_y", "accel_y"])
    z_col = _find_col(df, ["z", "acc_z", "accel_z"])

    df = df[[user_col, act_col, ts_col, x_col, y_col, z_col]].copy()
    df = df.dropna().sort_values([user_col, ts_col])

    users = sorted(df[user_col].unique().tolist())
    n_train_users = max(1, int(len(users) * args.train_user_frac))
    train_users = set(users[:n_train_users])

    acts = sorted(df[act_col].unique().tolist())
    act_to_id = {a: i for i, a in enumerate(acts)}
    df[act_col] = df[act_col].map(act_to_id).astype(np.int64)

    x_train_parts = []
    y_train_parts = []
    x_test_parts = []
    y_test_parts = []

    for uid, g in df.groupby(user_col):
        arr = g[[x_col, y_col, z_col]].to_numpy(dtype=np.float32)
        labels = g[act_col].to_numpy(dtype=np.int64)
        xw, yw = _windowize(arr, labels, args.window, args.stride)
        if uid in train_users:
            x_train_parts.append(xw)
            y_train_parts.append(yw)
        else:
            x_test_parts.append(xw)
            y_test_parts.append(yw)

    x_train = np.concatenate(x_train_parts, axis=0) if x_train_parts else np.empty((0, args.window, 3), dtype=np.float32)
    y_train = np.concatenate(y_train_parts, axis=0) if y_train_parts else np.empty((0,), dtype=np.int64)
    x_test = np.concatenate(x_test_parts, axis=0) if x_test_parts else np.empty((0, args.window, 3), dtype=np.float32)
    y_test = np.concatenate(y_test_parts, axis=0) if y_test_parts else np.empty((0,), dtype=np.int64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    print("Saved:")
    print(out)
    print("Shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)


if __name__ == "__main__":
    main()
