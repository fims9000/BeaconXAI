#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess UCI WISDM raw (phone accel+gyro) into NPZ")
    p.add_argument("--root", default="./data/wisdm_raw/wisdm-dataset/raw")
    p.add_argument("--out", default="./data/wisdm_phone_accel_gyro.npz")
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--min-purity", type=float, default=0.85)
    p.add_argument("--train-user-frac", type=float, default=0.7)
    p.add_argument("--merge-tol-ns", type=int, default=40_000_000)
    return p.parse_args()


def _read_sensor_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.endswith(";"):
                s = s[:-1]
            parts = s.split(",")
            if len(parts) != 6:
                continue
            try:
                uid = int(parts[0])
                act = parts[1]
                ts = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
            except Exception:
                continue
            rows.append((uid, act, ts, x, y, z))
    return pd.DataFrame(rows, columns=["uid", "act", "ts", "x", "y", "z"])


def _segment_ids(act: np.ndarray, ts: np.ndarray) -> np.ndarray:
    seg = np.zeros(len(act), dtype=np.int64)
    cur = 0
    for i in range(1, len(act)):
        if act[i] != act[i - 1] or (ts[i] <= ts[i - 1]):
            cur += 1
        seg[i] = cur
    return seg


def _windowize(df: pd.DataFrame, win: int, stride: int, act_to_id: dict[str, int], min_purity: float):
    xs = []
    ys = []
    if df.empty:
        return xs, ys

    df = df.sort_values("ts").reset_index(drop=True)
    seg = _segment_ids(df["act"].to_numpy(), df["ts"].to_numpy())
    df = df.assign(seg=seg)

    for _, g in df.groupby("seg"):
        arr = g[["ax", "ay", "az", "gx", "gy", "gz"]].to_numpy(dtype=np.float32)
        labels = g["act"].to_numpy()
        if len(arr) < win:
            continue
        for s in range(0, len(arr) - win + 1, stride):
            e = s + win
            lw = labels[s:e]
            vals, cnts = np.unique(lw, return_counts=True)
            k = int(np.argmax(cnts))
            act = vals[k]
            purity = float(cnts[k]) / float(len(lw))
            if purity < min_purity:
                continue
            xs.append(arr[s:e])
            ys.append(act_to_id[str(act)])
    return xs, ys


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    acc_dir = root / "phone" / "accel"
    gyr_dir = root / "phone" / "gyro"

    acc_files = sorted([p for p in acc_dir.glob("data_*_accel_phone.txt")])
    gyr_files = sorted([p for p in gyr_dir.glob("data_*_gyro_phone.txt")])

    gyr_map = {}
    for p in gyr_files:
        uid = int(p.name.split("_")[1])
        gyr_map[uid] = p

    users = sorted([int(p.name.split("_")[1]) for p in acc_files if int(p.name.split("_")[1]) in gyr_map])
    if not users:
        raise RuntimeError("No paired phone accel+gyro users found")

    all_acts = set()
    preview_n = min(5, len(users))
    for uid in users[:preview_n]:
        a = _read_sensor_file(acc_dir / f"data_{uid}_accel_phone.txt")
        g = _read_sensor_file(gyr_dir / f"data_{uid}_gyro_phone.txt")
        all_acts.update(a["act"].unique().tolist())
        all_acts.update(g["act"].unique().tolist())

    # Full activity map from all users (small overhead, ensures stable labels)
    all_acts = set()
    for uid in users:
        a = _read_sensor_file(acc_dir / f"data_{uid}_accel_phone.txt")
        all_acts.update(a["act"].unique().tolist())
    acts = sorted(all_acts)
    act_to_id = {a: i for i, a in enumerate(acts)}

    n_train = max(1, int(len(users) * args.train_user_frac))
    train_users = set(users[:n_train])
    test_users = set(users[n_train:])
    overlap = train_users.intersection(test_users)
    if overlap:
        raise RuntimeError(f"Subject split leakage detected: {sorted(overlap)}")
    print(f"subjects total={len(users)} train={len(train_users)} test={len(test_users)}")

    x_train, y_train = [], []
    x_test, y_test = [], []

    for k, uid in enumerate(users, start=1):
        acc = _read_sensor_file(acc_dir / f"data_{uid}_accel_phone.txt")
        gyr = _read_sensor_file(gyr_dir / f"data_{uid}_gyro_phone.txt")

        if acc.empty or gyr.empty:
            continue

        acc = acc.rename(columns={"x": "ax", "y": "ay", "z": "az"}).sort_values("ts")
        gyr = gyr.rename(columns={"x": "gx", "y": "gy", "z": "gz"}).sort_values("ts")

        merged = pd.merge_asof(
            acc,
            gyr[["uid", "act", "ts", "gx", "gy", "gz"]],
            on="ts",
            by=["uid", "act"],
            direction="nearest",
            tolerance=args.merge_tol_ns,
        )
        merged = merged.dropna(subset=["gx", "gy", "gz"]).reset_index(drop=True)
        if merged.empty:
            continue

        xs, ys = _windowize(merged, args.window, args.stride, act_to_id, args.min_purity)
        if not xs:
            continue

        if uid in train_users:
            x_train.extend(xs)
            y_train.extend(ys)
        else:
            x_test.extend(xs)
            y_test.extend(ys)

        if k % 10 == 0:
            print(f"processed users: {k}/{len(users)}")

    x_train_arr = np.stack(x_train).astype(np.float32) if x_train else np.empty((0, args.window, 6), dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int64)
    x_test_arr = np.stack(x_test).astype(np.float32) if x_test else np.empty((0, args.window, 6), dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.int64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x_train=x_train_arr, y_train=y_train_arr, x_test=x_test_arr, y_test=y_test_arr)

    print("Saved:")
    print(out)
    print("Shapes:", x_train_arr.shape, y_train_arr.shape, x_test_arr.shape, y_test_arr.shape)
    print("Classes:", len(act_to_id), sorted(act_to_id.items()))
    print("Train users head:", sorted(train_users)[:10])
    print("Test users head:", sorted(test_users)[:10])


if __name__ == "__main__":
    main()
