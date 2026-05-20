#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.preselect_surrogate import FEATURE_NAMES, SurrogatePack, component_features, full_deltas
from scripts.run_component_conflict_benchmark import _train_extratrees_local, _train_histgbt_local
from scripts.run_part2_extended import _margin, _neutralize_component, _time_slices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train surrogate model for low-budget risk preselect")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--model", choices=["extratrees", "histgbt", "cnn1d"], default="extratrees")
    p.add_argument("--n-windows", type=int, default=300)
    p.add_argument("--time-bins", type=int, default=16)
    p.add_argument("--neutralizer-mode", choices=["interp", "zero", "mean", "channel_mean", "class_mean"], default="interp")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs_composite/low_budget_surrogate_q16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    x_train, y_train, x_test, y_test = load_npz_dataset(args.dataset)
    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "histgbt":
        clf = _train_histgbt_local(x_train, y_train)
    elif args.model == "cnn1d":
        from beaconxai.models import train_1dcnn

        clf = train_1dcnn(
            x_train, y_train, epochs=12, batch_size=256, lr=1e-3, label_smoothing=0.0, use_class_weights=True, tta_shifts=(0,)
        )
    else:
        clf = _train_extratrees_local(x_train, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1)

    class_means = {int(c): np.mean(x_train[y_train == c], axis=(0, 1)).astype(np.float32) for c in np.unique(y_train)}
    global_means = np.mean(x_train, axis=(0, 1)).astype(np.float32)
    t_slices = _time_slices(x_train.shape[1], args.time_bins)

    idx = np.arange(len(x_train), dtype=np.int64)
    rng.shuffle(idx)
    idx = idx[: min(args.n_windows, len(idx))]

    rows_x = []
    rows_y = []
    rows_meta = []
    for i in idx:
        x = x_train[int(i)]
        lg = clf.logits(x)
        yhat = int(np.argmax(lg))
        tmp = lg.copy()
        tmp[yhat] = -1e18
        yrunner = int(np.argmax(tmp))

        if args.neutralizer_mode in ("mean", "channel_mean", "class_mean"):
            cm = class_means.get(yhat, global_means) if args.neutralizer_mode == "class_mean" else global_means
        else:
            cm = None

        d, _m0 = full_deltas(
            x=x,
            clf=clf,
            t_slices=t_slices,
            neutralize_fn=_neutralize_component,
            neutralizer_mode=args.neutralizer_mode,
            channel_means=cm,
        )
        feats = component_features(x, t_slices, class_means.get(yhat, global_means), class_means.get(yrunner, global_means))
        rows_x.append(feats)
        # Target conflict score: larger => more conflictive component.
        rows_y.append(-d)
        rows_meta.append(np.column_stack([np.full(len(d), int(i)), np.arange(len(d), dtype=np.int64)]))

    X = np.concatenate(rows_x, axis=0)
    y = np.concatenate(rows_y, axis=0)
    M = np.concatenate(rows_meta, axis=0)

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=args.seed,
        n_jobs=-1,
    )
    reg.fit(Xtr, ytr)
    pva = reg.predict(Xva)
    sign_acc = float(np.mean((pva > 0.0) == (yva > 0.0)))
    metrics = {
        "n_rows": int(len(X)),
        "n_windows": int(len(idx)),
        "r2_val": float(r2_score(yva, pva)),
        "mae_val": float(mean_absolute_error(yva, pva)),
        "sign_acc_val": sign_acc,
        "feature_names": list(FEATURE_NAMES),
        "feature_importances": [float(v) for v in reg.feature_importances_.tolist()],
    }

    with (out / "surrogate_model.pkl").open("wb") as f:
        pickle.dump(SurrogatePack(model=reg), f)
    with (out / "surrogate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    df["conflict_score_target"] = y
    df["sample_id"] = M[:, 0].astype(np.int64)
    df["component_index"] = M[:, 1].astype(np.int64)
    df.to_csv(out / "surrogate_training_rows.csv", index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
