#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn, train_extratrees_stats, train_histgbt_stats
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _stratified_split_idx(y: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr_idx = []
    va_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        va_idx.append(idx[:n_val])
        tr_idx.append(idx[n_val:])
    tr = np.concatenate(tr_idx)
    va = np.concatenate(va_idx)
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAMAP2 CE diagnostics (correct vs incorrect)")
    p.add_argument("--npz-path", default="./data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--model", choices=["histgbt", "extratrees", "cnn1d"], default="histgbt")
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--max-test", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cnn-epochs", type=int, default=16)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out", default="./outputs_composite/pamap2_ce_diagnostics_histgbt.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_train_full, y_train_full, x_test, y_test = load_npz_dataset(args.npz_path)
    tr_idx, _ = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]

    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed + 9)
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "histgbt":
        clf = train_histgbt_stats(
            x_tr, y_tr, max_iter=220, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=20
        )
    elif args.model == "extratrees":
        clf = train_extratrees_stats(x_tr, y_tr, n_estimators=1000, max_features=0.7, min_samples_leaf=1)
    else:
        clf = train_1dcnn(
            x_tr,
            y_tr,
            epochs=args.cnn_epochs,
            batch_size=args.cnn_batch_size,
            lr=args.cnn_lr,
            label_smoothing=0.0,
            use_class_weights=True,
            tta_shifts=(0, 50),
        )

    train_margins = []
    for i in range(min(len(x_tr), 2000)):
        lg = clf.logits(x_tr[i])
        y_hat = int(np.argmax(lg))
        m = float(lg[y_hat] - np.max(np.delete(lg, y_hat)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    cfg = BeaconConfig(
        q_max=args.q_max,
        k0=8,
        l_min=4,
        k_pos=3,
        k_neg=3,
        q_frag_ratio=0.25,
        alpha=1.0,
        beta=0.5,
        gamma=1.0,
        tau_s=0.10,
        tau_m=tau_m,
        refinement_mode="mixed",
        partition_mode="time_only",
        risk_policy="rho_only",
    )
    channel_means = x_tr.mean(axis=(0, 1)).astype(np.float32)
    neutralizer = Neutralizer(mode="zero", channel_means=channel_means)

    rows, local_rows, _ = evaluate_error_risk(
        x_test=x_test,
        y_test=y_test,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=[args.q_max],
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        composite_weights=None,
        methods={"negative_margin", "beacon_refine"},
    )

    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == args.q_max}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == args.q_max}
    ids = sorted(set(br).intersection(lm))
    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    ce = np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64)
    cm = np.array([lm[i].counter_mass for i in ids], dtype=np.float64)
    rho = np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64)

    c_mask = y == 0
    e_mask = y == 1
    out_rows = [
        {
            "feature": "CE",
            "mean_correct": float(np.mean(ce[c_mask])),
            "mean_incorrect": float(np.mean(ce[e_mask])),
            "auc": _auc(y, ce),
        },
        {
            "feature": "counter_mass",
            "mean_correct": float(np.mean(cm[c_mask])),
            "mean_incorrect": float(np.mean(cm[e_mask])),
            "auc": _auc(y, cm),
        },
        {
            "feature": "rho_cost",
            "mean_correct": float(np.mean(rho[c_mask])),
            "mean_incorrect": float(np.mean(rho[e_mask])),
            "auc": _auc(y, rho),
        },
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["feature", "mean_correct", "mean_incorrect", "auc"])
        wr.writeheader()
        wr.writerows(out_rows)
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()

