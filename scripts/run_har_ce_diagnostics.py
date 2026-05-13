#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_uci_har
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn
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
    p = argparse.ArgumentParser(description="CE diagnostics on HAR")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--cnn-epochs", type=int, default=25)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out", default="./outputs_composite/har_ce_diagnostics.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_train_full, y_train_full, x_test, y_test = load_uci_har(args.dataset_root)
    tr_idx, _ = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    clf = train_1dcnn(
        x_tr,
        y_tr,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        lr=args.cnn_lr,
        label_smoothing=0.0,
        use_class_weights=True,
        tta_shifts=(0, 64),
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
    neutralizer = Neutralizer(mode="zero", channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))

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

