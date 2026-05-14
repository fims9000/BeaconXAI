#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_uci_har
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig, RiskEvalRow


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


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    if pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(0.0, r - prev_recall)
        prev_recall = r
    return float(ap)


def _precision_recall_at_frac(y_true: np.ndarray, y_score: np.ndarray, frac: float) -> tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    k = max(1, int(np.ceil(frac * n)))
    order = np.argsort(-y_score)
    top = order[:k]
    y_top = y_true[top]
    tp_top = float(np.sum(y_top == 1))
    total_pos = float(np.sum(y_true == 1))
    precision = tp_top / k
    recall = tp_top / total_pos if total_pos > 0 else float("nan")
    return float(precision), float(recall)


def _bootstrap_metric_delta(
    y: np.ndarray,
    s: np.ndarray,
    b: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ds = metric_fn(y[idx], s[idx])
        db = metric_fn(y[idx], b[idx])
        if np.isfinite(ds) and np.isfinite(db):
            deltas.append(ds - db)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    d = np.asarray(deltas, dtype=np.float64)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(np.mean(d > 0.0))


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
    p = argparse.ArgumentParser(description="Q1 baselines on HAR")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--q-values", default="8,16,32,64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-test", type=int, default=512)
    p.add_argument("--cnn-epochs", type=int, default=8)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--n-bootstrap", type=int, default=400)
    p.add_argument("--latency-per-query", type=float, default=0.0010728925)
    p.add_argument("--out", default="./outputs_composite/har_q1_baselines_q8_16_32_64.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]
    k0_map = {8: 4, 16: 8, 32: 8, 64: 8}

    x_train_full, y_train_full, x_test, y_test = load_uci_har(args.dataset_root)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed + 20)
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_va = apply_standardizer(x_va, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    clf = train_1dcnn(
        x_tr,
        y_tr,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        lr=1e-3,
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

    neutralizer = Neutralizer(mode="zero", channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))
    methods = {
        "confidence",
        "entropy",
        "negative_margin",
        "beacon_flat",
        "uniform_refinement",
        "budgeted_shapley_like",
        "ig_topk",
        "beacon_refine",
    }

    out_rows: list[dict] = []
    for q in q_values:
        k0 = int(k0_map.get(q, 8))
        cfg = BeaconConfig(
            q_max=q,
            k0=k0,
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

        rows, _, _ = evaluate_error_risk(
            x_test=x_test,
            y_test=y_test,
            predict_fn=clf.predict,
            logits_fn=clf.logits,
            neutralizer=neutralizer,
            base_cfg=cfg,
            q_values=[q],
            margin_gradient_fn=getattr(clf, "margin_gradient", None),
            composite_weights=None,
            methods=methods,
        )

        by_method: dict[str, list[RiskEvalRow]] = {}
        for r in rows:
            if r.method not in by_method:
                by_method[r.method] = []
            by_method[r.method].append(r)

        base_rows = sorted(by_method["negative_margin"], key=lambda z: z.sample_id)
        y = np.array([r.is_error for r in base_rows], dtype=np.int64)
        b = np.array([r.risk_score for r in base_rows], dtype=np.float64)
        b_q = np.array([r.q_used for r in base_rows], dtype=np.float64)
        b_p10, b_r10 = _precision_recall_at_frac(y, b, 0.10)

        for m, mrows in by_method.items():
            if m not in methods:
                continue
            cur = sorted(mrows, key=lambda z: z.sample_id)
            if len(cur) != len(base_rows):
                continue
            s = np.array([r.risk_score for r in cur], dtype=np.float64)
            q_used = np.array([r.q_used for r in cur], dtype=np.float64)
            p10, r10 = _precision_recall_at_frac(y, s, 0.10)
            d_p10 = float(p10 - b_p10)
            d_r10 = float(r10 - b_r10)
            mean_q = float(max(np.mean(q_used), 1.0))
            qntg = float(d_p10 / mean_q)
            lat_obj = float(mean_q * args.latency_per_query)
            lntg = float(d_p10 / max(lat_obj, 1e-12))
            ci_auc_l, ci_auc_h, frac_auc = _bootstrap_metric_delta(y, s, b, _auc, args.n_bootstrap, args.seed + q + 1)
            ci_p10_l, ci_p10_h, frac_p10 = _bootstrap_metric_delta(
                y,
                s,
                b,
                lambda yy, ss: _precision_recall_at_frac(yy, ss, 0.10)[0],
                args.n_bootstrap,
                args.seed + q + 2,
            )
            out_rows.append(
                {
                    "dataset": "har",
                    "model": "cnn1d",
                    "q_max": q,
                    "k0": k0,
                    "method": m,
                    "auroc": _auc(y, s),
                    "auprc": _auprc(y, s),
                    "precision_at_10pct": p10,
                    "recall_at_10pct": r10,
                    "delta_auroc": _auc(y, s) - _auc(y, b),
                    "delta_auprc": _auprc(y, s) - _auprc(y, b),
                    "delta_p10": d_p10,
                    "delta_r10": d_r10,
                    "qntg_p10": qntg,
                    "lntg_p10": lntg,
                    "mean_q_used": mean_q,
                    "latency_per_object": lat_obj,
                    "ci_delta_auroc_low": ci_auc_l,
                    "ci_delta_auroc_high": ci_auc_h,
                    "frac_positive_auroc": frac_auc,
                    "ci_delta_p10_low": ci_p10_l,
                    "ci_delta_p10_high": ci_p10_h,
                    "frac_positive_p10": frac_p10,
                }
            )
        print(f"[q1-baselines] q={q} done", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()

