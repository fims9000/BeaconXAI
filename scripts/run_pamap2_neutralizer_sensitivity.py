#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn, train_extratrees_stats, train_histgbt_stats
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig, LocalMetricRow, RiskEvalRow


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


def _rank_norm(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(ranks)
    return ranks / (len(x) - 1)


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


def _bootstrap_ci(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        da = _auc(y[idx], a[idx])
        db = _auc(y[idx], b[idx])
        if np.isfinite(da) and np.isfinite(db):
            d.append(da - db)
    if not d:
        return float("nan"), float("nan"), float("nan")
    d = np.asarray(d, dtype=np.float64)
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


def _fit_weights(X: np.ndarray, y: np.ndarray, n_trials: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_auc = -1.0
    best_w = np.array([1.0, 0.0], dtype=np.float64)
    cands = [best_w]
    cands.extend(rng.uniform(-2.0, 2.0, size=(n_trials, 2)))
    for w in cands:
        a = _auc(y, X @ w)
        if np.isfinite(a) and a > best_auc:
            best_auc = a
            best_w = np.asarray(w, dtype=np.float64)
    return best_w


def _collect(
    rows: list[RiskEvalRow],
    local_rows: list[LocalMetricRow],
    q: int,
    method: str = "beacon_refine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    br = {r.sample_id: r for r in rows if r.method == method and r.q_max == q}
    nm = {r.sample_id: r for r in rows if r.method == "negative_margin" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == method and r.q_max == q}
    ids = sorted(set(br).intersection(nm).intersection(lm))
    if not ids:
        return None
    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    nm_s = _rank_norm(np.array([nm[i].risk_score for i in ids], dtype=np.float64))
    ce = _rank_norm(np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64))
    rho = _rank_norm(np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64))
    return y, nm_s, ce, rho


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAMAP2 neutralizer sensitivity on selected model")
    p.add_argument("--npz-path", default="./data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--model", choices=["histgbt", "extratrees", "cnn1d"], default="extratrees")
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--max-val", type=int, default=256)
    p.add_argument("--max-test", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-calib-trials", type=int, default=2000)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--cnn-epochs", type=int, default=16)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out", default="./outputs_composite/pamap2_neutralizer_sensitivity_extratrees_q16.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    x_train_full, y_train_full, x_test, y_test = load_npz_dataset(args.npz_path)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if args.max_val > 0 and args.max_val < len(x_va):
        rng = np.random.default_rng(args.seed + 1)
        idx = rng.choice(len(x_va), size=args.max_val, replace=False)
        x_va = x_va[idx]
        y_va = y_va[idx]
    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed + 2)
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_va = apply_standardizer(x_va, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "histgbt":
        clf = train_histgbt_stats(x_tr, y_tr, max_iter=220, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=20)
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
    base_cfg = BeaconConfig(
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

    out_rows: list[dict] = []
    for mode in ("zero", "mean", "interp"):
        neutralizer = Neutralizer(mode=mode, channel_means=channel_means)
        for refinement_mode, method_tag in (("support", "support+CE"), ("mixed", "mixed")):
            cfg = replace(base_cfg, refinement_mode=refinement_mode)
            rows_val, local_val, _ = evaluate_error_risk(
                x_test=x_va,
                y_test=y_va,
                predict_fn=clf.predict,
                logits_fn=clf.logits,
                neutralizer=neutralizer,
                base_cfg=cfg,
                q_values=[args.q_max],
                margin_gradient_fn=getattr(clf, "margin_gradient", None),
                composite_weights=None,
                methods={"negative_margin", "beacon_refine"},
            )
            rows_test, local_test, _ = evaluate_error_risk(
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

            dv = _collect(rows_val, local_val, args.q_max)
            dt = _collect(rows_test, local_test, args.q_max)
            if dv is None or dt is None:
                continue
            yv, nm_v, ce_v, rho_v = dv
            yt, nm_t, ce_t, rho_t = dt
            s_nm = nm_t

            w_ce = _fit_weights(np.stack([nm_v, ce_v], axis=1), yv, args.n_calib_trials, args.seed + len(mode) * 11)
            s_ce = np.stack([nm_t, ce_t], axis=1) @ w_ce
            p10, r10 = _precision_recall_at_frac(yt, s_ce, 0.10)
            lo, hi, fp = _bootstrap_ci(yt, s_ce, s_nm, args.n_bootstrap, args.seed + len(mode) * 101 + len(method_tag))
            out_rows.append(
                {
                    "neutralizer": mode,
                    "method": method_tag if refinement_mode == "support" else "mixed+CE",
                    "q_max": args.q_max,
                    "auroc": _auc(yt, s_ce),
                    "auprc": _auprc(yt, s_ce),
                    "delta_auroc": _auc(yt, s_ce) - _auc(yt, s_nm),
                    "delta_auprc": _auprc(yt, s_ce) - _auprc(yt, s_nm),
                    "precision_at_10pct": p10,
                    "recall_at_10pct": r10,
                    "ci_low": lo,
                    "ci_high": hi,
                    "frac_positive": fp,
                }
            )

            if refinement_mode == "mixed":
                w_rho = _fit_weights(
                    np.stack([nm_v, rho_v], axis=1), yv, args.n_calib_trials, args.seed + len(mode) * 17
                )
                s_rho = np.stack([nm_t, rho_t], axis=1) @ w_rho
                p10, r10 = _precision_recall_at_frac(yt, s_rho, 0.10)
                lo, hi, fp = _bootstrap_ci(
                    yt, s_rho, s_nm, args.n_bootstrap, args.seed + len(mode) * 201 + len(method_tag)
                )
                out_rows.append(
                    {
                        "neutralizer": mode,
                        "method": "mixed+rho_cost",
                        "q_max": args.q_max,
                        "auroc": _auc(yt, s_rho),
                        "auprc": _auprc(yt, s_rho),
                        "delta_auroc": _auc(yt, s_rho) - _auc(yt, s_nm),
                        "delta_auprc": _auprc(yt, s_rho) - _auprc(yt, s_nm),
                        "precision_at_10pct": p10,
                        "recall_at_10pct": r10,
                        "ci_low": lo,
                        "ci_high": hi,
                        "frac_positive": fp,
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            wr.writeheader()
            wr.writerows(out_rows)
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()
