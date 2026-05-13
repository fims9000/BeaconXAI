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


def _collect_margin_rho(
    rows: list[RiskEvalRow],
    local_rows: list[LocalMetricRow],
    q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == q}
    nm = {r.sample_id: r for r in rows if r.method == "negative_margin" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == q}
    ids = sorted(set(br).intersection(nm).intersection(lm))
    if not ids:
        return None
    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    neg_margin = _rank_norm(np.array([nm[i].risk_score for i in ids], dtype=np.float64))
    rho = _rank_norm(np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64))
    X = np.stack([neg_margin, rho], axis=1)
    return X, y, np.array(ids, dtype=np.int64)


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


def _bootstrap_delta(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        da = _auc(y[idx], score_a[idx])
        db = _auc(y[idx], score_b[idx])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
    if not deltas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = np.asarray(deltas, dtype=np.float64)
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(np.mean(d > 0.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict HAR final: margin + rho, val calibration -> test")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-values", default="16,32,64")
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--n-calib-trials", type=int, default=30000)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--cnn-epochs", type=int, default=25)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out", default="./outputs_composite/har_margin_rho_final.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]

    x_train_full, y_train_full, x_test, y_test = load_uci_har(args.dataset_root)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if args.max_val > 0 and args.max_val < len(x_va):
        rng = np.random.default_rng(args.seed + 11)
        idx = rng.choice(len(x_va), size=args.max_val, replace=False)
        x_va = x_va[idx]
        y_va = y_va[idx]

    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed)
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
        q_max=max(q_values),
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
        partition_mode="time_only",
        risk_policy="rho_only",
    )
    neutralizer = Neutralizer(mode="zero", channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))
    methods = {"negative_margin", "beacon_refine"}

    rows_va, local_va, _ = evaluate_error_risk(
        x_test=x_va,
        y_test=y_va,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=q_values,
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        composite_weights=None,
        methods=methods,
    )
    rows_te, local_te, _ = evaluate_error_risk(
        x_test=x_test,
        y_test=y_test,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=q_values,
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        composite_weights=None,
        methods=methods,
    )

    out_rows: list[dict] = []
    for q in q_values:
        data_va = _collect_margin_rho(rows_va, local_va, q)
        data_te = _collect_margin_rho(rows_te, local_te, q)
        if data_va is None or data_te is None:
            continue
        Xva, yva, _ = data_va
        Xte, yte, _ = data_te

        w = _fit_weights(Xva, yva, args.n_calib_trials, args.seed + q)
        s_te = Xte @ w
        s_nm = Xte[:, 0]

        a_comp = _auc(yte, s_te)
        a_nm = _auc(yte, s_nm)
        p_comp = _auprc(yte, s_te)
        p_nm = _auprc(yte, s_nm)
        d_mean, ci_lo, ci_hi, frac_pos = _bootstrap_delta(yte, s_te, s_nm, args.n_bootstrap, args.seed + q + 1000)

        out_rows.append(
            {
                "q_max": q,
                "auroc_margin_plus_rho": a_comp,
                "auroc_negative_margin": a_nm,
                "delta_auroc": a_comp - a_nm,
                "auprc_margin_plus_rho": p_comp,
                "auprc_negative_margin": p_nm,
                "bootstrap_delta_mean": d_mean,
                "bootstrap_ci_low": ci_lo,
                "bootstrap_ci_high": ci_hi,
                "bootstrap_frac_positive": frac_pos,
                "w_neg_margin": float(w[0]),
                "w_rho": float(w[1]),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_rows:
        with out_path.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            wr.writeheader()
            wr.writerows(out_rows)
    print("Saved:")
    print(out_path)


if __name__ == "__main__":
    main()
