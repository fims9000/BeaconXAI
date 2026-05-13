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


def _bootstrap_delta(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        da = _auc(y[idx], a[idx])
        db = _auc(y[idx], b[idx])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
    if not deltas:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = np.asarray(deltas, dtype=np.float64)
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(np.mean(d > 0.0))


def _collect_margin_rho_cost(
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
    x_nm = _rank_norm(np.array([nm[i].risk_score for i in ids], dtype=np.float64))
    x_rho = _rank_norm(np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64))
    X = np.stack([x_nm, x_rho], axis=1)
    return X, y, np.array(ids, dtype=np.int64)


def _noise(x: np.ndarray, std: float, seed: int) -> np.ndarray:
    if std <= 0:
        return x
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, std, size=x.shape).astype(x.dtype)


def _time_mask(x: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    if ratio <= 0:
        return x
    rng = np.random.default_rng(seed)
    out = x.copy()
    t = out.shape[1]
    w = max(1, int(round(t * ratio)))
    for i in range(out.shape[0]):
        s = int(rng.integers(0, max(1, t - w + 1)))
        out[i, s : s + w, :] = 0.0
    return out


def _channel_dropout(x: np.ndarray, p: float, seed: int) -> np.ndarray:
    if p <= 0:
        return x
    rng = np.random.default_rng(seed)
    out = x.copy()
    n, _, c = out.shape
    mask = rng.random((n, c)) < p
    for i in range(n):
        for j in range(c):
            if mask[i, j]:
                out[i, :, j] = 0.0
    return out


def _apply_stress(x: np.ndarray, mode: str, seed: int, noise_std: float, mask_ratio: float, drop_prob: float) -> np.ndarray:
    if mode == "clean":
        return x
    if mode == "noise":
        return _noise(x, noise_std, seed)
    if mode == "time_mask":
        return _time_mask(x, mask_ratio, seed)
    if mode == "channel_dropout":
        return _channel_dropout(x, drop_prob, seed)
    raise ValueError(f"unknown stress mode: {mode}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress HAR for mixed+rho_cost vs negative_margin")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--time-mask-ratio", type=float, default=0.10)
    p.add_argument("--channel-dropout-prob", type=float, default=0.10)
    p.add_argument("--n-calib-trials", type=int, default=12000)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--cnn-epochs", type=int, default=25)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out", default="./outputs_composite/har_stress_rho_cost.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q = int(args.q_max)

    x_train_full, y_train_full, x_test, y_test = load_uci_har(args.dataset_root)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

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
        yh = int(np.argmax(lg))
        m = float(lg[yh] - np.max(np.delete(lg, yh)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    cfg = BeaconConfig(
        q_max=q,
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

    out_rows: list[dict] = []
    for mode in ["clean", "noise", "time_mask", "channel_dropout"]:
        x_va_s = _apply_stress(
            x_va,
            mode=mode,
            seed=args.seed + 101,
            noise_std=args.noise_std,
            mask_ratio=args.time_mask_ratio,
            drop_prob=args.channel_dropout_prob,
        )
        x_te_s = _apply_stress(
            x_test,
            mode=mode,
            seed=args.seed + 202,
            noise_std=args.noise_std,
            mask_ratio=args.time_mask_ratio,
            drop_prob=args.channel_dropout_prob,
        )

        rows_va, local_va, _ = evaluate_error_risk(
            x_test=x_va_s,
            y_test=y_va,
            predict_fn=clf.predict,
            logits_fn=clf.logits,
            neutralizer=neutralizer,
            base_cfg=cfg,
            q_values=[q],
            margin_gradient_fn=getattr(clf, "margin_gradient", None),
            composite_weights=None,
            methods={"negative_margin", "beacon_refine"},
        )
        rows_te, local_te, _ = evaluate_error_risk(
            x_test=x_te_s,
            y_test=y_test,
            predict_fn=clf.predict,
            logits_fn=clf.logits,
            neutralizer=neutralizer,
            base_cfg=cfg,
            q_values=[q],
            margin_gradient_fn=getattr(clf, "margin_gradient", None),
            composite_weights=None,
            methods={"negative_margin", "beacon_refine"},
        )

        dv = _collect_margin_rho_cost(rows_va, local_va, q)
        dt = _collect_margin_rho_cost(rows_te, local_te, q)
        if dv is None or dt is None:
            continue
        Xv, yv, _ = dv
        Xt, yt, _ = dt

        w = _fit_weights(Xv, yv, args.n_calib_trials, args.seed + 333 + len(mode))
        s_mr = Xt @ w
        s_nm = Xt[:, 0]

        a_nm = _auc(yt, s_nm)
        a_mr = _auc(yt, s_mr)
        p_nm = _auprc(yt, s_nm)
        p_mr = _auprc(yt, s_mr)
        p5_nm, _ = _precision_recall_at_frac(yt, s_nm, 0.05)
        p10_nm, r10_nm = _precision_recall_at_frac(yt, s_nm, 0.10)
        p5_mr, _ = _precision_recall_at_frac(yt, s_mr, 0.05)
        p10_mr, r10_mr = _precision_recall_at_frac(yt, s_mr, 0.10)
        dm, lo, hi, fp = _bootstrap_delta(yt, s_mr, s_nm, args.n_bootstrap, args.seed + 909 + len(mode))

        out_rows.append(
            {
                "stress_mode": mode,
                "q_max": q,
                "auroc_negative_margin": a_nm,
                "auroc_margin_plus_rho_cost": a_mr,
                "delta_auroc": a_mr - a_nm,
                "auprc_negative_margin": p_nm,
                "auprc_margin_plus_rho_cost": p_mr,
                "delta_auprc": p_mr - p_nm,
                "p5_negative_margin": p5_nm,
                "p5_margin_plus_rho_cost": p5_mr,
                "p10_negative_margin": p10_nm,
                "p10_margin_plus_rho_cost": p10_mr,
                "r10_negative_margin": r10_nm,
                "r10_margin_plus_rho_cost": r10_mr,
                "bootstrap_delta_auc_mean": dm,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "bootstrap_frac_positive": fp,
                "w_neg_margin": float(w[0]),
                "w_rho_cost": float(w[1]),
            }
        )

    out = Path(args.out)
    _write_csv(out, out_rows)
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()

