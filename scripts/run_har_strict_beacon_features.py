#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


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
    best_w = np.concatenate([[1.0], np.zeros(X.shape[1] - 1)], axis=0)
    cands = [best_w]
    cands.extend(rng.uniform(-2.0, 2.0, size=(n_trials, X.shape[1])))
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


def _bootstrap_delta_auc(
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


def _collect_features(
    rows: list[RiskEvalRow],
    local_rows: list[LocalMetricRow],
    q: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == q}
    nm = {r.sample_id: r for r in rows if r.method == "negative_margin" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == q}
    ids = sorted(set(br).intersection(nm).intersection(lm))
    if not ids:
        return None

    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    feats = {
        "neg_margin": _rank_norm(np.array([nm[i].risk_score for i in ids], dtype=np.float64)),
        "rho_cost": _rank_norm(np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64)),
        "drop_ratio": _rank_norm(np.array([lm[i].drop_ratio for i in ids], dtype=np.float64)),
        "residual_bad": _rank_norm(np.array([-lm[i].residual_ratio for i in ids], dtype=np.float64)),
        "ce": _rank_norm(np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64)),
    }
    return y, feats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict HAR sweep for BEACON features: val calibration -> one final test")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-values", default="16,32,64")
    p.add_argument("--refinement-modes", default="support,mixed")
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--n-calib-trials", type=int, default=12000)
    p.add_argument("--n-bootstrap", type=int, default=800)
    p.add_argument("--cnn-epochs", type=int, default=25)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="./outputs_composite/har_strict_features")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]
    refinement_modes = [v.strip() for v in args.refinement_modes.split(",") if v.strip()]

    x_train_full, y_train_full, x_test, y_test = load_uci_har(args.dataset_root)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if args.max_val > 0 and args.max_val < len(x_va):
        rng = np.random.default_rng(args.seed + 7)
        idx = rng.choice(len(x_va), size=args.max_val, replace=False)
        x_va = x_va[idx]
        y_va = y_va[idx]

    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed + 11)
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

    base_cfg = BeaconConfig(
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

    feature_sets: dict[str, list[str]] = {
        "negative_margin": ["neg_margin"],
        "margin_plus_rho_cost": ["neg_margin", "rho_cost"],
        "margin_plus_drop_ratio": ["neg_margin", "drop_ratio"],
        "margin_plus_residual_ratio": ["neg_margin", "residual_bad"],
        "margin_plus_rho_drop": ["neg_margin", "rho_cost", "drop_ratio"],
        "margin_plus_ce": ["neg_margin", "ce"],
        "margin_plus_drop_ce": ["neg_margin", "drop_ratio", "ce"],
    }

    score_rows: list[dict] = []
    boot_rows: list[dict] = []

    for mode in refinement_modes:
        cfg = replace(base_cfg, refinement_mode=mode)
        rows_val, local_val, _ = evaluate_error_risk(
            x_test=x_va,
            y_test=y_va,
            predict_fn=clf.predict,
            logits_fn=clf.logits,
            neutralizer=neutralizer,
            base_cfg=cfg,
            q_values=q_values,
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
            q_values=q_values,
            margin_gradient_fn=getattr(clf, "margin_gradient", None),
            composite_weights=None,
            methods={"negative_margin", "beacon_refine"},
        )

        for q in q_values:
            fv = _collect_features(rows_val, local_val, q)
            ft = _collect_features(rows_test, local_test, q)
            if fv is None or ft is None:
                continue
            yv, fval = fv
            yt, fte = ft
            nm_score = fte["neg_margin"]

            method_scores: dict[str, np.ndarray] = {}

            for name, cols in feature_sets.items():
                Xv = np.stack([fval[c] for c in cols], axis=1)
                Xt = np.stack([fte[c] for c in cols], axis=1)
                if len(cols) == 1:
                    w = np.array([1.0], dtype=np.float64)
                else:
                    w = _fit_weights(Xv, yv, args.n_calib_trials, args.seed + q * 13 + len(cols) * 17 + len(name))
                s = Xt @ w
                method_scores[name] = s

                p5, _ = _precision_recall_at_frac(yt, s, 0.05)
                p10, r10 = _precision_recall_at_frac(yt, s, 0.10)
                score_rows.append(
                    {
                        "refinement_mode": mode,
                        "q_max": q,
                        "method": name,
                        "auroc": _auc(yt, s),
                        "auprc": _auprc(yt, s),
                        "precision_at_5pct": p5,
                        "precision_at_10pct": p10,
                        "recall_at_10pct": r10,
                        "delta_auroc_vs_negative_margin": _auc(yt, s) - _auc(yt, nm_score),
                        "delta_auprc_vs_negative_margin": _auprc(yt, s) - _auprc(yt, nm_score),
                        "weights": json.dumps([float(v) for v in w]),
                    }
                )

            for name, s in method_scores.items():
                if name == "negative_margin":
                    continue
                dm, lo, hi, fp = _bootstrap_delta_auc(yt, s, nm_score, args.n_bootstrap, args.seed + q + len(name) * 31)
                boot_rows.append(
                    {
                        "refinement_mode": mode,
                        "q_max": q,
                        "method": name,
                        "delta_auc_mean": dm,
                        "ci_low": lo,
                        "ci_high": hi,
                        "frac_positive": fp,
                    }
                )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "har_strict_feature_sweep.csv", score_rows)
    _write_csv(out / "har_strict_feature_bootstrap.csv", boot_rows)

    print("Saved:")
    print(out / "har_strict_feature_sweep.csv")
    print(out / "har_strict_feature_bootstrap.csv")


if __name__ == "__main__":
    main()
