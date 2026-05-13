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
    classes = np.unique(y)
    tr_idx = []
    va_idx = []
    for c in classes:
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


def _collect_frame(
    rows: list[RiskEvalRow],
    local_rows: list[LocalMetricRow],
    q: int,
) -> dict[str, np.ndarray]:
    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == q}
    conf = {r.sample_id: r for r in rows if r.method == "confidence" and r.q_max == 0}
    negm = {r.sample_id: r for r in rows if r.method == "negative_margin" and r.q_max == 0}
    entr = {r.sample_id: r for r in rows if r.method == "entropy" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == q}

    ids = sorted(set(br).intersection(conf).intersection(negm).intersection(entr).intersection(lm))
    if not ids:
        return {}

    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    q_used = np.array([br[i].q_used for i in ids], dtype=np.int64)
    cens = np.array([max(br[i].censored, lm[i].censored) for i in ids], dtype=np.float64)

    f = {
        "ids": np.array(ids, dtype=np.int64),
        "y": y,
        "q_used": q_used,
        "cens": cens,
        "beacon": _rank_norm(np.array([br[i].risk_score for i in ids], dtype=np.float64)),
        "conf": _rank_norm(np.array([conf[i].risk_score for i in ids], dtype=np.float64)),
        "neg_margin": _rank_norm(np.array([negm[i].risk_score for i in ids], dtype=np.float64)),
        "entropy": _rank_norm(np.array([entr[i].risk_score for i in ids], dtype=np.float64)),
        "rho": _rank_norm(np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64)),
        "nec": _rank_norm(np.array([lm[i].necessity for i in ids], dtype=np.float64)),
        "ce": _rank_norm(np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64)),
        "suff_bad": _rank_norm(np.array([-lm[i].sufficiency_margin for i in ids], dtype=np.float64)),
    }
    return f


def _fit_weights_random_search(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_auc = -1.0
    best_w = np.ones(X.shape[1], dtype=np.float64)
    # include simple baseline: weight only on first feature
    cands = [np.concatenate([[1.0], np.zeros(X.shape[1] - 1)])]
    cands.extend(rng.uniform(-2.0, 2.0, size=(n_trials, X.shape[1])))
    for w in cands:
        a = _auc(y, X @ w)
        if np.isfinite(a) and a > best_auc:
            best_auc = a
            best_w = np.asarray(w, dtype=np.float64)
    return best_w


def _bootstrap_delta_auc(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ya = y[idx]
        da = _auc(ya, score_a[idx])
        db = _auc(ya, score_b[idx])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
    if not deltas:
        return {"delta_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "frac_positive": float("nan")}
    d = np.array(deltas, dtype=np.float64)
    return {
        "delta_mean": float(np.mean(d)),
        "ci_low": float(np.quantile(d, 0.025)),
        "ci_high": float(np.quantile(d, 0.975)),
        "frac_positive": float(np.mean(d > 0.0)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR study: q-sweep, ablation, bootstrap, censored analysis")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=1024)
    p.add_argument("--q-values", default="16,32,64")
    p.add_argument("--k0", type=int, default=8)
    p.add_argument("--n-calib-trials", type=int, default=25000)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--cnn-epochs", type=int, default=25)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default="./outputs_composite/har_study")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]

    x_train_full, y_train_full, x_test_full, y_test_full = load_uci_har(args.dataset_root)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]
    x_te = x_test_full
    y_te = y_test_full

    if args.max_test > 0 and args.max_test < len(x_te):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(x_te), size=args.max_test, replace=False)
        x_te = x_te[idx]
        y_te = y_te[idx]

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_va = apply_standardizer(x_va, mu, sigma)
    x_te = apply_standardizer(x_te, mu, sigma)

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
        k0=args.k0,
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
    methods = {"confidence", "entropy", "negative_margin", "beacon_refine", "beacon_flat", "uniform_refinement"}

    rows_val, local_val, _ = evaluate_error_risk(
        x_test=x_va,
        y_test=y_va,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=base_cfg,
        q_values=q_values,
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        composite_weights=None,
        methods=methods,
    )
    rows_test, local_test, metrics_test = evaluate_error_risk(
        x_test=x_te,
        y_test=y_te,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=base_cfg,
        q_values=q_values,
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        composite_weights=None,
        methods=methods,
    )

    q_sweep_rows: list[dict] = [m for m in metrics_test]
    ablation_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    censored_rows: list[dict] = []
    calib_weights_rows: list[dict] = []

    for q in q_values:
        f_val = _collect_frame(rows_val, local_val, q)
        f_te = _collect_frame(rows_test, local_test, q)
        if not f_val or not f_te:
            continue

        yv = f_val["y"]
        yt = f_te["y"]

        X_val_margin_rho = np.stack([f_val["neg_margin"], f_val["rho"]], axis=1)
        X_te_margin_rho = np.stack([f_te["neg_margin"], f_te["rho"]], axis=1)
        w_mr = _fit_weights_random_search(X_val_margin_rho, yv, args.n_calib_trials, args.seed + q + 1)
        s_mr = X_te_margin_rho @ w_mr

        X_val_margin_ce = np.stack([f_val["neg_margin"], f_val["ce"]], axis=1)
        X_te_margin_ce = np.stack([f_te["neg_margin"], f_te["ce"]], axis=1)
        w_mc = _fit_weights_random_search(X_val_margin_ce, yv, args.n_calib_trials, args.seed + q + 2)
        s_mc = X_te_margin_ce @ w_mc

        X_val_full = np.stack(
            [
                f_val["beacon"],
                f_val["conf"],
                f_val["neg_margin"],
                f_val["rho"],
                f_val["nec"],
                f_val["ce"],
                f_val["suff_bad"],
                f_val["cens"],
            ],
            axis=1,
        )
        X_te_full = np.stack(
            [
                f_te["beacon"],
                f_te["conf"],
                f_te["neg_margin"],
                f_te["rho"],
                f_te["nec"],
                f_te["ce"],
                f_te["suff_bad"],
                f_te["cens"],
            ],
            axis=1,
        )
        w_full = _fit_weights_random_search(X_val_full, yv, args.n_calib_trials, args.seed + q + 3)
        s_full = X_te_full @ w_full

        s_nm = f_te["neg_margin"]

        variants = [
            ("negative_margin", s_nm, None),
            ("margin_plus_rho", s_mr, w_mr),
            ("margin_plus_ce", s_mc, w_mc),
            ("full_beacon_composite", s_full, w_full),
        ]
        for name, score, w in variants:
            row = {
                "method": name,
                "q_max": q,
                "auroc": _auc(yt, score),
                "auprc": _auprc(yt, score),
                "delta_vs_negative_margin": _auc(yt, score) - _auc(yt, s_nm),
            }
            if w is not None:
                row["weights"] = json.dumps([float(v) for v in w])
            ablation_rows.append(row)

        q_sweep_rows.append(
            {
                "method": "beacon_composite_calibrated",
                "q_max": float(q),
                "auroc": _auc(yt, s_full),
                "auprc": _auprc(yt, s_full),
                "mean_q_used": float(np.mean(f_te["q_used"])),
                "censored_rate": float(np.mean(f_te["cens"])),
            }
        )

        b = _bootstrap_delta_auc(yt, s_full, s_nm, args.n_bootstrap, args.seed + q + 100)
        bootstrap_rows.append(
            {
                "q_max": q,
                "delta_auc_mean": b["delta_mean"],
                "ci_low": b["ci_low"],
                "ci_high": b["ci_high"],
                "frac_positive": b["frac_positive"],
                "composite_auroc": _auc(yt, s_full),
                "negative_margin_auroc": _auc(yt, s_nm),
            }
        )

        correct = yt == 0
        incorrect = yt == 1
        for label, mask in [("correct", correct), ("incorrect", incorrect)]:
            if int(np.sum(mask)) == 0:
                continue
            censored_rows.append(
                {
                    "q_max": q,
                    "group": label,
                    "n": int(np.sum(mask)),
                    "censored_rate": float(np.mean(f_te["cens"][mask])),
                    "mean_rho": float(np.mean(f_te["rho"][mask])),
                    "mean_q_used": float(np.mean(f_te["q_used"][mask])),
                }
            )

        calib_weights_rows.append(
            {
                "q_max": q,
                "variant": "margin_plus_rho",
                "weights": json.dumps([float(v) for v in w_mr]),
            }
        )
        calib_weights_rows.append(
            {
                "q_max": q,
                "variant": "margin_plus_ce",
                "weights": json.dumps([float(v) for v in w_mc]),
            }
        )
        calib_weights_rows.append(
            {
                "q_max": q,
                "variant": "full_beacon_composite",
                "weights": json.dumps([float(v) for v in w_full]),
            }
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "har_q_sweep.csv", q_sweep_rows)
    _write_csv(out / "har_composite_ablation.csv", ablation_rows)
    _write_csv(out / "har_bootstrap_delta.csv", bootstrap_rows)
    _write_csv(out / "har_censored_analysis.csv", censored_rows)
    _write_csv(out / "har_calibrated_weights.csv", calib_weights_rows)

    print("Saved:")
    print(out / "har_q_sweep.csv")
    print(out / "har_composite_ablation.csv")
    print(out / "har_bootstrap_delta.csv")
    print(out / "har_censored_analysis.csv")
    print(out / "har_calibrated_weights.csv")


if __name__ == "__main__":
    main()
