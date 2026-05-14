#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
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
    d = int(X.shape[1])
    best_w = np.zeros(d, dtype=np.float64)
    best_w[0] = 1.0
    cands = [best_w]
    cands.extend(rng.uniform(-2.0, 2.0, size=(n_trials, d)))
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
) -> tuple[float, float, float]:
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
        return float("nan"), float("nan"), float("nan")
    d = np.asarray(deltas, dtype=np.float64)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float(np.mean(d > 0.0))


def _collect_features(
    rows: list[RiskEvalRow],
    local_rows: list[LocalMetricRow],
    q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == q}
    nm = {r.sample_id: r for r in rows if r.method == "negative_margin" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == q}
    ids = sorted(set(br).intersection(nm).intersection(lm))
    if not ids:
        return None
    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    neg_margin = _rank_norm(np.array([nm[i].risk_score for i in ids], dtype=np.float64))
    ce = _rank_norm(np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64))
    rho_cost = _rank_norm(np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64))
    counter_mass = _rank_norm(np.array([lm[i].counter_mass for i in ids], dtype=np.float64))
    return y, neg_margin, ce, rho_cost, counter_mass


def _collect_baseline(rows: list[RiskEvalRow]) -> tuple[np.ndarray, np.ndarray]:
    nm = sorted((r for r in rows if r.method == "negative_margin" and r.q_max == 0), key=lambda z: z.sample_id)
    y = np.array([r.is_error for r in nm], dtype=np.int64)
    s = _rank_norm(np.array([r.risk_score for r in nm], dtype=np.float64))
    return y, s


def _append_row(
    out_rows: list[dict],
    refinement_mode: str,
    q_max: int,
    method: str,
    y: np.ndarray,
    score: np.ndarray,
    score_baseline: np.ndarray,
    n_boot: int,
    seed: int,
) -> None:
    auroc = _auc(y, score)
    auprc = _auprc(y, score)
    auroc_b = _auc(y, score_baseline)
    auprc_b = _auprc(y, score_baseline)
    p5, _ = _precision_recall_at_frac(y, score, 0.05)
    p10, r10 = _precision_recall_at_frac(y, score, 0.10)
    if method == "negative_margin":
        ci_low, ci_high, frac_pos = 0.0, 0.0, 0.0
    else:
        ci_low, ci_high, frac_pos = _bootstrap_delta(y, score, score_baseline, n_boot, seed)
    out_rows.append(
        {
            "refinement_mode": refinement_mode,
            "q_max": q_max,
            "method": method,
            "auroc": auroc,
            "auprc": auprc,
            "precision_at_5pct": p5,
            "precision_at_10pct": p10,
            "recall_at_10pct": r10,
            "delta_auroc": auroc - auroc_b,
            "delta_auprc": auprc - auprc_b,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "frac_positive": frac_pos,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAMAP2 strict CE q-sweep")
    p.add_argument("--npz-path", default="./data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--model", choices=["histgbt", "extratrees", "cnn1d"], default="histgbt")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--q-values", default="8,16,32,64")
    p.add_argument("--k0", type=int, default=8)
    p.add_argument("--n-calib-trials", type=int, default=8000)
    p.add_argument("--n-bootstrap", type=int, default=600)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--cnn-epochs", type=int, default=16)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out", default="./outputs_composite/pamap2_ce_q_sweep.csv")
    return p.parse_args()


def main() -> None:
    def _log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

    def _log_last() -> None:
        r = out_rows[-1]
        _log(
            f"result q={r['q_max']} {r['method']} ({r['refinement_mode']}): "
            f"auroc={r['auroc']:.4f} auprc={r['auprc']:.4f} d_auc={r['delta_auroc']:.4f}"
        )

    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]
    _log(f"start model={args.model} q_values={q_values} k0={args.k0} max_val={args.max_val} max_test={args.max_test}")
    if args.model == "cnn1d":
        _log(f"epochs={args.cnn_epochs} batch={args.cnn_batch_size} lr={args.cnn_lr}")
    else:
        _log("epochs=N/A for tree boosting models")
    x_train_full, y_train_full, x_test, y_test = load_npz_dataset(args.npz_path)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if args.max_val > 0 and args.max_val < len(x_va):
        rng = np.random.default_rng(args.seed + 5)
        idx = rng.choice(len(x_va), size=args.max_val, replace=False)
        x_va = x_va[idx]
        y_va = y_va[idx]

    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed + 7)
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]
    _log(
        f"split sizes: train={len(x_tr)} val={len(x_va)} test={len(x_test)} "
        f"classes_train={len(np.unique(y_tr))}"
    )

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_va = apply_standardizer(x_va, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "histgbt":
        _log("training histgbt...")
        clf = train_histgbt_stats(
            x_tr, y_tr, max_iter=220, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=20
        )
    elif args.model == "extratrees":
        _log("training extratrees...")
        clf = train_extratrees_stats(x_tr, y_tr, n_estimators=1000, max_features=0.7, min_samples_leaf=1)
    else:
        _log("training cnn1d...")
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
    _log("model training done")

    _log("computing tau_m from train margins...")
    train_margins = []
    n_tau = min(len(x_tr), 2000)
    for i in range(n_tau):
        lg = clf.logits(x_tr[i])
        y_hat = int(np.argmax(lg))
        m = float(lg[y_hat] - np.max(np.delete(lg, y_hat)))
        if m > 0:
            train_margins.append(m)
        if (i + 1) % 500 == 0 or (i + 1) == n_tau:
            _log(f"tau_m progress: {i + 1}/{n_tau}")
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0
    _log(f"tau_m computed: {tau_m:.6f} from {len(train_margins)} positive margins")

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

    modes = {}
    for mode in ("support", "mixed"):
        _log(f"evaluate mode={mode} on val/test ...")
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
        modes[mode] = (rows_val, local_val, rows_test, local_test)
        _log(
            f"mode={mode} evaluated: val_rows={len(rows_val)} val_local={len(local_val)} "
            f"test_rows={len(rows_test)} test_local={len(local_test)}"
        )

    y_base, s_base = _collect_baseline(modes["mixed"][2])
    out_rows: list[dict] = []

    for q in q_values:
        _log(f"processing q={q}")
        _append_row(out_rows, "baseline", q, "negative_margin", y_base, s_base, s_base, args.n_bootstrap, args.seed + q)
        _log_last()

        if q <= args.k0:
            _append_row(out_rows, "support", q, "support+CE", y_base, s_base, s_base, args.n_bootstrap, args.seed + 100 + q)
            _log_last()
            _append_row(
                out_rows, "support", q, "support+counter_mass", y_base, s_base, s_base, args.n_bootstrap, args.seed + 110 + q
            )
            _log_last()
            _append_row(out_rows, "mixed", q, "mixed+CE", y_base, s_base, s_base, args.n_bootstrap, args.seed + 200 + q)
            _log_last()
            _append_row(
                out_rows, "mixed", q, "mixed+counter_mass", y_base, s_base, s_base, args.n_bootstrap, args.seed + 210 + q
            )
            _log_last()
            _append_row(
                out_rows, "mixed", q, "mixed+rho_cost", y_base, s_base, s_base, args.n_bootstrap, args.seed + 300 + q
            )
            _log_last()
            _append_row(
                out_rows,
                "mixed",
                q,
                "mixed+CE+counter_mass",
                y_base,
                s_base,
                s_base,
                args.n_bootstrap,
                args.seed + 410 + q,
            )
            _log_last()
            continue

        s_val, l_val, s_test, l_test = modes["support"]
        m_val, ml_val, m_test, ml_test = modes["mixed"]
        sup_v = _collect_features(s_val, l_val, q)
        sup_t = _collect_features(s_test, l_test, q)
        mix_v = _collect_features(m_val, ml_val, q)
        mix_t = _collect_features(m_test, ml_test, q)
        if sup_v is None or sup_t is None or mix_v is None or mix_t is None:
            continue

        yv_s, nm_v_s, ce_v_s, _, cm_v_s = sup_v
        yt_s, nm_t_s, ce_t_s, _, cm_t_s = sup_t
        yv_m, nm_v_m, ce_v_m, rho_v_m, cm_v_m = mix_v
        yt_m, nm_t_m, ce_t_m, rho_t_m, cm_t_m = mix_t

        ws = _fit_weights(np.stack([nm_v_s, ce_v_s], axis=1), yv_s, args.n_calib_trials, args.seed + q + 11)
        ss = np.stack([nm_t_s, ce_t_s], axis=1) @ ws
        ws_cm = _fit_weights(np.stack([nm_v_s, cm_v_s], axis=1), yv_s, args.n_calib_trials, args.seed + q + 15)
        ss_cm = np.stack([nm_t_s, cm_t_s], axis=1) @ ws_cm
        wm_ce = _fit_weights(np.stack([nm_v_m, ce_v_m], axis=1), yv_m, args.n_calib_trials, args.seed + q + 21)
        sm_ce = np.stack([nm_t_m, ce_t_m], axis=1) @ wm_ce
        wm_cm = _fit_weights(np.stack([nm_v_m, cm_v_m], axis=1), yv_m, args.n_calib_trials, args.seed + q + 25)
        sm_cm = np.stack([nm_t_m, cm_t_m], axis=1) @ wm_cm
        wm_rho = _fit_weights(np.stack([nm_v_m, rho_v_m], axis=1), yv_m, args.n_calib_trials, args.seed + q + 31)
        sm_rho = np.stack([nm_t_m, rho_t_m], axis=1) @ wm_rho
        wm_ce_cm = _fit_weights(
            np.stack([nm_v_m, ce_v_m, cm_v_m], axis=1), yv_m, args.n_calib_trials, args.seed + q + 41
        )
        sm_ce_cm = np.stack([nm_t_m, ce_t_m, cm_t_m], axis=1) @ wm_ce_cm

        _append_row(out_rows, "support", q, "support+CE", yt_s, ss, nm_t_s, args.n_bootstrap, args.seed + q + 111)
        _log_last()
        _append_row(
            out_rows, "support", q, "support+counter_mass", yt_s, ss_cm, nm_t_s, args.n_bootstrap, args.seed + q + 151
        )
        _log_last()
        _append_row(out_rows, "mixed", q, "mixed+CE", yt_m, sm_ce, nm_t_m, args.n_bootstrap, args.seed + q + 211)
        _log_last()
        _append_row(
            out_rows, "mixed", q, "mixed+counter_mass", yt_m, sm_cm, nm_t_m, args.n_bootstrap, args.seed + q + 251
        )
        _log_last()
        _append_row(out_rows, "mixed", q, "mixed+rho_cost", yt_m, sm_rho, nm_t_m, args.n_bootstrap, args.seed + q + 311)
        _log_last()
        _append_row(
            out_rows,
            "mixed",
            q,
            "mixed+CE+counter_mass",
            yt_m,
            sm_ce_cm,
            nm_t_m,
            args.n_bootstrap,
            args.seed + q + 411,
        )
        _log_last()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "refinement_mode",
                "q_max",
                "method",
                "auroc",
                "auprc",
                "precision_at_5pct",
                "precision_at_10pct",
                "recall_at_10pct",
                "delta_auroc",
                "delta_auprc",
                "ci_low",
                "ci_high",
                "frac_positive",
            ],
        )
        wr.writeheader()
        wr.writerows(out_rows)
    _log(f"Saved: {out}")


if __name__ == "__main__":
    main()
