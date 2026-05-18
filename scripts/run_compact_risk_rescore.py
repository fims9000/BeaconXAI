#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import (
    apply_standardizer,
    fit_channel_standardizer,
    load_npz_dataset,
    load_uci_har,
)
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn, train_extratrees_stats
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


def _rank_norm(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(ranks)
    return ranks / (len(x) - 1)


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


def _collect_features(
    rows: list[RiskEvalRow],
    local_rows: list[LocalMetricRow],
    q: int,
) -> dict[str, np.ndarray] | None:
    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == q}
    nm = {r.sample_id: r for r in rows if r.method == "negative_margin" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == q}
    ids = sorted(set(br).intersection(nm).intersection(lm))
    if not ids:
        return None
    eps = 1e-8
    y = np.array([br[i].is_error for i in ids], dtype=np.int64)
    neg_margin = np.array([nm[i].risk_score for i in ids], dtype=np.float64)
    ce = np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64)
    counter_mass = np.array([lm[i].counter_mass for i in ids], dtype=np.float64)
    support_mass = np.array([lm[i].support_mass for i in ids], dtype=np.float64)
    rho_cost = np.array([lm[i].rho_b_cost for i in ids], dtype=np.float64)
    m0 = np.array([lm[i].m0 for i in ids], dtype=np.float64)
    m_last = np.array([lm[i].m_last for i in ids], dtype=np.float64)
    conflict_ratio = counter_mass / np.maximum(counter_mass + support_mass, eps)
    frag_drop = np.maximum(0.0, m0 - m_last)
    frag_residual = m_last / (np.abs(m0) + eps)
    data = {
        "y": y,
        "neg_margin": _rank_norm(neg_margin),
        "ce": _rank_norm(ce),
        "counter_mass": _rank_norm(counter_mass),
        "conflict_ratio": _rank_norm(conflict_ratio),
        "rho_cost": _rank_norm(rho_cost),
        "frag_drop": _rank_norm(frag_drop),
        "frag_residual": _rank_norm(frag_residual),
        "q_used": np.array([br[i].q_used for i in ids], dtype=np.float64),
        "censored": np.array([br[i].censored for i in ids], dtype=np.float64),
    }
    return data


def _fit_and_score(
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    xv = scaler.fit_transform(x_val)
    xt = scaler.transform(x_test)
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
        max_iter=1000,
    )
    clf.fit(xv, y_val)
    return clf.predict_proba(xt)[:, 1]


def _append_eval(
    out_rows: list[dict],
    dataset: str,
    model: str,
    q: int,
    method: str,
    y: np.ndarray,
    score: np.ndarray,
    score_base: np.ndarray,
    p10_base: float,
    r10_base: float,
    mean_q_used: float,
    latency_per_query: float,
    n_boot: int,
    seed: int,
) -> None:
    auroc = _auc(y, score)
    auprc = _auprc(y, score)
    auroc_b = _auc(y, score_base)
    auprc_b = _auprc(y, score_base)
    p10, r10 = _precision_recall_at_frac(y, score, 0.10)
    delta_p10 = float(p10 - p10_base)
    delta_r10 = float(r10 - r10_base)
    mean_q_used = float(max(mean_q_used, 1.0))
    qntg_p10 = float(delta_p10 / mean_q_used)
    if latency_per_query > 0:
        latency_per_object = float(mean_q_used * latency_per_query)
        lntg_p10 = float(delta_p10 / max(latency_per_object, 1e-12))
    else:
        latency_per_object = float("nan")
        lntg_p10 = float("nan")
    if method == "negative_margin":
        ci_low, ci_high, frac_positive = 0.0, 0.0, 0.0
    else:
        ci_low, ci_high, frac_positive = _bootstrap_delta(y, score, score_base, n_boot, seed)
    out_rows.append(
        {
            "dataset": dataset,
            "model": model,
            "q_max": q,
            "method": method,
            "auroc": auroc,
            "auprc": auprc,
            "precision_at_10pct": p10,
            "recall_at_10pct": r10,
            "delta_p10": delta_p10,
            "delta_r10": delta_r10,
            "qntg_p10": qntg_p10,
            "lntg_p10": lntg_p10,
            "mean_q_used": mean_q_used,
            "latency_per_object": latency_per_object,
            "delta_auroc": auroc - auroc_b,
            "delta_auprc": auprc - auprc_b,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "frac_positive": frac_positive,
        }
    )


def _run_dataset(
    dataset: str,
    model: str,
    q_values: list[int],
    seed: int,
    n_bootstrap: int,
    val_frac: float,
    max_val: int,
    max_test: int,
    pamap_npz: str,
    har_root: str,
    cnn_epochs: int,
    cnn_batch_size: int,
    latency_per_query: float,
    priority_mode: str,
    switch_eta: float,
    budget_mode: str,
    tau_conflict: float,
    margin_mode: str,
    audit_mode: str,
    partition_mode: str,
) -> list[dict]:
    print(f"[rescore] dataset={dataset} model={model} q={q_values}", flush=True)
    if dataset == "har":
        x_train_full, y_train_full, x_test, y_test = load_uci_har(har_root)
    else:
        x_train_full, y_train_full, x_test, y_test = load_npz_dataset(pamap_npz)

    tr_idx, va_idx = _stratified_split_idx(y_train_full, val_frac, seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if max_val > 0 and max_val < len(x_va):
        rng = np.random.default_rng(seed + 10)
        idx = rng.choice(len(x_va), size=max_val, replace=False)
        x_va = x_va[idx]
        y_va = y_va[idx]
    if max_test > 0 and max_test < len(x_test):
        rng = np.random.default_rng(seed + 20)
        idx = rng.choice(len(x_test), size=max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_tr)
    x_tr = apply_standardizer(x_tr, mu, sigma)
    x_va = apply_standardizer(x_va, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if model == "cnn1d":
        print(f"[rescore] train cnn epochs={cnn_epochs} batch={cnn_batch_size}", flush=True)
        tta = (0, 64) if dataset == "har" else (0, 50)
        clf = train_1dcnn(
            x_tr,
            y_tr,
            epochs=cnn_epochs,
            batch_size=cnn_batch_size,
            lr=1e-3,
            label_smoothing=0.0,
            use_class_weights=True,
            tta_shifts=tta,
        )
    elif model == "extratrees":
        print("[rescore] train extratrees", flush=True)
        clf = train_extratrees_stats(
            x_tr,
            y_tr,
            n_estimators=1000,
            max_features=0.7,
            min_samples_leaf=1,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    train_margins = []
    for i in range(min(len(x_tr), 2000)):
        lg = clf.logits(x_tr[i])
        y_hat = int(np.argmax(lg))
        m = float(lg[y_hat] - np.max(np.delete(lg, y_hat)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0
    neutralizer = Neutralizer(mode="zero", channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))
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
        refinement_mode="mixed",
        priority_mode=priority_mode,
        switch_eta=switch_eta,
        budget_mode=budget_mode,
        tau_conflict=tau_conflict,
        margin_mode=margin_mode,
        audit_mode=audit_mode,
        partition_mode=partition_mode,
        risk_policy="rho_only",
    )

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
        methods={"negative_margin", "beacon_refine"},
    )
    rows_test, local_test, _ = evaluate_error_risk(
        x_test=x_test,
        y_test=y_test,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=base_cfg,
        q_values=q_values,
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        composite_weights=None,
        methods={"negative_margin", "beacon_refine"},
    )

    out_rows: list[dict] = []
    print("[rescore] beacon features collected, start compact rescoring", flush=True)
    methods = {
        "margin + CE_B": ("neg_margin", "ce"),
        "margin + counter_mass": ("neg_margin", "counter_mass"),
        "margin + conflict_ratio": ("neg_margin", "conflict_ratio"),
        "margin + CE_B + counter_mass": ("neg_margin", "ce", "counter_mass"),
        "margin + counter_mass + conflict_ratio": ("neg_margin", "counter_mass", "conflict_ratio"),
        "margin + counter_mass + rho_B_cost": ("neg_margin", "counter_mass", "rho_cost"),
        "margin + counter_mass + conflict_ratio + rho_B_cost": (
            "neg_margin",
            "counter_mass",
            "conflict_ratio",
            "rho_cost",
        ),
        "margin + frag_drop": ("neg_margin", "frag_drop"),
        "margin + frag_residual": ("neg_margin", "frag_residual"),
        "margin + counter_mass + frag_drop": ("neg_margin", "counter_mass", "frag_drop"),
        "margin + conflict_ratio + frag_drop": ("neg_margin", "conflict_ratio", "frag_drop"),
    }

    for q in q_values:
        fv = _collect_features(rows_val, local_val, q)
        ft = _collect_features(rows_test, local_test, q)
        if fv is None or ft is None:
            continue
        yv = fv["y"]
        yt = ft["y"]
        base_v = fv["neg_margin"]
        base_t = ft["neg_margin"]
        p10_base, r10_base = _precision_recall_at_frac(yt, base_t, 0.10)
        _append_eval(
            out_rows,
            dataset,
            model,
            q,
            "negative_margin",
            yt,
            base_t,
            base_t,
            p10_base,
            r10_base,
            1.0,
            latency_per_query,
            n_bootstrap,
            seed + q,
        )
        mean_q_used = float(np.mean(ft["q_used"]))
        for name, keys in methods.items():
            xv = np.stack([fv[k] for k in keys], axis=1)
            xt = np.stack([ft[k] for k in keys], axis=1)
            score_t = _fit_and_score(xv, yv, xt, seed + 100 + q)
            _append_eval(
                out_rows,
                dataset,
                model,
                q,
                name,
                yt,
                score_t,
                base_t,
                p10_base,
                r10_base,
                mean_q_used,
                latency_per_query,
                n_bootstrap,
                seed + 200 + q,
            )
        print(f"[rescore] q={q} done", flush=True)
    return out_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact risk rescoring with counter-mass/conflict features")
    p.add_argument("--dataset", choices=["har", "pamap2"], required=True)
    p.add_argument("--model", choices=["cnn1d", "extratrees"], required=True)
    p.add_argument("--q-values", default="16,32,64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--n-bootstrap", type=int, default=400)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--pamap-npz", default="./data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--har-root", default="./data")
    p.add_argument("--cnn-epochs", type=int, default=25)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--latency-per-query", type=float, default=-1.0)
    p.add_argument("--priority-mode", choices=["base", "switch"], default="base")
    p.add_argument("--switch-eta", type=float, default=0.0)
    p.add_argument("--budget-mode", choices=["fixed", "conflict_first"], default="fixed")
    p.add_argument("--tau-conflict", type=float, default=0.0)
    p.add_argument("--margin-mode", choices=["adaptive_all", "nearest_competitor"], default="adaptive_all")
    p.add_argument("--audit-mode", choices=["full", "counter_only"], default="full")
    p.add_argument(
        "--partition-mode",
        choices=["time_only", "time_channel", "channel_time", "sensor_group_time", "fuzzy_chunks"],
        default="time_only",
    )
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]
    rows = _run_dataset(
        dataset=args.dataset,
        model=args.model,
        q_values=q_values,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        val_frac=args.val_frac,
        max_val=args.max_val,
        max_test=args.max_test,
        pamap_npz=args.pamap_npz,
        har_root=args.har_root,
        cnn_epochs=args.cnn_epochs,
        cnn_batch_size=args.cnn_batch_size,
        latency_per_query=args.latency_per_query,
        priority_mode=args.priority_mode,
        switch_eta=args.switch_eta,
        budget_mode=args.budget_mode,
        tau_conflict=args.tau_conflict,
        margin_mode=args.margin_mode,
        audit_mode=args.audit_mode,
        partition_mode=args.partition_mode,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "q_max",
                "method",
                "auroc",
                "auprc",
                "precision_at_10pct",
                "recall_at_10pct",
                "delta_p10",
                "delta_r10",
                "qntg_p10",
                "lntg_p10",
                "mean_q_used",
                "latency_per_object",
                "delta_auroc",
                "delta_auprc",
                "ci_low",
                "ci_high",
                "frac_positive",
            ],
        )
        wr.writeheader()
        wr.writerows(rows)
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()
