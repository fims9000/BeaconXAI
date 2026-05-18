#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.core import BeaconAudit
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_uci_har
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn
from beaconxai.neutralization import Neutralizer
from beaconxai.partition import make_initial_partition_time
from beaconxai.types import BeaconConfig, LocalMetricRow, RiskEvalRow


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


def _p10_r10(y: np.ndarray, s: np.ndarray) -> tuple[float, float, np.ndarray]:
    k = max(1, int(np.ceil(0.10 * len(y))))
    idx = np.argsort(-s)[:k]
    p10 = float(np.mean(y[idx] == 1))
    den = float(np.sum(y == 1))
    r10 = float(np.sum(y[idx] == 1) / den) if den > 0 else float("nan")
    return p10, r10, idx


def _bootstrap_delta(y: np.ndarray, s: np.ndarray, b: np.ndarray, metric_fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ds = metric_fn(y[idx], s[idx])
        db = metric_fn(y[idx], b[idx])
        if np.isfinite(ds) and np.isfinite(db):
            d.append(ds - db)
    if not d:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(d, dtype=np.float64)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(np.mean(arr > 0.0))


def _fit_lr_score(yv: np.ndarray, Xv: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
    sc = StandardScaler()
    Xv2 = sc.fit_transform(Xv)
    Xt2 = sc.transform(Xt)
    clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=seed, solver="lbfgs", max_iter=1200)
    clf.fit(Xv2, yv)
    return clf.predict_proba(Xt2)[:, 1]


def _collect_rows(rows: list[RiskEvalRow], local_rows: list[LocalMetricRow], q: int):
    br = {r.sample_id: r for r in rows if r.method == "beacon_refine" and r.q_max == q}
    ent = {r.sample_id: r for r in rows if r.method == "entropy" and r.q_max == 0}
    lm = {r.sample_id: r for r in local_rows if r.method == "beacon_refine" and r.q_max == q}
    ids = sorted(set(br).intersection(ent).intersection(lm))
    if not ids:
        return None
    return {
        "ids": np.array(ids, dtype=np.int64),
        "y": np.array([br[i].is_error for i in ids], dtype=np.int64),
        "entropy": np.array([ent[i].risk_score for i in ids], dtype=np.float64),
        "counter_mass": np.array([lm[i].counter_mass for i in ids], dtype=np.float64),
        "q_used": np.array([br[i].q_used for i in ids], dtype=np.float64),
    }


def _fit_excess_map(ent: np.ndarray, cm: np.ndarray, pred: np.ndarray):
    stats: dict[int, dict[str, np.ndarray | float | None]] = {}
    gmu = float(np.mean(cm))
    gsd = float(np.std(cm) + 1e-8)
    for c in np.unique(pred):
        m = pred == c
        e = ent[m]
        x = cm[m]
        if len(e) < 40:
            stats[int(c)] = {"q": None, "mu": gmu, "sd": gsd}
            continue
        q = np.quantile(e, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        q[0] -= 1e-12
        q[-1] += 1e-12
        mu = []
        sd = []
        for i in range(5):
            b = (e > q[i]) & (e <= q[i + 1])
            if np.sum(b) < 5:
                mu.append(float(np.mean(x)))
                sd.append(float(np.std(x) + 1e-8))
            else:
                mu.append(float(np.mean(x[b])))
                sd.append(float(np.std(x[b]) + 1e-8))
        stats[int(c)] = {"q": q, "mu": np.array(mu, dtype=np.float64), "sd": np.array(sd, dtype=np.float64)}
    return stats


def _apply_excess_map(ent: np.ndarray, cm: np.ndarray, pred: np.ndarray, stats):
    gmu = float(np.mean(cm))
    gsd = float(np.std(cm) + 1e-8)
    mu = np.zeros_like(cm, dtype=np.float64)
    sd = np.ones_like(cm, dtype=np.float64)
    for i, (e, c) in enumerate(zip(ent, pred)):
        st = stats.get(int(c))
        if st is None or st["q"] is None:
            mu[i] = gmu
            sd[i] = gsd
            continue
        q = st["q"]
        bi = np.searchsorted(q, e, side="right") - 1
        bi = max(0, min(4, bi))
        mu[i] = float(st["mu"][bi])
        sd[i] = max(float(st["sd"][bi]), 1e-8)
    ex = cm - mu
    exz = ex / sd
    return ex, exz


def _counter_prob_mass_k0(x: np.ndarray, clf, neutralizer: Neutralizer, k0: int) -> float:
    lg0 = clf.logits(x)
    y_hat = int(np.argmax(lg0))
    z0 = lg0 - np.max(lg0)
    p0 = np.exp(z0)
    p0 = p0 / np.sum(p0)
    c = int(np.argmax(np.where(np.arange(len(lg0)) == y_hat, -np.inf, lg0)))
    base = float(p0[c])

    comps = make_initial_partition_time(x.shape[0], x.shape[1], k0)
    total = 0.0
    for g in comps:
        xn = neutralizer(x, [g])
        lgn = clf.logits(xn)
        zn = lgn - np.max(lgn)
        pn = np.exp(zn)
        pn = pn / np.sum(pn)
        dp = float(pn[c] - base)
        if dp > 0:
            total += dp
    return float(total)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR full Q16: entropy vs C_excess")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--q-max", type=int, default=16)
    p.add_argument("--k0", type=int, default=8)
    p.add_argument(
        "--partition-mode",
        default="time_only",
        choices=["time_only", "time_channel", "channel_time", "sensor_group_time", "fuzzy_chunks"],
    )
    p.add_argument("--neutralizer", default="zero", choices=["zero", "mean", "interp"])
    p.add_argument("--cnn-epochs", type=int, default=8)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--latency-per-query", type=float, default=0.0010728925)
    p.add_argument("--out", default="./outputs_composite/har_q16_full_entropy_excess.csv")
    p.add_argument("--out-sanity", default="./outputs_composite/har_q16_full_entropy_excess_sanity.json")
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

    # Strict isolation: fit only on train split
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
        q_max=q,
        k0=int(args.k0),
        l_min=4,
        k_pos=3,
        k_neg=3,
        q_frag_ratio=0.0,  # enforce refinement budget for Q=16, K0=8 -> 8 refinement queries
        alpha=1.0,
        beta=0.5,
        gamma=1.0,
        tau_s=0.10,
        tau_m=tau_m,
        refinement_mode="mixed",
        partition_mode=str(args.partition_mode),
        risk_policy="rho_only",
        margin_mode="adaptive_all",
        audit_mode="full",
    )

    if args.neutralizer == "mean":
        neutralizer = Neutralizer(mode="mean", channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))
    else:
        neutralizer = Neutralizer(mode=args.neutralizer, channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))

    methods = {"entropy", "beacon_refine"}

    rows_va, local_va, _ = evaluate_error_risk(
        x_test=x_va,
        y_test=y_va,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=[q],
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        methods=methods,
    )
    rows_te, local_te, _ = evaluate_error_risk(
        x_test=x_test,
        y_test=y_test,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=[q],
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        methods=methods,
    )

    dv = _collect_rows(rows_va, local_va, q)
    dt = _collect_rows(rows_te, local_te, q)
    if dv is None or dt is None:
        raise RuntimeError("Failed to collect aligned rows")

    ids_v = dv["ids"]
    ids_t = dt["ids"]
    yv = dv["y"]
    yt = dt["y"]
    ent_v = dv["entropy"]
    ent_t = dt["entropy"]
    cm_v = dv["counter_mass"]
    cm_t = dt["counter_mass"]
    q_used_t = dt["q_used"]

    pred_v = np.array([clf.predict(x_va[i]) for i in ids_v], dtype=np.int64)
    pred_t = np.array([clf.predict(x_test[i]) for i in ids_t], dtype=np.int64)

    stats_cm = _fit_excess_map(ent_v, cm_v, pred_v)
    ex_v, exz_v = _apply_excess_map(ent_v, cm_v, pred_v, stats_cm)
    ex_t, exz_t = _apply_excess_map(ent_t, cm_t, pred_t, stats_cm)

    # Optional probability-based conflict mass (k0 only) for contrastive C_excess_prob
    cm_prob_v = np.array([_counter_prob_mass_k0(x_va[i], clf, neutralizer, int(args.k0)) for i in ids_v], dtype=np.float64)
    cm_prob_t = np.array([_counter_prob_mass_k0(x_test[i], clf, neutralizer, int(args.k0)) for i in ids_t], dtype=np.float64)
    stats_prob = _fit_excess_map(ent_v, cm_prob_v, pred_v)
    ex_prob_v, _ = _apply_excess_map(ent_v, cm_prob_v, pred_v, stats_prob)
    ex_prob_t, _ = _apply_excess_map(ent_t, cm_prob_t, pred_t, stats_prob)

    score_entropy = ent_t
    score_raw = _fit_lr_score(yv, np.stack([ent_v, cm_v], axis=1), np.stack([ent_t, cm_t], axis=1), args.seed + 1)
    score_ex = _fit_lr_score(yv, np.stack([ent_v, ex_v], axis=1), np.stack([ent_t, ex_t], axis=1), args.seed + 2)
    score_ex_prob = _fit_lr_score(
        yv, np.stack([ent_v, ex_prob_v], axis=1), np.stack([ent_t, ex_prob_t], axis=1), args.seed + 3
    )

    methods_scores = [
        ("entropy_only", score_entropy),
        ("entropy_plus_raw_counter", score_raw),
        ("entropy_plus_C_excess", score_ex),
        ("entropy_plus_C_excess_prob", score_ex_prob),
    ]

    bp10, br10, bidx = _p10_r10(yt, score_entropy)
    bauprc = _auprc(yt, score_entropy)
    mean_q = float(np.mean(q_used_t))

    out_rows = []
    for name, s in methods_scores:
        p10, r10, idx = _p10_r10(yt, s)
        d_p10 = float(p10 - bp10)
        d_r10 = float(r10 - br10)
        d_auprc = float(_auprc(yt, s) - bauprc)
        ci_p10_l, ci_p10_h, frac_p10 = _bootstrap_delta(
            yt, s, score_entropy, lambda yy, ss: _p10_r10(yy, ss)[0], args.n_bootstrap, args.seed + 101
        )
        ci_auprc_l, ci_auprc_h, frac_auprc = _bootstrap_delta(
            yt, s, score_entropy, _auprc, args.n_bootstrap, args.seed + 202
        )
        err = np.where(yt == 1)[0]
        rescued = int(np.sum(np.isin(err, idx) & ~np.isin(err, bidx)))
        q_eff = 1.0 if name == "entropy_only" else max(mean_q, 1.0)
        qntg = float(d_p10 / q_eff)
        lat_obj = float(q_eff * args.latency_per_query)
        lntg = float(d_p10 / max(lat_obj, 1e-12))

        out_rows.append(
            {
                "protocol": "HAR_full_Q16_entropy_excess",
                "method": name,
                "q_max": q,
                "k0": int(args.k0),
                "partition_mode": str(args.partition_mode),
                "neutralizer": str(args.neutralizer),
                "n_test": int(len(yt)),
                "n_errors": int(np.sum(yt == 1)),
                "auroc": _auc(yt, s),
                "auprc": _auprc(yt, s),
                "p10": p10,
                "r10": r10,
                "delta_p10_vs_entropy": d_p10,
                "delta_r10_vs_entropy": d_r10,
                "delta_auprc_vs_entropy": d_auprc,
                "ci_delta_p10_low": ci_p10_l,
                "ci_delta_p10_high": ci_p10_h,
                "frac_positive_delta_p10": frac_p10,
                "ci_delta_auprc_low": ci_auprc_l,
                "ci_delta_auprc_high": ci_auprc_h,
                "frac_positive_delta_auprc": frac_auprc,
                "rescued_to_top10": rescued,
                "mean_q_used": q_eff,
                "qntg_p10": qntg,
                "latency_per_object": lat_obj,
                "lntg_p10": lntg,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)

    # Sanity audit: verify query allocation and one-object trace
    one_id = int(ids_t[0])
    one_x = x_test[one_id]
    dbg_audit = BeaconAudit(model_logits=clf.logits, neutralizer=neutralizer, config=cfg).audit(one_x)
    sanity = {
        "split_isolation": {
            "fit_standardizer_on": "train_split_only",
            "calibrate_excess_on": "validation_split_only",
            "use_test_labels_in_calibration": False,
        },
        "sizes": {
            "n_train": int(len(x_tr)),
            "n_val": int(len(x_va)),
            "n_test": int(len(x_test)),
            "n_eval_common_test": int(len(yt)),
        },
        "budget_check": {
            "q_max": int(cfg.q_max),
            "k0": int(cfg.k0),
            "q_frag_ratio": float(cfg.q_frag_ratio),
            "expected_q_remaining": int(cfg.q_max - cfg.k0),
            "sample_id": one_id,
            "q_init_used": int(dbg_audit.q_init),
            "q_ref_used": int(dbg_audit.q_ref_used),
            "q_frag_used": int(dbg_audit.q_frag_used),
            "q_total_used": int(dbg_audit.q_used),
            "counter_mass_sample": float(dbg_audit.counter_mass),
            "m0_sample": float(dbg_audit.m0),
            "top_s_minus": [
                {
                    "cid": s.component.cid,
                    "delta": float(s.delta),
                    "t0": int(s.component.t0),
                    "t1": int(s.component.t1),
                    "c0": int(s.component.c0),
                    "c1": int(s.component.c1),
                }
                for s in dbg_audit.s_minus
            ],
        },
    }
    out_sanity = Path(args.out_sanity)
    out_sanity.parent.mkdir(parents=True, exist_ok=True)
    out_sanity.write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out}")
    print(f"Saved: {out_sanity}")
    for r in out_rows:
        print(
            f"{r['method']}: AUROC={r['auroc']:.4f} AUPRC={r['auprc']:.4f} "
            f"P10={r['p10']:.4f} dP10={r['delta_p10_vs_entropy']:.4f} "
            f"CI_dP10=[{r['ci_delta_p10_low']:.4f},{r['ci_delta_p10_high']:.4f}] "
            f"rescued={r['rescued_to_top10']} frac+={r['frac_positive_delta_p10']:.3f}"
        )


if __name__ == "__main__":
    main()
