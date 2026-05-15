#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

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


def _p10_r10(y: np.ndarray, s: np.ndarray) -> tuple[float, float, np.ndarray]:
    k = max(1, int(np.ceil(0.10 * len(y))))
    idx = np.argsort(-s)[:k]
    p10 = float(np.mean(y[idx] == 1))
    denom = float(np.sum(y == 1))
    r10 = float(np.sum(y[idx] == 1) / denom) if denom > 0 else float("nan")
    return p10, r10, idx


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


def _collect(
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
    return {
        "ids": np.array(ids, dtype=np.int64),
        "y": np.array([br[i].is_error for i in ids], dtype=np.int64),
        "neg_margin": np.array([nm[i].risk_score for i in ids], dtype=np.float64),
        "counter_mass": np.array([lm[i].counter_mass for i in ids], dtype=np.float64),
        "ce": np.array([lm[i].counter_evidence_gain for i in ids], dtype=np.float64),
        "q_used": np.array([br[i].q_used for i in ids], dtype=np.float64),
        "censored": np.array([br[i].censored for i in ids], dtype=np.float64),
    }


def _fit_logreg_score(yv: np.ndarray, Xv: np.ndarray, Xt: np.ndarray, seed: int) -> np.ndarray:
    sc = StandardScaler()
    Xv2 = sc.fit_transform(Xv)
    Xt2 = sc.transform(Xt)
    clf = LogisticRegression(C=1.0, class_weight="balanced", random_state=seed, solver="lbfgs", max_iter=1000)
    clf.fit(Xv2, yv)
    return clf.predict_proba(Xt2)[:, 1]


def _fit_cz(mv: np.ndarray, cmv: np.ndarray, mt: np.ndarray, cmt: np.ndarray):
    spl = SplineTransformer(n_knots=6, degree=3, include_bias=False)
    Xv = spl.fit_transform(mv.reshape(-1, 1))
    mu_model = Ridge(alpha=1.0).fit(Xv, cmv)
    mu_v = mu_model.predict(Xv)
    resid = cmv - mu_v
    std_model = Ridge(alpha=1.0).fit(Xv, resid * resid)
    Xt = spl.transform(mt.reshape(-1, 1))
    mu_t = mu_model.predict(Xt)
    var_t = std_model.predict(Xt)
    sig_t = np.sqrt(np.maximum(var_t, 1e-8))
    mu_v2 = mu_model.predict(Xv)
    var_v = std_model.predict(Xv)
    sig_v = np.sqrt(np.maximum(var_v, 1e-8))
    cz_v = (cmv - mu_v2) / np.maximum(sig_v, 1e-8)
    cz_t = (cmt - mu_t) / np.maximum(sig_t, 1e-8)
    return cz_v, cz_t


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


def _margin_matched_score(y: np.ndarray, margin: np.ndarray, cm: np.ndarray) -> float:
    err = np.where(y == 1)[0]
    cor = np.where(y == 0)[0]
    if len(err) == 0 or len(cor) == 0:
        return float("nan")
    win = 0
    total = 0
    for i in err:
        j = cor[np.argmin(np.abs(margin[cor] - margin[i]))]
        win += int(cm[i] > cm[j])
        total += 1
    return float(win / total) if total > 0 else float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR targeted conflict package")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--q-values", default="8,16")
    p.add_argument("--max-test", type=int, default=0)  # 0 => full test
    p.add_argument("--cnn-epochs", type=int, default=8)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--n-bootstrap", type=int, default=800)
    p.add_argument("--latency-per-query", type=float, default=0.0010728925)
    p.add_argument("--out-main", default="./outputs_composite/har_targeted_conflict_package.csv")
    p.add_argument("--out-subset", default="./outputs_composite/har_targeted_conflict_subsets.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]
    k0_map = {8: 4, 16: 8}

    x_train_full, y_train_full, x_test, y_test = load_uci_har(args.dataset_root)
    tr_idx, va_idx = _stratified_split_idx(y_train_full, args.val_frac, args.seed)
    x_tr = x_train_full[tr_idx]
    y_tr = y_train_full[tr_idx]
    x_va = x_train_full[va_idx]
    y_va = y_train_full[va_idx]

    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed + 21)
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

    base = BeaconConfig(
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
        partition_mode="time_only",
        risk_policy="rho_only",
    )
    neutralizer = Neutralizer(mode="zero", channel_means=np.zeros(x_tr.shape[-1], dtype=np.float32))
    configs = [
        ("A_adaptive_time", dict(margin_mode="adaptive_all", audit_mode="full", partition_mode="time_only")),
        ("B_nearest_time", dict(margin_mode="nearest_competitor", audit_mode="full", partition_mode="time_only")),
        (
            "C_adaptive_sensor_group",
            dict(margin_mode="adaptive_all", audit_mode="full", partition_mode="sensor_group_time"),
        ),
        (
            "D_nearest_counter_sensor_group",
            dict(margin_mode="nearest_competitor", audit_mode="counter_only", partition_mode="sensor_group_time"),
        ),
    ]

    out_main = []
    out_subset = []

    for cfg_name, extra in configs:
        print(f"[targeted] {cfg_name}", flush=True)
        for q in q_values:
            cfg = replace(base, q_max=q, k0=k0_map.get(q, 8), **extra)
            rows_v, local_v, _ = evaluate_error_risk(
                x_test=x_va,
                y_test=y_va,
                predict_fn=clf.predict,
                logits_fn=clf.logits,
                neutralizer=neutralizer,
                base_cfg=cfg,
                q_values=[q],
                margin_gradient_fn=getattr(clf, "margin_gradient", None),
                methods={"negative_margin", "beacon_refine"},
            )
            rows_t, local_t, _ = evaluate_error_risk(
                x_test=x_test,
                y_test=y_test,
                predict_fn=clf.predict,
                logits_fn=clf.logits,
                neutralizer=neutralizer,
                base_cfg=cfg,
                q_values=[q],
                margin_gradient_fn=getattr(clf, "margin_gradient", None),
                methods={"negative_margin", "beacon_refine"},
            )
            fv = _collect(rows_v, local_v, q)
            ft = _collect(rows_t, local_t, q)
            if fv is None or ft is None:
                continue
            yv, yt = fv["y"], ft["y"]
            mv, mt = fv["neg_margin"], ft["neg_margin"]
            cmv, cmt = fv["counter_mass"], ft["counter_mass"]
            cev, cet = fv["ce"], ft["ce"]
            cz_v, cz_t = _fit_cz(mv, cmv, mt, cmt)

            scores = {
                "margin": mt,
                "margin+counter_mass": _fit_logreg_score(yv, np.stack([mv, cmv], axis=1), np.stack([mt, cmt], axis=1), args.seed + q + 1),
                "margin+CE": _fit_logreg_score(yv, np.stack([mv, cev], axis=1), np.stack([mt, cet], axis=1), args.seed + q + 2),
                "margin+counter_mass+CE": _fit_logreg_score(
                    yv, np.stack([mv, cmv, cev], axis=1), np.stack([mt, cmt, cet], axis=1), args.seed + q + 3
                ),
                "margin+counter_mass+C_z": _fit_logreg_score(
                    yv, np.stack([mv, cmv, cz_v], axis=1), np.stack([mt, cmt, cz_t], axis=1), args.seed + q + 4
                ),
            }
            b = scores["margin"]
            p10_b, r10_b, idx_b = _p10_r10(yt, b)
            rank_b = np.argsort(np.argsort(-b))
            mean_q = float(np.mean(ft["q_used"]))
            mm_score = _margin_matched_score(yt, mt, cmt)

            for name, s in scores.items():
                p10, r10, idx_s = _p10_r10(yt, s)
                d_p10 = float(p10 - p10_b)
                d_r10 = float(r10 - r10_b)
                q_used = 1.0 if name == "margin" else max(mean_q, 1.0)
                qntg = float(d_p10 / q_used)
                lat_obj = float(q_used * args.latency_per_query)
                lntg = float(d_p10 / max(lat_obj, 1e-12))
                ci_p10_l, ci_p10_h, frac_p10 = _bootstrap_delta(
                    yt, s, b, lambda yy, ss: _p10_r10(yy, ss)[0], args.n_bootstrap, args.seed + 100 + q
                )
                ci_r10_l, ci_r10_h, frac_r10 = _bootstrap_delta(
                    yt, s, b, lambda yy, ss: _p10_r10(yy, ss)[1], args.n_bootstrap, args.seed + 200 + q
                )
                err = np.where(yt == 1)[0]
                rescued = int(np.sum(np.isin(err, idx_s) & ~np.isin(err, idx_b)))
                rank_s = np.argsort(np.argsort(-s))
                rank_imp = float(np.mean(rank_b[err] - rank_s[err])) if len(err) > 0 else float("nan")
                out_main.append(
                    {
                        "config": cfg_name,
                        "q_max": q,
                        "k0": cfg.k0,
                        "method": name,
                        "auroc": _auc(yt, s),
                        "auprc": _auprc(yt, s),
                        "p10": p10,
                        "r10": r10,
                        "delta_p10": d_p10,
                        "delta_r10": d_r10,
                        "qntg_p10": qntg,
                        "lntg_p10": lntg,
                        "mean_q_used": q_used,
                        "latency_per_object": lat_obj,
                        "ci_delta_p10_low": ci_p10_l,
                        "ci_delta_p10_high": ci_p10_h,
                        "frac_positive_p10": frac_p10,
                        "ci_delta_r10_low": ci_r10_l,
                        "ci_delta_r10_high": ci_r10_h,
                        "frac_positive_r10": frac_r10,
                        "rescued_to_top10": rescued,
                        "mean_error_rank_improvement": rank_imp,
                        "margin_matched_counter_score": mm_score,
                    }
                )

                # subset analysis by margin bands with errors
                cuts = np.quantile(mt, [0.0, 0.25, 0.5, 0.75, 1.0])
                cuts[0] -= 1e-12
                cuts[-1] += 1e-12
                for bi in range(4):
                    msk = (mt > cuts[bi]) & (mt <= cuts[bi + 1])
                    if np.sum(msk) < 30:
                        continue
                    y_sub = yt[msk]
                    if np.sum(y_sub == 1) == 0:
                        continue
                    p10_m, _, _ = _p10_r10(y_sub, b[msk])
                    p10_s, _, _ = _p10_r10(y_sub, s[msk])
                    out_subset.append(
                        {
                            "config": cfg_name,
                            "q_max": q,
                            "method": name,
                            "band": bi,
                            "n": int(np.sum(msk)),
                            "n_errors": int(np.sum(y_sub)),
                            "p10_margin": p10_m,
                            "p10_method": p10_s,
                            "delta_p10": float(p10_s - p10_m),
                        }
                    )
            print(f"[targeted] {cfg_name} q={q} done", flush=True)

    for path, rows in [(Path(args.out_main), out_main), (Path(args.out_subset), out_subset)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            if rows:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
            else:
                # keep pipeline stable even if a split/band produced no rows
                f.write("")
    print("Saved:")
    print(args.out_main)
    print(args.out_subset)


if __name__ == "__main__":
    main()
