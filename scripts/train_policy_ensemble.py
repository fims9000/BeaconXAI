#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from beaconxai.tan_policy import TANModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stacking ensemble + cost-optimized thresholds")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--delta-file", default="delta_vectors_beacon_core.csv")
    p.add_argument("--target", choices=["binary", "ce", "ordinal"], default="ce")
    p.add_argument("--ce-quantile", type=float, default=0.65)
    p.add_argument("--sample-weight", action="store_true")
    p.add_argument("--weight-alpha", type=float, default=5.0)
    p.add_argument("--w-fp", type=float, default=1.0)
    p.add_argument("--w-fn", type=float, default=2.0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-results", default="ensemble_results.csv")
    p.add_argument("--out-bootstrap", default="ensemble_bootstrap.csv")
    return p.parse_args()


def _f1_10(y: np.ndarray, s: np.ndarray) -> float:
    n = len(y)
    k = max(1, int(np.ceil(0.10 * n)))
    order = np.argsort(-s)
    yp = np.zeros(n, dtype=np.int64)
    yp[order[:k]] = 1
    tp = float(np.sum((yp == 1) & (y == 1)))
    fp = float(np.sum((yp == 1) & (y == 0)))
    fn = float(np.sum((yp == 0) & (y == 1)))
    p = tp / max(1.0, tp + fp)
    r = tp / max(1.0, tp + fn)
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        da = fn(y[idx], a[idx])
        db = fn(y[idx], b[idx])
        if np.isfinite(da) and np.isfinite(db):
            vals.append(float(da - db))
    arr = np.asarray(vals, dtype=float)
    p = 2.0 * min(float(np.mean(arr < 0.0)), float(np.mean(arr > 0.0)))
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), float(min(1.0, max(0.0, p)))


def _build_ordinal_targets(y_ce: np.ndarray, tr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y_ce = np.maximum(np.asarray(y_ce, dtype=float), 0.0)
    tr_pos = y_ce[tr_idx][y_ce[tr_idx] > 1e-12]
    med_pos = float(np.median(tr_pos)) if tr_pos.size > 0 else 0.0
    y_ord = np.zeros_like(y_ce, dtype=np.int64)
    weak = (y_ce > 1e-12) & (y_ce <= med_pos)
    strong = y_ce > med_pos
    y_ord[weak] = 1
    y_ord[strong] = 2
    y_high = (y_ord == 2).astype(np.int64)
    return y_ord, y_high, med_pos


def _best_cost_threshold(y: np.ndarray, s: np.ndarray, w_fp: float, w_fn: float) -> tuple[float, float]:
    qs = np.linspace(0.01, 0.99, 99)
    thr_list = np.unique(np.quantile(s, qs))
    best_t, best_c = float(thr_list[0]), 1e18
    for t in thr_list:
        yp = (s >= t).astype(np.int64)
        fp = float(np.sum((yp == 1) & (y == 0)))
        fn = float(np.sum((yp == 0) & (y == 1)))
        tn = float(np.sum((yp == 0) & (y == 0)))
        tp = float(np.sum((yp == 1) & (y == 1)))
        fpr = fp / max(1.0, fp + tn)
        fnr = fn / max(1.0, fn + tp)
        c = w_fp * fpr + w_fn * fnr
        if c < best_c:
            best_c = c
            best_t = float(t)
    return best_t, best_c


def _cost_metrics(y: np.ndarray, s: np.ndarray, thr: float, w_fp: float, w_fn: float) -> dict[str, float]:
    yp = (s >= thr).astype(np.int64)
    fp = float(np.sum((yp == 1) & (y == 0)))
    fn = float(np.sum((yp == 0) & (y == 1)))
    tn = float(np.sum((yp == 0) & (y == 0)))
    tp = float(np.sum((yp == 1) & (y == 1)))
    fpr = fp / max(1.0, fp + tn)
    fnr = fn / max(1.0, fn + tp)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"cost": w_fp * fpr + w_fn * fnr, "fpr": fpr, "fnr": fnr, "f1_thr": f1}


def main() -> None:
    args = parse_args()
    bdir = Path(args.bundle_dir)
    df = pd.read_csv(bdir / "audit_features_beacon_core.csv").set_index("sample_id").sort_index()
    ddf = pd.read_csv(bdir / args.delta_file).set_index("sample_id").sort_index()
    with (bdir / "split_manifest.json").open("r", encoding="utf-8") as f:
        man = json.load(f)
    tr = np.asarray(man["train_ids"], dtype=np.int64)
    va = np.asarray(man["val_ids"], dtype=np.int64)
    te = np.asarray(man["test_ids"], dtype=np.int64)

    # targets
    y_eval_default = df["is_hidden_conflict"].to_numpy(dtype=np.int64)
    y_ce = np.maximum(df["CE_B"].to_numpy(dtype=float), 0.0)
    if args.target == "ce":
        thr_ce = float(np.quantile(y_ce[tr], float(np.clip(args.ce_quantile, 0.05, 0.95))))
        y_train_target = (y_ce >= thr_ce).astype(np.int64)
        y_eval = y_eval_default
        ord_med = float("nan")
    elif args.target == "ordinal":
        y_train_target, y_eval, ord_med = _build_ordinal_targets(y_ce, tr)
        thr_ce = float("nan")
    else:
        y_train_target = y_eval_default
        y_eval = y_eval_default
        thr_ce = float("nan")
        ord_med = float("nan")
    sw = 1.0 + float(args.weight_alpha) * y_ce if args.sample_weight else np.ones_like(y_ce)

    feat_cols = [
        "m_neg", "M_B_minus", "r_B_minus", "CE_B", "rho_B_cost", "frag_drop",
        "top1_delta", "top3_sum_delta", "top3_conflict_count", "delta_entropy", "margin_entropy",
        "var_conflict", "conflict_connectivity", "delta_frag_proxy", "r_cf",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df.loc[:, feat_cols].to_numpy(dtype=float)
    delta_cols = [c for c in ddf.columns if c.startswith("d")]
    Xd = ddf.loc[:, delta_cols].to_numpy(dtype=float)

    # base 1: logit compact
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    if args.sample_weight:
        logit.fit(X[tr], y_train_target[tr], logisticregression__sample_weight=sw[tr])
    else:
        logit.fit(X[tr], y_train_target[tr])
    p_logit_va = logit.predict_proba(X[va])
    p_logit_te = logit.predict_proba(X[te])
    if args.target == "ordinal":
        cls = list(np.asarray(logit.named_steps["logisticregression"].classes_, dtype=int))
        idx = cls.index(2) if 2 in cls else int(np.argmax(cls))
        s_logit_va = p_logit_va[:, idx]
        s_logit_te = p_logit_te[:, idx]
    else:
        s_logit_va = p_logit_va[:, 1]
        s_logit_te = p_logit_te[:, 1]

    # base 2: TAN compact
    kb = KBinsDiscretizer(n_bins=6, encode="ordinal", strategy="quantile")
    Xtr_d = np.asarray(kb.fit_transform(X[tr]), dtype=np.int64)
    Xva_d = np.asarray(kb.transform(X[va]), dtype=np.int64)
    Xte_d = np.asarray(kb.transform(X[te]), dtype=np.int64)
    sel = SelectKBest(score_func=chi2, k=max(1, min(3, Xtr_d.shape[1])))
    sel.fit(Xtr_d, y_train_target[tr])
    keep = np.where(sel.get_support())[0]
    tan = TANModel(n_bins=6, alpha=1.0).fit(Xtr_d[:, keep], y_train_target[tr], sample_weight=sw[tr])
    p_tan_va = tan.predict_proba(Xva_d[:, keep])
    p_tan_te = tan.predict_proba(Xte_d[:, keep])
    if args.target == "ordinal":
        cls = list(np.asarray(tan.classes_, dtype=int))
        idx = cls.index(2) if 2 in cls else int(np.argmax(cls))
        s_tan_va = p_tan_va[:, idx]
        s_tan_te = p_tan_te[:, idx]
    else:
        s_tan_va = p_tan_va[:, 1]
        s_tan_te = p_tan_te[:, 1]

    # base 3: xgboost compact
    from xgboost import XGBClassifier  # type: ignore

    if args.target == "ordinal":
        xgb_c = XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=args.seed, n_jobs=4, eval_metric="mlogloss"
        )
    else:
        xgb_c = XGBClassifier(
            objective="binary:logistic", n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=args.seed, n_jobs=4, eval_metric="logloss"
        )
    xgb_c.fit(X[tr], y_train_target[tr], sample_weight=sw[tr] if args.sample_weight else None)
    p_xc_va = xgb_c.predict_proba(X[va])
    p_xc_te = xgb_c.predict_proba(X[te])
    if args.target == "ordinal":
        cls = list(np.asarray(xgb_c.classes_, dtype=int))
        idx = cls.index(2) if 2 in cls else int(np.argmax(cls))
        s_xc_va = p_xc_va[:, idx]
        s_xc_te = p_xc_te[:, idx]
    else:
        s_xc_va = p_xc_va[:, 1]
        s_xc_te = p_xc_te[:, 1]

    # base 4: xgboost delta
    if args.target == "ordinal":
        xgb_d = XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=args.seed + 9, n_jobs=4, eval_metric="mlogloss"
        )
    else:
        xgb_d = XGBClassifier(
            objective="binary:logistic", n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=args.seed + 9, n_jobs=4, eval_metric="logloss"
        )
    xgb_d.fit(Xd[tr], y_train_target[tr], sample_weight=sw[tr] if args.sample_weight else None)
    p_xd_va = xgb_d.predict_proba(Xd[va])
    p_xd_te = xgb_d.predict_proba(Xd[te])
    if args.target == "ordinal":
        cls = list(np.asarray(xgb_d.classes_, dtype=int))
        idx = cls.index(2) if 2 in cls else int(np.argmax(cls))
        s_xd_va = p_xd_va[:, idx]
        s_xd_te = p_xd_te[:, idx]
    else:
        s_xd_va = p_xd_va[:, 1]
        s_xd_te = p_xd_te[:, 1]

    # stacking meta on validation scores (+ optional CE_B context)
    Z_va = np.column_stack([s_logit_va, s_tan_va, s_xc_va, s_xd_va, y_ce[va]])
    Z_te = np.column_stack([s_logit_te, s_tan_te, s_xc_te, s_xd_te, y_ce[te]])
    meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed))
    if args.sample_weight:
        meta.fit(Z_va, y_eval[va], logisticregression__sample_weight=sw[va])
    else:
        meta.fit(Z_va, y_eval[va])
    s_ens = meta.predict_proba(Z_te)[:, 1]
    s_base = s_logit_te

    # threshold by cost on validation
    s_ens_va = meta.predict_proba(Z_va)[:, 1]
    t_ens, c_ens_val = _best_cost_threshold(y_eval[va], s_ens_va, args.w_fp, args.w_fn)
    t_log, c_log_val = _best_cost_threshold(y_eval[va], s_logit_va, args.w_fp, args.w_fn)
    cm_ens = _cost_metrics(y_eval[te], s_ens, t_ens, args.w_fp, args.w_fn)
    cm_log = _cost_metrics(y_eval[te], s_base, t_log, args.w_fp, args.w_fn)

    res = {
        "bundle": bdir.name,
        "target": args.target,
        "ce_quantile": float(args.ce_quantile),
        "ce_threshold_train": float(thr_ce),
        "ordinal_median_pos": float(ord_med),
        "sample_weight": int(args.sample_weight),
        "weight_alpha": float(args.weight_alpha),
        "w_fp": float(args.w_fp),
        "w_fn": float(args.w_fn),
        "auroc_ens": _auroc(y_eval[te], s_ens),
        "auprc_ens": _auprc(y_eval[te], s_ens),
        "f1_10_ens": _f1_10(y_eval[te], s_ens),
        "auroc_logit": _auroc(y_eval[te], s_base),
        "auprc_logit": _auprc(y_eval[te], s_base),
        "f1_10_logit": _f1_10(y_eval[te], s_base),
        "thr_ens": float(t_ens),
        "thr_logit": float(t_log),
        "val_cost_ens": float(c_ens_val),
        "val_cost_logit": float(c_log_val),
        "test_cost_ens": float(cm_ens["cost"]),
        "test_cost_logit": float(cm_log["cost"]),
        "test_f1_thr_ens": float(cm_ens["f1_thr"]),
        "test_f1_thr_logit": float(cm_log["f1_thr"]),
    }
    pd.DataFrame([res]).to_csv(bdir / args.out_results, index=False)

    rows = []
    for metric, fn in [("delta_auroc", _auroc), ("delta_auprc", _auprc), ("delta_f1_10", _f1_10)]:
        d, lo, hi, p = _bootstrap_delta(y_eval[te], s_ens, s_base, fn, n_boot=args.n_boot, seed=args.seed + len(metric) * 13)
        rows.append(
            {
                "bundle": bdir.name,
                "comparison": "ensemble_vs_logit_panel",
                "metric": metric,
                "delta": d,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
            }
        )
    pd.DataFrame(rows).to_csv(bdir / args.out_bootstrap, index=False)
    print(f"saved: {bdir / args.out_results}")
    print(f"saved: {bdir / args.out_bootstrap}")


if __name__ == "__main__":
    main()
