#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 1: policy on full delta vector")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--delta-file", default="delta_vectors_beacon_core.csv")
    p.add_argument("--compare-to", choices=["logit_panel"], default="logit_panel")
    p.add_argument("--model", choices=["xgboost", "histgbt", "mlp"], default="xgboost")
    p.add_argument("--target", choices=["binary", "ce", "ordinal"], default="ce")
    p.add_argument("--ce-quantile", type=float, default=0.65)
    p.add_argument("--sample-weight", action="store_true")
    p.add_argument("--weight-alpha", type=float, default=5.0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-results", default="delta_policy_results.csv")
    p.add_argument("--out-bootstrap", default="delta_policy_bootstrap.csv")
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
    return float(roc_auc_score(y, s))


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        da = fn(yy, a[idx])
        db = fn(yy, b[idx])
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


def _make_model(model_name: str, target: str, seed: int):
    if model_name == "xgboost":
        from xgboost import XGBClassifier  # type: ignore
        if target == "ordinal":
            return XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=4,
                eval_metric="mlogloss",
            )
        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=4,
            eval_metric="logloss",
        )
    if model_name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=500,
                random_state=seed,
            ),
        )
    # histgbt fallback
    return HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=1e-4,
        random_state=seed,
    )


def main() -> None:
    args = parse_args()
    bdir = Path(args.bundle_dir)
    ddf = pd.read_csv(bdir / args.delta_file).set_index("sample_id").sort_index()
    adf = pd.read_csv(bdir / "audit_features_beacon_core.csv").set_index("sample_id").sort_index()
    with (bdir / "split_manifest.json").open("r", encoding="utf-8") as f:
        man = json.load(f)
    tr = np.asarray(man["train_ids"], dtype=np.int64)
    te = np.asarray(man["test_ids"], dtype=np.int64)

    delta_cols = [c for c in ddf.columns if c.startswith("d")]
    X_delta = ddf.loc[:, delta_cols].to_numpy(dtype=float)

    feat_cols = [
        "m_neg", "M_B_minus", "r_B_minus", "CE_B", "rho_B_cost",
        "frag_drop", "top1_delta", "top3_sum_delta", "top3_conflict_count",
        "delta_entropy", "margin_entropy", "var_conflict", "conflict_connectivity",
        "delta_frag_proxy", "r_cf",
    ]
    feat_cols = [c for c in feat_cols if c in adf.columns]
    X_base = adf.loc[:, feat_cols].to_numpy(dtype=float)

    y_eval_default = adf["is_hidden_conflict"].to_numpy(dtype=np.int64)
    y_ce = np.maximum(adf["CE_B"].to_numpy(dtype=float), 0.0)
    if args.target == "ce":
        q = float(np.clip(args.ce_quantile, 0.05, 0.95))
        thr_ce = float(np.quantile(y_ce[tr], q))
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

    if args.sample_weight:
        sw_all = 1.0 + float(args.weight_alpha) * y_ce
    else:
        sw_all = np.ones_like(y_ce, dtype=float)

    # Baseline: logit on compact panel.
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    if args.sample_weight:
        base.fit(X_base[tr], y_train_target[tr], logisticregression__sample_weight=sw_all[tr])
    else:
        base.fit(X_base[tr], y_train_target[tr])
    pb = base.predict_proba(X_base[te])
    if args.target == "ordinal":
        cls_b = list(np.asarray(base.named_steps["logisticregression"].classes_, dtype=int))
        idx_hi_b = cls_b.index(2) if 2 in cls_b else int(np.argmax(cls_b))
        s_base = pb[:, idx_hi_b]
    else:
        s_base = pb[:, 1]

    # Delta model.
    model = _make_model(args.model, args.target, args.seed)
    fit_kwargs = {}
    if args.sample_weight and args.model != "mlp":
        fit_kwargs["sample_weight"] = sw_all[tr]
    if args.sample_weight and args.model == "xgboost":
        fit_kwargs["sample_weight"] = sw_all[tr]
    model.fit(X_delta[tr], y_train_target[tr], **fit_kwargs)
    pdm = model.predict_proba(X_delta[te])
    if args.target == "ordinal":
        cls = list(np.asarray(getattr(model, "classes_"), dtype=int))
        idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
        s_model = pdm[:, idx_hi]
    else:
        s_model = pdm[:, 1]

    res = {
        "bundle": bdir.name,
        "delta_file": args.delta_file,
        "model": args.model,
        "compare_to": args.compare_to,
        "target": args.target,
        "ce_quantile": float(args.ce_quantile),
        "ce_threshold_train": float(thr_ce),
        "ordinal_median_pos": float(ord_med),
        "sample_weight": bool(args.sample_weight),
        "weight_alpha": float(args.weight_alpha),
        "n_delta_features": int(len(delta_cols)),
        "auroc_model": _auroc(y_eval[te], s_model),
        "auprc_model": _auprc(y_eval[te], s_model),
        "f1_10_model": _f1_10(y_eval[te], s_model),
        "auroc_baseline": _auroc(y_eval[te], s_base),
        "auprc_baseline": _auprc(y_eval[te], s_base),
        "f1_10_baseline": _f1_10(y_eval[te], s_base),
    }
    pd.DataFrame([res]).to_csv(bdir / args.out_results, index=False)

    rows = []
    for metric, fn in [("delta_auroc", _auroc), ("delta_auprc", _auprc), ("delta_f1_10", _f1_10)]:
        d, lo, hi, p = _bootstrap_delta(y_eval[te], s_model, s_base, fn, n_boot=args.n_boot, seed=args.seed + len(metric) * 11)
        rows.append(
            {
                "bundle": bdir.name,
                "comparison": f"{args.model}_vs_logit_panel",
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
