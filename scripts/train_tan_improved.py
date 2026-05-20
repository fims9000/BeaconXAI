#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow running as `python scripts/train_tan_improved.py ...` without package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beaconxai.tan_policy import TANModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train improved TAN policy (MI feature selection + optional MDLP + Platt)")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--k-features", type=int, default=5)
    p.add_argument("--discretizer", choices=["mdlp", "quantile", "uniform", "kmeans"], default="mdlp")
    p.add_argument("--bins", default="6")
    p.add_argument("--alpha", default="0.1,0.5,1.0,2.0")
    p.add_argument("--strategy", default="quantile")
    p.add_argument("--no-platt", action="store_true")
    p.add_argument("--compare-to", choices=["logit_panel", "uniform"], default="logit_panel")
    p.add_argument("--target", choices=["binary", "ce", "ordinal"], default="binary")
    p.add_argument("--ce-quantile", type=float, default=0.65)
    p.add_argument("--sample-weight", action="store_true")
    p.add_argument("--weight-alpha", type=float, default=5.0)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-results", default="tan_improved_results.csv")
    p.add_argument("--out-bootstrap", default="tan_improved_bootstrap.csv")
    return p.parse_args()


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


def _fit_discretizer(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    mode: str,
    n_bins: int,
    strategy: str,
) -> tuple[object, np.ndarray, callable, int, str]:
    if mode == "mdlp":
        mdlp_exc = None
        MDLP = None
        try:
            from mdlp.discretization import MDLP as _MDLP  # type: ignore

            MDLP = _MDLP
        except Exception as e1:
            mdlp_exc = e1
            try:
                from mdlp import MDLP as _MDLP  # type: ignore

                MDLP = _MDLP
            except Exception as e2:
                mdlp_exc = e2 if mdlp_exc is None else mdlp_exc
        if MDLP is not None:
            disc = MDLP()
            Xtr_d = np.asarray(disc.fit_transform(Xtr, ytr), dtype=np.int64)

            def _transform(X: np.ndarray) -> np.ndarray:
                return np.asarray(disc.transform(X), dtype=np.int64)

            n_bins_eff = int(np.max(Xtr_d) + 1) if Xtr_d.size > 0 else 2
            n_bins_eff = max(2, n_bins_eff)
            return disc, Xtr_d, _transform, n_bins_eff, "mdlp"

        print(f"[warn] MDLP unavailable ({mdlp_exc}); fallback to quantile bins.")
        mode = "quantile"

    disc_strategy = mode if mode in ("uniform", "kmeans", "quantile") else strategy
    disc = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=disc_strategy,
    )
    Xtr_d = np.asarray(disc.fit_transform(Xtr), dtype=np.int64)

    def _transform(X: np.ndarray) -> np.ndarray:
        return np.asarray(disc.transform(X), dtype=np.int64)

    return disc, Xtr_d, _transform, int(n_bins), mode


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "delta_entropy" not in d.columns and "rank_entropy" in d.columns:
        d["delta_entropy"] = d["rank_entropy"]
    if "margin_entropy" not in d.columns:
        m = -d["m_neg"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-m))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        d["margin_entropy"] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    # Extended conflict descriptors (cheap to compute from existing audit columns).
    t3c = np.clip(d.get("top3_conflict_count", 0.0).to_numpy(dtype=float), 0.0, 3.0)
    t3s = np.maximum(d.get("top3_sum_delta", 0.0).to_numpy(dtype=float), 0.0)
    c1 = np.maximum(d.get("top1_delta", 0.0).to_numpy(dtype=float), 0.0)
    ce = np.maximum(d.get("CE_B", 0.0).to_numpy(dtype=float), 0.0)
    mb = np.maximum(d.get("M_B_minus", 0.0).to_numpy(dtype=float), 0.0)
    frag = d.get("frag_drop", 0.0).to_numpy(dtype=float)
    denom = np.maximum(t3c, 1.0)
    d["mean_conflict"] = t3s / denom
    d["var_conflict_proxy"] = np.maximum(c1 - d["mean_conflict"].to_numpy(dtype=float), 0.0)
    d["frac_conflict_top3"] = t3c / 3.0
    d["fragility_gap"] = frag - ce
    d["ce_density"] = ce / (mb + 1e-6)
    # New v9 descriptors; fallback derivation keeps old bundles runnable.
    if "var_conflict" not in d.columns:
        d["var_conflict"] = d["var_conflict_proxy"]
    if "conflict_connectivity" not in d.columns:
        d["conflict_connectivity"] = d["frac_conflict_top3"]
    if "delta_frag_proxy" not in d.columns:
        d["delta_frag_proxy"] = d["fragility_gap"]
    if "r_cf" not in d.columns:
        d["r_cf"] = mb / (np.maximum(d.get("rho_B_cost", 1.0).to_numpy(dtype=float), 1e-6))
    return d


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


def main() -> None:
    args = parse_args()
    bdir = Path(args.bundle_dir)

    df_b = _prepare(pd.read_csv(bdir / "audit_features_beacon_core.csv")).set_index("sample_id").sort_index()
    df_u = _prepare(pd.read_csv(bdir / "audit_features_uniform.csv")).set_index("sample_id").sort_index()
    with (bdir / "split_manifest.json").open("r", encoding="utf-8") as f:
        man = json.load(f)
    tr = np.asarray(man["train_ids"], dtype=np.int64)
    va = np.asarray(man["val_ids"], dtype=np.int64)
    te = np.asarray(man["test_ids"], dtype=np.int64)

    feature_cols = [
        "m_neg", "M_B_minus", "r_B_minus", "CE_B", "rho_B_cost",
        "frag_drop", "top1_delta", "top3_sum_delta", "top3_conflict_count", "margin_entropy",
        "mean_conflict", "var_conflict_proxy", "frac_conflict_top3", "fragility_gap", "ce_density",
        "var_conflict", "conflict_connectivity", "delta_frag_proxy", "r_cf",
    ]
    y_eval_default = df_b["is_hidden_conflict"].to_numpy(dtype=np.int64)
    y_ce = np.maximum(df_b["CE_B"].to_numpy(dtype=float), 0.0)
    if args.target == "ce":
        q = float(np.clip(args.ce_quantile, 0.05, 0.95))
        thr_ce = float(np.quantile(y_ce[tr], q))
        y_train_target = (y_ce >= thr_ce).astype(np.int64)
        y_eval = y_eval_default
        ordinal_median_pos = float("nan")
    elif args.target == "ordinal":
        y_train_target, y_eval, ordinal_median_pos = _build_ordinal_targets(y_ce, tr)
        thr_ce = float("nan")
    else:
        thr_ce = float("nan")
        y_train_target = y_eval_default
        y_eval = y_eval_default
        ordinal_median_pos = float("nan")
    if args.sample_weight:
        sw_all = 1.0 + float(args.weight_alpha) * y_ce
    else:
        sw_all = np.ones_like(y_ce, dtype=float)
    Xb = df_b.loc[:, feature_cols].to_numpy(dtype=float)
    Xu = df_u.loc[:, feature_cols].to_numpy(dtype=float)

    bins = [int(v.strip()) for v in args.bins.split(",") if v.strip()]
    alphas = [float(v.strip()) for v in args.alpha.split(",") if v.strip()]
    use_platt = not bool(args.no_platt)

    best = None
    bin_grid = bins if args.discretizer != "mdlp" else [bins[0] if bins else 4]
    for nb in bin_grid:
        disc, Xtr_d_full, transform, n_bins_eff, used_disc = _fit_discretizer(
            Xb[tr], y_train_target[tr], mode=args.discretizer, n_bins=nb, strategy=args.strategy
        )
        selector = SelectKBest(score_func=chi2, k=max(1, min(args.k_features, Xtr_d_full.shape[1])))
        selector.fit(Xtr_d_full, y_train_target[tr])
        keep = np.where(selector.get_support())[0]
        kept_cols = [feature_cols[i] for i in keep]

        Xtr_d = Xtr_d_full[:, keep]
        Xva_d = transform(Xb[va])[:, keep]
        Xte_d = transform(Xb[te])[:, keep]
        Xb_k = Xb[:, keep]
        Xu_k = Xu[:, keep]

        for al in alphas:
            model = TANModel(n_bins=n_bins_eff, alpha=al).fit(Xtr_d, y_train_target[tr], sample_weight=sw_all[tr])
            p_va_all = model.predict_proba(Xva_d)
            p_te_all = model.predict_proba(Xte_d)
            if args.target == "ordinal":
                cls = list(np.asarray(model.classes_, dtype=int))
                idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
                p_va_raw = p_va_all[:, idx_hi]
                p_te_raw = p_te_all[:, idx_hi]
            else:
                p_va_raw = p_va_all[:, 1]
                p_te_raw = p_te_all[:, 1]
            calib = None
            if use_platt:
                calib = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed)
                y_cal = y_eval[va] if args.target == "ordinal" else y_train_target[va]
                calib.fit(p_va_raw.reshape(-1, 1), y_cal, sample_weight=sw_all[va])
                p_va = calib.predict_proba(p_va_raw.reshape(-1, 1))[:, 1]
                p_te = calib.predict_proba(p_te_raw.reshape(-1, 1))[:, 1]
            else:
                p_va = p_va_raw
                p_te = p_te_raw

            score = _f1_10(y_eval[va], p_va)
            row = {
                "n_bins": n_bins_eff,
                "alpha": al,
                "f1_10_val": score,
                "f1_10_test": _f1_10(y_eval[te], p_te),
                "discretizer": used_disc,
            }
            if best is None or row["f1_10_val"] > best["f1_10_val"]:
                best = row
                best["policy_b"] = model
                best["disc_b"] = disc
                best["calib_b"] = calib
                best["transform_b"] = transform
                best["keep_idx"] = keep.copy()
                best["kept_cols"] = kept_cols.copy()

    model_b = best.pop("policy_b")
    disc_b = best.pop("disc_b")
    calib_b = best.pop("calib_b")
    transform_b = best.pop("transform_b")
    keep_best = np.asarray(best.pop("keep_idx"), dtype=np.int64)
    kept_cols = best.pop("kept_cols")
    Xb_k = Xb[:, keep_best]
    Xu_k = Xu[:, keep_best]
    Xte_b_d = transform_b(Xb[te])[:, keep_best]
    sb_all = model_b.predict_proba(Xte_b_d)
    if args.target == "ordinal":
        cls = list(np.asarray(model_b.classes_, dtype=int))
        idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
        sb_raw = sb_all[:, idx_hi]
    else:
        sb_raw = sb_all[:, 1]
    sb = calib_b.predict_proba(sb_raw.reshape(-1, 1))[:, 1] if calib_b is not None else sb_raw

    # Baseline for comparison: logit panel (default) or uniform TAN.
    s_base = None
    base_name = ""
    if args.compare_to == "logit_panel":
        if args.target == "ordinal":
            logit = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed),
            )
        else:
            logit = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed),
            )
        fit_kwargs = {}
        if args.sample_weight:
            fit_kwargs = {"logisticregression__sample_weight": sw_all[tr]}
        logit.fit(Xb_k[tr], y_train_target[tr], **fit_kwargs)
        p_base = logit.predict_proba(Xb_k[te])
        if args.target == "ordinal":
            cls = list(np.unique(y_train_target[tr]))
            idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
            s_base = p_base[:, idx_hi]
        else:
            s_base = p_base[:, 1]
        base_name = "logit_panel"
    else:
        Xtr_u_d = np.asarray(disc_b.transform(Xu_k[tr]), dtype=np.int64)
        Xva_u_d = np.asarray(disc_b.transform(Xu_k[va]), dtype=np.int64)
        Xte_u_d = np.asarray(disc_b.transform(Xu_k[te]), dtype=np.int64)
        model_u = TANModel(n_bins=int(best["n_bins"]), alpha=float(best["alpha"])).fit(
            Xtr_u_d, y_train_target[tr], sample_weight=sw_all[tr]
        )
        pu_all = model_u.predict_proba(Xte_u_d)
        if args.target == "ordinal":
            cls = list(np.asarray(model_u.classes_, dtype=int))
            idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
            su_raw = pu_all[:, idx_hi]
        else:
            su_raw = pu_all[:, 1]
        if use_platt:
            p_va_u_all = model_u.predict_proba(Xva_u_d)
            if args.target == "ordinal":
                cls = list(np.asarray(model_u.classes_, dtype=int))
                idx_hi = cls.index(2) if 2 in cls else int(np.argmax(cls))
                p_va_u_raw = p_va_u_all[:, idx_hi]
                y_cal_u = y_eval[va]
            else:
                p_va_u_raw = p_va_u_all[:, 1]
                y_cal_u = y_train_target[va]
            calib_u = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed + 17)
            calib_u.fit(p_va_u_raw.reshape(-1, 1), y_cal_u, sample_weight=sw_all[va])
            su = calib_u.predict_proba(su_raw.reshape(-1, 1))[:, 1]
        else:
            su = su_raw
        s_base = su
        base_name = "uniform_tan"

    res = pd.DataFrame([
        {
            "bundle": bdir.name,
            "selected_features": ",".join(kept_cols),
            "compare_to": base_name,
            "target": args.target,
            "ce_quantile": args.ce_quantile,
            "ce_threshold_train": thr_ce,
            "ce_median_pos_train": ordinal_median_pos,
            "sample_weight": int(args.sample_weight),
            "weight_alpha": float(args.weight_alpha),
            "platt": int(use_platt),
            **best,
            "auroc_tan": _auroc(y_eval[te], sb),
            "auprc_tan": _auprc(y_eval[te], sb),
            "f1_10_tan": _f1_10(y_eval[te], sb),
            "auroc_baseline": _auroc(y_eval[te], s_base),
            "auprc_baseline": _auprc(y_eval[te], s_base),
            "f1_10_baseline": _f1_10(y_eval[te], s_base),
        }
    ])
    res.to_csv(bdir / args.out_results, index=False)

    boot_rows = []
    metric_seeds = {"delta_auroc": 101, "delta_auprc": 211, "delta_f1_10": 307}
    for mname, fn in (
        ("delta_auroc", _auroc),
        ("delta_auprc", _auprc),
        ("delta_f1_10", _f1_10),
    ):
        m, lo, hi, p = _bootstrap_delta(y_eval[te], sb, s_base, fn, args.n_boot, args.seed + metric_seeds[mname])
        boot_rows.append(
            {
                "bundle": bdir.name,
                "comparison": f"tan_vs_{base_name}",
                "metric": mname,
                "delta": m,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
            }
        )
    pd.DataFrame(boot_rows).to_csv(bdir / args.out_bootstrap, index=False)
    print(f"saved: {bdir / args.out_results}")
    print(f"saved: {bdir / args.out_bootstrap}")


if __name__ == "__main__":
    main()
