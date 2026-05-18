#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beaconxai.calibration import brier_score, expected_calibration_error
from beaconxai.fuzzy_policy_v2 import eval_at_budget
from beaconxai.fuzzy_policy_v5 import FEATURES_V5, build_fuzzy_inputs_v5, fit_fuzzy_policy_v5, predict_fuzzy_policy_v5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fuzzy_v5 with full BEACON vector and compare vs uniform/logit")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--uniform-features", default="outputs_composite/part2_extended_v2/audit_features_uniform.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--n-rules", type=int, default=7)
    p.add_argument("--n-terms", type=int, default=3)
    p.add_argument("--epochs", type=int, default=350)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/fuzzy_v5_results.csv")
    p.add_argument("--out-bootstrap", default="outputs_composite/part2_extended_v2/fuzzy_v5_bootstrap.csv")
    return p.parse_args()


def _metrics(y: np.ndarray, s: np.ndarray):
    p10, r10, f10 = eval_at_budget(y, s, 0.10)
    p20, r20, f20 = eval_at_budget(y, s, 0.20)
    return {
        "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) >= 2 else float("nan"),
        "auprc": float(average_precision_score(y, s)),
        "precision_10": p10,
        "recall_10": r10,
        "f1_10": f10,
        "precision_20": p20,
        "recall_20": r20,
        "f1_20": f20,
        "ece": float(expected_calibration_error(y, s, n_bins=10)),
        "brier": float(brier_score(y, s)),
    }


def _bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        try:
            da = fn(yy, a[idx])
            db = fn(yy, b[idx])
        except Exception:
            continue
        if np.isfinite(da) and np.isfinite(db):
            vals.append(float(da - db))
    if not vals:
        return float("nan"), float("nan"), float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    p = 2.0 * min(float(np.mean(arr < 0.0)), float(np.mean(arr > 0.0)))
    p = float(min(1.0, max(0.0, p)))
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)), p


def _f1_budget(frac: float):
    def fn(y: np.ndarray, s: np.ndarray):
        _p, _r, f = eval_at_budget(y, s, frac)
        return float(f)

    return fn


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "delta_entropy" not in d.columns and "rank_entropy" in d.columns:
        d["delta_entropy"] = d["rank_entropy"]
    if "margin_entropy" not in d.columns:
        m = -d["m_neg"].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-m))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        d["margin_entropy"] = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return d


def main() -> None:
    args = parse_args()

    df_b = _prepare_df(pd.read_csv(args.beacon_features)).set_index("sample_id").sort_index()
    df_u = _prepare_df(pd.read_csv(args.uniform_features)).set_index("sample_id").sort_index()
    with Path(args.split_manifest).open("r", encoding="utf-8") as f:
        man = json.load(f)
    tr = np.asarray(man["train_ids"], dtype=np.int64)
    va = np.asarray(man["val_ids"], dtype=np.int64)
    te = np.asarray(man["test_ids"], dtype=np.int64)

    y = df_b["is_hidden_conflict"].to_numpy(dtype=np.int64)

    Xb = build_fuzzy_inputs_v5(df_b)
    Xu = build_fuzzy_inputs_v5(df_u)

    # Logit on full 10 features (stronger baseline for fair comparison)
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    logit.fit(Xb[tr], y[tr])
    s_logit = logit.predict_proba(Xb[te])[:, 1]

    pol_b = fit_fuzzy_policy_v5(
        Xb[tr], y[tr], Xb[va], y[va],
        n_terms=args.n_terms,
        n_rules=args.n_rules,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    pol_u = fit_fuzzy_policy_v5(
        Xu[tr], y[tr], Xu[va], y[va],
        n_terms=args.n_terms,
        n_rules=args.n_rules,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed + 17,
        device=args.device,
    )

    s_fb_va = predict_fuzzy_policy_v5(pol_b, Xb[va])
    s_fl_va = logit.predict_proba(Xb[va])[:, 1]

    # fixed lambda from validation target
    grid = np.linspace(0.0, 1.0, 21)
    best_lam = 0.5
    best_t = -1.0
    for lam in grid:
        sm = lam * s_fb_va + (1.0 - lam) * s_fl_va
        p10, _r10, f10 = eval_at_budget(y[va], sm, 0.10)
        _p20, _r20, f20 = eval_at_budget(y[va], sm, 0.20)
        t = 0.6 * f10 + 0.2 * p10 + 0.2 * f20
        if t > best_t:
            best_t = t
            best_lam = float(lam)

    s_fb = predict_fuzzy_policy_v5(pol_b, Xb[te])
    s_fu = predict_fuzzy_policy_v5(pol_u, Xu[te])
    s_mix = best_lam * s_fb + (1.0 - best_lam) * s_logit

    mb = _metrics(y[te], s_fb)
    mu = _metrics(y[te], s_fu)
    ml = _metrics(y[te], s_logit)
    mm = _metrics(y[te], s_mix)

    rows = []
    for name, m in [
        ("logit_full10_beacon", ml),
        ("fuzzy_v5_beacon", mb),
        ("fuzzy_v5_uniform", mu),
        ("soft_mix_v5_fixed", mm),
    ]:
        rows.append(
            {
                "policy": name,
                "n_features": len(FEATURES_V5),
                "n_rules": args.n_rules,
                "n_terms": args.n_terms,
                "lambda_fixed": best_lam if name == "soft_mix_v5_fixed" else np.nan,
                **m,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    boot_rows = []
    comparisons = {
        "fuzzy_v5_beacon_vs_uniform": (s_fb, s_fu),
        "soft_mix_v5_fixed_vs_logit_full10": (s_mix, s_logit),
    }
    metric_fns = {
        "delta_auroc": lambda yy, ss: float(roc_auc_score(yy, ss)) if len(np.unique(yy)) >= 2 else float("nan"),
        "delta_auprc": lambda yy, ss: float(average_precision_score(yy, ss)),
        "delta_f1_10": _f1_budget(0.10),
        "delta_f1_20": _f1_budget(0.20),
    }
    for cname, (a, b) in comparisons.items():
        for mname, fn in metric_fns.items():
            d, lo, hi, p = _bootstrap_delta(y[te], a, b, fn, n_boot=args.n_boot, seed=args.seed + abs(hash((cname, mname))) % 100000)
            boot_rows.append(
                {
                    "comparison": cname,
                    "metric": mname,
                    "delta": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                    "n_test": int(len(te)),
                }
            )

    pd.DataFrame(boot_rows).to_csv(args.out_bootstrap, index=False)
    print(f"saved: {out}")
    print(f"saved: {args.out_bootstrap}")


if __name__ == "__main__":
    main()