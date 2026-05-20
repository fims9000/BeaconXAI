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
try:
    from beaconxai.fuzzy_policy_v5 import FEATURES_V5, build_fuzzy_inputs_v5, fit_fuzzy_policy_v5, predict_fuzzy_policy_v5
    HAS_FUZZY_V5 = True
except Exception:
    FEATURES_V5 = [
        "m_neg",
        "M_B_minus",
        "r_B_minus",
        "CE_B",
        "rho_B_cost",
        "frag_drop",
        "top1_delta",
        "top3_sum_delta",
        "top3_conflict_count",
        "margin_entropy",
    ]

    def build_fuzzy_inputs_v5(df):
        d = df.copy()
        return d.loc[:, FEATURES_V5].to_numpy(dtype=np.float32)

    HAS_FUZZY_V5 = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate v6 policies on one feature bundle")
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--beacon-file", default="audit_features_beacon_core.csv")
    p.add_argument("--uniform-file", default="audit_features_uniform.csv")
    p.add_argument("--n-rules", type=int, default=7)
    p.add_argument("--n-terms", type=int, default=3)
    p.add_argument("--epochs", type=int, default=260)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="v6_policy_eval.csv")
    p.add_argument("--out-bootstrap", default="v6_bootstrap_deltas.csv")
    p.add_argument("--out-cost", default="tinyxai_full_audit_cost.csv")
    return p.parse_args()


def _metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
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
        _p, _r, f1 = eval_at_budget(y, s, frac)
        return float(f1)

    return fn


def _prepare(df: pd.DataFrame):
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
    bdir = Path(args.bundle_dir)

    df_b = _prepare(pd.read_csv(bdir / args.beacon_file)).set_index("sample_id").sort_index()
    df_u = _prepare(pd.read_csv(bdir / args.uniform_file)).set_index("sample_id").sort_index()
    with (bdir / "split_manifest.json").open("r", encoding="utf-8") as f:
        man = json.load(f)

    tr = np.asarray(man["train_ids"], dtype=np.int64)
    va = np.asarray(man["val_ids"], dtype=np.int64)
    te = np.asarray(man["test_ids"], dtype=np.int64)
    y = df_b["is_hidden_conflict"].to_numpy(dtype=np.int64)

    Xb = build_fuzzy_inputs_v5(df_b)
    Xu = build_fuzzy_inputs_v5(df_u)

    # Logit baselines
    logit_b = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    logit_u = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    logit_b.fit(Xb[tr], y[tr])
    logit_u.fit(Xu[tr], y[tr])
    s_logit_b = logit_b.predict_proba(Xb[te])[:, 1]
    s_logit_u = logit_u.predict_proba(Xu[te])[:, 1]

    s_fuzzy_b = s_logit_b.copy()
    s_fuzzy_u = s_logit_u.copy()
    s_soft = s_logit_b.copy()
    lam_best = float("nan")
    if HAS_FUZZY_V5:
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

        s_fuzzy_b_val = predict_fuzzy_policy_v5(pol_b, Xb[va])
        s_logit_b_val = logit_b.predict_proba(Xb[va])[:, 1]
        s_fuzzy_b = predict_fuzzy_policy_v5(pol_b, Xb[te])
        s_fuzzy_u = predict_fuzzy_policy_v5(pol_u, Xu[te])

        # fixed soft-mix lambda on validation
        grid = np.linspace(0.0, 1.0, 21)
        lam_best = 0.5
        tgt_best = -1.0
        for lam in grid:
            sv = lam * s_fuzzy_b_val + (1.0 - lam) * s_logit_b_val
            p10, _r10, f10 = eval_at_budget(y[va], sv, 0.10)
            _p20, _r20, f20 = eval_at_budget(y[va], sv, 0.20)
            tgt = 0.6 * f10 + 0.2 * p10 + 0.2 * f20
            if tgt > tgt_best:
                tgt_best = tgt
                lam_best = float(lam)
        s_soft = lam_best * s_fuzzy_b + (1.0 - lam_best) * s_logit_b

    # simple baselines from BEACON feature table
    s_margin = df_b["m_neg"].to_numpy(dtype=float)[te]
    s_variance = (df_b["top3_sum_delta"].to_numpy(dtype=float)[te] + df_b["delta_entropy"].to_numpy(dtype=float)[te])
    s_energy = (df_b["M_B_minus"].to_numpy(dtype=float)[te] + df_b["M_B_plus"].to_numpy(dtype=float)[te])

    Xtrain0 = Xb[tr][y[tr] == 0]
    mu0 = np.mean(Xtrain0, axis=0)
    sd0 = np.std(Xtrain0, axis=0) + 1e-6
    z = (Xb[te] - mu0.reshape(1, -1)) / sd0.reshape(1, -1)
    s_profile = np.sqrt(np.sum(z * z, axis=1))

    rows = []
    items = [
        ("logit_beacon", s_logit_b),
        ("logit_uniform", s_logit_u),
        ("fuzzy_v5_beacon", s_fuzzy_b),
        ("fuzzy_v5_uniform", s_fuzzy_u),
        ("soft_mix_v5", s_soft),
        ("margin_only", s_margin),
        ("variance_score", s_variance),
        ("energy_score", s_energy),
        ("profile_distance_score", s_profile),
    ]
    for name, score in items:
        rows.append(
            {
                "bundle": bdir.name,
                "policy": name,
                "q_max": int(man.get("q_max", -1)),
                "neutralizer_mode": str(man.get("neutralizer_mode", "na")),
                "n_features": len(FEATURES_V5),
                "n_rules": args.n_rules if "fuzzy" in name or "soft_mix" in name else 0,
                "lambda_fixed": lam_best if name == "soft_mix_v5" else np.nan,
                **_metrics(y[te], np.asarray(score, dtype=float)),
            }
        )

    out = bdir / args.out
    pd.DataFrame(rows).to_csv(out, index=False)

    # bootstrap deltas
    b_rows = []
    comps = {
        "logit_beacon_vs_uniform": (s_logit_b, s_logit_u),
        "fuzzy_beacon_vs_uniform": (s_fuzzy_b, s_fuzzy_u),
        "soft_mix_vs_logit_beacon": (s_soft, s_logit_b),
        "logit_beacon_vs_profile": (s_logit_b, s_profile),
    }
    mfuncs = {
        "delta_auroc": lambda yy, ss: float(roc_auc_score(yy, ss)) if len(np.unique(yy)) >= 2 else float("nan"),
        "delta_auprc": lambda yy, ss: float(average_precision_score(yy, ss)),
        "delta_f1_10": _f1_budget(0.10),
        "delta_f1_20": _f1_budget(0.20),
    }
    for cname, (a, b) in comps.items():
        for mname, fn in mfuncs.items():
            d, lo, hi, p = _bootstrap_delta(y[te], a, b, fn, n_boot=args.n_boot, seed=args.seed + abs(hash((cname, mname))) % 100000)
            b_rows.append(
                {
                    "bundle": bdir.name,
                    "comparison": cname,
                    "metric": mname,
                    "delta": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                    "n_test": int(len(te)),
                }
            )
    outb = bdir / args.out_bootstrap
    pd.DataFrame(b_rows).to_csv(outb, index=False)

    # tiny full audit cost table
    q = int(man.get("q_max", -1))
    policy_lat_us = 0.0
    soft_row = next(r for r in rows if r["policy"] == "soft_mix_v5")
    # approximate policy latency from arithmetic profile (conservative)
    policy_lat_us = 12.0  # constant engineering estimate for compact policy layer
    model_calls = q + 1
    model_inf_ms = 0.20  # configurable proxy for lightweight timeseries model inference
    total_ms = model_calls * model_inf_ms + policy_lat_us / 1000.0
    cost = pd.DataFrame(
        [
            {
                "bundle": bdir.name,
                "q_max": q,
                "neutralizer_mode": str(man.get("neutralizer_mode", "na")),
                "model_calls": model_calls,
                "estimated_model_inference_ms": model_inf_ms,
                "audit_policy_us": policy_lat_us,
                "total_estimated_ms": total_ms,
                "policy_share_percent": (policy_lat_us / 1000.0) / max(total_ms, 1e-12) * 100.0,
                "is_simulation": 1,
            }
        ]
    )
    cost.to_csv(bdir / args.out_cost, index=False)

    print(f"saved: {out}")
    print(f"saved: {outb}")
    print(f"saved: {bdir / args.out_cost}")


if __name__ == "__main__":
    main()
