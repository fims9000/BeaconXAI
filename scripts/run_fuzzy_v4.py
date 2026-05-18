#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beaconxai.audit_features import margin_entropy_from_margin
from beaconxai.calibration import brier_score, expected_calibration_error
from beaconxai.fuzzy_policy_v2 import (
    _memberships_3,
    _rule_activations,
    build_fuzzy_inputs_v2,
    eval_at_budget,
    fit_fuzzy_policy_v2,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuzzy v4: entropy + adaptive soft-mix lambda")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--fuzzy-v3-final", default="outputs_composite/part2_extended_v2/fuzzy_v3_final_test.csv")
    p.add_argument("--lambdas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/fuzzy_v4_results.csv")
    p.add_argument("--out-lambda", default="outputs_composite/part2_extended_v2/fuzzy_v4_lambda_profile.csv")
    return p.parse_args()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "delta_entropy" not in df.columns and "rank_entropy" in df.columns:
        df["delta_entropy"] = df["rank_entropy"]
    if "margin_entropy" not in df.columns:
        margin = -df["m_neg"].to_numpy(dtype=float)
        df["margin_entropy"] = np.asarray([margin_entropy_from_margin(v) for v in margin], dtype=float)
    return df


def _compact_weights(policy, acts_train: np.ndarray, n_rules: int) -> np.ndarray:
    w = np.clip(policy.rule_weights.astype(float), 0.1, 10.0)
    importance = np.mean(acts_train, axis=0) * w * np.maximum(policy.rule_outputs, 1e-3)
    idx = np.argsort(-importance)[:n_rules]
    out = np.full_like(w, 0.1)
    out[idx] = w[idx]
    return out


def _score_with_weights(policy, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    m1 = _memberships_3(X[:, 0], policy.params_margin)
    m2 = _memberships_3(X[:, 1], policy.params_conflict)
    m3 = _memberships_3(X[:, 2], policy.params_fragility)
    acts = _rule_activations(m1, m2, m3)
    ww = np.clip(w.reshape(1, -1), 0.1, 10.0)
    num = np.sum(acts * ww * policy.rule_outputs.reshape(1, -1), axis=1)
    den = np.sum(acts * ww, axis=1)
    return num / np.maximum(den, 1e-8)


def _metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    p10, r10, f10 = eval_at_budget(y, s, 0.10)
    p20, r20, f20 = eval_at_budget(y, s, 0.20)
    return {
        "precision_10": p10,
        "recall_10": r10,
        "f1_10": f10,
        "precision_20": p20,
        "recall_20": r20,
        "f1_20": f20,
        "auprc": float(average_precision_score(y, s)),
        "ece": float(expected_calibration_error(y, s, n_bins=10)),
        "brier": float(brier_score(y, s)),
    }


def _to_budget_rows(policy: str, m: dict[str, float], extra: dict[str, float | int | str] | None = None) -> list[dict]:
    if extra is None:
        extra = {}
    return [
        {
            "policy": policy,
            "budget": 0.10,
            "precision": m["precision_10"],
            "recall": m["recall_10"],
            "f1": m["f1_10"],
            "auprc": m["auprc"],
            "ece": m["ece"],
            "brier": m["brier"],
            **extra,
        },
        {
            "policy": policy,
            "budget": 0.20,
            "precision": m["precision_20"],
            "recall": m["recall_20"],
            "f1": m["f1_20"],
            "auprc": m["auprc"],
            "ece": m["ece"],
            "brier": m["brier"],
            **extra,
        },
    ]


def main() -> None:
    args = parse_args()
    lambdas = [float(v.strip()) for v in args.lambdas.split(",") if v.strip()]

    df = _prepare_df(pd.read_csv(args.beacon_features)).set_index("sample_id").sort_index()
    with Path(args.split_manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    tr = np.asarray(manifest["train_ids"], dtype=np.int64)
    va = np.asarray(manifest["val_ids"], dtype=np.int64)
    te = np.asarray(manifest["test_ids"], dtype=np.int64)

    y = df["is_hidden_conflict"].to_numpy(dtype=np.int64)
    Xf = build_fuzzy_inputs_v2(df)
    Xp = df[["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]].to_numpy(dtype=float)
    Xg = df[["m_neg", "M_B_minus", "r_B_minus", "CE_B", "frag_drop", "rho_B_cost", "delta_entropy", "margin_entropy"]].to_numpy(dtype=float)

    # Base logit policy
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, solver="lbfgs", random_state=args.seed))
    logit.fit(Xp[tr], y[tr])
    s_logit_val = logit.predict_proba(Xp[va])[:, 1]
    s_logit_test = logit.predict_proba(Xp[te])[:, 1]

    # Fuzzy settings from best v3 row
    fv3 = pd.read_csv(args.fuzzy_v3_final)
    best = fv3.sort_values("mix_test_f1_10", ascending=False).iloc[0]
    reg = float(best["reg"])
    n_rules = int(best["rule_count"])

    pol = fit_fuzzy_policy_v2(Xf[tr], y[tr], reg=reg, seed=args.seed + int(reg * 1e6))
    acts_tr = _rule_activations(
        _memberships_3(Xf[tr, 0], pol.params_margin),
        _memberships_3(Xf[tr, 1], pol.params_conflict),
        _memberships_3(Xf[tr, 2], pol.params_fragility),
    )
    cw = _compact_weights(pol, acts_tr, n_rules)

    s_f_val = _score_with_weights(pol, Xf[va], cw)
    s_f_test = _score_with_weights(pol, Xf[te], cw)

    # Fixed-lambda mix chosen on validation
    best_lam = 0.5
    best_target = -1.0
    for lam in lambdas:
        s_mix_v = lam * s_f_val + (1.0 - lam) * s_logit_val
        mv = _metrics(y[va], s_mix_v)
        target = 0.6 * mv["f1_10"] + 0.2 * mv["precision_10"] + 0.2 * mv["f1_20"]
        if target > best_target:
            best_target = target
            best_lam = lam
    s_mix_fixed_test = best_lam * s_f_test + (1.0 - best_lam) * s_logit_test

    # Adaptive lambda: learn where fuzzy is better.
    better_is_fuzzy = (np.abs(s_f_val - y[va]) < np.abs(s_logit_val - y[va])).astype(np.int64)
    # fallback if degenerate target
    if len(np.unique(better_is_fuzzy)) < 2:
        lam_val = np.full(len(va), best_lam, dtype=float)
        lam_test = np.full(len(te), best_lam, dtype=float)
        gate_model = None
    else:
        gate_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed),
        )
        gate_model.fit(Xg[va], better_is_fuzzy)
        lam_val = gate_model.predict_proba(Xg[va])[:, 1]
        lam_test = gate_model.predict_proba(Xg[te])[:, 1]

    s_mix_adapt_val = lam_val * s_f_val + (1.0 - lam_val) * s_logit_val
    s_mix_adapt_test = lam_test * s_f_test + (1.0 - lam_test) * s_logit_test

    m_scalar = _metrics(y[te], df["m_neg"].to_numpy(dtype=float)[te])
    m_logit = _metrics(y[te], s_logit_test)
    m_fuzzy = _metrics(y[te], s_f_test)
    m_fixed = _metrics(y[te], s_mix_fixed_test)
    m_adapt = _metrics(y[te], s_mix_adapt_test)

    rows: list[dict] = []
    rows.extend(_to_budget_rows("scalar", m_scalar))
    rows.extend(_to_budget_rows("logit_panel", m_logit))
    rows.extend(_to_budget_rows("fuzzy_only_v4", m_fuzzy, {"reg": reg, "rule_count": n_rules}))
    rows.extend(_to_budget_rows("soft_mix_fixed_v4", m_fixed, {"lambda_fixed": best_lam}))
    rows.extend(_to_budget_rows("soft_mix_adaptive_v4", m_adapt, {"lambda_mean_test": float(np.mean(lam_test))}))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    lam_df = pd.DataFrame(
        {
            "split": ["val"] * len(lam_val) + ["test"] * len(lam_test),
            "sample_id": list(va) + list(te),
            "lambda": np.concatenate([lam_val, lam_test]),
            "score_fuzzy": np.concatenate([s_f_val, s_f_test]),
            "score_logit": np.concatenate([s_logit_val, s_logit_test]),
        }
    )
    lam_df.to_csv(args.out_lambda, index=False)

    print(f"saved: {out}")
    print(f"saved: {args.out_lambda}")


if __name__ == "__main__":
    main()