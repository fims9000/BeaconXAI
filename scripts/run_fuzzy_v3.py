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

from beaconxai.calibration import brier_score, expected_calibration_error
from beaconxai.fuzzy_policy_v2 import (
    _memberships_3,
    _rule_activations,
    build_fuzzy_inputs_v2,
    eval_at_budget,
    fit_fuzzy_policy_v2,
    predict_fuzzy_policy_v2,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fuzzy v3 sweep with compact rule sets and soft mix")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--regs", default="1e-4,1e-3,1e-2")
    p.add_argument("--rule-counts", default="5,7,9,27")
    p.add_argument("--lambdas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-sweep", default="outputs_composite/part2_extended_v2/fuzzy_v3_sweep_results.csv")
    p.add_argument("--out-final", default="outputs_composite/part2_extended_v2/fuzzy_v3_final_test.csv")
    p.add_argument("--out-grid", default="outputs_composite/part2_extended_v2/fuzzy_v3_softmix_grid.csv")
    return p.parse_args()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if "delta_entropy" not in df.columns and "rank_entropy" in df.columns:
        df = df.copy()
        df["delta_entropy"] = df["rank_entropy"]
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


def _row_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
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


def main() -> None:
    args = parse_args()
    regs = [float(v.strip()) for v in args.regs.split(",") if v.strip()]
    rule_counts = [int(v.strip()) for v in args.rule_counts.split(",") if v.strip()]
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

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, solver="lbfgs", random_state=args.seed))
    logit.fit(Xp[tr], y[tr])
    s_logit_val = logit.predict_proba(Xp[va])[:, 1]
    s_logit_test = logit.predict_proba(Xp[te])[:, 1]

    sweep_rows = []
    final_rows = []
    softmix_rows = []

    for reg in regs:
        pol = fit_fuzzy_policy_v2(Xf[tr], y[tr], reg=reg, seed=args.seed + int(reg * 1e6))

        m1_tr = _memberships_3(Xf[tr, 0], pol.params_margin)
        m2_tr = _memberships_3(Xf[tr, 1], pol.params_conflict)
        m3_tr = _memberships_3(Xf[tr, 2], pol.params_fragility)
        acts_tr = _rule_activations(m1_tr, m2_tr, m3_tr)

        for nr in rule_counts:
            cw = _compact_weights(pol, acts_tr, nr)
            s_f_val = _score_with_weights(pol, Xf[va], cw)
            s_f_test = _score_with_weights(pol, Xf[te], cw)

            m_val = _row_metrics(y[va], s_f_val)
            m_test = _row_metrics(y[te], s_f_test)

            best_lam = 1.0
            best_target = -1.0
            best_val = None
            best_test = None
            for lam in lambdas:
                s_mix_val = lam * s_logit_val + (1.0 - lam) * s_f_val
                s_mix_test = lam * s_logit_test + (1.0 - lam) * s_f_test
                mv = _row_metrics(y[va], s_mix_val)
                mt = _row_metrics(y[te], s_mix_test)
                target = 0.6 * mv["f1_10"] + 0.2 * mv["precision_10"] + 0.2 * mv["f1_20"]
                softmix_rows.append(
                    {
                        "reg": reg,
                        "rule_count": nr,
                        "lambda": lam,
                        "val_f1_10": mv["f1_10"],
                        "val_precision_10": mv["precision_10"],
                        "val_f1_20": mv["f1_20"],
                        "val_auprc": mv["auprc"],
                        "test_f1_10": mt["f1_10"],
                        "test_precision_10": mt["precision_10"],
                        "test_f1_20": mt["f1_20"],
                        "test_auprc": mt["auprc"],
                        "val_target": target,
                    }
                )
                if target > best_target:
                    best_target = target
                    best_lam = lam
                    best_val = mv
                    best_test = mt

            assert best_val is not None and best_test is not None

            sweep_rows.append(
                {
                    "policy": "fuzzy_only",
                    "membership": "kmeans_3bin",
                    "inference": "sugeno_weighted",
                    "rule_set": f"compact{nr}" if nr < 27 else "full27",
                    "rule_count": nr,
                    "reg": reg,
                    **{f"val_{k}": v for k, v in m_val.items()},
                    **{f"test_{k}": v for k, v in m_test.items()},
                }
            )
            sweep_rows.append(
                {
                    "policy": "soft_mix_logit_fuzzy",
                    "membership": "kmeans_3bin",
                    "inference": "sugeno_weighted",
                    "rule_set": f"compact{nr}" if nr < 27 else "full27",
                    "rule_count": nr,
                    "reg": reg,
                    "lambda": best_lam,
                    **{f"val_{k}": v for k, v in best_val.items()},
                    **{f"test_{k}": v for k, v in best_test.items()},
                }
            )

            final_rows.append(
                {
                    "reg": reg,
                    "rule_count": nr,
                    "rule_set": f"compact{nr}" if nr < 27 else "full27",
                    "fuzzy_test_f1_10": m_test["f1_10"],
                    "fuzzy_test_precision_10": m_test["precision_10"],
                    "fuzzy_test_f1_20": m_test["f1_20"],
                    "fuzzy_test_auprc": m_test["auprc"],
                    "mix_lambda": best_lam,
                    "mix_test_f1_10": best_test["f1_10"],
                    "mix_test_precision_10": best_test["precision_10"],
                    "mix_test_f1_20": best_test["f1_20"],
                    "mix_test_auprc": best_test["auprc"],
                }
            )

    out_sweep = Path(args.out_sweep)
    out_sweep.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sweep_rows).to_csv(out_sweep, index=False)
    pd.DataFrame(final_rows).sort_values("mix_test_f1_10", ascending=False).to_csv(args.out_final, index=False)
    pd.DataFrame(softmix_rows).to_csv(args.out_grid, index=False)

    print(f"saved: {out_sweep}")
    print(f"saved: {args.out_final}")
    print(f"saved: {args.out_grid}")


if __name__ == "__main__":
    main()