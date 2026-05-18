#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beaconxai.calibration import brier_score, calibration_slope, expected_calibration_error
from beaconxai.fuzzy_policy_v2 import build_fuzzy_inputs_v2, eval_at_budget, fit_fuzzy_policy_v2, predict_fuzzy_policy_v2
from beaconxai.tan_policy import FEATURE_SETS, fit_tan_policy, predict_proba_tan


def _eval(y: np.ndarray, s: np.ndarray, budget: float) -> dict[str, float]:
    p, r, f1 = eval_at_budget(y, s, budget)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "auprc": float(average_precision_score(y, s)),
        "ece": float(expected_calibration_error(y, s, n_bins=10)),
        "brier": float(brier_score(y, s)),
        "calibration_slope": float(calibration_slope(y, s)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Soft mix policy sweep: logit + fuzzy")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--tan-final", default="outputs_composite/part2_extended_v2/tan_final_test.csv")
    p.add_argument("--fuzzy-results", default="outputs_composite/part2_extended_v2/fuzzy_policy_results.csv")
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/soft_mix_results.csv")
    p.add_argument("--out-scores", default="outputs_composite/part2_extended_v2/policy_scores_long.csv")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.beacon_features).set_index("sample_id").sort_index()
    with Path(args.split_manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    tr = np.asarray(manifest["train_ids"], dtype=np.int64)
    va = np.asarray(manifest["val_ids"], dtype=np.int64)
    te = np.asarray(manifest["test_ids"], dtype=np.int64)

    y = df["is_hidden_conflict"].to_numpy(dtype=np.int64)
    x_scalar = df[["m_neg"]].to_numpy(dtype=float)
    x_panel = df[["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]].to_numpy(dtype=float)
    x_fuzzy = build_fuzzy_inputs_v2(df.reset_index())

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=args.seed))
    logit.fit(x_panel[tr], y[tr])
    p_logit_val = logit.predict_proba(x_panel[va])[:, 1]
    p_logit_test = logit.predict_proba(x_panel[te])[:, 1]

    fuzzy_cfg = pd.read_csv(args.fuzzy_results).iloc[0]
    pol_f = fit_fuzzy_policy_v2(x_fuzzy[tr], y[tr], reg=float(fuzzy_cfg["reg"]), seed=args.seed)
    p_fuzzy_val = predict_fuzzy_policy_v2(pol_f, x_fuzzy[va])
    p_fuzzy_test = predict_fuzzy_policy_v2(pol_f, x_fuzzy[te])

    tan_cfg = pd.read_csv(args.tan_final).iloc[0]
    fs = FEATURE_SETS[str(tan_cfg["feature_set"])]
    x_tan = df[fs].to_numpy(dtype=float)
    pol_tan = fit_tan_policy(
        x_tan[tr], y[tr], x_tan[va], y[va],
        n_bins=int(tan_cfg["n_bins"]), alpha=float(tan_cfg["alpha"])
    )
    p_tan_test = predict_proba_tan(pol_tan, x_tan[te])

    # validation sweep for lambda
    best_l = 1.0
    best_score = -1.0
    grid_rows = []
    for lmb in np.linspace(0.0, 1.0, 21):
        s = lmb * p_logit_val + (1.0 - lmb) * p_fuzzy_val
        m10 = _eval(y[va], s, 0.10)
        m20 = _eval(y[va], s, 0.20)
        target = 0.5 * m10["f1"] + 0.25 * m10["precision"] + 0.25 * m20["f1"]
        grid_rows.append({"lambda": float(lmb), "val_f1_10": m10["f1"], "val_precision_10": m10["precision"], "val_f1_20": m20["f1"], "val_target": target})
        if target > best_score:
            best_score = target
            best_l = float(lmb)

    p_mix_test = best_l * p_logit_test + (1.0 - best_l) * p_fuzzy_test
    p_scalar_test = x_scalar[te, 0]

    policies = {
        "scalar": p_scalar_test,
        "logit_panel": p_logit_test,
        "fuzzy_only": p_fuzzy_test,
        "tan_only": p_tan_test,
        "soft_mix_logit_fuzzy": p_mix_test,
    }

    rows = []
    for name, s in policies.items():
        for b in [0.10, 0.20]:
            m = _eval(y[te], s, b)
            rows.append({
                "policy": name,
                "budget": b,
                "lambda": best_l if name == "soft_mix_logit_fuzzy" else np.nan,
                **m,
                "n_test": int(len(te)),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    score_rows = []
    for idx in te:
        score_rows.append({"sample_id": int(idx), "split": "test", "y_true": int(y[idx]), "policy": "scalar", "score": float(x_scalar[idx, 0])})
        score_rows.append({"sample_id": int(idx), "split": "test", "y_true": int(y[idx]), "policy": "logit_panel", "score": float(logit.predict_proba(x_panel[idx:idx+1])[:, 1][0])})
        score_rows.append({"sample_id": int(idx), "split": "test", "y_true": int(y[idx]), "policy": "fuzzy_only", "score": float(predict_fuzzy_policy_v2(pol_f, x_fuzzy[idx:idx+1])[0])})
        score_rows.append({"sample_id": int(idx), "split": "test", "y_true": int(y[idx]), "policy": "tan_only", "score": float(predict_proba_tan(pol_tan, x_tan[idx:idx+1])[0])})
        score_rows.append({"sample_id": int(idx), "split": "test", "y_true": int(y[idx]), "policy": "soft_mix_logit_fuzzy", "score": float(best_l * logit.predict_proba(x_panel[idx:idx+1])[:, 1][0] + (1.0 - best_l) * predict_fuzzy_policy_v2(pol_f, x_fuzzy[idx:idx+1])[0])})

    pd.DataFrame(score_rows).to_csv(args.out_scores, index=False)
    pd.DataFrame(grid_rows).to_csv(out.with_name("soft_mix_lambda_grid.csv"), index=False)
    print(f"saved: {out}")
    print(f"saved: {args.out_scores}")
    print(f"saved: {out.with_name('soft_mix_lambda_grid.csv')}")


if __name__ == "__main__":
    main()
