#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from beaconxai.tan_policy import bootstrap_delta_auroc, fit_tan_policy, metrics_binary, predict_proba_tan


FEATURE_SETS_V3 = {
    "tan_a_conflict_min": ["m_neg", "M_B_minus", "r_B_minus"],
    "tan_b_conflict_ce": ["m_neg", "M_B_minus", "CE_B"],
    "tan_c_rank_only": ["m_neg", "top1_delta", "top3_sum_delta", "top3_conflict_count"],
    "tan_d_conflict_frag": ["m_neg", "r_B_minus", "frag_drop"],
    "tan_e_full_compact": ["m_neg", "M_B_minus", "r_B_minus", "CE_B", "rho_B_cost", "frag_drop"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TAN v3 sweep")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--uniform-features", default="outputs_composite/part2_extended_v2/audit_features_uniform.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--bins", default="2,3,4,5,6")
    p.add_argument("--alpha", default="0.01,0.05,0.1,0.5,1.0,2.0,5.0")
    p.add_argument("--strategies", default="quantile,kmeans,uniform")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/tan_v3_sweep_results.csv")
    p.add_argument("--out-best", default="outputs_composite/part2_extended_v2/tan_v3_best.csv")
    return p.parse_args()


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if "delta_entropy" not in df.columns and "rank_entropy" in df.columns:
        df = df.copy()
        df["delta_entropy"] = df["rank_entropy"]
    return df


def main() -> None:
    args = parse_args()
    bins = [int(v.strip()) for v in args.bins.split(",") if v.strip()]
    alphas = [float(v.strip()) for v in args.alpha.split(",") if v.strip()]
    strategies = [v.strip() for v in args.strategies.split(",") if v.strip()]

    df_b = _prepare_df(pd.read_csv(args.beacon_features)).set_index("sample_id").sort_index()
    df_u = _prepare_df(pd.read_csv(args.uniform_features)).set_index("sample_id").sort_index()

    with Path(args.split_manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    tr = np.asarray(manifest["train_ids"], dtype=np.int64)
    va = np.asarray(manifest["val_ids"], dtype=np.int64)
    te = np.asarray(manifest["test_ids"], dtype=np.int64)

    y = df_b["is_hidden_conflict"].to_numpy(dtype=np.int64)

    rows = []
    for fs_name, fs_cols in FEATURE_SETS_V3.items():
        xb = df_b.loc[:, fs_cols].to_numpy(dtype=float)
        xu = df_u.loc[:, fs_cols].to_numpy(dtype=float)
        for strategy in strategies:
            for nb in bins:
                for al in alphas:
                    pol_b = fit_tan_policy(
                        xb[tr], y[tr], xb[va], y[va], n_bins=nb, alpha=al, strategy=strategy
                    )
                    pol_u = fit_tan_policy(
                        xu[tr], y[tr], xu[va], y[va], n_bins=nb, alpha=al, strategy=strategy
                    )

                    p_b = predict_proba_tan(pol_b, xb[te])
                    p_u = predict_proba_tan(pol_u, xu[te])
                    pred_b = (p_b >= float(pol_b["threshold"])).astype(np.int64)
                    pred_u = (p_u >= float(pol_u["threshold"])).astype(np.int64)
                    mb = metrics_binary(y[te], p_b, pred_b)
                    mu = metrics_binary(y[te], p_u, pred_u)
                    d_mean, d_lo, d_hi, pval = bootstrap_delta_auroc(
                        y[te], p_b, p_u, n_boot=args.bootstrap, seed=args.seed + nb * 29 + int(al * 1000)
                    )

                    rows.append(
                        {
                            "feature_set": fs_name,
                            "discretizer": strategy,
                            "n_bins": nb,
                            "alpha": al,
                            "val_auroc": pol_b["val_metrics"]["auroc"],
                            "val_auprc": pol_b["val_metrics"]["auprc"],
                            "val_f1": pol_b["val_metrics"]["f1"],
                            "test_auroc": mb["auroc"],
                            "test_auprc": mb["auprc"],
                            "test_f1": mb["f1"],
                            "test_precision": mb["precision"],
                            "test_recall": mb["recall"],
                            "uniform_test_auroc": mu["auroc"],
                            "uniform_test_auprc": mu["auprc"],
                            "uniform_test_f1": mu["f1"],
                            "delta_auroc_vs_uniform": d_mean,
                            "ci_low": d_lo,
                            "ci_high": d_hi,
                            "p_value": pval,
                        }
                    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values(["val_auroc", "val_auprc", "val_f1"], ascending=False)
    res.to_csv(out, index=False)

    best = res.head(1).copy()
    best.to_csv(args.out_best, index=False)
    print(f"saved: {out}")
    print(f"saved: {args.out_best}")


if __name__ == "__main__":
    main()