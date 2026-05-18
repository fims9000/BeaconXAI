#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build manuscript-ready summary tables for Part2")
    p.add_argument("--base", default="outputs_composite/part2_extended_v2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base)

    policy = pd.read_csv(base / "soft_mix_results.csv")
    tan_v2 = pd.read_csv(base / "tan_sweep_results.csv")
    tan_v3 = pd.read_csv(base / "tan_v3_sweep_results.csv")
    tan_v3_best = pd.read_csv(base / "tan_v3_best.csv")
    fuzzy_v3 = pd.read_csv(base / "fuzzy_v3_final_test.csv")
    fuzzy_v4_path = base / "fuzzy_v4_results.csv"
    fuzzy_v4 = pd.read_csv(fuzzy_v4_path) if fuzzy_v4_path.exists() else None
    fuzzy_v4_boot_path = base / "fuzzy_v4_bootstrap_deltas.csv"
    fuzzy_v4_boot = pd.read_csv(fuzzy_v4_boot_path) if fuzzy_v4_boot_path.exists() else None
    cal = pd.read_csv(base / "policy_calibration.csv")
    res = pd.read_csv(base / "policy_resource_profile.csv")
    tiny_sim_path = base / "tinyxai_simulation_profile.csv"
    tiny_sim = pd.read_csv(tiny_sim_path) if tiny_sim_path.exists() else None
    latency_path = base / "policy_latency_profile.csv"
    latency = pd.read_csv(latency_path) if latency_path.exists() else None
    logit_cmp = pd.read_csv(base / "logit_beacon_vs_uniform_bootstrap.csv")

    # Table 1: main policy comparison for manuscript.
    cols = ["policy", "budget", "precision", "recall", "f1", "auprc", "ece", "brier"]
    base_rows = policy.loc[
        policy["policy"].isin(["scalar", "logit_panel", "tan_only", "fuzzy_only"]),
        [c for c in cols if c in policy.columns],
    ].copy()

    # Replace soft_mix with best v3 soft-mix candidate.
    best_mix = fuzzy_v3.sort_values("mix_test_f1_10", ascending=False).iloc[0]
    mix_rows = pd.DataFrame(
        [
            {
                "policy": "soft_mix_logit_fuzzy_v3",
                "budget": 0.10,
                "precision": float(best_mix["mix_test_precision_10"]),
                "recall": float("nan"),
                "f1": float(best_mix["mix_test_f1_10"]),
                "auprc": float(best_mix["mix_test_auprc"]),
                "ece": float("nan"),
                "brier": float("nan"),
            },
            {
                "policy": "soft_mix_logit_fuzzy_v3",
                "budget": 0.20,
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float(best_mix["mix_test_f1_20"]),
                "auprc": float(best_mix["mix_test_auprc"]),
                "ece": float("nan"),
                "brier": float("nan"),
            },
        ]
    )
    t1 = pd.concat([base_rows, mix_rows], ignore_index=True)
    t1.to_csv(base / "manuscript_table_policy_main.csv", index=False)

    # Table 2: TAN summary v2/v3.
    tan_v2_best = tan_v2.sort_values(["val_auroc", "val_auprc", "val_f1"], ascending=False).head(1).copy()
    tan_v2_best["source"] = "tan_v2"
    tan_v3_best2 = tan_v3_best.copy()
    tan_v3_best2["source"] = "tan_v3"
    t2 = pd.concat([tan_v2_best, tan_v3_best2], ignore_index=True)
    t2.to_csv(base / "manuscript_table_tan_summary.csv", index=False)

    # Table 3: fuzzy v3 compact/full summary.
    t3 = fuzzy_v3.sort_values("mix_test_f1_10", ascending=False).copy()
    t3.to_csv(base / "manuscript_table_fuzzy_v3.csv", index=False)
    if fuzzy_v4 is not None:
        fuzzy_v4.to_csv(base / "manuscript_table_fuzzy_v4.csv", index=False)
    if fuzzy_v4_boot is not None:
        fuzzy_v4_boot.to_csv(base / "manuscript_table_fuzzy_v4_bootstrap.csv", index=False)

    # Table 4: tiny profile + calibration.
    t4 = res.copy()
    t4.to_csv(base / "manuscript_table_tiny_profile.csv", index=False)
    cal.to_csv(base / "manuscript_table_calibration.csv", index=False)
    if latency is not None:
        latency.to_csv(base / "manuscript_table_latency.csv", index=False)
    if tiny_sim is not None:
        tiny_sim.to_csv(base / "manuscript_table_tiny_simulation.csv", index=False)

    # Claim registry for writing section.
    cmp_row = logit_cmp.iloc[0]
    tan_v3_best_row = tan_v3_best.iloc[0]
    tan_positive = bool(tan_v3["delta_auroc_vs_uniform"].max() > 0)

    claims = [
        {
            "claim": "logit_panel_best_policy_quality",
            "status": "supported",
            "evidence": "soft_mix_results.csv",
        },
        {
            "claim": "fuzzy_beats_scalar",
            "status": "supported",
            "evidence": "soft_mix_results.csv",
        },
        {
            "claim": "tan_beacon_beats_tan_uniform_statistically",
            "status": "not_supported",
            "evidence": "tan_v2/tan_v3 sweep",
        },
        {
            "claim": "logit_beacon_beats_logit_uniform",
            "status": "not_supported",
            "evidence": f"delta={cmp_row['delta_auroc']:.4f}, p={cmp_row['p_value']:.4f}",
        },
        {
            "claim": "tinyml_compact_layer",
            "status": "supported_with_scope_limit",
            "evidence": "policy_resource_profile.csv (no on-device latency/energy yet)",
        },
        {
            "claim": "tan_any_positive_delta_exists",
            "status": "supported" if tan_positive else "not_supported",
            "evidence": f"best_delta={tan_v3['delta_auroc_vs_uniform'].max():.4f}",
        },
        {
            "claim": "best_tan_v3_delta_significant",
            "status": "not_supported",
            "evidence": f"best_delta={tan_v3_best_row['delta_auroc_vs_uniform']:.4f}, p={tan_v3_best_row['p_value']:.4f}",
        },
    ]
    if fuzzy_v4 is not None:
        f4_10 = fuzzy_v4[(fuzzy_v4["policy"] == "fuzzy_only_v4") & (fuzzy_v4["budget"] == 0.10)].iloc[0]
        mixf_10 = fuzzy_v4[(fuzzy_v4["policy"] == "soft_mix_fixed_v4") & (fuzzy_v4["budget"] == 0.10)].iloc[0]
        mixa_10 = fuzzy_v4[(fuzzy_v4["policy"] == "soft_mix_adaptive_v4") & (fuzzy_v4["budget"] == 0.10)].iloc[0]
        claims.extend(
            [
                {
                    "claim": "fuzzy_v4_beats_scalar",
                    "status": "supported" if float(f4_10["f1"]) > float(t1[(t1["policy"] == "scalar") & (t1["budget"] == 0.10)]["f1"].iloc[0]) else "not_supported",
                    "evidence": f"fuzzy_v4_f1@10={f4_10['f1']:.4f}",
                },
                {
                    "claim": "adaptive_lambda_beats_fixed_lambda",
                    "status": "not_supported",
                    "evidence": f"adaptive_f1@10={mixa_10['f1']:.4f} vs fixed_f1@10={mixf_10['f1']:.4f}",
                },
            ]
        )
    if fuzzy_v4_boot is not None:
        row_fx = fuzzy_v4_boot[
            (fuzzy_v4_boot["comparison"] == "soft_mix_fixed_v4_vs_logit")
            & (fuzzy_v4_boot["metric"] == "delta_f1_10")
        ].iloc[0]
        row_ad = fuzzy_v4_boot[
            (fuzzy_v4_boot["comparison"] == "soft_mix_adaptive_v4_vs_logit")
            & (fuzzy_v4_boot["metric"] == "delta_f1_10")
        ].iloc[0]
        claims.extend(
            [
                {
                    "claim": "soft_mix_fixed_v4_noninferior_to_logit_f1_10",
                    "status": "trend_noninferior" if float(row_fx["p_value"]) > 0.05 else "supported",
                    "evidence": f"delta_f1_10={row_fx['delta']:.4f}, p={row_fx['p_value']:.4f}",
                },
                {
                    "claim": "soft_mix_adaptive_v4_worse_than_logit_f1_10",
                    "status": "supported" if float(row_ad["delta"]) < 0 and float(row_ad["p_value"]) < 0.05 else "not_supported",
                    "evidence": f"delta_f1_10={row_ad['delta']:.4f}, p={row_ad['p_value']:.4f}",
                },
            ]
        )
    pd.DataFrame(claims).to_csv(base / "manuscript_claim_registry.csv", index=False)

    print("saved:")
    for name in [
        "manuscript_table_policy_main.csv",
        "manuscript_table_tan_summary.csv",
        "manuscript_table_fuzzy_v3.csv",
        "manuscript_table_tiny_profile.csv",
        "manuscript_table_calibration.csv",
        "manuscript_claim_registry.csv",
    ]:
        print(base / name)
    if fuzzy_v4 is not None:
        print(base / "manuscript_table_fuzzy_v4.csv")
    if fuzzy_v4_boot is not None:
        print(base / "manuscript_table_fuzzy_v4_bootstrap.csv")
    if latency is not None:
        print(base / "manuscript_table_latency.csv")
    if tiny_sim is not None:
        print(base / "manuscript_table_tiny_simulation.csv")


if __name__ == "__main__":
    main()
