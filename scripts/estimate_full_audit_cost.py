#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate full BEACON audit cost envelope (model calls + policy layer).")
    p.add_argument("--profile-csv", default="outputs_composite/edge_portability_profile.csv")
    p.add_argument("--resource-csv", default="outputs_composite/edge_resource_budget_table.csv")
    p.add_argument("--out", default="outputs_composite/tinyxai_full_audit_cost.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prof = pd.read_csv(args.profile_csv)
    res = pd.read_csv(args.resource_csv)

    inf = prof[prof["method"] == "inference_only"]
    if inf.empty:
        raise ValueError("inference_only row not found in profile csv")
    inf_p50_ms = float(inf["latency_p50_ms"].iloc[0])

    out_rows = []
    for _, r in res.iterrows():
        method = str(r["method"])
        calls = float(r["mean_model_calls"])
        latency_p50 = float(r["latency_p50_ms"])
        latency_p95 = float(r["latency_p95_ms"])
        model_infer_est = calls * inf_p50_ms
        audit_extract_est = max(0.0, latency_p50 - model_infer_est)
        state_kb = float(r["audit_state_kb_est"])
        rss_mb = float(r["rss_delta_mb"])

        out_rows.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "method": method,
                "q_max": int(r["q_max"]),
                "model_calls": calls,
                "inference_only_p50_ms": inf_p50_ms,
                "model_inference_est_p50_ms": model_infer_est,
                "audit_feature_extraction_est_p50_ms": audit_extract_est,
                "policy_layer_est_us": 10.0 if "core" in method else 12.0,
                "total_audit_p50_ms": latency_p50,
                "total_audit_p95_ms": latency_p95,
                "policy_share_percent_est": (0.010 / max(latency_p50, 1e-9)) * 100.0 if "core" in method else (0.012 / max(latency_p50, 1e-9)) * 100.0,
                "audit_state_kb_est": state_kb,
                "rss_delta_mb_est": rss_mb,
                "note": "simulation-based split; full audit cost dominated by model calls",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

