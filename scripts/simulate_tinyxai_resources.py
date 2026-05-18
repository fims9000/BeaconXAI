#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate TinyXAI resource profile for MCU targets")
    p.add_argument("--resource-profile", default="outputs_composite/part2_extended_v2/policy_resource_profile.csv")
    p.add_argument("--latency-profile", default="outputs_composite/part2_extended_v2/policy_latency_profile.csv")
    p.add_argument("--freq-mhz", type=float, default=48.0)
    p.add_argument("--voltage-v", type=float, default=3.3)
    p.add_argument("--current-ma", type=float, default=5.0)
    p.add_argument("--audit-rate-hz", type=float, default=1000.0)
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/tinyxai_simulation_profile.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rp = pd.read_csv(args.resource_profile)
    lp_path = Path(args.latency_profile)
    lp = pd.read_csv(lp_path) if lp_path.exists() else pd.DataFrame(columns=["policy", "mean_us", "p95_us"])

    flash_base = {
        "scalar": 512,
        "logit_panel": 1200,
        "fuzzy_only": 1800,
        "tan": 1700,
        "soft_mix": 2200,
    }
    runtime_buf = {
        "scalar": 32,
        "logit_panel": 64,
        "fuzzy_only": 128,
        "tan": 96,
        "soft_mix": 160,
    }

    rows = []
    for _, r in rp.iterrows():
        pol = str(r["policy"])
        comps = float(r.get("n_comparisons", 0.0))
        adds = float(r.get("n_additions", 0.0))
        muls = float(r.get("n_multiplications", 0.0))
        lookups = float(r.get("n_table_lookups", 0.0))
        branches = float(r.get("branch_count", 0.0))

        cycles = comps * 1.0 + adds * 1.0 + muls * 2.0 + lookups * 2.0 + branches * 1.0 + 20.0
        time_us_est = cycles / max(args.freq_mhz, 1e-9)

        energy_uj_est = args.voltage_v * args.current_ma * time_us_est / 1000.0
        power_mw_at_rate = energy_uj_est * args.audit_rate_hz / 1000.0

        state_f32 = float(r.get("state_bytes_float32", 0.0))
        ram_est = state_f32 + float(runtime_buf.get(pol, 96))
        flash_est = float(r.get("n_params", 0.0)) * 4.0 + float(flash_base.get(pol, 1400))

        rows.append(
            {
                "policy": pol,
                "state_bytes_float32": state_f32,
                "ram_estimated_bytes": ram_est,
                "flash_estimated_bytes": flash_est,
                "cycles_estimated": cycles,
                "time_estimated_us_at_freq": time_us_est,
                "energy_estimated_uJ": energy_uj_est,
                "power_estimated_mW_at_audit_rate": power_mw_at_rate,
                "freq_mhz": args.freq_mhz,
                "current_ma": args.current_ma,
                "voltage_v": args.voltage_v,
                "audit_rate_hz": args.audit_rate_hz,
                "is_simulation": 1,
            }
        )

    out = pd.DataFrame(rows)
    if not lp.empty:
        out = out.merge(lp[["policy", "mean_us", "p95_us"]], on="policy", how="left")
    out = out.sort_values("time_estimated_us_at_freq")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()