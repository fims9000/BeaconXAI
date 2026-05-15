#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build compact edge-resource budget table from portability profile CSV."
    )
    p.add_argument(
        "--profile-csv",
        default="outputs_composite/edge_portability_profile.csv",
        help="Input CSV from scripts/measure_portability.py",
    )
    p.add_argument(
        "--out",
        default="outputs_composite/edge_resource_budget_table.csv",
        help="Output compact table",
    )
    p.add_argument(
        "--window-ms",
        type=float,
        default=100.0,
        help="Reference sensor window budget in ms for utilization ratio",
    )
    return p.parse_args()


def estimate_state_bytes(model_calls: int) -> int:
    # Algorithmic state estimate (array-level, language-agnostic):
    # delta(float64) + comp_id(int32) + rank(int32) + queue(score,id; float64+int32, padded)
    per_component_bytes = 8 + 4 + 4 + 16
    return int(model_calls * per_component_bytes)


def main() -> None:
    args = parse_args()
    in_path = Path(args.profile_csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty profile CSV: {in_path}")

    inference = next((r for r in rows if r.get("method") == "inference_only"), None)
    inf_p50 = _f(inference or {}, "latency_p50_ms", 1.0)

    keep = []
    for r in rows:
        method = (r.get("method") or "").strip()
        if not method.startswith("beacon_"):
            continue
        calls = int(round(_f(r, "mean_model_calls", 0.0)))
        p50 = _f(r, "latency_p50_ms", 0.0)
        p95 = _f(r, "latency_p95_ms", 0.0)
        state_b = estimate_state_bytes(calls)
        keep.append(
            {
                "dataset": r.get("dataset", ""),
                "model": r.get("model", ""),
                "method": method,
                "q_max": r.get("q_max", ""),
                "mean_model_calls": f"{_f(r, 'mean_model_calls', 0.0):.3f}",
                "latency_p50_ms": f"{p50:.3f}",
                "latency_p95_ms": f"{p95:.3f}",
                "cpu_p50_ms": f"{_f(r, 'cpu_p50_ms', 0.0):.3f}",
                "rss_delta_mb": f"{_f(r, 'rss_delta_mb', 0.0):.3f}",
                "audit_state_bytes_est": str(state_b),
                "audit_state_kb_est": f"{state_b / 1024.0:.3f}",
                "overhead_vs_inference_x": f"{(p50 / inf_p50) if inf_p50 > 0 else 0.0:.3f}",
                "window_utilization_p95_vs_100ms": f"{(p95 / args.window_ms):.3f}",
                "affinity_applied": r.get("affinity_applied", ""),
                "nice_applied": r.get("nice_applied", ""),
                "cpu_model": r.get("cpu_model", ""),
                "cpu_mhz_snapshot": r.get("cpu_mhz_snapshot", ""),
            }
        )

    fields = [
        "dataset",
        "model",
        "method",
        "q_max",
        "mean_model_calls",
        "latency_p50_ms",
        "latency_p95_ms",
        "cpu_p50_ms",
        "rss_delta_mb",
        "audit_state_bytes_est",
        "audit_state_kb_est",
        "overhead_vs_inference_x",
        "window_utilization_p95_vs_100ms",
        "affinity_applied",
        "nice_applied",
        "cpu_model",
        "cpu_mhz_snapshot",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
