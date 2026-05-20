#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "outputs_composite" / "part1_localization_q_sweep" / "raw"
OUT_DIR = ROOT / "outputs_composite" / "part1_localization_q_sweep"


def _f(x: str, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _i(x: str, default: int = -1) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset", "") == "dataset":
                continue
            out.append(row)
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + (z * z) / n
    c = p + (z * z) / (2.0 * n)
    m = z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    lo = (c - m) / den
    hi = (c + m) / den
    return (max(0.0, lo), min(1.0, hi))


def _neutralizer_from_name(name: str) -> str:
    if "_interp" in name:
        return "interp"
    if "_zero" in name:
        return "zero"
    return "unknown"


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = sorted(
        p for p in RAW_DIR.glob("*.csv") if not p.name.endswith("_per_sample.csv")
    )
    per_csv = sorted(p for p in RAW_DIR.glob("*_per_sample.csv"))

    summary_rows: list[dict[str, object]] = []
    boot_rows: list[dict[str, object]] = []
    claim_rows: list[dict[str, object]] = []

    # (dataset, neutralizer, q) -> method -> list[int]
    hit_store: dict[tuple[str, str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # per-sample delta export
    wisdm_delta_rows: list[dict[str, object]] = []

    for f in raw_csv:
        neutralizer = _neutralizer_from_name(f.name)
        rows = _load_csv_rows(f)
        for r in rows:
            dataset = r["dataset"]
            q = _i(r["q_max"])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "neutralizer": neutralizer,
                    "q_max": q,
                    "model": r.get("model", ""),
                    "partition_mode": r.get("partition_mode", ""),
                    "group_mode": r.get("group_mode", ""),
                    "time_bins": _i(r.get("time_bins", "")),
                    "n_components": _i(r.get("n_components", "")),
                    "n_eval": _i(r.get("n_eval", "")),
                    "n_conflict": _i(r.get("n_conflict", "")),
                    "loc_top1_beacon": _f(r.get("loc_top1_beacon", "")),
                    "loc_top1_uniform": _f(r.get("loc_top1_uniform", "")),
                    "loc_top1_logo": _f(r.get("loc_top1_logo", "")),
                    "loc_hit3_beacon": _f(r.get("loc_hit3_beacon", "")),
                    "loc_hit3_uniform": _f(r.get("loc_hit3_uniform", "")),
                    "loc_hit3_logo": _f(r.get("loc_hit3_logo", "")),
                    "loc_hit5_beacon": _f(r.get("loc_hit5_beacon", "")),
                    "loc_hit5_uniform": _f(r.get("loc_hit5_uniform", "")),
                    "loc_hit5_logo": _f(r.get("loc_hit5_logo", "")),
                    "loc_mrr_beacon": _f(r.get("loc_mrr_beacon", "")),
                    "loc_mrr_uniform": _f(r.get("loc_mrr_uniform", "")),
                    "loc_mrr_logo": _f(r.get("loc_mrr_logo", "")),
                    "loc_mean_rank_beacon": _f(r.get("loc_mean_rank_beacon", "")),
                    "loc_mean_rank_uniform": _f(r.get("loc_mean_rank_uniform", "")),
                    "loc_mean_rank_logo": _f(r.get("loc_mean_rank_logo", "")),
                    "loc_nrg_beacon": _f(r.get("loc_nrg_beacon", "")),
                    "loc_nrg_uniform": _f(r.get("loc_nrg_uniform", "")),
                    "loc_nrg_logo": _f(r.get("loc_nrg_logo", "")),
                    "delta_loc_top1_beacon_minus_uniform": _f(
                        r.get("delta_loc_top1_beacon_minus_uniform", "")
                    ),
                    "ci_delta_loc_top1_beacon_minus_uniform_low": _f(
                        r.get("ci_delta_loc_top1_beacon_minus_uniform_low", "")
                    ),
                    "ci_delta_loc_top1_beacon_minus_uniform_high": _f(
                        r.get("ci_delta_loc_top1_beacon_minus_uniform_high", "")
                    ),
                    "delta_loc_top1_beacon_minus_logo": _f(
                        r.get("delta_loc_top1_beacon_minus_logo", "")
                    ),
                    "ci_delta_loc_top1_beacon_minus_logo_low": _f(
                        r.get("ci_delta_loc_top1_beacon_minus_logo_low", "")
                    ),
                    "ci_delta_loc_top1_beacon_minus_logo_high": _f(
                        r.get("ci_delta_loc_top1_beacon_minus_logo_high", "")
                    ),
                    "mean_q_used": _f(r.get("mean_q_used", "")),
                    "latency_per_object_sec": _f(r.get("latency_per_object_sec", "")),
                }
            )

            boot_rows.append(
                {
                    "dataset": dataset,
                    "neutralizer": neutralizer,
                    "q_max": q,
                    "comparison": "beacon_minus_uniform",
                    "metric": "loc_top1",
                    "delta": _f(r.get("delta_loc_top1_beacon_minus_uniform", "")),
                    "ci_low": _f(r.get("ci_delta_loc_top1_beacon_minus_uniform_low", "")),
                    "ci_high": _f(r.get("ci_delta_loc_top1_beacon_minus_uniform_high", "")),
                    "frac_positive": _f(
                        r.get("frac_positive_delta_loc_top1_beacon_minus_uniform", "")
                    ),
                }
            )
            boot_rows.append(
                {
                    "dataset": dataset,
                    "neutralizer": neutralizer,
                    "q_max": q,
                    "comparison": "beacon_minus_logo",
                    "metric": "loc_top1",
                    "delta": _f(r.get("delta_loc_top1_beacon_minus_logo", "")),
                    "ci_low": _f(r.get("ci_delta_loc_top1_beacon_minus_logo_low", "")),
                    "ci_high": _f(r.get("ci_delta_loc_top1_beacon_minus_logo_high", "")),
                    "frac_positive": _f(
                        r.get("frac_positive_delta_loc_top1_beacon_minus_logo", "")
                    ),
                }
            )

            ci_low = _f(r.get("ci_delta_loc_top1_beacon_minus_uniform_low", ""))
            q1_signal = int(ci_low > 0.0)
            claim_rows.append(
                {
                    "dataset": dataset,
                    "neutralizer": neutralizer,
                    "q_max": q,
                    "claim": "beacon_gt_uniform_loc_top1",
                    "delta": _f(r.get("delta_loc_top1_beacon_minus_uniform", "")),
                    "ci_low": ci_low,
                    "ci_high": _f(
                        r.get("ci_delta_loc_top1_beacon_minus_uniform_high", "")
                    ),
                    "q1_signal": q1_signal,
                }
            )

    for f in per_csv:
        neutralizer = _neutralizer_from_name(f.name)
        rows = _load_csv_rows(f)
        for r in rows:
            dataset = r["dataset"]
            q = _i(r["q_max"])
            key = (dataset, neutralizer, q)
            b1 = _i(r.get("is_correct_beacon", "0"), 0)
            u1 = _i(r.get("is_correct_uniform", "0"), 0)
            l1 = _i(r.get("is_correct_logo", "0"), 0)
            b3 = _i(r.get("hit3_beacon", "0"), 0)
            u3 = _i(r.get("hit3_uniform", "0"), 0)
            l3 = _i(r.get("hit3_logo", "0"), 0)
            b5 = _i(r.get("hit5_beacon", "0"), 0)
            u5 = _i(r.get("hit5_uniform", "0"), 0)
            l5 = _i(r.get("hit5_logo", "0"), 0)

            hit_store[key]["top1_beacon"].append(b1)
            hit_store[key]["top1_uniform"].append(u1)
            hit_store[key]["top1_logo"].append(l1)
            hit_store[key]["hit3_beacon"].append(b3)
            hit_store[key]["hit3_uniform"].append(u3)
            hit_store[key]["hit3_logo"].append(l3)
            hit_store[key]["hit5_beacon"].append(b5)
            hit_store[key]["hit5_uniform"].append(u5)
            hit_store[key]["hit5_logo"].append(l5)

            if dataset == "wisdm":
                wisdm_delta_rows.append(
                    {
                        "dataset": dataset,
                        "neutralizer": neutralizer,
                        "q_max": q,
                        "sample_index_eval": _i(r.get("sample_index_eval", "")),
                        "delta_top1_beacon_minus_uniform": b1 - u1,
                        "delta_hit3_beacon_minus_uniform": b3 - u3,
                        "delta_hit5_beacon_minus_uniform": b5 - u5,
                    }
                )

    # Table with full CI for all rows/metrics
    ci_rows: list[dict[str, object]] = []
    for (dataset, neutralizer, q), cols in sorted(hit_store.items()):
        n = len(cols["top1_beacon"])
        for method in ("beacon", "uniform", "logo"):
            t1 = cols[f"top1_{method}"]
            h3 = cols[f"hit3_{method}"]
            h5 = cols[f"hit5_{method}"]
            k1, k3, k5 = sum(t1), sum(h3), sum(h5)
            lo1, hi1 = _wilson(k1, n)
            lo3, hi3 = _wilson(k3, n)
            lo5, hi5 = _wilson(k5, n)
            ci_rows.append(
                {
                    "dataset": dataset,
                    "neutralizer": neutralizer,
                    "q_max": q,
                    "method": method,
                    "n_conflict": n,
                    "loc_top1": k1 / n if n else float("nan"),
                    "loc_top1_ci_low": lo1,
                    "loc_top1_ci_high": hi1,
                    "loc_hit3": k3 / n if n else float("nan"),
                    "loc_hit3_ci_low": lo3,
                    "loc_hit3_ci_high": hi3,
                    "loc_hit5": k5 / n if n else float("nan"),
                    "loc_hit5_ci_low": lo5,
                    "loc_hit5_ci_high": hi5,
                }
            )

    # WISDM failure analysis (compact summary)
    wisdm_summary: list[dict[str, object]] = []
    for (dataset, neutralizer, q), cols in sorted(hit_store.items()):
        if dataset != "wisdm":
            continue
        n = len(cols["top1_beacon"])
        b1 = sum(cols["top1_beacon"]) / n
        u1 = sum(cols["top1_uniform"]) / n
        b3 = sum(cols["hit3_beacon"]) / n
        u3 = sum(cols["hit3_uniform"]) / n
        b5 = sum(cols["hit5_beacon"]) / n
        u5 = sum(cols["hit5_uniform"]) / n
        win = sum(
            1
            for bi, ui in zip(cols["top1_beacon"], cols["top1_uniform"])
            if bi > ui
        ) / n
        wisdm_summary.append(
            {
                "dataset": "wisdm",
                "neutralizer": neutralizer,
                "q_max": q,
                "n_conflict": n,
                "loc_top1_beacon": b1,
                "loc_top1_uniform": u1,
                "delta_top1": b1 - u1,
                "loc_hit3_beacon": b3,
                "loc_hit3_uniform": u3,
                "delta_hit3": b3 - u3,
                "loc_hit5_beacon": b5,
                "loc_hit5_uniform": u5,
                "delta_hit5": b5 - u5,
                "beacon_strict_win_rate_top1": win,
            }
        )

    def _write(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            with path.open("w", newline="", encoding="utf-8") as f:
                f.write("")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write(OUT_DIR / "localization_q_sweep.csv", summary_rows)
    _write(OUT_DIR / "localization_q_sweep_bootstrap.csv", boot_rows)
    _write(OUT_DIR / "claim_registry_part1.csv", claim_rows)
    _write(ROOT / "outputs_composite" / "part1_table1_with_ci.csv", ci_rows)
    _write(ROOT / "outputs_composite" / "wisdm_failure_analysis.csv", wisdm_summary)
    _write(ROOT / "outputs_composite" / "wisdm_delta_distribution.csv", wisdm_delta_rows)

    print("saved:", OUT_DIR / "localization_q_sweep.csv")
    print("saved:", OUT_DIR / "localization_q_sweep_bootstrap.csv")
    print("saved:", OUT_DIR / "claim_registry_part1.csv")
    print("saved:", ROOT / "outputs_composite" / "part1_table1_with_ci.csv")
    print("saved:", ROOT / "outputs_composite" / "wisdm_failure_analysis.csv")
    print("saved:", ROOT / "outputs_composite" / "wisdm_delta_distribution.csv")


if __name__ == "__main__":
    build()
