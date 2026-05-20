#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "outputs_composite" / "part1_localization_q_sweep" / "raw"
OUT_DIR = ROOT / "outputs_composite" / "part1_localization_q_sweep"


METRICS = ("loc@1", "hit@3", "hit@5", "mrr", "nrg", "mean_rank")


def _i(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _bootstrap_delta(
    beacon: np.ndarray,
    uniform: np.ndarray,
    higher_better: bool,
    seed: int = 42,
    n_boot: int = 4000,
) -> tuple[float, float, float, float]:
    # delta > 0 means BEACON better
    if higher_better:
        delta = beacon - uniform
    else:
        delta = uniform - beacon
    base = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    n = len(delta)
    vals = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = float(np.mean(delta[idx]))
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    # bootstrap two-sided p around zero
    p_pos = float(np.mean(vals > 0.0))
    p_neg = float(np.mean(vals < 0.0))
    p_two = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    return base, lo, hi, p_two


def _neutralizer_from_name(name: str) -> str:
    if "_interp_" in name or name.endswith("_interp_per_sample.csv"):
        return "interp"
    if "_zero_" in name or name.endswith("_zero_per_sample.csv"):
        return "zero"
    return "unknown"


def _load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("dataset", "") == "dataset":
                continue
            rows.append(r)
    return rows


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_files = sorted(RAW_DIR.glob("*_per_sample.csv"))

    out_rows: list[dict[str, object]] = []

    for f in per_files:
        neutralizer = _neutralizer_from_name(f.name)
        rows = _load_rows(f)
        grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
        for r in rows:
            dataset = r.get("dataset", "")
            q = _i(r.get("q_max", "0"))
            grouped.setdefault((dataset, q), []).append(r)

        for (dataset, q), grp in sorted(grouped.items()):
            # conflict-only
            grp = [r for r in grp if _i(r.get("true_component", "-1"), -1) >= 0]
            if not grp:
                continue

            n_components = max(_i(grp[0].get("n_components", "1"), 1), 1)
            rand_mean_rank = (n_components + 1.0) / 2.0
            denom = max(rand_mean_rank - 1.0, 1e-12)

            b_top1 = np.array([_i(r.get("is_correct_beacon", "0")) for r in grp], dtype=float)
            u_top1 = np.array([_i(r.get("is_correct_uniform", "0")) for r in grp], dtype=float)
            b_h3 = np.array([_i(r.get("hit3_beacon", "0")) for r in grp], dtype=float)
            u_h3 = np.array([_i(r.get("hit3_uniform", "0")) for r in grp], dtype=float)
            b_h5 = np.array([_i(r.get("hit5_beacon", "0")) for r in grp], dtype=float)
            u_h5 = np.array([_i(r.get("hit5_uniform", "0")) for r in grp], dtype=float)

            rb = np.array([max(_i(r.get("rank_true_beacon", "1"), 1), 1) for r in grp], dtype=float)
            ru = np.array([max(_i(r.get("rank_true_uniform", "1"), 1), 1) for r in grp], dtype=float)
            b_mrr = 1.0 / rb
            u_mrr = 1.0 / ru
            b_mrank = rb
            u_mrank = ru
            b_nrg = (rand_mean_rank - b_mrank) / denom
            u_nrg = (rand_mean_rank - u_mrank) / denom

            spec = [
                ("loc@1", b_top1, u_top1, True),
                ("hit@3", b_h3, u_h3, True),
                ("hit@5", b_h5, u_h5, True),
                ("mrr", b_mrr, u_mrr, True),
                ("nrg", b_nrg, u_nrg, True),
                ("mean_rank", b_mrank, u_mrank, False),
            ]

            for metric, bvals, uvals, higher_better in spec:
                delta, ci_low, ci_high, p = _bootstrap_delta(
                    bvals, uvals, higher_better=higher_better, seed=42 + q
                )
                out_rows.append(
                    {
                        "dataset": dataset,
                        "neutralizer": neutralizer,
                        "q_max": q,
                        "method": "beacon_vs_uniform",
                        "metric": metric,
                        "beacon_value": float(np.mean(bvals)),
                        "uniform_value": float(np.mean(uvals)),
                        "delta_beacon_minus_uniform": delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_value_bootstrap_two_sided": p,
                        "n_conflict": len(grp),
                        "higher_better": int(higher_better),
                        "supported_positive_signal": int(ci_low > 0.0 and p < 0.05),
                    }
                )

    out_csv = OUT_DIR / "part1_best_claims_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Build allowed/not allowed claims markdown
    by_metric: dict[str, list[dict[str, object]]] = {}
    for r in out_rows:
        by_metric.setdefault(str(r["metric"]), []).append(r)

    lines: list[str] = []
    lines.append("# Part1 allowed claims (from q-sweep)")
    lines.append("")
    lines.append("## Best config per metric (delta > 0 means BEACON better)")
    lines.append("")
    lines.append("| metric | dataset | neutralizer | Q | BEACON | uniform | delta | 95% CI | p | supported |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for m in METRICS:
        rows = by_metric.get(m, [])
        if not rows:
            continue
        best = max(rows, key=lambda x: float(x["delta_beacon_minus_uniform"]))
        lines.append(
            f"| {m} | {best['dataset']} | {best['neutralizer']} | {best['q_max']} | "
            f"{float(best['beacon_value']):.4f} | {float(best['uniform_value']):.4f} | "
            f"{float(best['delta_beacon_minus_uniform']):+.4f} | "
            f"[{float(best['ci_low']):+.4f}; {float(best['ci_high']):+.4f}] | "
            f"{float(best['p_value_bootstrap_two_sided']):.3f} | {int(best['supported_positive_signal'])} |"
        )

    lines.append("")
    lines.append("## Dataset-level status")
    lines.append("")
    for dataset in sorted({str(r['dataset']) for r in out_rows}):
        drows = [r for r in out_rows if str(r["dataset"]) == dataset]
        pos = [r for r in drows if int(r["supported_positive_signal"]) == 1]
        lines.append(f"- `{dataset}`: supported_positive_signals = {len(pos)}")
        if pos:
            for r in pos:
                lines.append(
                    f"  - {r['metric']}, neutralizer={r['neutralizer']}, Q={r['q_max']}, "
                    f"delta={float(r['delta_beacon_minus_uniform']):+.4f}, "
                    f"CI=[{float(r['ci_low']):+.4f}; {float(r['ci_high']):+.4f}], "
                    f"p={float(r['p_value_bootstrap_two_sided']):.3f}"
                )

    lines.append("")
    lines.append("## Allowed claims")
    lines.append("")
    if any(int(r["supported_positive_signal"]) == 1 for r in out_rows):
        lines.append("- Можно заявлять только те метрики/конфигурации, где `supported_positive_signal=1`.")
    else:
        lines.append("- Статистически подтверждённых положительных сигналов BEACON>uniform не обнаружено в текущем Part1 q-sweep.")
    lines.append("- Для PAMAP2 и WISDM обязательно писать sensitivity к neutralizer/Q и наличие negative/flat cases.")
    lines.append("- Для WISDM использовать формулировку boundary condition (не universal improvement).")

    out_md = OUT_DIR / "part1_allowed_claims.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved:", out_csv)
    print("saved:", out_md)


if __name__ == "__main__":
    build()
