#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot AUROC/AUPRC vs Q from har_ce_q_sweep.csv")
    p.add_argument("--in-csv", default="./outputs_composite/har_ce_q_sweep.csv")
    p.add_argument("--out-dir", default="./outputs_composite")
    return p.parse_args()


def _load(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] not in {"negative_margin", "support+CE", "mixed+CE", "mixed+rho_cost"}:
                continue
            rows.append(
                {
                    "q": int(r["q_max"]),
                    "method": r["method"],
                    "auroc": float(r["auroc"]),
                    "auprc": float(r["auprc"]),
                }
            )
    return rows


def _plot(rows, metric: str, out_path: Path) -> None:
    methods = ["negative_margin", "support+CE", "mixed+CE", "mixed+rho_cost"]
    plt.figure(figsize=(7.2, 4.6))
    for m in methods:
        cur = sorted([r for r in rows if r["method"] == m], key=lambda z: z["q"])
        if not cur:
            continue
        xs = [r["q"] for r in cur]
        ys = [r[metric] for r in cur]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=m)
    plt.xlabel("Qmax")
    plt.ylabel(metric.upper())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    rows = _load(Path(args.in_csv))
    out = Path(args.out_dir)
    _plot(rows, "auroc", out / "figure_auroc_vs_q.png")
    _plot(rows, "auprc", out / "figure_auprc_vs_q.png")
    print("Saved:")
    print(out / "figure_auroc_vs_q.png")
    print(out / "figure_auprc_vs_q.png")


if __name__ == "__main__":
    main()

