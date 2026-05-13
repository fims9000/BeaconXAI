#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-panel AUROC/AUPRC vs Q plot")
    p.add_argument("--in-csv", default="./outputs_composite/har_ce_q_sweep.csv")
    p.add_argument("--out", default="./outputs_composite/figure_risk_vs_q.png")
    return p.parse_args()


def _load(path: Path):
    keep = {"negative_margin", "support+CE", "mixed+CE", "mixed+rho_cost"}
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["method"] not in keep:
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


def _plot_metric(ax, rows, metric: str):
    methods = ["negative_margin", "support+CE", "mixed+CE", "mixed+rho_cost"]
    for m in methods:
        cur = sorted([r for r in rows if r["method"] == m], key=lambda z: z["q"])
        if not cur:
            continue
        xs = [r["q"] for r in cur]
        ys = [r[metric] for r in cur]
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=m)
    ax.set_xlabel("Qmax")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.3)


def main() -> None:
    args = parse_args()
    rows = _load(Path(args.in_csv))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    _plot_metric(axes[0], rows, "auroc")
    _plot_metric(axes[1], rows, "auprc")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()

