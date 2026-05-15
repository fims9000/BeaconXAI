#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make practical validation figures")
    p.add_argument("--in-summary", default="outputs_composite/har_sensor_fault_localization_table.csv")
    p.add_argument("--in-per-sample", default="outputs_composite/har_sensor_fault_localization_per_sample.csv")
    p.add_argument("--in-eval-npz", default="outputs_composite/har_sensor_fault_eval.npz")
    p.add_argument("--out-hit3", default="outputs_composite/figure_har_fault_hit3_vs_calls.png")
    p.add_argument("--out-case", default="outputs_composite/figure_har_panel_case_study.png")
    return p.parse_args()


def _decode_comp(comp: int, n_bins: int) -> tuple[int, int]:
    return comp // n_bins, comp % n_bins


def main() -> None:
    args = parse_args()
    Path(args.out_hit3).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_summary)
    order = ["random", "amplitude_heuristic", "energy_heuristic", "variance_heuristic", "uniform_occlusion", "beacon_xai"]
    dfx = df.set_index("method").reindex(order).reset_index()

    # Figure 1: hit@3 vs calls
    plt.figure(figsize=(7.2, 4.5))
    plt.plot(dfx["calls"], dfx["hit@3"], marker="o", linewidth=1.8)
    for _, r in dfx.iterrows():
        plt.annotate(r["method"], (r["calls"], r["hit@3"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Model calls")
    plt.ylabel("Hit@3")
    plt.title("HAR sensor-fault localization: quality vs calls")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out_hit3, dpi=220)
    plt.close()

    # Figure 2: single-case visualization (true vs BEACON)
    per = pd.read_csv(args.in_per_sample)
    arr = np.load(args.in_eval_npz, allow_pickle=True)
    x_eval = arr["x_eval"]
    true_comp = arr["true_comp"]
    n_bins = int(arr["time_bins"])

    beacon_rows = per[per["method"] == "beacon_xai"].copy()
    # prefer correct case; fallback first
    ok = beacon_rows[beacon_rows["is_correct"] == 1]
    row = ok.iloc[0] if len(ok) else beacon_rows.iloc[0]

    sid = int(row["sample_index_eval"])
    tc = int(row["true_component"])
    pc = int(row["pred_component"])
    x = x_eval[sid]  # [T, C]

    t_len, n_ch = x.shape
    edges = np.linspace(0, t_len, n_bins + 1, dtype=int)

    tc_ch, tc_b = _decode_comp(tc, n_bins)
    pc_ch, pc_b = _decode_comp(pc, n_bins)
    t0_true, t1_true = int(edges[tc_b]), int(edges[tc_b + 1])
    t0_pred, t1_pred = int(edges[pc_b]), int(edges[pc_b + 1])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(x.T, aspect="auto", origin="lower", interpolation="nearest")
    ax.add_patch(plt.Rectangle((t0_true, tc_ch - 0.5), t1_true - t0_true, 1.0, fill=False, linewidth=2.0, edgecolor="lime", label="true"))
    ax.add_patch(plt.Rectangle((t0_pred, pc_ch - 0.5), t1_pred - t0_pred, 1.0, fill=False, linewidth=2.0, edgecolor="red", label="beacon top1"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Channel")
    ax.set_title(f"HAR fault case study: sample={sid}, true={tc}, pred={pc}")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(args.out_case, dpi=220)
    plt.close()

    print(f"saved: {args.out_hit3}")
    print(f"saved: {args.out_case}")


if __name__ == "__main__":
    main()
