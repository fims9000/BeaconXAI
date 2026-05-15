#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset


def group_slices(n_channels: int, mode: str) -> list[tuple[int, int]]:
    if mode == "per_channel":
        return [(i, i + 1) for i in range(n_channels)]
    if mode == "split2":
        if n_channels == 6:
            return [(0, 3), (3, 6)]
        mid = max(1, n_channels // 2)
        return [(0, mid), (mid, n_channels)]
    if mode == "pamap3":
        if n_channels == 9:
            return [(0, 3), (3, 6), (6, 9)]
        g = max(1, n_channels // 3)
        return [(0, g), (g, min(2 * g, n_channels)), (min(2 * g, n_channels), n_channels)]
    if n_channels == 9:
        return [(0, 3), (3, 6), (6, 9)]
    return [(i, i + 1) for i in range(n_channels)]


def time_slices(t_len: int, n_bins: int) -> list[tuple[int, int]]:
    e = np.linspace(0, t_len, n_bins + 1, dtype=int)
    out = []
    for i in range(n_bins):
        t0, t1 = int(e[i]), int(e[i + 1])
        if t1 <= t0:
            t1 = min(t_len, t0 + 1)
        out.append((t0, t1))
    return out


def inject_component(x_test, y_test, g_slices, t_slices, conflict_ratio, seed):
    rng = np.random.default_rng(seed)
    n = len(x_test)
    n_conf = int(round(conflict_ratio * n))
    conf_idx = rng.choice(n, size=n_conf, replace=False)
    x_eval = x_test.copy()
    conf_present = np.zeros(n, dtype=np.int64)
    true_comp = -np.ones(n, dtype=np.int64)
    b_count = len(t_slices)
    for i in conf_idx:
        yi = int(y_test[i])
        donors = np.where(y_test != yi)[0]
        if len(donors) == 0:
            continue
        j = int(rng.choice(donors))
        g = int(rng.integers(0, len(g_slices)))
        b = int(rng.integers(0, b_count))
        c0, c1 = g_slices[g]
        t0, t1 = t_slices[b]
        x_eval[i, t0:t1, c0:c1] = x_test[j, t0:t1, c0:c1]
        conf_present[i] = 1
        true_comp[i] = g * b_count + b
    return x_eval, conf_present, true_comp


def comp_to_rect(comp: int, n_bins: int, g_slices, t_slices):
    g = comp // n_bins
    b = comp % n_bins
    c0, c1 = g_slices[g]
    t0, t1 = t_slices[b]
    return t0, t1, c0, c1


@dataclass
class Case:
    label: str
    dataset: str
    model: str
    q_max: int
    npz_path: str
    seed: int
    group_mode: str
    time_bins: int
    sample_index_eval: int
    true_component: int
    pred_component_beacon: int
    is_correct_beacon: int


def read_rows(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_cases(rows_a, rows_b):
    def filt(rows, dataset, q, ok):
        out = [r for r in rows if r["dataset"] == dataset and int(r["q_max"]) == q and int(r["is_correct_beacon"]) == ok]
        return out

    c1 = filt(rows_a, "pamap2", 8, 1)[0]
    c2 = filt(rows_a, "pamap2", 16, 1)[0]
    c3 = filt(rows_b, "wisdm", 16, 1)[0]
    c4 = filt(rows_a, "pamap2", 16, 0)[0]

    picked = [
        ("case1_pamap2_q8_correct", c1),
        ("case2_pamap2_q16_correct", c2),
        ("case3_wisdm_q16_correct", c3),
        ("case4_pamap2_q16_error", c4),
    ]

    out = []
    for label, r in picked:
        out.append(
            Case(
                label=label,
                dataset=r["dataset"],
                model=r["model"],
                q_max=int(r["q_max"]),
                npz_path=r["npz_path"],
                seed=int(r["seed"]),
                group_mode=r["group_mode"],
                time_bins=int(r["time_bins"]),
                sample_index_eval=int(r["sample_index_eval"]),
                true_component=int(r["true_component"]),
                pred_component_beacon=int(r["pred_component_beacon"]),
                is_correct_beacon=int(r["is_correct_beacon"]),
            )
        )
    return out


def make_plot(case: Case, out_dir: Path):
    x_train, y_train, x_test, y_test = load_npz_dataset(case.npz_path)

    rng = np.random.default_rng(case.seed)
    if len(x_test) > 512:
        idx = rng.choice(len(x_test), size=512, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_train)
    x_test = apply_standardizer(x_test, mu, sigma)

    g_s = group_slices(x_test.shape[-1], case.group_mode)
    t_s = time_slices(x_test.shape[1], case.time_bins)
    x_eval, conf_present, true_comp = inject_component(x_test, y_test, g_s, t_s, 0.5, case.seed + 101)

    i = case.sample_index_eval
    x = x_eval[i]

    n_bins = case.time_bins
    t0, t1, c0, c1 = comp_to_rect(case.true_component, n_bins, g_s, t_s)
    p0, p1, pc0, pc1 = comp_to_rect(case.pred_component_beacon, n_bins, g_s, t_s)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(x.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax.add_patch(patches.Rectangle((t0, c0 - 0.5), t1 - t0, c1 - c0, fill=False, edgecolor="lime", linewidth=2.0, label="True"))
    ax.add_patch(patches.Rectangle((p0, pc0 - 0.5), p1 - p0, pc1 - pc0, fill=False, edgecolor="red", linewidth=2.0, linestyle="--", label="BEACON pred"))
    ax.set_xlabel("Time index")
    ax.set_ylabel("Channel")
    tag = "correct" if case.is_correct_beacon == 1 else "error"
    ax.set_title(f"{case.dataset}/{case.model} Q={case.q_max} ({tag})")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()

    out_pdf = out_dir / f"{case.label}.pdf"
    out_png = out_dir / f"{case.label}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return out_pdf, out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pamap2-per-sample", required=True)
    ap.add_argument("--wisdm-per-sample", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows_a = read_rows(args.pamap2_per_sample)
    rows_b = read_rows(args.wisdm_per_sample)
    cases = pick_cases(rows_a, rows_b)

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    manifest = []
    for c in cases:
        pdf, png = make_plot(c, od)
        manifest.append({
            "label": c.label,
            "dataset": c.dataset,
            "model": c.model,
            "q_max": c.q_max,
            "sample_index_eval": c.sample_index_eval,
            "is_correct_beacon": c.is_correct_beacon,
            "true_component": c.true_component,
            "pred_component_beacon": c.pred_component_beacon,
            "pdf": str(pdf),
            "png": str(png),
        })

    mf = od / "case_manifest.csv"
    with mf.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        wr.writeheader()
        wr.writerows(manifest)
    print(f"saved: {mf}")


if __name__ == "__main__":
    main()
