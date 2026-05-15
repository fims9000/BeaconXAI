#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from scripts.run_component_conflict_benchmark import (
    _group_slices,
    _time_slices,
    _inject_component_conflicts,
    _train_extratrees_local,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAMAP2 time-series attribution-style baseline comparison")
    p.add_argument("--npz-path", default="data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=512)
    p.add_argument("--conflict-ratio", type=float, default=0.5)
    p.add_argument("--group-mode", default="pamap3")
    p.add_argument("--time-bins", type=int, default=12)
    p.add_argument("--neutralizer", choices=["zero", "mean", "interp"], default="interp")
    p.add_argument("--rise-budgets", default="64,128,256,512")
    p.add_argument("--shap-budgets", default="128,256,512")
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--et-n-estimators", type=int, default=120)
    p.add_argument("--et-max-features", type=float, default=0.7)
    p.add_argument("--et-min-samples-leaf", type=int, default=1)
    p.add_argument("--beacon-row-csv", default="outputs_composite/pamap2_component_tb12_q16_interp_shortlisting.csv")
    p.add_argument("--out-summary", default="outputs_composite/pamap2_tsxai_baselines_summary.csv")
    p.add_argument("--out-per-sample", default="outputs_composite/pamap2_tsxai_baselines_per_sample.csv")
    p.add_argument("--plot-hit3", default="outputs_composite/figure_pamap2_calls_vs_hit3.png")
    p.add_argument("--plot-mrr", default="outputs_composite/figure_pamap2_calls_vs_mrr.png")
    return p.parse_args()


def _neutralize_component_independent(
    x_conf: np.ndarray,
    active_mask: np.ndarray,
    comp_slices: list[tuple[int, int, int, int]],
    comp_patches: list[np.ndarray],
) -> np.ndarray:
    x = x_conf.copy()
    for j in np.where(active_mask > 0.5)[0]:
        t0, t1, c0, c1 = comp_slices[int(j)]
        x[t0:t1, c0:c1] = comp_patches[int(j)]
    return x


def _predict_proba_batch(clf, x_batch: np.ndarray) -> np.ndarray:
    from scripts.run_component_conflict_benchmark import _anfis_features

    f = _anfis_features(x_batch)
    return clf.model.predict_proba(f)


def _margin_ref_from_proba(proba: np.ndarray, y_ref: int) -> np.ndarray:
    lp = np.log(np.clip(proba, 1e-12, 1.0))
    others = np.max(np.delete(lp, y_ref, axis=1), axis=1)
    return lp[:, y_ref] - others


def _fit_linear_scores(a: np.ndarray, y: np.ndarray, w: np.ndarray | None, ridge: float) -> np.ndarray:
    x = np.concatenate([np.ones((a.shape[0], 1), dtype=np.float64), a.astype(np.float64)], axis=1)
    if w is None:
        xtx = x.T @ x
        xty = x.T @ y
    else:
        ww = np.sqrt(w)[:, None]
        xw = x * ww
        yw = y * np.sqrt(w)
        xtx = xw.T @ xw
        xty = xw.T @ yw
    reg = ridge * np.eye(xtx.shape[0], dtype=np.float64)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtx + reg, xty)
    return beta[1:]


def _true_ranks(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    inv = np.empty_like(order, dtype=np.int64)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    return inv[np.arange(order.shape[0]), y_true] + 1


def _metrics_from_ranks(r: np.ndarray, m: int) -> dict[str, float]:
    rr = r.astype(np.float64)
    rand_mean = (m + 1.0) / 2.0
    return {
        "loc_top1": float(np.mean(rr <= 1)),
        "hit3": float(np.mean(rr <= 3)),
        "hit5": float(np.mean(rr <= 5)),
        "mrr": float(np.mean(1.0 / rr)),
        "mean_rank": float(np.mean(rr)),
        "nrg": float((rand_mean - float(np.mean(rr))) / max(rand_mean - 1.0, 1e-12)),
    }


def _shap_kernel_weight(m: int, s: int) -> float:
    if s <= 0 or s >= m:
        return 1e6
    return float((m - 1) / (math.comb(m, s) * s * (m - s)))


def run() -> None:
    args = parse_args()
    rise_budgets = [int(v) for v in args.rise_budgets.split(",") if v.strip()]
    shap_budgets = [int(v) for v in args.shap_budgets.split(",") if v.strip()]

    x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)
    if 0 < args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    clf = _train_extratrees_local(
        x_train,
        y_train,
        n_estimators=args.et_n_estimators,
        max_features=args.et_max_features,
        min_samples_leaf=args.et_min_samples_leaf,
    )

    g_slices = _group_slices(x_test.shape[-1], args.group_mode)
    t_slices = _time_slices(x_test.shape[1], args.time_bins)
    m_components = len(g_slices) * len(t_slices)

    x_eval, conflict_present, true_comp = _inject_component_conflicts(
        x_test,
        y_test,
        g_slices,
        t_slices,
        conflict_ratio=args.conflict_ratio,
        seed=args.seed + 101,
    )
    cidx = np.where(conflict_present == 1)[0]
    x_conf = x_eval[cidx]
    y_true_comp = true_comp[cidx].astype(np.int64)

    proba_conf = _predict_proba_batch(clf, x_conf)
    y_ref = np.argmax(proba_conf, axis=1).astype(np.int64)
    m0 = _margin_ref_from_proba(proba_conf, 0)
    # m0 per-sample with own y_ref
    m0 = np.array(
        [
            float(_margin_ref_from_proba(proba_conf[i : i + 1], int(y_ref[i]))[0])
            for i in range(len(y_ref))
        ],
        dtype=np.float64,
    )

    # Component indexing (group-major then time-bin)
    comp_slices: list[tuple[int, int, int, int]] = []
    for gi, (c0, c1) in enumerate(g_slices):
        for bi, (t0, t1) in enumerate(t_slices):
            comp_slices.append((t0, t1, c0, c1))

    # Precompute independent patches for each sample/component under chosen neutralizer
    from scripts.run_component_conflict_benchmark import _neutralize_component

    sample_patches: list[list[np.ndarray]] = []
    for i in range(len(x_conf)):
        patches_i: list[np.ndarray] = []
        xi = x_conf[i]
        for (t0, t1, c0, c1) in comp_slices:
            xj = _neutralize_component(xi, t0, t1, c0, c1, args.neutralizer)
            patches_i.append(xj[t0:t1, c0:c1].copy())
        sample_patches.append(patches_i)

    per_rows = []
    summary_rows = []

    def evaluate_budget(method: str, budget: int, make_masks_fn):
        scores = np.zeros((len(x_conf), m_components), dtype=np.float64)
        calls = np.zeros(len(x_conf), dtype=np.float64)
        rng = np.random.default_rng(args.seed + budget + (0 if method == "rise" else 10000))
        for i in range(len(x_conf)):
            masks, w = make_masks_fn(rng, budget, m_components)
            # Build masked batch
            xb = np.repeat(x_conf[i][None, :, :], masks.shape[0], axis=0)
            for r in range(masks.shape[0]):
                xb[r] = _neutralize_component_independent(x_conf[i], masks[r], comp_slices, sample_patches[i])
            pb = _predict_proba_batch(clf, xb)
            mb = _margin_ref_from_proba(pb, int(y_ref[i]))
            y = mb - m0[i]
            phi = _fit_linear_scores(masks, y, w, args.ridge)
            scores[i, :] = phi
            calls[i] = float(masks.shape[0])
        ranks = _true_ranks(scores, y_true_comp)
        mm = _metrics_from_ranks(ranks, m_components)
        summary_rows.append(
            {
                "dataset": "pamap2",
                "model": "extratrees",
                "method": method,
                "budget_calls": int(budget),
                "n_eval": int(len(x_conf)),
                "time_bins": int(args.time_bins),
                "n_components": int(m_components),
                "neutralizer": args.neutralizer,
                "loc_top1": mm["loc_top1"],
                "hit3": mm["hit3"],
                "hit5": mm["hit5"],
                "mrr": mm["mrr"],
                "mean_rank": mm["mean_rank"],
                "nrg": mm["nrg"],
                "mean_model_calls": float(np.mean(calls)),
            }
        )
        for i in range(len(x_conf)):
            per_rows.append(
                {
                    "dataset": "pamap2",
                    "model": "extratrees",
                    "method": method,
                    "budget_calls": int(budget),
                    "sample_id": int(i),
                    "rank_true": int(ranks[i]),
                    "top1_hit": int(ranks[i] <= 1),
                    "hit3": int(ranks[i] <= 3),
                    "hit5": int(ranks[i] <= 5),
                    "model_calls": int(calls[i]),
                }
            )

    def rise_masks(rng: np.random.Generator, budget: int, m: int):
        a = np.zeros((budget, m), dtype=np.float64)
        for i in range(1, budget):
            k = int(rng.integers(1, m + 1))
            sel = rng.choice(m, size=k, replace=False)
            a[i, sel] = 1.0
        return a, None

    def shap_masks(rng: np.random.Generator, budget: int, m: int):
        a = np.zeros((budget, m), dtype=np.float64)
        w = np.ones(budget, dtype=np.float64)
        if budget >= 2:
            a[1, :] = 1.0
            w[0] = _shap_kernel_weight(m, 0)
            w[1] = _shap_kernel_weight(m, m)
            start = 2
        else:
            w[0] = _shap_kernel_weight(m, 0)
            start = 1
        for i in range(start, budget):
            s = int(rng.integers(1, m))
            sel = rng.choice(m, size=s, replace=False)
            a[i, sel] = 1.0
            w[i] = _shap_kernel_weight(m, s)
        return a, w

    for b in rise_budgets:
        evaluate_budget(f"rise_masking_{b}", b, rise_masks)
    for b in shap_budgets:
        evaluate_budget(f"kernelshap_components_{b}", b, shap_masks)

    # Append BEACON / uniform / random reference row from existing csv
    with open(args.beacon_row_csv, newline="", encoding="utf-8") as f:
        ref = next(csv.DictReader(f))
    summary_rows.extend(
        [
            {
                "dataset": "pamap2",
                "model": "extratrees",
                "method": "beacon_q16",
                "budget_calls": 16,
                "n_eval": int(ref["n_conflict"]),
                "time_bins": int(ref["time_bins"]),
                "n_components": int(ref["n_components"]),
                "neutralizer": ref["neutralizer"],
                "loc_top1": float(ref["loc_top1_beacon"]),
                "hit3": float(ref["loc_hit3_beacon"]),
                "hit5": float(ref["loc_hit5_beacon"]),
                "mrr": float(ref["loc_mrr_beacon"]),
                "mean_rank": float(ref["loc_mean_rank_beacon"]),
                "nrg": float(ref["loc_nrg_beacon"]),
                "mean_model_calls": float(ref["mean_q_used"]),
            },
            {
                "dataset": "pamap2",
                "model": "extratrees",
                "method": "uniform_occlusion_q16",
                "budget_calls": 16,
                "n_eval": int(ref["n_conflict"]),
                "time_bins": int(ref["time_bins"]),
                "n_components": int(ref["n_components"]),
                "neutralizer": ref["neutralizer"],
                "loc_top1": float(ref["loc_top1_uniform"]),
                "hit3": float(ref["loc_hit3_uniform"]),
                "hit5": float(ref["loc_hit5_uniform"]),
                "mrr": float(ref["loc_mrr_uniform"]),
                "mean_rank": float(ref["loc_mean_rank_uniform"]),
                "nrg": float(ref["loc_nrg_uniform"]),
                "mean_model_calls": 16.0,
            },
            {
                "dataset": "pamap2",
                "model": "extratrees",
                "method": "random",
                "budget_calls": 0,
                "n_eval": int(ref["n_conflict"]),
                "time_bins": int(ref["time_bins"]),
                "n_components": int(ref["n_components"]),
                "neutralizer": ref["neutralizer"],
                "loc_top1": float(ref["loc_top1_random"]),
                "hit3": float(ref["loc_hit3_random"]),
                "hit5": float(ref["loc_hit5_random"]),
                "mrr": float(ref["loc_mrr_random"]),
                "mean_rank": float(ref["loc_mean_rank_random"]),
                "nrg": float(ref["loc_nrg_random"]),
                "mean_model_calls": 0.0,
            },
        ]
    )

    # Write outputs
    out = Path(args.out_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"saved: {out}")

    po = Path(args.out_per_sample)
    po.parent.mkdir(parents=True, exist_ok=True)
    with po.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)
    print(f"saved: {po}")

    # Plots (quality vs calls)
    def _plot(y_key: str, out_path: str, title: str):
        rows = [r for r in summary_rows if r["method"] != "random"]
        rows = sorted(rows, key=lambda r: float(r["mean_model_calls"]))
        x = [float(r["mean_model_calls"]) for r in rows]
        y = [float(r[y_key]) for r in rows]
        labels = [r["method"] for r in rows]
        plt.figure(figsize=(7.0, 4.4))
        for xi, yi, lb in zip(x, y, labels):
            plt.scatter([xi], [yi], s=50)
            plt.text(xi, yi, lb, fontsize=8, ha="left", va="bottom")
        rand = [r for r in summary_rows if r["method"] == "random"][0]
        plt.axhline(float(rand[y_key]), linestyle="--", linewidth=1.1, color="gray", label="random")
        plt.xlabel("Model calls")
        plt.ylabel(y_key)
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(op, dpi=220)
        plt.close()
        print(f"saved: {op}")

    _plot("hit3", args.plot_hit3, "PAMAP2: hit@3 vs model calls")
    _plot("mrr", args.plot_mrr, "PAMAP2: MRR vs model calls")


if __name__ == "__main__":
    run()
