#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
import warnings
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.core import BeaconAudit
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.models import train_1dcnn, train_extratrees_stats, train_histgbt_stats
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used",
    category=UserWarning,
)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return float((sum_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = np.sum(y_true == 1)
    if pos == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(0.0, r - prev_recall)
        prev_recall = r
    return float(ap)


def _group_slices(n_channels: int, mode: str = "auto") -> list[tuple[int, int]]:
    if mode == "per_channel":
        return [(i, i + 1) for i in range(n_channels)]
    if mode == "split2":
        if n_channels == 6:
            return [(0, 3), (3, 6)]
        mid = max(1, n_channels // 2)
        return [(0, mid), (mid, n_channels)]
    if mode == "pamap3":
        if n_channels < 3:
            return [(i, i + 1) for i in range(n_channels)]
        if n_channels == 9:
            return [(0, 3), (3, 6), (6, 9)]
        g = max(1, n_channels // 3)
        return [(0, g), (g, min(2 * g, n_channels)), (min(2 * g, n_channels), n_channels)]
    # auto
    if n_channels == 9:
        return [(0, 3), (3, 6), (6, 9)]
    return [(i, i + 1) for i in range(n_channels)]


def _topk_hits(pred_scores: np.ndarray, true_group: np.ndarray, k: int) -> float:
    if len(true_group) == 0:
        return float("nan")
    k = int(max(1, min(k, pred_scores.shape[1])))
    topk = np.argpartition(-pred_scores, kth=k - 1, axis=1)[:, :k]
    hits = np.any(topk == true_group[:, None], axis=1)
    return float(np.mean(hits))


def _bootstrap_ci_topk(
    pred_scores: np.ndarray,
    true_group: np.ndarray,
    k: int,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    base = _topk_hits(pred_scores, true_group, k)
    if len(true_group) == 0 or n_boot <= 0:
        return base, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.zeros(n_boot, dtype=np.float64)
    n = len(true_group)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = _topk_hits(pred_scores[idx], true_group[idx], k)
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return base, lo, hi


def _paired_bootstrap_delta_against_random(
    beacon_hits: np.ndarray,
    random_p: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if len(beacon_hits) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    beacon_hits = beacon_hits.astype(np.float64)
    random_p = random_p.astype(np.float64)
    delta = float(np.mean(beacon_hits - random_p))
    if n_boot <= 0:
        frac_pos = float(delta > 0.0)
        return delta, float("nan"), float("nan"), frac_pos
    rng = np.random.default_rng(seed)
    n = len(beacon_hits)
    vals = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = (rng.random(n) < random_p[idx]).astype(np.float64)
        vals[b] = float(np.mean(beacon_hits[idx] - r))
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    frac_pos = float(np.mean(vals > 0.0))
    return delta, lo, hi, frac_pos


def _paired_bootstrap_delta_between_hits(
    a_hits: np.ndarray,
    b_hits: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if len(a_hits) == 0 or len(b_hits) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    a_hits = a_hits.astype(np.float64)
    b_hits = b_hits.astype(np.float64)
    delta = float(np.mean(a_hits - b_hits))
    if n_boot <= 0:
        frac_pos = float(delta > 0.0)
        return delta, float("nan"), float("nan"), frac_pos
    rng = np.random.default_rng(seed)
    n = len(a_hits)
    vals = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = float(np.mean(a_hits[idx] - b_hits[idx]))
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    frac_pos = float(np.mean(vals > 0.0))
    return delta, lo, hi, frac_pos


def _binom_ci_approx(p: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = float(np.clip(p, 0.0, 1.0))
    se = np.sqrt(max(0.0, p * (1.0 - p) / float(n)))
    z = 1.96
    lo = max(0.0, p - z * se)
    hi = min(1.0, p + z * se)
    return float(lo), float(hi)


def _entropy_from_logits(logits: np.ndarray) -> float:
    z = logits - np.max(logits)
    p = np.exp(z)
    p = p / np.sum(p)
    eps = 1e-12
    return float(-np.sum(p * np.log(np.clip(p, eps, 1.0))))


def _inject_sensor_conflicts(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    seed: int,
    group_slices: list[tuple[int, int]],
    conflict_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      X_eval: clean+conflict set
      conflict_present: 0/1 labels for detection
      conflict_group: -1 for clean else injected group index
    """
    rng = np.random.default_rng(seed)
    n = len(x_clean)
    n_conf = int(round(conflict_ratio * n))
    conf_idx = rng.choice(n, size=n_conf, replace=False)

    x_eval = x_clean.copy()
    conf_present = np.zeros(n, dtype=np.int64)
    conf_group = -np.ones(n, dtype=np.int64)

    by_class: dict[int, np.ndarray] = {}
    classes = np.unique(y_clean)
    for c in classes:
        by_class[int(c)] = np.where(y_clean == c)[0]

    for i in conf_idx:
        yi = int(y_clean[i])
        donor_pool = np.where(y_clean != yi)[0]
        if len(donor_pool) == 0:
            continue
        j = int(rng.choice(donor_pool))
        g = int(rng.integers(0, len(group_slices)))
        c0, c1 = group_slices[g]
        x_eval[i, :, c0:c1] = x_clean[j, :, c0:c1]
        conf_present[i] = 1
        conf_group[i] = g

    return x_eval, conf_present, conf_group


def _predict_conflict_group_from_audit(audit, group_slices: list[tuple[int, int]]) -> int:
    # aggregate counter contribution by sensor-group from negative deltas
    masses = np.zeros(len(group_slices), dtype=np.float64)
    leaf = audit.metadata.get("leaf_components", [])
    deltas = audit.metadata.get("leaf_deltas", [])
    for comp_tuple, d in zip(leaf, deltas):
        if d >= 0:
            continue
        _, _t0, _t1, c0, c1 = comp_tuple
        mass = -float(d)
        width = max(1, c1 - c0)
        for gi, (g0, g1) in enumerate(group_slices):
            ov = max(0, min(c1, g1) - max(c0, g0))
            if ov > 0:
                masses[gi] += mass * (ov / width)
    if np.all(masses <= 0):
        return -1
    return int(np.argmax(masses))


def _counter_masses_by_group_from_audit(audit, group_slices: list[tuple[int, int]]) -> np.ndarray:
    masses = np.zeros(len(group_slices), dtype=np.float64)
    leaf = audit.metadata.get("leaf_components", [])
    deltas = audit.metadata.get("leaf_deltas", [])
    for comp_tuple, d in zip(leaf, deltas):
        if d >= 0:
            continue
        _, _t0, _t1, c0, c1 = comp_tuple
        mass = -float(d)
        width = max(1, c1 - c0)
        for gi, (g0, g1) in enumerate(group_slices):
            ov = max(0, min(c1, g1) - max(c0, g0))
            if ov > 0:
                masses[gi] += mass * (ov / width)
    return masses


def _ig_group_scores(
    clf,
    x: np.ndarray,
    y_hat: int,
    group_slices: list[tuple[int, int]],
    steps: int,
) -> np.ndarray | None:
    if not hasattr(clf, "margin_gradient"):
        return None
    steps = int(max(1, steps))
    baseline = np.zeros_like(x, dtype=np.float64)
    dx = x.astype(np.float64) - baseline
    gsum = np.zeros_like(dx, dtype=np.float64)
    for s in range(1, steps + 1):
        a = float(s) / float(steps)
        z = baseline + a * dx
        gsum += clf.margin_gradient(z, y_hat=y_hat)
    avg = gsum / float(steps)
    ig = dx * avg
    out = np.zeros(len(group_slices), dtype=np.float64)
    for gi, (c0, c1) in enumerate(group_slices):
        out[gi] = float(np.sum(np.abs(ig[:, c0:c1])))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PAMAP2 controlled conflict benchmark for BEACON")
    p.add_argument("--npz-path", default="./data/pamap2_acc9_w200s100_p095.npz")
    p.add_argument("--dataset-name", default="pamap2")
    p.add_argument("--model", choices=["extratrees", "histgbt", "cnn1d"], default="extratrees")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=512)
    p.add_argument("--conflict-ratio", type=float, default=0.5)
    p.add_argument("--q-values", default="8,16")
    p.add_argument("--neutralizer", choices=["zero", "mean", "interp"], default="zero")
    p.add_argument(
        "--partition-mode",
        choices=["time_only", "time_channel", "channel_time", "sensor_group_time", "fuzzy_chunks"],
        default="sensor_group_time",
    )
    p.add_argument("--group-mode", choices=["auto", "pamap3", "per_channel", "split2"], default="auto")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--paired-bootstrap", type=int, default=2000)
    p.add_argument("--ig-baseline", action="store_true", help="Compute budgeted IG localization baseline when available")
    p.add_argument("--per-sample-out", default="", help="Optional CSV path for per-sample localization export")
    p.add_argument("--plot-out", default="./outputs_composite/figure_pamap2_loc_acc_vs_q.png")
    p.add_argument("--cnn-epochs", type=int, default=16)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--et-n-estimators", type=int, default=300)
    p.add_argument("--et-max-features", type=float, default=0.7)
    p.add_argument("--et-min-samples-leaf", type=int, default=1)
    p.add_argument("--out", default="./outputs_composite/pamap2_conflict_benchmark.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]

    x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)
    if args.max_test > 0 and args.max_test < len(x_test):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "extratrees":
        clf = train_extratrees_stats(
            x_train,
            y_train,
            n_estimators=args.et_n_estimators,
            max_features=args.et_max_features,
            min_samples_leaf=args.et_min_samples_leaf,
        )
    elif args.model == "histgbt":
        clf = train_histgbt_stats(
            x_train, y_train, max_iter=220, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=20
        )
    else:
        clf = train_1dcnn(
            x_train,
            y_train,
            epochs=args.cnn_epochs,
            batch_size=args.cnn_batch_size,
            lr=args.cnn_lr,
            label_smoothing=0.0,
            use_class_weights=True,
            tta_shifts=(0, 50),
        )

    # tau_m for config
    train_margins = []
    for i in range(min(len(x_train), 2000)):
        lg = clf.logits(x_train[i])
        y_hat = int(np.argmax(lg))
        m = float(lg[y_hat] - np.max(np.delete(lg, y_hat)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    n_channels = x_test.shape[-1]
    g_slices = _group_slices(n_channels, mode=args.group_mode)
    x_eval, conflict_present, conflict_group = _inject_sensor_conflicts(
        x_test, y_test, seed=args.seed + 11, group_slices=g_slices, conflict_ratio=args.conflict_ratio
    )

    if args.neutralizer == "mean":
        neutralizer = Neutralizer(mode="mean", channel_means=np.zeros(n_channels, dtype=np.float32))
    else:
        neutralizer = Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32))

    rows: list[dict] = []
    per_rows: list[dict] = []
    plot_q: list[int] = []
    plot_beacon: list[float] = []
    plot_random: list[float] = []
    plot_ig: list[float] = []

    for q in q_values:
        k0 = 4 if q <= 8 else 8
        cfg = BeaconConfig(
            q_max=q,
            k0=k0,
            l_min=4,
            k_pos=3,
            k_neg=3,
            q_frag_ratio=0.0,  # all refinement budget
            alpha=1.0,
            beta=0.5,
            gamma=1.0,
            tau_s=0.10,
            tau_m=tau_m,
            refinement_mode="mixed",
            partition_mode=args.partition_mode,
            risk_policy="rho_only",
            margin_mode="adaptive_all",
            audit_mode="full",
        )
        audit = BeaconAudit(model_logits=clf.logits, neutralizer=neutralizer, config=cfg)

        s_entropy = np.zeros(len(x_eval), dtype=np.float64)
        s_counter = np.zeros(len(x_eval), dtype=np.float64)
        pred_group = -np.ones(len(x_eval), dtype=np.int64)
        score_beacon = np.zeros((len(x_eval), len(g_slices)), dtype=np.float64)
        score_ig = np.full((len(x_eval), len(g_slices)), np.nan, dtype=np.float64)
        q_used = np.zeros(len(x_eval), dtype=np.float64)
        q_used_ig = np.zeros(len(x_eval), dtype=np.float64)

        t0 = time.time()
        t_ig = 0.0
        for i in range(len(x_eval)):
            lg = clf.logits(x_eval[i])
            y_hat = int(np.argmax(lg))
            s_entropy[i] = _entropy_from_logits(lg)
            ar = audit.audit(x_eval[i])
            s_counter[i] = float(ar.counter_mass)
            pred_group[i] = _predict_conflict_group_from_audit(ar, g_slices)
            score_beacon[i, :] = _counter_masses_by_group_from_audit(ar, g_slices)
            q_used[i] = float(ar.q_used)

            if args.ig_baseline:
                tig0 = time.time()
                igs = _ig_group_scores(clf, x_eval[i], y_hat=y_hat, group_slices=g_slices, steps=q)
                t_ig += time.time() - tig0
                if igs is not None:
                    score_ig[i, :] = igs
                    q_used_ig[i] = float(q)
        elapsed = time.time() - t0
        lat_obj = float(elapsed / max(len(x_eval), 1))
        lat_obj_ig = float(t_ig / max(len(x_eval), 1)) if args.ig_baseline else float("nan")

        # conflict detection metrics
        y_conf = conflict_present.astype(np.int64)
        au_entropy = _auc(y_conf, s_entropy)
        ap_entropy = _auprc(y_conf, s_entropy)
        au_counter = _auc(y_conf, s_counter)
        ap_counter = _auprc(y_conf, s_counter)

        # localization only on injected conflicts
        m = y_conf == 1
        loc_n = int(np.sum(m))
        loc_acc = float(np.mean(pred_group[m] == conflict_group[m])) if loc_n > 0 else float("nan")

        true_g = conflict_group[m]
        sb = score_beacon[m]
        b_top1, b_top1_lo, b_top1_hi = _bootstrap_ci_topk(sb, true_g, 1, args.bootstrap, args.seed + 17 * q + 1)
        b_top2, b_top2_lo, b_top2_hi = _bootstrap_ci_topk(sb, true_g, 2, args.bootstrap, args.seed + 17 * q + 2)
        g = max(1, len(g_slices))
        r_top1 = 1.0 / float(g)
        r_top2 = min(1.0, 2.0 / float(g))
        r_top1_lo, r_top1_hi = _binom_ci_approx(r_top1, loc_n)
        r_top2_lo, r_top2_hi = _binom_ci_approx(r_top2, loc_n)

        ig_ok = bool(args.ig_baseline and np.any(np.isfinite(score_ig[m])))
        b_hits = (np.argmax(sb, axis=1) == true_g).astype(np.float64) if loc_n > 0 else np.array([], dtype=np.float64)
        r_p = np.full(loc_n, r_top1, dtype=np.float64)
        delta_b_r_pair, delta_b_r_lo, delta_b_r_hi, frac_pos_b_r = _paired_bootstrap_delta_against_random(
            b_hits, r_p, args.paired_bootstrap, args.seed + 19 * q + 11
        )

        if ig_ok:
            si = score_ig[m]
            i_top1, i_top1_lo, i_top1_hi = _bootstrap_ci_topk(si, true_g, 1, args.bootstrap, args.seed + 17 * q + 5)
            i_top2, i_top2_lo, i_top2_hi = _bootstrap_ci_topk(si, true_g, 2, args.bootstrap, args.seed + 17 * q + 6)
            i_hits = (np.argmax(si, axis=1) == true_g).astype(np.float64)
            delta_b_i_pair, delta_b_i_lo, delta_b_i_hi, frac_pos_b_i = _paired_bootstrap_delta_between_hits(
                b_hits, i_hits, args.paired_bootstrap, args.seed + 19 * q + 13
            )
            delta_b_r = b_top1 - r_top1
            delta_b_i = b_top1 - i_top1
        else:
            i_top1 = i_top1_lo = i_top1_hi = float("nan")
            i_top2 = i_top2_lo = i_top2_hi = float("nan")
            delta_b_i_pair = delta_b_i_lo = delta_b_i_hi = frac_pos_b_i = float("nan")
            delta_b_r = b_top1 - r_top1
            delta_b_i = float("nan")

        rows.append(
            {
                "dataset": args.dataset_name,
                "model": args.model,
                "q_max": q,
                "k0": k0,
                "partition_mode": args.partition_mode,
                "neutralizer": args.neutralizer,
                "n_eval": int(len(x_eval)),
                "n_conflict": int(np.sum(y_conf == 1)),
                "conflict_ratio": float(args.conflict_ratio),
                "conflict_det_auroc_entropy": au_entropy,
                "conflict_det_auprc_entropy": ap_entropy,
                "conflict_det_auroc_beacon_counter_mass": au_counter,
                "conflict_det_auprc_beacon_counter_mass": ap_counter,
                "delta_auroc_beacon_minus_entropy": au_counter - au_entropy,
                "delta_auprc_beacon_minus_entropy": ap_counter - ap_entropy,
                # keep single authoritative top-1 from group score ranking
                "loc_top1_acc_beacon": b_top1,
                "loc_top1_acc_beacon_legacy_predgroup": loc_acc,
                "loc_top1_acc_beacon_boot": b_top1,
                "loc_top1_ci_low_beacon": b_top1_lo,
                "loc_top1_ci_high_beacon": b_top1_hi,
                "loc_recall_at2_beacon": b_top2,
                "loc_recall_at2_ci_low_beacon": b_top2_lo,
                "loc_recall_at2_ci_high_beacon": b_top2_hi,
                "loc_top1_acc_random": r_top1,
                "loc_top1_ci_low_random": r_top1_lo,
                "loc_top1_ci_high_random": r_top1_hi,
                "loc_recall_at2_random": r_top2,
                "loc_recall_at2_ci_low_random": r_top2_lo,
                "loc_recall_at2_ci_high_random": r_top2_hi,
                "loc_top1_acc_ig": i_top1,
                "loc_top1_ci_low_ig": i_top1_lo,
                "loc_top1_ci_high_ig": i_top1_hi,
                "loc_recall_at2_ig": i_top2,
                "loc_recall_at2_ci_low_ig": i_top2_lo,
                "loc_recall_at2_ci_high_ig": i_top2_hi,
                "delta_loc_top1_beacon_minus_random": delta_b_r,
                "delta_loc_top1_beacon_minus_ig": delta_b_i,
                "delta_loc_top1_beacon_minus_random_pair": delta_b_r_pair,
                "ci_delta_loc_top1_beacon_minus_random_pair_low": delta_b_r_lo,
                "ci_delta_loc_top1_beacon_minus_random_pair_high": delta_b_r_hi,
                "frac_positive_delta_loc_top1_beacon_minus_random_pair": frac_pos_b_r,
                "delta_loc_top1_beacon_minus_ig_pair": delta_b_i_pair,
                "ci_delta_loc_top1_beacon_minus_ig_pair_low": delta_b_i_lo,
                "ci_delta_loc_top1_beacon_minus_ig_pair_high": delta_b_i_hi,
                "frac_positive_delta_loc_top1_beacon_minus_ig_pair": frac_pos_b_i,
                "mean_q_used": float(np.mean(q_used)),
                "mean_q_used_ig": float(np.mean(q_used_ig)) if args.ig_baseline else float("nan"),
                "latency_per_object_sec": lat_obj,
                "latency_per_object_ig_sec": lat_obj_ig,
                "group_mode": args.group_mode,
            }
        )
        if args.per_sample_out:
            g_count = len(g_slices)
            ig_argmax = np.argmax(score_ig, axis=1) if args.ig_baseline else np.full(len(x_eval), -1, dtype=np.int64)
            beacon_argmax = np.argmax(score_beacon, axis=1)
            for i in range(len(x_eval)):
                is_conf = int(y_conf[i] == 1)
                tg = int(conflict_group[i]) if is_conf else -1
                bpred = int(beacon_argmax[i]) if is_conf else -1
                igpred = int(ig_argmax[i]) if (is_conf and args.ig_baseline and np.all(np.isfinite(score_ig[i]))) else -1
                b_ok = int(bpred == tg) if is_conf else -1
                ig_ok_i = int(igpred == tg) if igpred >= 0 else -1
                per_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "model": args.model,
                        "q_max": int(q),
                        "seed": int(args.seed),
                        "npz_path": str(args.npz_path),
                        "group_mode": args.group_mode,
                        "partition_mode": args.partition_mode,
                        "sample_index_eval": int(i),
                        "is_conflict": is_conf,
                        "true_conflict_group": tg,
                        "n_groups": int(g_count),
                        "random_p_top1": float(1.0 / max(g_count, 1)),
                        "pred_group_beacon": bpred,
                        "is_correct_beacon": b_ok,
                        "pred_group_ig": igpred,
                        "is_correct_ig": ig_ok_i,
                        "beacon_top1_score": float(np.max(score_beacon[i])) if is_conf else float("nan"),
                        "beacon_true_group_score": float(score_beacon[i, tg]) if is_conf and tg >= 0 else float("nan"),
                    }
                )
        print(
            f"q={q}: det_auc entropy={au_entropy:.4f} beacon={au_counter:.4f} "
            f"det_ap entropy={ap_entropy:.4f} beacon={ap_counter:.4f} "
            f"loc@1 beacon={b_top1:.4f} random={r_top1:.4f} ig={i_top1 if np.isfinite(i_top1) else float('nan'):.4f} "
            f"mean_q={np.mean(q_used):.2f} lat_obj={lat_obj:.6f}s",
            flush=True,
        )
        plot_q.append(q)
        plot_beacon.append(b_top1)
        plot_random.append(r_top1)
        plot_ig.append(i_top1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved: {out}")
    if args.per_sample_out and per_rows:
        per_out = Path(args.per_sample_out)
        per_out.parent.mkdir(parents=True, exist_ok=True)
        with per_out.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            wr.writeheader()
            wr.writerows(per_rows)
        print(f"saved: {per_out}")

    # Plot localization top-1 vs budget
    pth = Path(args.plot_out)
    pth.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(plot_q, plot_beacon, marker="o", label="BEACON")
    plt.plot(plot_q, plot_random, marker="o", label="Random")
    if any(np.isfinite(v) for v in plot_ig):
        q_ig = [q for q, v in zip(plot_q, plot_ig) if np.isfinite(v)]
        v_ig = [v for v in plot_ig if np.isfinite(v)]
        plt.plot(q_ig, v_ig, marker="o", label="Budgeted IG")
    plt.xlabel("Query Budget Q")
    plt.ylabel("Localization Top-1 Accuracy")
    plt.title("PAMAP2 Conflict Localization vs Budget")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pth, dpi=220)
    plt.close()
    print(f"saved: {pth}")


if __name__ == "__main__":
    main()
