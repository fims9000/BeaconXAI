#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import warnings

import numpy as np

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


def _group_slices(n_channels: int, mode: str) -> list[tuple[int, int]]:
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
    return [(i, i + 1) for i in range(n_channels)]


def _time_slices(t_len: int, n_bins: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, t_len, n_bins + 1, dtype=int)
    out = []
    for i in range(n_bins):
        t0, t1 = int(edges[i]), int(edges[i + 1])
        if t1 <= t0:
            t1 = min(t_len, t0 + 1)
        out.append((t0, t1))
    return out


def _component_from_gb(g: int, b: int, n_bins: int) -> int:
    return g * n_bins + b


def _component_to_gb(comp: int, n_bins: int) -> tuple[int, int]:
    return comp // n_bins, comp % n_bins


def _inject_component_conflicts(
    x_clean: np.ndarray,
    y_clean: np.ndarray,
    group_slices: list[tuple[int, int]],
    time_slices: list[tuple[int, int]],
    conflict_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(x_clean)
    n_conf = int(round(conflict_ratio * n))
    conf_idx = rng.choice(n, size=n_conf, replace=False)

    x_eval = x_clean.copy()
    conf_present = np.zeros(n, dtype=np.int64)
    true_comp = -np.ones(n, dtype=np.int64)

    g_count = len(group_slices)
    b_count = len(time_slices)
    for i in conf_idx:
        yi = int(y_clean[i])
        donors = np.where(y_clean != yi)[0]
        if len(donors) == 0:
            continue
        j = int(rng.choice(donors))
        g = int(rng.integers(0, g_count))
        b = int(rng.integers(0, b_count))
        c0, c1 = group_slices[g]
        t0, t1 = time_slices[b]
        x_eval[i, t0:t1, c0:c1] = x_clean[j, t0:t1, c0:c1]
        conf_present[i] = 1
        true_comp[i] = _component_from_gb(g, b, b_count)
    return x_eval, conf_present, true_comp


def _neutralize_component(x: np.ndarray, t0: int, t1: int, c0: int, c1: int, mode: str) -> np.ndarray:
    y = x.copy()
    if mode in ("zero", "mean"):
        y[t0:t1, c0:c1] = 0.0
        return y
    for c in range(c0, c1):
        if t0 > 0 and t1 < y.shape[0]:
            left = y[t0 - 1, c]
            right = y[t1, c]
            y[t0:t1, c] = np.linspace(left, right, t1 - t0, endpoint=False)
        else:
            y[t0:t1, c] = 0.0
    return y


def _margin(logits: np.ndarray) -> tuple[int, float]:
    y_hat = int(np.argmax(logits))
    tmp = np.copy(logits)
    tmp[y_hat] = -1e18
    return y_hat, float(logits[y_hat] - np.max(tmp))


def _counter_scores_by_component(audit_res, group_slices, time_slices) -> np.ndarray:
    out = np.zeros(len(group_slices) * len(time_slices), dtype=np.float64)
    leaf = audit_res.metadata.get("leaf_components", [])
    deltas = audit_res.metadata.get("leaf_deltas", [])
    for comp_tuple, d in zip(leaf, deltas):
        if d >= 0:
            continue
        mass = -float(d)
        _cid, lt0, lt1, lc0, lc1 = comp_tuple
        area = max(1, (lt1 - lt0) * (lc1 - lc0))
        for gi, (g0, g1) in enumerate(group_slices):
            ov_c = max(0, min(lc1, g1) - max(lc0, g0))
            if ov_c <= 0:
                continue
            for bi, (t0, t1) in enumerate(time_slices):
                ov_t = max(0, min(lt1, t1) - max(lt0, t0))
                if ov_t <= 0:
                    continue
                w = float((ov_t * ov_c) / area)
                out[_component_from_gb(gi, bi, len(time_slices))] += mass * w
    return out


def _true_ranks_from_scores(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    inv = np.empty_like(order, dtype=np.int64)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    return inv[np.arange(order.shape[0]), y_true] + 1


def _rank_metrics(ranks: np.ndarray, n_components: int) -> dict[str, float]:
    r = ranks.astype(np.float64)
    rand_mean_rank = (float(n_components) + 1.0) / 2.0
    denom = max(rand_mean_rank - 1.0, 1e-12)
    return {
        "loc@1": float(np.mean(r <= 1.0)),
        "hit@3": float(np.mean(r <= 3.0)),
        "hit@5": float(np.mean(r <= 5.0)),
        "MRR": float(np.mean(1.0 / r)),
        "mean_rank": float(np.mean(r)),
        "NRG": float((rand_mean_rank - float(np.mean(r))) / denom),
    }


def _zscore(v: np.ndarray) -> np.ndarray:
    mu = float(np.mean(v))
    sd = float(np.std(v))
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


def _component_cheap_scores(
    x: np.ndarray,
    class_profile: np.ndarray,
    g_slices: list[tuple[int, int]],
    t_slices: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_comp = len(g_slices) * len(t_slices)
    energy = np.zeros(n_comp, dtype=np.float64)
    var = np.zeros(n_comp, dtype=np.float64)
    pdist = np.zeros(n_comp, dtype=np.float64)
    mdev = np.zeros(n_comp, dtype=np.float64)
    for comp in range(n_comp):
        g, b = _component_to_gb(comp, len(t_slices))
        c0, c1 = g_slices[g]
        t0, t1 = t_slices[b]
        patch = x[t0:t1, c0:c1]
        ppatch = class_profile[t0:t1, c0:c1]
        energy[comp] = float(np.mean(patch * patch))
        var[comp] = float(np.var(patch))
        pdist[comp] = float(np.mean(np.abs(patch - ppatch)))
        mdev[comp] = float(abs(float(np.mean(patch)) - float(np.mean(ppatch))))
    ze, zv, zp, zm = _zscore(energy), _zscore(var), _zscore(pdist), _zscore(mdev)
    score_energy = ze + zv
    score_profile = zp + zm
    score_combined = ze + zv + zp + zm
    return score_energy, score_profile, score_combined


def _fill_scores_for_candidates(
    clf,
    x: np.ndarray,
    m0: float,
    cand: np.ndarray,
    g_slices: list[tuple[int, int]],
    t_slices: list[tuple[int, int]],
    neutralizer_mode: str,
    n_components: int,
) -> np.ndarray:
    scores = np.full(n_components, -1e18, dtype=np.float64)
    for cidx in cand:
        g, b = _component_to_gb(int(cidx), len(t_slices))
        c0, c1 = g_slices[g]
        t0, t1 = t_slices[b]
        x_mod = _neutralize_component(x, t0, t1, c0, c1, neutralizer_mode)
        lg1 = clf.logits(x_mod)
        _y1, m1 = _margin(lg1)
        scores[int(cidx)] = abs(m0 - m1)
    return scores


def _bootstrap_delta(a: np.ndarray, b: np.ndarray, higher_better: bool, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    delta = (a - b) if higher_better else (b - a)
    base = float(np.mean(delta))
    rng = np.random.default_rng(seed)
    n = len(delta)
    vals = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(delta[idx]))
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    p_pos = float(np.mean(vals > 0.0))
    p_neg = float(np.mean(vals < 0.0))
    p_two = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    return base, lo, hi, p_two


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--npz-path", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--model", choices=["extratrees", "histgbt", "cnn1d"], default="extratrees")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=512)
    p.add_argument("--conflict-ratio", type=float, default=0.5)
    p.add_argument("--q-values", default="16,32,64")
    p.add_argument("--neutralizer", choices=["zero", "interp", "mean"], default="interp")
    p.add_argument("--group-mode", choices=["pamap3", "per_channel", "split2", "auto"], default="auto")
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out-results", required=True)
    p.add_argument("--out-bootstrap", required=True)
    p.add_argument("--out-claims", required=True)
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
        clf = train_extratrees_stats(x_train, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1, random_state=args.seed)
    elif args.model == "histgbt":
        clf = train_histgbt_stats(x_train, y_train, random_state=args.seed)
    else:
        clf = train_1dcnn(x_train, y_train, epochs=16, batch_size=256, lr=1e-3, label_smoothing=0.0, use_class_weights=True, tta_shifts=(0, 50))

    train_margins = []
    for i in range(min(len(x_train), 2000)):
        lg = clf.logits(x_train[i])
        _y, m = _margin(lg)
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    n_classes = int(np.max(y_train)) + 1
    class_profiles = np.zeros((n_classes, x_train.shape[1], x_train.shape[2]), dtype=np.float64)
    for c in range(n_classes):
        idx = np.where(y_train == c)[0]
        if len(idx) > 0:
            class_profiles[c] = np.mean(x_train[idx], axis=0)

    n_channels = x_test.shape[-1]
    t_len = x_test.shape[1]
    g_slices = _group_slices(n_channels, args.group_mode)
    t_slices = _time_slices(t_len, args.time_bins)
    n_components = len(g_slices) * len(t_slices)

    x_eval, conflict_present, true_comp = _inject_component_conflicts(
        x_test, y_test, g_slices, t_slices, args.conflict_ratio, args.seed + 101
    )
    mask_conf = conflict_present == 1
    y_true = true_comp[mask_conf]

    if args.neutralizer == "mean":
        neutralizer = Neutralizer(mode="mean", channel_means=np.zeros(n_channels, dtype=np.float32))
    else:
        neutralizer = Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32))

    methods = [
        "uniform_occlusion",
        "beacon_core",
        "adaptive_old",
        "adaptive_v2_energy",
        "adaptive_v2_profile",
        "adaptive_v2_combined",
    ]
    metric_arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    result_rows: list[dict[str, object]] = []

    for q in q_values:
        k0 = 4 if q <= 8 else 8
        cfg = BeaconConfig(
            q_max=q,
            k0=k0,
            l_min=4,
            k_pos=3,
            k_neg=3,
            q_frag_ratio=0.0,
            alpha=1.0,
            beta=0.5,
            gamma=1.0,
            tau_s=0.10,
            tau_m=tau_m,
            refinement_mode="mixed",
            partition_mode="sensor_group_time",
            risk_policy="rho_only",
            margin_mode="adaptive_all",
            audit_mode="full",
        )
        audit = BeaconAudit(model_logits=clf.logits, neutralizer=neutralizer, config=cfg)

        scores: dict[str, np.ndarray] = {m: np.full((len(x_eval), n_components), -1e18, dtype=np.float64) for m in methods}
        rng_q = np.random.default_rng(args.seed + 313 * q)

        for i in range(len(x_eval)):
            x = x_eval[i]
            lg0 = clf.logits(x)
            yhat, m0 = _margin(lg0)
            budget = min(q, n_components)

            # beacon core
            ar = audit.audit(x)
            scores["beacon_core"][i, :] = _counter_scores_by_component(ar, g_slices, t_slices)

            # uniform baseline
            cand_u = rng_q.choice(n_components, size=budget, replace=False)
            scores["uniform_occlusion"][i, :] = _fill_scores_for_candidates(
                clf, x, m0, cand_u, g_slices, t_slices, args.neutralizer, n_components
            )

            # adaptive_old = group-first then random bins
            group_scores = np.full(len(g_slices), -1e18, dtype=np.float64)
            for gi, (c0, c1) in enumerate(g_slices):
                x_g = _neutralize_component(x, 0, t_len, c0, c1, args.neutralizer)
                lg_g = clf.logits(x_g)
                _yg, mg = _margin(lg_g)
                group_scores[gi] = abs(m0 - mg)
            g_hat = int(np.argmax(group_scores))
            rem = max(0, budget - len(g_slices))
            cand_old: list[int] = []
            if rem > 0:
                b_budget = min(rem, len(t_slices))
                cand_bins = rng_q.choice(len(t_slices), size=b_budget, replace=False)
                for bi in cand_bins:
                    cand_old.append(_component_from_gb(g_hat, int(bi), len(t_slices)))
            else:
                cand_old.append(_component_from_gb(g_hat, 0, len(t_slices)))
            scores["adaptive_old"][i, :] = _fill_scores_for_candidates(
                clf, x, m0, np.array(cand_old, dtype=int), g_slices, t_slices, args.neutralizer, n_components
            )

            # adaptive v2 variants
            prof = class_profiles[yhat]
            sc_e, sc_p, sc_c = _component_cheap_scores(x, prof, g_slices, t_slices)
            for name, sc in (
                ("adaptive_v2_energy", sc_e),
                ("adaptive_v2_profile", sc_p),
                ("adaptive_v2_combined", sc_c),
            ):
                cand = np.argpartition(-sc, kth=budget - 1)[:budget] if budget < n_components else np.arange(n_components)
                scores[name][i, :] = _fill_scores_for_candidates(
                    clf, x, m0, cand, g_slices, t_slices, args.neutralizer, n_components
                )

        for method in methods:
            pred = np.argmax(scores[method][mask_conf], axis=1)
            ranks = _true_ranks_from_scores(scores[method][mask_conf], y_true)
            mets = _rank_metrics(ranks, n_components)
            rand_mean_rank = (float(n_components) + 1.0) / 2.0
            denom = max(rand_mean_rank - 1.0, 1e-12)
            nrg_per_sample = (rand_mean_rank - ranks.astype(np.float64)) / denom
            metric_arrays[(q, method)] = {
                "loc@1": (pred == y_true).astype(np.float64),
                "hit@3": (ranks <= 3).astype(np.float64),
                "hit@5": (ranks <= 5).astype(np.float64),
                "MRR": (1.0 / ranks.astype(np.float64)),
                "mean_rank": ranks.astype(np.float64),
                "NRG": nrg_per_sample,
            }
            result_rows.append(
                {
                    "dataset": args.dataset_name,
                    "neutralizer": args.neutralizer,
                    "q": q,
                    "method": method,
                    "n": int(np.sum(mask_conf)),
                    **mets,
                }
            )

    comparisons = [
        ("adaptive_v2_energy", "uniform_occlusion"),
        ("adaptive_v2_profile", "uniform_occlusion"),
        ("adaptive_v2_combined", "uniform_occlusion"),
        ("adaptive_v2_combined", "beacon_core"),
        ("adaptive_v2_combined", "adaptive_old"),
    ]
    higher_better = {"loc@1": True, "hit@3": True, "hit@5": True, "MRR": True, "NRG": True, "mean_rank": False}
    boot_rows: list[dict[str, object]] = []
    claim_rows: list[dict[str, object]] = []

    for q in q_values:
        for m1, m0 in comparisons:
            for metric in ("loc@1", "hit@3", "hit@5", "MRR", "mean_rank", "NRG"):
                a = metric_arrays[(q, m1)][metric]
                b = metric_arrays[(q, m0)][metric]
                d, lo, hi, p = _bootstrap_delta(a, b, higher_better[metric], args.bootstrap, seed=1000 + q)
                supp = int(lo > 0.0 and p < 0.05)
                row = {
                    "dataset": args.dataset_name,
                    "neutralizer": args.neutralizer,
                    "q": q,
                    "metric": metric,
                    "comparison": f"{m1}_vs_{m0}",
                    "delta": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": p,
                    "supported_positive": supp,
                }
                boot_rows.append(row)
                claim_rows.append(row)

    out_results = Path(args.out_results)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    with out_results.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
        w.writeheader()
        w.writerows(result_rows)

    out_boot = Path(args.out_bootstrap)
    with out_boot.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        w.writeheader()
        w.writerows(boot_rows)

    out_claims = Path(args.out_claims)
    with out_claims.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(claim_rows[0].keys()))
        w.writeheader()
        w.writerows(claim_rows)

    print("saved:", out_results)
    print("saved:", out_boot)
    print("saved:", out_claims)


if __name__ == "__main__":
    main()
