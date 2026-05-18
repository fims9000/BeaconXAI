#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
import warnings
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.core import BeaconAudit
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig

warnings.filterwarnings("ignore")


def _group_slices(n_channels: int, mode: str) -> list[tuple[int, int]]:
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
    if n_channels == 9:
        return [(0, 3), (3, 6), (6, 9)]
    return [(i, i + 1) for i in range(n_channels)]


def _ts_stat_features(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[None, ...]
        squeeze = True
    else:
        squeeze = False
    feats = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        np.median(x, axis=1),
        (x**2).mean(axis=1),
        (np.diff(x, axis=1) ** 2).mean(axis=1),
    ]
    out = np.concatenate(feats, axis=1)
    return out[0] if squeeze else out


def _anfis_features(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[None, ...]
        squeeze = True
    else:
        squeeze = False

    n, t, c = x.shape
    base = _ts_stat_features(x)
    mean_abs = np.mean(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x * x, axis=1))
    q25 = np.quantile(x, 0.25, axis=1)
    q75 = np.quantile(x, 0.75, axis=1)
    iqr = q75 - q25
    sma = np.sum(np.abs(x), axis=1) / float(t)

    xc = x - x.mean(axis=1, keepdims=True)
    std = x.std(axis=1) + 1e-8
    corrs = []
    for i in range(c):
        for j in range(i + 1, c):
            num = np.mean(xc[:, :, i] * xc[:, :, j], axis=1)
            den = std[:, i] * std[:, j]
            corrs.append((num / den)[:, None])
    corr_feat = np.concatenate(corrs, axis=1) if corrs else np.zeros((n, 0), dtype=x.dtype)

    fx = np.fft.rfft(x, axis=1)
    pwr = (fx.real * fx.real + fx.imag * fx.imag).astype(np.float64, copy=False)
    pwr_nd = pwr[:, 1:, :] if pwr.shape[1] > 1 else pwr
    spec_energy = np.mean(pwr_nd, axis=1)
    ps = pwr_nd + 1e-12
    ps = ps / np.sum(ps, axis=1, keepdims=True)
    spec_entropy = -np.sum(ps * np.log(ps), axis=1) / np.log(ps.shape[1] + 1e-12)
    dom_bin = np.argmax(pwr_nd, axis=1).astype(np.float64) / float(max(1, pwr_nd.shape[1] - 1))

    out = np.concatenate(
        [base, mean_abs, rms, iqr, sma, corr_feat, spec_energy, spec_entropy, dom_bin],
        axis=1,
    )
    return out[0] if squeeze else out


class _TreeStatsClassifierLocal:
    def __init__(self, model: ExtraTreesClassifier):
        self.model = model

    def logits(self, x: np.ndarray) -> np.ndarray:
        f = _anfis_features(x).reshape(1, -1)
        probs = self.model.predict_proba(f)[0]
        return np.log(np.clip(probs, 1e-12, 1.0))


class _BoostingStatsClassifierLocal:
    def __init__(self, model: HistGradientBoostingClassifier):
        self.model = model

    def logits(self, x: np.ndarray) -> np.ndarray:
        f = _anfis_features(x).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(f)[0]
            return np.log(np.clip(probs, 1e-12, 1.0))
        out = self.model.decision_function(f)
        if np.ndim(out) == 1:
            score = float(out[0])
            return np.array([-score, score], dtype=np.float64)
        return np.asarray(out[0], dtype=np.float64)


def _train_extratrees_local(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_features: float, min_samples_leaf: int):
    f_train = _anfis_features(x_train)
    m = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=1,
    )
    m.fit(f_train, y_train)
    return _TreeStatsClassifierLocal(m)


def _train_histgbt_local(x_train: np.ndarray, y_train: np.ndarray):
    f_train = _anfis_features(x_train)
    m = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        random_state=42,
    )
    m.fit(f_train, y_train)
    return _BoostingStatsClassifierLocal(m)


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


def _counter_scores_by_component(audit_res, group_slices, time_slices, score_mode: str) -> np.ndarray:
    g_count = len(group_slices)
    b_count = len(time_slices)
    out = np.zeros(g_count * b_count, dtype=np.float64)
    leaf = audit_res.metadata.get("leaf_components", [])
    deltas = audit_res.metadata.get("leaf_deltas", [])
    for comp_tuple, d in zip(leaf, deltas):
        if score_mode == "neg_only":
            if d >= 0:
                continue
            mass = -float(d)
        else:
            mass = abs(float(d))
        if mass <= 0:
            continue
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


def _neutralize_component(x: np.ndarray, t0: int, t1: int, c0: int, c1: int, mode: str) -> np.ndarray:
    y = x.copy()
    if mode in ("zero", "mean"):
        y[t0:t1, c0:c1] = 0.0
        return y
    # interp
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


def _bootstrap_ci_hits(hits: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    if len(hits) == 0:
        return float("nan"), float("nan"), float("nan")
    base = float(np.mean(hits))
    if n_boot <= 0:
        return base, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(hits)
    vals = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = float(np.mean(hits[idx]))
    return base, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _paired_bootstrap_delta(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float, float]:
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = float(np.mean(a - b))
    if n_boot <= 0:
        return d, float("nan"), float("nan"), float(d > 0)
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.zeros(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(np.mean(a[idx] - b[idx]))
    return d, float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)), float(np.mean(vals > 0.0))


def _true_ranks_from_scores(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    inv = np.empty_like(order, dtype=np.int64)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    return inv[np.arange(order.shape[0]), y_true] + 1


def _rank_metrics(ranks: np.ndarray, n_components: int) -> dict[str, float]:
    r = ranks.astype(np.float64)
    out = {
        "hit_at_3": float(np.mean(r <= 3.0)),
        "hit_at_5": float(np.mean(r <= 5.0)),
        "mrr": float(np.mean(1.0 / r)),
        "mean_rank": float(np.mean(r)),
    }
    rand_mean_rank = (float(n_components) + 1.0) / 2.0
    denom = max(rand_mean_rank - 1.0, 1e-12)
    out["normalized_rank_gain"] = float((rand_mean_rank - out["mean_rank"]) / denom)
    return out


def _binom_sf(k: int, n: int, p: float) -> float:
    if n <= 0:
        return float("nan")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lp = math.log(max(p, 1e-15))
    lq = math.log(max(1.0 - p, 1e-15))
    vals = []
    for i in range(k, n + 1):
        lv = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        vals.append(lv)
    m = max(vals)
    s = sum(math.exp(v - m) for v in vals)
    return float(min(1.0, math.exp(m) * s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Component-level conflict localization benchmark")
    p.add_argument("--npz-path", required=True)
    p.add_argument("--dataset-name", required=True)
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
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--paired-bootstrap", type=int, default=2000)
    p.add_argument("--beacon-score-mode", choices=["neg_only", "abs_delta"], default="neg_only")
    p.add_argument("--beacon-q-frag-ratio", type=float, default=0.0)
    p.add_argument("--beacon-alpha", type=float, default=1.0)
    p.add_argument("--beacon-beta", type=float, default=0.5)
    p.add_argument("--beacon-gamma", type=float, default=1.0)
    p.add_argument("--beacon-tau-s", type=float, default=0.10)
    p.add_argument("--beacon-risk-policy", choices=["rho_only", "rho_plus_conflict"], default="rho_only")
    p.add_argument("--beacon-refinement-mode", choices=["none", "counter_mass", "conflict_ratio", "mixed"], default="mixed")
    p.add_argument("--beacon-margin-mode", choices=["adaptive_all", "adaptive_time", "nearest_time", "adaptive_sensor_group", "nearest_counter_sensor_group"], default="adaptive_all")
    p.add_argument("--plot-out", default="")
    p.add_argument("--per-sample-out", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--cnn-epochs", type=int, default=16)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--et-n-estimators", type=int, default=300)
    p.add_argument("--et-max-features", type=float, default=0.7)
    p.add_argument("--et-min-samples-leaf", type=int, default=1)
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
        clf = _train_extratrees_local(
            x_train,
            y_train,
            n_estimators=args.et_n_estimators,
            max_features=args.et_max_features,
            min_samples_leaf=args.et_min_samples_leaf,
        )
    elif args.model == "histgbt":
        clf = _train_histgbt_local(x_train, y_train)
    else:
        from beaconxai.models import train_1dcnn

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

    train_margins = []
    for i in range(min(len(x_train), 2000)):
        lg = clf.logits(x_train[i])
        yhat, m = _margin(lg)
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    n_channels = x_test.shape[-1]
    t_len = x_test.shape[1]
    g_slices = _group_slices(n_channels, args.group_mode)
    t_slices = _time_slices(t_len, args.time_bins)
    n_components = len(g_slices) * len(t_slices)

    x_eval, conflict_present, true_comp = _inject_component_conflicts(
        x_test,
        y_test,
        g_slices,
        t_slices,
        conflict_ratio=args.conflict_ratio,
        seed=args.seed + 101,
    )

    if args.neutralizer == "mean":
        neutralizer = Neutralizer(mode="mean", channel_means=np.zeros(n_channels, dtype=np.float32))
    else:
        neutralizer = Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32))

    rows = []
    per_rows = []
    plot_q, plot_b, plot_u, plot_l, plot_r = [], [], [], [], []

    for q in q_values:
        k0 = 4 if q <= 8 else 8
        cfg = BeaconConfig(
            q_max=q,
            k0=k0,
            l_min=4,
            k_pos=3,
            k_neg=3,
            q_frag_ratio=args.beacon_q_frag_ratio,
            alpha=args.beacon_alpha,
            beta=args.beacon_beta,
            gamma=args.beacon_gamma,
            tau_s=args.beacon_tau_s,
            tau_m=tau_m,
            refinement_mode=args.beacon_refinement_mode,
            partition_mode=args.partition_mode,
            risk_policy=args.beacon_risk_policy,
            margin_mode=args.beacon_margin_mode,
            audit_mode="full",
        )
        audit = BeaconAudit(model_logits=clf.logits, neutralizer=neutralizer, config=cfg)

        scores_beacon = np.zeros((len(x_eval), n_components), dtype=np.float64)
        scores_uniform = np.full((len(x_eval), n_components), -1e18, dtype=np.float64)
        scores_logo = np.full((len(x_eval), n_components), -1e18, dtype=np.float64)
        pred_group_logo = np.full(len(x_eval), -1, dtype=np.int64)
        q_used = np.zeros(len(x_eval), dtype=np.float64)

        t0 = time.time()
        rng_q = np.random.default_rng(args.seed + 313 * q)
        for i in range(len(x_eval)):
            ar = audit.audit(x_eval[i])
            scores_beacon[i, :] = _counter_scores_by_component(ar, g_slices, t_slices, args.beacon_score_mode)
            q_used[i] = float(ar.q_used)

            lg0 = clf.logits(x_eval[i])
            yhat, m0 = _margin(lg0)
            budget = min(q, n_components)
            cand = rng_q.choice(n_components, size=budget, replace=False)
            for cidx in cand:
                g, b = _component_to_gb(int(cidx), len(t_slices))
                c0, c1 = g_slices[g]
                tt0, tt1 = t_slices[b]
                x_mod = _neutralize_component(x_eval[i], tt0, tt1, c0, c1, args.neutralizer)
                lg1 = clf.logits(x_mod)
                _y1, m1 = _margin(lg1)
                scores_uniform[i, cidx] = abs(m0 - m1)

            group_scores = np.full(len(g_slices), -1e18, dtype=np.float64)
            for gi, (c0, c1) in enumerate(g_slices):
                x_g = _neutralize_component(x_eval[i], 0, t_len, c0, c1, args.neutralizer)
                lg_g = clf.logits(x_g)
                _yg, mg = _margin(lg_g)
                group_scores[gi] = abs(m0 - mg)
            g_hat = int(np.argmax(group_scores))
            pred_group_logo[i] = g_hat

            rem = max(0, int(q) - len(g_slices))
            if rem > 0:
                b_budget = min(rem, len(t_slices))
                cand_bins = rng_q.choice(len(t_slices), size=b_budget, replace=False)
                c0, c1 = g_slices[g_hat]
                for bi in cand_bins:
                    tt0, tt1 = t_slices[int(bi)]
                    x_mod = _neutralize_component(x_eval[i], tt0, tt1, c0, c1, args.neutralizer)
                    lg1 = clf.logits(x_mod)
                    _y1, m1 = _margin(lg1)
                    comp = _component_from_gb(g_hat, int(bi), len(t_slices))
                    scores_logo[i, comp] = abs(m0 - m1)
            else:
                comp = _component_from_gb(g_hat, 0, len(t_slices))
                scores_logo[i, comp] = 0.0
        lat = float((time.time() - t0) / max(len(x_eval), 1))

        m = conflict_present == 1
        y_true = true_comp[m]

        pred_b = np.argmax(scores_beacon[m], axis=1)
        pred_u = np.argmax(scores_uniform[m], axis=1)
        pred_l = np.argmax(scores_logo[m], axis=1)
        rb = _true_ranks_from_scores(scores_beacon[m], y_true)
        ru = _true_ranks_from_scores(scores_uniform[m], y_true)
        rl = _true_ranks_from_scores(scores_logo[m], y_true)
        mb = _rank_metrics(rb, n_components)
        mu = _rank_metrics(ru, n_components)
        ml = _rank_metrics(rl, n_components)
        rand_hit3 = float(min(3.0 / n_components, 1.0))
        rand_hit5 = float(min(5.0 / n_components, 1.0))
        rand_mrr = float(np.mean(1.0 / np.arange(1, n_components + 1)))
        rand_mean_rank = (float(n_components) + 1.0) / 2.0
        hit_b = (pred_b == y_true).astype(np.float64)
        hit_u = (pred_u == y_true).astype(np.float64)
        hit_l = (pred_l == y_true).astype(np.float64)
        true_group = y_true // len(t_slices)
        logo_group = pred_group_logo[m]
        hit_lg = (logo_group == true_group).astype(np.float64)
        p_rand = 1.0 / float(n_components)
        hit_r = np.full_like(hit_b, p_rand, dtype=np.float64)

        b_top1, b_lo, b_hi = _bootstrap_ci_hits(hit_b, args.bootstrap, args.seed + 17 * q + 1)
        u_top1, u_lo, u_hi = _bootstrap_ci_hits(hit_u, args.bootstrap, args.seed + 17 * q + 2)
        l_top1, l_lo, l_hi = _bootstrap_ci_hits(hit_l, args.bootstrap, args.seed + 17 * q + 4)
        lg_top1, lg_lo, lg_hi = _bootstrap_ci_hits(hit_lg, args.bootstrap, args.seed + 17 * q + 5)
        r_top1 = p_rand

        d_bu, d_bu_lo, d_bu_hi, frac_bu = _paired_bootstrap_delta(hit_b, hit_u, args.paired_bootstrap, args.seed + 17 * q + 3)
        d_bl, d_bl_lo, d_bl_hi, frac_bl = _paired_bootstrap_delta(hit_b, hit_l, args.paired_bootstrap, args.seed + 17 * q + 6)
        d_br = float(np.mean(hit_b - hit_r))
        p_binom = _binom_sf(int(np.sum(hit_b)), len(hit_b), p_rand)

        rows.append(
            {
                "dataset": args.dataset_name,
                "model": args.model,
                "q_max": int(q),
                "k0": int(k0),
                "group_mode": args.group_mode,
                "time_bins": int(args.time_bins),
                "n_components": int(n_components),
                "partition_mode": args.partition_mode,
                "neutralizer": args.neutralizer,
                "n_eval": int(len(x_eval)),
                "n_conflict": int(np.sum(m)),
                "conflict_ratio": float(args.conflict_ratio),
                "loc_top1_beacon": b_top1,
                "loc_hit3_beacon": mb["hit_at_3"],
                "loc_hit5_beacon": mb["hit_at_5"],
                "loc_mrr_beacon": mb["mrr"],
                "loc_mean_rank_beacon": mb["mean_rank"],
                "loc_nrg_beacon": mb["normalized_rank_gain"],
                "loc_top1_ci_low_beacon": b_lo,
                "loc_top1_ci_high_beacon": b_hi,
                "loc_top1_uniform": u_top1,
                "loc_hit3_uniform": mu["hit_at_3"],
                "loc_hit5_uniform": mu["hit_at_5"],
                "loc_mrr_uniform": mu["mrr"],
                "loc_mean_rank_uniform": mu["mean_rank"],
                "loc_nrg_uniform": mu["normalized_rank_gain"],
                "loc_top1_ci_low_uniform": u_lo,
                "loc_top1_ci_high_uniform": u_hi,
                "loc_top1_logo": l_top1,
                "loc_hit3_logo": ml["hit_at_3"],
                "loc_hit5_logo": ml["hit_at_5"],
                "loc_mrr_logo": ml["mrr"],
                "loc_mean_rank_logo": ml["mean_rank"],
                "loc_nrg_logo": ml["normalized_rank_gain"],
                "loc_top1_ci_low_logo": l_lo,
                "loc_top1_ci_high_logo": l_hi,
                "loc_group_top1_logo": lg_top1,
                "loc_group_ci_low_logo": lg_lo,
                "loc_group_ci_high_logo": lg_hi,
                "loc_top1_random": r_top1,
                "loc_hit3_random": rand_hit3,
                "loc_hit5_random": rand_hit5,
                "loc_mrr_random": rand_mrr,
                "loc_mean_rank_random": rand_mean_rank,
                "loc_nrg_random": 0.0,
                "delta_loc_top1_beacon_minus_uniform": d_bu,
                "ci_delta_loc_top1_beacon_minus_uniform_low": d_bu_lo,
                "ci_delta_loc_top1_beacon_minus_uniform_high": d_bu_hi,
                "frac_positive_delta_loc_top1_beacon_minus_uniform": frac_bu,
                "delta_loc_top1_beacon_minus_logo": d_bl,
                "ci_delta_loc_top1_beacon_minus_logo_low": d_bl_lo,
                "ci_delta_loc_top1_beacon_minus_logo_high": d_bl_hi,
                "frac_positive_delta_loc_top1_beacon_minus_logo": frac_bl,
                "delta_loc_top1_beacon_minus_random": d_br,
                "pvalue_binom_beacon_gt_random": p_binom,
                "mean_q_used": float(np.mean(q_used)),
                "latency_per_object_sec": lat,
            }
        )

        if args.per_sample_out:
            for j, idx_eval in enumerate(np.where(m)[0]):
                per_rows.append(
                    {
                        "dataset": args.dataset_name,
                        "model": args.model,
                        "q_max": int(q),
                        "sample_index_eval": int(idx_eval),
                        "true_component": int(y_true[j]),
                        "pred_component_beacon": int(pred_b[j]),
                        "pred_component_uniform": int(pred_u[j]),
                        "pred_component_logo": int(pred_l[j]),
                        "rank_true_beacon": int(rb[j]),
                        "rank_true_uniform": int(ru[j]),
                        "rank_true_logo": int(rl[j]),
                        "hit3_beacon": int(rb[j] <= 3),
                        "hit5_beacon": int(rb[j] <= 5),
                        "hit3_uniform": int(ru[j] <= 3),
                        "hit5_uniform": int(ru[j] <= 5),
                        "hit3_logo": int(rl[j] <= 3),
                        "hit5_logo": int(rl[j] <= 5),
                        "is_correct_beacon": int(hit_b[j]),
                        "is_correct_uniform": int(hit_u[j]),
                        "is_correct_logo": int(hit_l[j]),
                        "pred_group_logo": int(logo_group[j]),
                        "is_correct_group_logo": int(hit_lg[j]),
                        "random_p_top1": float(p_rand),
                        "n_components": int(n_components),
                        "group_mode": args.group_mode,
                        "time_bins": int(args.time_bins),
                        "seed": int(args.seed),
                        "npz_path": str(args.npz_path),
                    }
                )

        print(
            f"q={q} loc@1 beacon={b_top1:.4f} uniform={u_top1:.4f} logo={l_top1:.4f} random={r_top1:.4f} "
            f"delta(b-u)={d_bu:.4f} [{d_bu_lo:.4f},{d_bu_hi:.4f}] delta(b-l)={d_bl:.4f} [{d_bl_lo:.4f},{d_bl_hi:.4f}] p_binom={p_binom:.3g}",
            flush=True,
        )

        plot_q.append(q)
        plot_b.append(b_top1)
        plot_u.append(u_top1)
        plot_l.append(l_top1)
        plot_r.append(r_top1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved: {out}")

    if args.per_sample_out and per_rows:
        po = Path(args.per_sample_out)
        po.parent.mkdir(parents=True, exist_ok=True)
        with po.open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
            wr.writeheader()
            wr.writerows(per_rows)
        print(f"saved: {po}")

    if args.plot_out:
        import matplotlib.pyplot as plt

        pp = Path(args.plot_out)
        pp.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 4.5))
        plt.plot(plot_q, plot_b, marker="o", label="BEACON")
        plt.plot(plot_q, plot_u, marker="o", label="Uniform budgeted occlusion")
        plt.plot(plot_q, plot_l, marker="o", label="Leave-one-group-out")
        plt.plot(plot_q, plot_r, marker="o", label="Random")
        plt.xlabel("Query Budget Q")
        plt.ylabel("Component Top-1 Accuracy")
        plt.title(f"{args.dataset_name.upper()} component-level localization")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(pp, dpi=220)
        plt.close()
        print(f"saved: {pp}")


if __name__ == "__main__":
    main()
