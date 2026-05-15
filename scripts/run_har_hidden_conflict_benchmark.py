#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.core import BeaconAudit
from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig
from scripts.run_component_conflict_benchmark import _train_extratrees_local, _train_histgbt_local


def _time_slices(t_len: int, n_bins: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, t_len, n_bins + 1, dtype=int)
    out = []
    for i in range(n_bins):
        t0, t1 = int(edges[i]), int(edges[i + 1])
        if t1 <= t0:
            t1 = min(t_len, t0 + 1)
        out.append((t0, t1))
    return out


def _component_idx(ch: int, b: int, n_bins: int) -> int:
    return ch * n_bins + b


def _component_decode(comp: int, n_bins: int) -> tuple[int, int]:
    return comp // n_bins, comp % n_bins


def _neighbors(comp: int, n_channels: int, n_bins: int) -> list[int]:
    c, b = _component_decode(comp, n_bins)
    out: list[int] = []
    if b > 0:
        out.append(_component_idx(c, b - 1, n_bins))
    if b + 1 < n_bins:
        out.append(_component_idx(c, b + 1, n_bins))
    if c > 0:
        out.append(_component_idx(c - 1, b, n_bins))
    if c + 1 < n_channels:
        out.append(_component_idx(c + 1, b, n_bins))
    return out


def _margin(logits: np.ndarray) -> tuple[int, float]:
    y = int(np.argmax(logits))
    tmp = logits.copy()
    tmp[y] = -1e18
    return y, float(logits[y] - np.max(tmp))


def _neutralize_component(x: np.ndarray, t0: int, t1: int, c: int, mode: str) -> np.ndarray:
    y = x.copy()
    if mode in ("zero", "mean"):
        y[t0:t1, c:c+1] = 0.0
        return y
    if t0 > 0 and t1 < y.shape[0]:
        left = y[t0 - 1, c]
        right = y[t1, c]
        y[t0:t1, c] = np.linspace(left, right, t1 - t0, endpoint=False)
    else:
        y[t0:t1, c] = 0.0
    return y


def _counter_scores_by_component(audit_res, n_channels: int, t_slices: list[tuple[int, int]], score_mode: str) -> np.ndarray:
    n_bins = len(t_slices)
    out = np.zeros(n_channels * n_bins, dtype=np.float64)
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
        for c in range(max(0, lc0), min(n_channels, lc1)):
            for bi, (t0, t1) in enumerate(t_slices):
                ov_t = max(0, min(lt1, t1) - max(lt0, t0))
                if ov_t <= 0:
                    continue
                out[_component_idx(c, bi, n_bins)] += mass * float(ov_t / area)
    return out


def _true_ranks(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    inv = np.empty_like(order, dtype=np.int64)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    return inv[np.arange(order.shape[0]), y_true] + 1


def _rank_metrics(ranks: np.ndarray) -> dict[str, float]:
    r = ranks.astype(np.float64)
    return {
        "loc@1": float(np.mean(r == 1)),
        "hit@3": float(np.mean(r <= 3)),
        "hit@5": float(np.mean(r <= 5)),
        "mrr": float(np.mean(1.0 / r)),
        "mean_rank": float(np.mean(r)),
    }


def _safe_norm(v: np.ndarray) -> np.ndarray:
    lo = float(np.min(v))
    hi = float(np.max(v))
    if hi <= lo + 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _corr_prefilter_scores(x: np.ndarray, t_slices: list[tuple[int, int]]) -> np.ndarray:
    t_len, n_channels = x.shape
    n_bins = len(t_slices)
    out = np.zeros(n_channels * n_bins, dtype=np.float64)
    for bi, (t0, t1) in enumerate(t_slices):
        block = x[t0:t1, :]
        if block.shape[0] < 2:
            for c in range(n_channels):
                out[_component_idx(c, bi, n_bins)] = 0.0
            continue
        for c in range(n_channels):
            vc = block[:, c]
            vals = []
            for cc in range(n_channels):
                if cc == c:
                    continue
                vo = block[:, cc]
                vals.append(_safe_corr(vc, vo))
            mean_abs_corr = float(np.mean(np.abs(vals))) if vals else 0.0
            out[_component_idx(c, bi, n_bins)] = 1.0 - mean_abs_corr
    return out


def _build_component_refs(x_train: np.ndarray, t_slices: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_channels = x_train.shape[2]
    n_bins = len(t_slices)
    ref_amp = np.zeros((n_channels, n_bins), dtype=np.float64)
    ref_energy = np.zeros((n_channels, n_bins), dtype=np.float64)
    ref_var = np.zeros((n_channels, n_bins), dtype=np.float64)
    ref_corr = np.zeros((n_bins, n_channels, n_channels), dtype=np.float64)
    n = float(max(1, x_train.shape[0]))

    for xi in x_train:
        for b, (t0, t1) in enumerate(t_slices):
            blk = xi[t0:t1, :]
            for c in range(n_channels):
                v = blk[:, c]
                ref_amp[c, b] += float(np.mean(np.abs(v)))
                ref_energy[c, b] += float(np.mean(v * v))
                ref_var[c, b] += float(np.var(v))
            for c0 in range(n_channels):
                for c1 in range(n_channels):
                    if c0 == c1:
                        ref_corr[b, c0, c1] += 1.0
                    else:
                        ref_corr[b, c0, c1] += _safe_corr(blk[:, c0], blk[:, c1])

    ref_amp /= n
    ref_energy /= n
    ref_var /= n
    ref_corr /= n
    return ref_amp, ref_energy, ref_var, ref_corr


def _inject_hidden_conflict(
    x: np.ndarray,
    donor: np.ndarray,
    c: int,
    t0: int,
    t1: int,
    alpha: float,
) -> np.ndarray:
    y = x.copy()
    src = y[t0:t1, c].copy()
    d = donor[t0:t1, c].copy()
    eps = 1e-6
    d = (d - np.mean(d)) / (np.std(d) + eps)
    d = d * (np.std(src) + eps) + np.mean(src)
    mix = (1.0 - alpha) * src + alpha * d
    mix = (mix - np.mean(mix)) / (np.std(mix) + eps)
    mix = mix * (np.std(src) + eps) + np.mean(src)
    y[t0:t1, c] = mix
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR hidden-conflict benchmark")
    p.add_argument("--npz-path", default="data/uci_har_shifted.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=512)
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--q", type=int, default=16)
    p.add_argument("--model", choices=["cnn1d", "extratrees", "histgbt"], default="cnn1d")
    p.add_argument("--neutralizer", choices=["zero", "mean", "interp"], default="interp")
    p.add_argument("--cnn-epochs", type=int, default=12)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--beacon-score-mode", choices=["neg_only", "abs_delta"], default="neg_only")
    p.add_argument("--hidden-margin-drop-min", type=float, default=0.05)
    p.add_argument("--hidden-true-effect-min", type=float, default=0.03)
    p.add_argument("--hidden-effect-cands", type=int, default=8)
    p.add_argument("--hidden-alpha-min", type=float, default=0.35)
    p.add_argument("--hidden-alpha-max", type=float, default=0.65)
    p.add_argument("--hidden-max-tries", type=int, default=20)
    p.add_argument("--hybrid-prefilter-weight", type=float, default=0.45)
    p.add_argument("--route-k0", type=int, default=8)
    p.add_argument("--route-expand-quantile", type=float, default=0.75)
    p.add_argument("--adaptive-phase1-ratio", type=float, default=0.4)
    p.add_argument("--adaptive-early-eff", type=float, default=0.12)
    p.add_argument("--out-summary", default="outputs_composite/har_hidden_conflict_localization_table.csv")
    p.add_argument("--out-per-sample", default="outputs_composite/har_hidden_conflict_localization_per_sample.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)
    if args.max_test > 0 and args.max_test < len(x_test):
        idx = rng.choice(len(x_test), size=args.max_test, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "extratrees":
        clf = _train_extratrees_local(x_train, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1)
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
            tta_shifts=(0,),
        )

    t_len, n_channels = x_test.shape[1], x_test.shape[2]
    t_slices = _time_slices(t_len, args.time_bins)
    n_bins = len(t_slices)
    n_components = n_channels * n_bins
    ref_amp, ref_energy, ref_var, ref_corr = _build_component_refs(x_train, t_slices)

    x_eval = x_test.copy()
    true_comp = -np.ones(len(x_eval), dtype=np.int64)
    accepted_drop = np.zeros(len(x_eval), dtype=np.float64)

    # Build class->indices for donor sampling
    class_to_ids: dict[int, np.ndarray] = {}
    for cl in np.unique(y_test):
        class_to_ids[int(cl)] = np.where(y_test == cl)[0]

    accepted = 0
    for i in range(len(x_eval)):
        lg0 = clf.logits(x_eval[i])
        _, m0 = _margin(lg0)
        yi = int(y_test[i])
        donor_pool = np.where(y_test != yi)[0]
        if len(donor_pool) == 0:
            continue
        ok = False
        for _ in range(args.hidden_max_tries):
            c = int(rng.integers(0, n_channels))
            b = int(rng.integers(0, n_bins))
            t0, t1 = t_slices[b]
            d_id = int(donor_pool[int(rng.integers(0, len(donor_pool)))])
            alpha = float(rng.uniform(args.hidden_alpha_min, args.hidden_alpha_max))
            xc = _inject_hidden_conflict(x_eval[i], x_test[d_id], c, t0, t1, alpha)
            _, m1 = _margin(clf.logits(xc))
            drop = float(m0 - m1)
            # model-relevant filter: true component neutralization should have strong local effect
            xm_true = _neutralize_component(xc, t0, t1, c, args.neutralizer)
            _, m_true = _margin(clf.logits(xm_true))
            true_eff = abs(float(m1 - m_true))
            cand_eff = []
            for _k in range(max(1, args.hidden_effect_cands)):
                cc = int(rng.integers(0, n_channels))
                bb = int(rng.integers(0, n_bins))
                tt0, tt1 = t_slices[bb]
                xm = _neutralize_component(xc, tt0, tt1, cc, args.neutralizer)
                _, mm = _margin(clf.logits(xm))
                cand_eff.append(abs(float(m1 - mm)))
            thr = float(np.quantile(np.asarray(cand_eff, dtype=np.float64), 0.75))
            if drop >= args.hidden_margin_drop_min and true_eff >= max(args.hidden_true_effect_min, thr):
                x_eval[i] = xc
                true_comp[i] = _component_idx(c, b, n_bins)
                accepted_drop[i] = drop
                accepted += 1
                ok = True
                break
        if not ok:
            # keep best-effort candidate with maximal drop among a few tries
            best_drop = -1e9
            best = None
            for _ in range(6):
                c = int(rng.integers(0, n_channels))
                b = int(rng.integers(0, n_bins))
                t0, t1 = t_slices[b]
                d_id = int(donor_pool[int(rng.integers(0, len(donor_pool)))])
                alpha = float(rng.uniform(args.hidden_alpha_min, args.hidden_alpha_max))
                xc = _inject_hidden_conflict(x_eval[i], x_test[d_id], c, t0, t1, alpha)
                _, m1 = _margin(clf.logits(xc))
                drop = float(m0 - m1)
                if drop > best_drop:
                    best_drop = drop
                    best = (xc, c, b, drop)
            if best is not None:
                x_eval[i] = best[0]
                true_comp[i] = _component_idx(best[1], best[2], n_bins)
                accepted_drop[i] = float(best[3])
                accepted += 1

    m = true_comp >= 0
    idx_eval = np.where(m)[0]
    y_true = true_comp[m]

    cfg = BeaconConfig(
        q_max=args.q,
        k0=8 if args.q >= 16 else 4,
        l_min=4,
        k_pos=3,
        k_neg=3,
        partition_mode="sensor_group_time",
        refinement_mode="mixed",
        margin_mode="adaptive_all",
        risk_policy="rho_only",
        audit_mode="full",
    )
    neutralizer = Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32))
    audit = BeaconAudit(model_logits=clf.logits, neutralizer=neutralizer, config=cfg)

    scores_beacon = np.zeros((len(idx_eval), n_components), dtype=np.float64)
    q_used_beacon = np.zeros(len(idx_eval), dtype=np.float64)
    scores_pref = np.zeros((len(idx_eval), n_components), dtype=np.float64)

    t_start = time.time()
    for j, i in enumerate(idx_eval):
        ar = audit.audit(x_eval[i])
        q_used_beacon[j] = float(ar.q_used)
        scores_beacon[j] = _counter_scores_by_component(ar, n_channels, t_slices, args.beacon_score_mode)
    lat_beacon = float((time.time() - t_start) / max(1, len(idx_eval)))

    t_start = time.time()
    for j, i in enumerate(idx_eval):
        raw = _corr_prefilter_scores(x_eval[i], t_slices)
        # template-disagreement bonus: how far channel correlation profile is from clean HAR baseline
        bonus = np.zeros_like(raw)
        xi = x_eval[i]
        for b, (t0, t1) in enumerate(t_slices):
            blk = xi[t0:t1, :]
            for c in range(n_channels):
                vals = []
                for cc in range(n_channels):
                    vals.append(abs(_safe_corr(blk[:, c], blk[:, cc]) - ref_corr[b, c, cc]))
                bonus[_component_idx(c, b, n_bins)] = float(np.mean(vals))
        scores_pref[j] = raw + bonus
    lat_pref = float((time.time() - t_start) / max(1, len(idx_eval)))

    scores_hybrid = np.zeros_like(scores_beacon)
    w = float(args.hybrid_prefilter_weight)
    for j in range(len(idx_eval)):
        sb = _safe_norm(scores_beacon[j])
        sp = _safe_norm(scores_pref[j])
        scores_hybrid[j] = (1.0 - w) * sb + w * sp

    scores_uniform = np.full((len(idx_eval), n_components), -1e18, dtype=np.float64)
    t_start = time.time()
    for j, i in enumerate(idx_eval):
        lg0 = clf.logits(x_eval[i])
        _, m0 = _margin(lg0)
        cand = rng.choice(n_components, size=min(args.q, n_components), replace=False)
        for comp in cand:
            c, b = _component_decode(int(comp), n_bins)
            tt0, tt1 = t_slices[b]
            xm = _neutralize_component(x_eval[i], tt0, tt1, c, args.neutralizer)
            _, m1 = _margin(clf.logits(xm))
            scores_uniform[j, comp] = abs(m0 - m1)
    lat_uniform = float((time.time() - t_start) / max(1, len(idx_eval)))

    # Prior-routed budgeted audit (v2): top-K0 routing + local expansion by directed margin gain
    scores_route = np.full((len(idx_eval), n_components), -1e18, dtype=np.float64)
    route_calls = np.zeros(len(idx_eval), dtype=np.float64)
    t_start = time.time()
    for j, i in enumerate(idx_eval):
        prior = scores_pref[j]
        lg0 = clf.logits(x_eval[i])
        _, m0 = _margin(lg0)
        order = np.argsort(-prior)
        k0 = min(max(1, int(args.route_k0)), n_components, int(args.q))
        queue = list(order[:k0].tolist())
        in_queue = set(queue)
        seen: set[int] = set()
        eff_vals: list[float] = []
        calls = 0

        while queue and calls < int(args.q):
            comp = int(queue.pop(0))
            in_queue.discard(comp)
            if comp in seen:
                continue
            seen.add(comp)

            c, b = _component_decode(comp, n_bins)
            tt0, tt1 = t_slices[b]
            xm = _neutralize_component(x_eval[i], tt0, tt1, c, args.neutralizer)
            _, m1 = _margin(clf.logits(xm))
            # robust route score: absolute margin response to component neutralization
            eff = float(abs(m1 - m0))
            scores_route[j, comp] = eff
            eff_vals.append(eff)
            calls += 1

            thr = float(np.quantile(np.asarray(eff_vals, dtype=np.float64), args.route_expand_quantile))
            if eff >= thr and calls < int(args.q):
                nbs = _neighbors(comp, n_channels, n_bins)
                nbs = sorted(nbs, key=lambda z: float(prior[z]), reverse=True)
                for nb in nbs:
                    if nb not in seen and nb not in in_queue:
                        queue.append(nb)
                        in_queue.add(nb)

        if calls < int(args.q):
            for comp in order:
                cc = int(comp)
                if cc in seen:
                    continue
                c, b = _component_decode(cc, n_bins)
                tt0, tt1 = t_slices[b]
                xm = _neutralize_component(x_eval[i], tt0, tt1, c, args.neutralizer)
                _, m1 = _margin(clf.logits(xm))
                scores_route[j, cc] = float(abs(m1 - m0))
                calls += 1
                if calls >= int(args.q):
                    break
        route_calls[j] = float(calls)
    lat_route = float((time.time() - t_start) / max(1, len(idx_eval)))

    # Adaptive two-phase audit: route early then coverage fallback
    scores_adapt = np.full((len(idx_eval), n_components), -1e18, dtype=np.float64)
    adapt_calls = np.zeros(len(idx_eval), dtype=np.float64)
    t_start = time.time()
    for j, i in enumerate(idx_eval):
        prior = scores_pref[j]
        lg0 = clf.logits(x_eval[i])
        _, m0 = _margin(lg0)
        order = np.argsort(-prior)
        q_total = int(args.q)
        q1 = min(max(1, int(round(args.adaptive_phase1_ratio * q_total))), q_total)
        seen: set[int] = set()
        calls = 0
        best_eff = -1.0

        # Phase 1: fast route on top-prior components
        for comp in order[:q1]:
            cc = int(comp)
            c, b = _component_decode(cc, n_bins)
            tt0, tt1 = t_slices[b]
            xm = _neutralize_component(x_eval[i], tt0, tt1, c, args.neutralizer)
            _, m1 = _margin(clf.logits(xm))
            eff = float(abs(m1 - m0))
            scores_adapt[j, cc] = eff
            seen.add(cc)
            calls += 1
            if eff > best_eff:
                best_eff = eff

        # Early-stop-like behavior for ranking: if very strong signal, prioritize local neighborhood
        if best_eff >= float(args.adaptive_early_eff) and calls < q_total:
            seed = int(np.argmax(scores_adapt[j]))
            nbs = _neighbors(seed, n_channels, n_bins)
            nbs = sorted(nbs, key=lambda z: float(prior[z]), reverse=True)
            for nb in nbs:
                if calls >= q_total:
                    break
                if nb in seen:
                    continue
                c, b = _component_decode(int(nb), n_bins)
                tt0, tt1 = t_slices[b]
                xm = _neutralize_component(x_eval[i], tt0, tt1, c, args.neutralizer)
                _, m1 = _margin(clf.logits(xm))
                scores_adapt[j, int(nb)] = float(abs(m1 - m0))
                seen.add(int(nb))
                calls += 1

        # Phase 2: coverage fallback (round-robin over channels with local best prior)
        if calls < q_total:
            per_ch = [[] for _ in range(n_channels)]
            for comp in order:
                cc = int(comp)
                if cc in seen:
                    continue
                c, _b = _component_decode(cc, n_bins)
                per_ch[c].append(cc)

            cursor = [0 for _ in range(n_channels)]
            while calls < q_total:
                progress = False
                for c in range(n_channels):
                    while cursor[c] < len(per_ch[c]) and per_ch[c][cursor[c]] in seen:
                        cursor[c] += 1
                    if cursor[c] >= len(per_ch[c]):
                        continue
                    cc = int(per_ch[c][cursor[c]])
                    cursor[c] += 1
                    if cc in seen:
                        continue
                    _c, b = _component_decode(cc, n_bins)
                    tt0, tt1 = t_slices[b]
                    xm = _neutralize_component(x_eval[i], tt0, tt1, _c, args.neutralizer)
                    _, m1 = _margin(clf.logits(xm))
                    scores_adapt[j, cc] = float(abs(m1 - m0))
                    seen.add(cc)
                    calls += 1
                    progress = True
                    if calls >= q_total:
                        break
                if not progress:
                    break
        adapt_calls[j] = float(calls)
    lat_adapt = float((time.time() - t_start) / max(1, len(idx_eval)))

    scores_amp = np.zeros((len(idx_eval), n_components), dtype=np.float64)
    scores_energy = np.zeros((len(idx_eval), n_components), dtype=np.float64)
    scores_var = np.zeros((len(idx_eval), n_components), dtype=np.float64)
    scores_rand = rng.random((len(idx_eval), n_components))
    for j, i in enumerate(idx_eval):
        xi = x_eval[i]
        for c in range(n_channels):
            for b, (tt0, tt1) in enumerate(t_slices):
                v = xi[tt0:tt1, c]
                cid = _component_idx(c, b, n_bins)
                # anomaly-style zero-query baselines against clean HAR references
                scores_amp[j, cid] = abs(float(np.mean(np.abs(v))) - float(ref_amp[c, b]))
                scores_energy[j, cid] = abs(float(np.mean(v * v)) - float(ref_energy[c, b]))
                scores_var[j, cid] = abs(float(np.var(v)) - float(ref_var[c, b]))

    methods = {
        "random": (scores_rand, 0.0, 0.0),
        "amplitude_heuristic": (scores_amp, 0.0, 0.0),
        "energy_heuristic": (scores_energy, 0.0, 0.0),
        "variance_heuristic": (scores_var, 0.0, 0.0),
        "corr_prefilter": (scores_pref, 0.0, lat_pref),
        "uniform_occlusion": (scores_uniform, float(args.q), lat_uniform),
        "beacon_xai": (scores_beacon, float(np.mean(q_used_beacon)), lat_beacon),
        "beacon_prefilter_hybrid": (scores_hybrid, float(np.mean(q_used_beacon)), lat_beacon + lat_pref),
        "beacon_prefilter_route": (scores_route, float(np.mean(route_calls)), lat_route),
        "beacon_adaptive": (scores_adapt, float(np.mean(adapt_calls)), lat_adapt),
    }

    rows = []
    per_rows = []
    for name, (sc, calls, lat) in methods.items():
        ranks = _true_ranks(sc, y_true)
        mm = _rank_metrics(ranks)
        rows.append({
            "dataset": "har",
            "scenario": "hidden_cross_channel_conflict",
            "model": args.model,
            "q_max": int(args.q),
            "time_bins": int(args.time_bins),
            "n_components": int(n_components),
            "method": name,
            "calls": float(calls),
            "loc@1": mm["loc@1"],
            "hit@3": mm["hit@3"],
            "hit@5": mm["hit@5"],
            "mrr": mm["mrr"],
            "mean_rank": mm["mean_rank"],
            "latency_per_object_sec": float(lat),
            "n_eval": int(len(idx_eval)),
            "mean_margin_drop_hidden": float(np.mean(accepted_drop[idx_eval])) if len(idx_eval) > 0 else 0.0,
            "accepted_ratio": float(len(idx_eval) / max(1, len(x_eval))),
        })
        pred = np.argmax(sc, axis=1)
        for jj, i in enumerate(idx_eval):
            per_rows.append({
                "sample_index_eval": int(i),
                "method": name,
                "true_component": int(y_true[jj]),
                "pred_component": int(pred[jj]),
                "rank_true": int(ranks[jj]),
                "is_correct": int(ranks[jj] == 1),
                "hit3": int(ranks[jj] <= 3),
                "hit5": int(ranks[jj] <= 5),
                "margin_drop_hidden": float(accepted_drop[i]),
            })

    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    out_per = Path(args.out_per_sample)
    with out_per.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        wr.writeheader()
        wr.writerows(per_rows)

    print(f"accepted={accepted}/{len(x_eval)}")
    print(f"saved: {out_summary}")
    print(f"saved: {out_per}")


if __name__ == "__main__":
    main()
