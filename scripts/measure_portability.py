#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
import time
from pathlib import Path

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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _corr_prefilter_scores(x: np.ndarray, t_slices: list[tuple[int, int]]) -> np.ndarray:
    n_channels = x.shape[1]
    n_bins = len(t_slices)
    out = np.zeros(n_channels * n_bins, dtype=np.float64)
    for bi, (t0, t1) in enumerate(t_slices):
        block = x[t0:t1, :]
        if block.shape[0] < 2:
            continue
        for c in range(n_channels):
            vc = block[:, c]
            vals = [_safe_corr(vc, block[:, cc]) for cc in range(n_channels) if cc != c]
            mean_abs_corr = float(np.mean(np.abs(vals))) if vals else 0.0
            out[_component_idx(c, bi, n_bins)] = 1.0 - mean_abs_corr
    return out


def _build_ref_corr(x_train: np.ndarray, t_slices: list[tuple[int, int]]) -> np.ndarray:
    n_channels = x_train.shape[2]
    n_bins = len(t_slices)
    ref_corr = np.zeros((n_bins, n_channels, n_channels), dtype=np.float64)
    n = float(max(1, x_train.shape[0]))
    for xi in x_train:
        for b, (t0, t1) in enumerate(t_slices):
            blk = xi[t0:t1, :]
            for c0 in range(n_channels):
                for c1 in range(n_channels):
                    if c0 == c1:
                        ref_corr[b, c0, c1] += 1.0
                    else:
                        ref_corr[b, c0, c1] += _safe_corr(blk[:, c0], blk[:, c1])
    ref_corr /= n
    return ref_corr


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


def _margin(logits: np.ndarray) -> tuple[int, float]:
    y = int(np.argmax(logits))
    tmp = logits.copy()
    tmp[y] = -1e18
    return y, float(logits[y] - np.max(tmp))


def _rss_mb() -> float:
    try:
        with open('/proc/self/statm', 'r', encoding='utf-8') as f:
            parts = f.read().strip().split()
        rss_pages = int(parts[1])
        return float(rss_pages * os.sysconf('SC_PAGE_SIZE') / (1024 * 1024))
    except Exception:
        return float('nan')


def _qstats(v: list[float]) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=np.float64)
    return float(np.mean(a)), float(np.quantile(a, 0.50)), float(np.quantile(a, 0.95))


def _cpu_model() -> str:
    try:
        for line in Path('/proc/cpuinfo').read_text(encoding='utf-8', errors='ignore').splitlines():
            if 'model name' in line:
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return 'unknown'


def _cpu_mhz() -> float:
    try:
        for line in Path('/proc/cpuinfo').read_text(encoding='utf-8', errors='ignore').splitlines():
            if 'cpu MHz' in line:
                return float(line.split(':', 1)[1].strip())
    except Exception:
        pass
    return float('nan')


def _set_constraints(core: int, nice_level: int) -> tuple[int, int]:
    affinity_ok = 0
    nice_ok = 0
    try:
        os.sched_setaffinity(0, {core})
        affinity_ok = 1
    except Exception:
        pass
    try:
        os.nice(nice_level)
        nice_ok = 1
    except Exception:
        pass
    return affinity_ok, nice_ok


def _adaptive_calls(
    x: np.ndarray,
    logits_fn,
    q: int,
    neutralizer_mode: str,
    t_slices: list[tuple[int, int]],
    ref_corr: np.ndarray,
    phase1_ratio: float,
    early_eff: float,
) -> int:
    n_channels = x.shape[1]
    n_bins = len(t_slices)

    prior_raw = _corr_prefilter_scores(x, t_slices)
    bonus = np.zeros_like(prior_raw)
    for b, (t0, t1) in enumerate(t_slices):
        blk = x[t0:t1, :]
        for c in range(n_channels):
            vals = [abs(_safe_corr(blk[:, c], blk[:, cc]) - ref_corr[b, c, cc]) for cc in range(n_channels)]
            bonus[_component_idx(c, b, n_bins)] = float(np.mean(vals))
    prior = prior_raw + bonus

    lg0 = logits_fn(x)
    _, m0 = _margin(lg0)
    calls = 1  # base inference

    order = np.argsort(-prior)
    q1 = min(max(1, int(round(phase1_ratio * q))), q)
    seen: set[int] = set()
    best_eff = -1.0

    for comp in order[:q1]:
        cc = int(comp)
        c, b = _component_decode(cc, n_bins)
        tt0, tt1 = t_slices[b]
        xm = _neutralize_component(x, tt0, tt1, c, neutralizer_mode)
        _, m1 = _margin(logits_fn(xm))
        eff = float(abs(m1 - m0))
        if eff > best_eff:
            best_eff = eff
        seen.add(cc)
        calls += 1

    if best_eff >= early_eff and calls - 1 < q:
        seed = int(order[0])
        nbs = _neighbors(seed, n_channels, n_bins)
        nbs = sorted(nbs, key=lambda z: float(prior[z]), reverse=True)
        for nb in nbs:
            if calls - 1 >= q:
                break
            if nb in seen:
                continue
            c, b = _component_decode(nb, n_bins)
            tt0, tt1 = t_slices[b]
            xm = _neutralize_component(x, tt0, tt1, c, neutralizer_mode)
            _ = logits_fn(xm)
            seen.add(nb)
            calls += 1

    if calls - 1 < q:
        per_ch = [[] for _ in range(n_channels)]
        for comp in order:
            cc = int(comp)
            if cc in seen:
                continue
            c, _b = _component_decode(cc, n_bins)
            per_ch[c].append(cc)
        cursor = [0 for _ in range(n_channels)]
        while calls - 1 < q:
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
                xm = _neutralize_component(x, tt0, tt1, _c, neutralizer_mode)
                _ = logits_fn(xm)
                seen.add(cc)
                calls += 1
                progress = True
                if calls - 1 >= q:
                    break
            if not progress:
                break
    return calls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Portability profiling under constrained CPU resources')
    p.add_argument('--npz-path', default='data/uci_har_shifted.npz')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-profile', type=int, default=200)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--q-values', default='8,16')
    p.add_argument('--model', choices=['cnn1d', 'extratrees', 'histgbt'], default='cnn1d')
    p.add_argument('--neutralizer', choices=['zero', 'mean', 'interp'], default='interp')
    p.add_argument('--cnn-epochs', type=int, default=10)
    p.add_argument('--cnn-batch-size', type=int, default=256)
    p.add_argument('--cnn-lr', type=float, default=1e-3)
    p.add_argument('--core-id', type=int, default=0)
    p.add_argument('--nice-level', type=int, default=10)
    p.add_argument('--adaptive-phase1-ratio', type=float, default=0.25)
    p.add_argument('--adaptive-early-eff', type=float, default=0.12)
    p.add_argument('--out', default='outputs_composite/edge_portability_profile.csv')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    q_values = [int(x.strip()) for x in args.q_values.split(',') if x.strip()]

    affinity_ok, nice_ok = _set_constraints(args.core_id, args.nice_level)

    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch_ok = 1
    except Exception:
        torch_ok = 0

    x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)
    if args.n_profile > 0 and args.n_profile < len(x_test):
        idx = rng.choice(len(x_test), size=args.n_profile, replace=False)
        x_test = x_test[idx]

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == 'extratrees':
        clf = _train_extratrees_local(x_train, y_train, n_estimators=300, max_features=0.7, min_samples_leaf=1)
    elif args.model == 'histgbt':
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

    t_slices = _time_slices(x_test.shape[1], 8)
    ref_corr = _build_ref_corr(x_train, t_slices)
    n_channels = x_test.shape[-1]

    rows: list[dict[str, float | int | str]] = []

    def run_profile(name: str, fn_call, q_max: int, call_counter_fn=None):
        wall_ms = []
        cpu_ms = []
        calls = []
        rss_peak = _rss_mb()
        rss_before = _rss_mb()

        n_warm = min(args.warmup, len(x_test))
        for i in range(n_warm):
            _ = fn_call(x_test[i])

        for i in range(n_warm, len(x_test)):
            t0 = time.perf_counter()
            c0 = time.process_time()
            res = fn_call(x_test[i])
            c1 = time.process_time()
            t1 = time.perf_counter()
            wall_ms.append((t1 - t0) * 1000.0)
            cpu_ms.append((c1 - c0) * 1000.0)
            if call_counter_fn is not None:
                calls.append(float(call_counter_fn(res)))
            rss_now = _rss_mb()
            if np.isfinite(rss_now) and (not np.isfinite(rss_peak) or rss_now > rss_peak):
                rss_peak = rss_now

        w_mean, w_p50, w_p95 = _qstats(wall_ms)
        c_mean, c_p50, c_p95 = _qstats(cpu_ms)
        rows.append({
            'dataset': 'har',
            'model': args.model,
            'method': name,
            'q_max': int(q_max),
            'n_profile': int(len(x_test) - n_warm),
            'warmup': int(n_warm),
            'latency_mean_ms': w_mean,
            'latency_p50_ms': w_p50,
            'latency_p95_ms': w_p95,
            'cpu_mean_ms': c_mean,
            'cpu_p50_ms': c_p50,
            'cpu_p95_ms': c_p95,
            'rss_before_mb': float(rss_before),
            'rss_peak_mb': float(rss_peak),
            'rss_delta_mb': float(rss_peak - rss_before) if np.isfinite(rss_peak) and np.isfinite(rss_before) else float('nan'),
            'mean_model_calls': float(np.mean(calls)) if calls else 1.0,
            'affinity_core': int(args.core_id),
            'affinity_applied': int(affinity_ok),
            'nice_level': int(args.nice_level),
            'nice_applied': int(nice_ok),
            'torch_single_thread': int(torch_ok),
            'cpu_model': _cpu_model(),
            'cpu_mhz_snapshot': float(_cpu_mhz()),
            'energy_proxy_cpu_ms': c_mean,
        })

    run_profile('inference_only', lambda x: clf.logits(x), 0)

    for q in q_values:
        cfg = BeaconConfig(
            q_max=q,
            k0=4 if q <= 8 else 8,
            l_min=4,
            k_pos=3,
            k_neg=3,
            partition_mode='sensor_group_time',
            refinement_mode='mixed',
            margin_mode='adaptive_all',
            risk_policy='rho_only',
            audit_mode='full',
        )
        audit = BeaconAudit(
            model_logits=clf.logits,
            neutralizer=Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32)),
            config=cfg,
        )
        run_profile(
            f'beacon_core_q{q}',
            lambda x, _a=audit: _a.audit(x),
            q,
            call_counter_fn=lambda r: float(getattr(r, 'q_used', q)) + 1.0,
        )
        cfg_fast = BeaconConfig(
            q_max=q,
            k0=4 if q <= 8 else 8,
            l_min=4,
            k_pos=3,
            k_neg=3,
            partition_mode='sensor_group_time',
            refinement_mode='mixed',
            margin_mode='adaptive_all',
            risk_policy='rho_only',
            audit_mode='full',
            fast_core=True,
        )
        audit_fast = BeaconAudit(
            model_logits=clf.logits,
            neutralizer=Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32)),
            config=cfg_fast,
        )
        run_profile(
            f'beacon_core_fast_q{q}',
            lambda x, _a=audit_fast: _a.audit(x),
            q,
            call_counter_fn=lambda r: float(getattr(r, 'q_used', q)) + 1.0,
        )

        run_profile(
            f'beacon_adaptive_q{q}',
            lambda x, _q=q: _adaptive_calls(
                x,
                clf.logits,
                _q,
                args.neutralizer,
                t_slices,
                ref_corr,
                phase1_ratio=args.adaptive_phase1_ratio,
                early_eff=args.adaptive_early_eff,
            ),
            q,
            call_counter_fn=lambda c: float(c),
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f'saved: {out}')


if __name__ == '__main__':
    main()
