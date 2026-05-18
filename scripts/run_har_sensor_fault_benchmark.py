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


def _margin(logits: np.ndarray) -> tuple[int, float]:
    y = int(np.argmax(logits))
    tmp = logits.copy()
    tmp[y] = -1e18
    return y, float(logits[y] - np.max(tmp))


def _neutralize_component(x: np.ndarray, t0: int, t1: int, c: int, mode: str) -> np.ndarray:
    y = x.copy()
    if mode in ("zero", "mean"):
        y[t0:t1, c : c + 1] = 0.0
        return y
    # interp
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
                w = float(ov_t / area)
                out[_component_idx(c, bi, n_bins)] += mass * w
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
    }


def _inject_fault(
    x: np.ndarray,
    c: int,
    t0: int,
    t1: int,
    fault_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    y = x.copy()
    seg = y[t0:t1, c]
    if fault_type == "channel_dropout":
        y[t0:t1, c] = 0.0
    elif fault_type == "dropout":
        y[t0:t1, c] = 0.0
    elif fault_type == "stuck_sensor":
        fill = float(y[t0 - 1, c]) if t0 > 0 else float(np.mean(y[:, c]))
        y[t0:t1, c] = fill
    elif fault_type == "spike":
        tt = int(rng.integers(t0, t1))
        s = float(np.std(y[:, c]) + 1e-6)
        y[tt, c] = y[tt, c] + float(rng.choice([-1.0, 1.0])) * 6.0 * s
    elif fault_type == "scale_drift":
        scale = float(rng.uniform(1.6, 2.6))
        y[t0:t1, c] = seg * scale
    elif fault_type == "drift":
        s = float(np.std(y[:, c]) + 1e-6)
        ramp = np.linspace(0.0, float(rng.choice([-1.0, 1.0])) * 2.0 * s, t1 - t0)
        y[t0:t1, c] = seg + ramp
    elif fault_type == "additive_noise":
        s = float(np.std(y[:, c]) + 1e-6)
        y[t0:t1, c] = seg + rng.normal(0.0, 0.8 * s, size=(t1 - t0))
    elif fault_type == "temporal_shift":
        shift = int(max(1, (t1 - t0) // 4))
        ch = y[:, c].copy()
        y[t0:t1, c] = np.roll(ch, shift)[t0:t1]
    else:
        raise ValueError(f"unknown fault_type={fault_type}")
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR sensor-fault component localization benchmark")
    p.add_argument("--npz-path", default="data/uci_har_shifted.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=512)
    p.add_argument("--fault-ratio", type=float, default=0.5)
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--q", type=int, default=16)
    p.add_argument("--model", choices=["cnn1d", "extratrees", "histgbt"], default="cnn1d")
    p.add_argument("--neutralizer", choices=["zero", "mean", "interp"], default="interp")
    p.add_argument("--cnn-epochs", type=int, default=12)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--beacon-score-mode", choices=["neg_only", "abs_delta"], default="neg_only")
    p.add_argument("--fault-types", default="spike,drift,stuck_sensor,dropout")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out-summary", default="outputs_composite/har_sensor_fault_localization_table.csv")
    p.add_argument("--out-per-sample", default="outputs_composite/har_sensor_fault_localization_per_sample.csv")
    p.add_argument("--out-bootstrap", default="outputs_composite/har_sensor_fault_bootstrap.csv")
    p.add_argument("--out-eval-npz", default="outputs_composite/har_sensor_fault_eval.npz")
    return p.parse_args()


def _component_reference_stats(x_train: np.ndarray, t_slices: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    n_channels = x_train.shape[2]
    n_components = n_channels * len(t_slices)
    means = np.zeros((n_components, 3), dtype=np.float64)
    stds = np.ones((n_components, 3), dtype=np.float64)
    vals_by_comp: list[list[list[float]]] = [[] for _ in range(n_components)]
    for xi in x_train:
        for c in range(n_channels):
            for b, (tt0, tt1) in enumerate(t_slices):
                v = xi[tt0:tt1, c]
                cid = _component_idx(c, b, len(t_slices))
                vals_by_comp[cid].append([float(np.mean(v)), float(np.var(v)), float(np.mean(v * v))])
    for cid, vals in enumerate(vals_by_comp):
        arr = np.asarray(vals, dtype=np.float64)
        means[cid] = np.mean(arr, axis=0)
        stds[cid] = np.std(arr, axis=0) + 1e-6
    return means, stds


def _bootstrap_method_delta(per_rows: list[dict], method_a: str, method_b: str, metric: str, n_boot: int, seed: int) -> dict:
    by_method: dict[str, dict[int, float]] = {}
    for r in per_rows:
        by_method.setdefault(str(r["method"]), {})[int(r["sample_index_eval"])] = float(r[metric])
    common = sorted(set(by_method.get(method_a, {})) & set(by_method.get(method_b, {})))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(common, size=len(common), replace=True)
        va = np.mean([by_method[method_a][int(i)] for i in idx])
        vb = np.mean([by_method[method_b][int(i)] for i in idx])
        vals.append(float(va - vb))
    arr = np.asarray(vals, dtype=np.float64)
    p = 2.0 * min(float(np.mean(arr <= 0.0)), float(np.mean(arr >= 0.0)))
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "delta": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_value": float(min(1.0, max(0.0, p))),
        "n": int(len(common)),
    }


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
        clf = _train_extratrees_local(
            x_train,
            y_train,
            n_estimators=300,
            max_features=0.7,
            min_samples_leaf=1,
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
            tta_shifts=(0,),
        )

    t_len, n_channels = x_test.shape[1], x_test.shape[2]
    t_slices = _time_slices(t_len, args.time_bins)
    n_components = n_channels * len(t_slices)

    n = len(x_test)
    n_fault = int(round(args.fault_ratio * n))
    fault_idx = rng.choice(n, size=n_fault, replace=False)
    x_eval = x_test.copy()
    true_comp = -np.ones(n, dtype=np.int64)
    fault_type_arr = np.array(["none"] * n, dtype=object)
    faults = [v.strip() for v in args.fault_types.split(",") if v.strip()]
    if not faults:
        raise ValueError("--fault-types must contain at least one fault")

    for i in fault_idx:
        c = int(rng.integers(0, n_channels))
        b = int(rng.integers(0, len(t_slices)))
        t0, t1 = t_slices[b]
        ft = faults[int(rng.integers(0, len(faults)))]
        x_eval[i] = _inject_fault(x_eval[i], c, t0, t1, ft, rng)
        true_comp[i] = _component_idx(c, b, len(t_slices))
        fault_type_arr[i] = ft

    m = true_comp >= 0
    y_true = true_comp[m]
    idx_fault = np.where(m)[0]

    # BEACON
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

    scores_beacon = np.zeros((len(idx_fault), n_components), dtype=np.float64)
    q_used_beacon = np.zeros(len(idx_fault), dtype=np.float64)

    t0 = time.time()
    for j, i in enumerate(idx_fault):
        ar = audit.audit(x_eval[i])
        q_used_beacon[j] = float(ar.q_used)
        scores_beacon[j] = _counter_scores_by_component(ar, n_channels, t_slices, args.beacon_score_mode)
    lat_beacon = float((time.time() - t0) / max(1, len(idx_fault)))

    # Uniform occlusion (equal budget)
    scores_uniform = np.full((len(idx_fault), n_components), -1e18, dtype=np.float64)
    t0 = time.time()
    for j, i in enumerate(idx_fault):
        lg0 = clf.logits(x_eval[i])
        _y, m0 = _margin(lg0)
        cand = rng.choice(n_components, size=min(args.q, n_components), replace=False)
        for comp in cand:
            c, b = _component_decode(int(comp), len(t_slices))
            tt0, tt1 = t_slices[b]
            xm = _neutralize_component(x_eval[i], tt0, tt1, c, args.neutralizer)
            _y1, m1 = _margin(clf.logits(xm))
            scores_uniform[j, comp] = abs(m0 - m1)
    lat_uniform = float((time.time() - t0) / max(1, len(idx_fault)))

    # Zero-query baselines
    ref_mean, ref_std = _component_reference_stats(x_train, t_slices)
    scores_amp = np.zeros((len(idx_fault), n_components), dtype=np.float64)
    scores_energy = np.zeros((len(idx_fault), n_components), dtype=np.float64)
    scores_var = np.zeros((len(idx_fault), n_components), dtype=np.float64)
    scores_profile = np.zeros((len(idx_fault), n_components), dtype=np.float64)
    scores_rand = rng.random((len(idx_fault), n_components))

    for j, i in enumerate(idx_fault):
        xi = x_eval[i]
        for c in range(n_channels):
            for b, (tt0, tt1) in enumerate(t_slices):
                v = xi[tt0:tt1, c]
                cid = _component_idx(c, b, len(t_slices))
                scores_amp[j, cid] = float(np.mean(np.abs(v)))
                scores_energy[j, cid] = float(np.mean(v * v))
                scores_var[j, cid] = float(np.var(v))
                desc = np.asarray([float(np.mean(v)), float(np.var(v)), float(np.mean(v * v))])
                ref = ref_mean[cid]
                sd = ref_std[cid]
                scores_profile[j, cid] = float(np.mean(np.abs((desc - ref) / sd)))

    methods = {
        "random": (scores_rand, 0.0, 0.0),
        "amplitude_heuristic": (scores_amp, 0.0, 0.0),
        "energy_heuristic": (scores_energy, 0.0, 0.0),
        "variance_heuristic": (scores_var, 0.0, 0.0),
        "profile_distance": (scores_profile, 0.0, 0.0),
        "uniform_occlusion": (scores_uniform, float(args.q), lat_uniform),
        "beacon_xai": (scores_beacon, float(np.mean(q_used_beacon)), lat_beacon),
    }

    rows = []
    per_rows = []
    for name, (sc, calls, lat) in methods.items():
        ranks = _true_ranks(sc, y_true)
        mm = _rank_metrics(ranks)
        rows.append(
            {
                "dataset": "har",
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
                "latency_per_object_sec": float(lat),
                "n_fault_eval": int(len(idx_fault)),
            }
        )
        pred = np.argmax(sc, axis=1)
        for jj, i in enumerate(idx_fault):
            per_rows.append(
                {
                    "sample_index_eval": int(i),
                    "fault_type": str(fault_type_arr[i]),
                    "method": name,
                    "true_component": int(y_true[jj]),
                    "pred_component": int(pred[jj]),
                    "rank_true": int(ranks[jj]),
                    "is_correct": int(ranks[jj] == 1),
                    "hit3": int(ranks[jj] <= 3),
                    "hit5": int(ranks[jj] <= 5),
                }
            )

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

    boot_rows = []
    for baseline in ["uniform_occlusion", "variance_heuristic", "energy_heuristic", "profile_distance"]:
        for metric in ["is_correct", "hit3", "hit5"]:
            boot_rows.append(
                _bootstrap_method_delta(
                    per_rows,
                    method_a="beacon_xai",
                    method_b=baseline,
                    metric=metric,
                    n_boot=args.n_boot,
                    seed=args.seed + abs(hash((baseline, metric))) % 100000,
                )
            )
    out_boot = Path(args.out_bootstrap)
    with out_boot.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(boot_rows[0].keys()))
        wr.writeheader()
        wr.writerows(boot_rows)

    np.savez_compressed(
        args.out_eval_npz,
        x_eval=x_eval,
        y_test=y_test,
        true_comp=true_comp,
        fault_mask=m.astype(np.int64),
        fault_type=fault_type_arr,
        time_bins=np.int64(args.time_bins),
    )

    print(f"saved: {out_summary}")
    print(f"saved: {out_per}")
    print(f"saved: {out_boot}")
    print(f"saved: {args.out_eval_npz}")


if __name__ == "__main__":
    main()
