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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HAR latency profiling (CPU, batch=1)")
    p.add_argument("--npz-path", default="data/uci_har_shifted.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-profile", type=int, default=256)
    p.add_argument("--q-values", default="8,16")
    p.add_argument("--model", choices=["cnn1d", "extratrees", "histgbt"], default="cnn1d")
    p.add_argument("--neutralizer", choices=["zero", "mean", "interp"], default="interp")
    p.add_argument("--cnn-epochs", type=int, default=10)
    p.add_argument("--cnn-batch-size", type=int, default=256)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--out", default="outputs_composite/har_latency_profile_table.csv")
    return p.parse_args()


def _quantiles(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.quantile(arr, 0.50)), float(np.quantile(arr, 0.95))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    q_values = [int(v) for v in args.q_values.split(",") if v.strip()]

    x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)
    if args.n_profile > 0 and args.n_profile < len(x_test):
        idx = rng.choice(len(x_test), size=args.n_profile, replace=False)
        x_test = x_test[idx]

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

    inf_times = []
    for i in range(len(x_test)):
        t0 = time.perf_counter()
        _ = clf.logits(x_test[i])
        inf_times.append((time.perf_counter() - t0) * 1000.0)
    inf_p50, _ = _quantiles(inf_times)

    n_channels = x_test.shape[-1]
    rows = []

    for q in q_values:
        cfg = BeaconConfig(
            q_max=q,
            k0=4 if q <= 8 else 8,
            l_min=4,
            k_pos=3,
            k_neg=3,
            partition_mode="sensor_group_time",
            refinement_mode="mixed",
            margin_mode="adaptive_all",
            risk_policy="rho_only",
            audit_mode="full",
        )
        audit = BeaconAudit(
            model_logits=clf.logits,
            neutralizer=Neutralizer(mode=args.neutralizer, channel_means=np.zeros(n_channels, dtype=np.float32)),
            config=cfg,
        )

        audit_times = []
        for i in range(len(x_test)):
            t0 = time.perf_counter()
            _ = audit.audit(x_test[i])
            audit_times.append((time.perf_counter() - t0) * 1000.0)

        a50, a95 = _quantiles(audit_times)
        rows.append(
            {
                "dataset": "har",
                "model": args.model,
                "q_max": int(q),
                "inference_p50_ms": inf_p50,
                "audit_p50_ms": a50,
                "audit_p95_ms": a95,
                "overhead_x": (a50 / max(inf_p50, 1e-9)),
                "profiling_mode": "cpu_single_thread_batch1",
                "n_profile": int(len(x_test)),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
