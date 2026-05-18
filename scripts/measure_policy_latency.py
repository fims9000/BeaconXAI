#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beaconxai.fuzzy_policy_v2 import build_fuzzy_inputs_v2, fit_fuzzy_policy_v2, predict_fuzzy_policy_v2
from beaconxai.tan_policy import FEATURE_SETS, fit_tan_policy, predict_proba_tan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure policy-layer latency")
    p.add_argument("--beacon-features", default="outputs_composite/part2_extended_v2/audit_features_beacon_core.csv")
    p.add_argument("--split-manifest", default="outputs_composite/part2_extended_v2/split_manifest.json")
    p.add_argument("--tan-final", default="outputs_composite/part2_extended_v2/tan_final_test.csv")
    p.add_argument("--fuzzy-v3-final", default="outputs_composite/part2_extended_v2/fuzzy_v3_final_test.csv")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/policy_latency_profile.csv")
    return p.parse_args()


def _measure(fn, x: np.ndarray, n: int) -> tuple[float, float]:
    times = np.zeros(n, dtype=float)
    # Warmup
    for _ in range(200):
        fn(x)
    for i in range(n):
        t0 = time.perf_counter()
        fn(x)
        t1 = time.perf_counter()
        times[i] = (t1 - t0) * 1e6
    return float(np.mean(times)), float(np.quantile(times, 0.95))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.beacon_features).set_index("sample_id").sort_index()
    with Path(args.split_manifest).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    tr = np.asarray(manifest["train_ids"], dtype=np.int64)

    y = df["is_hidden_conflict"].to_numpy(dtype=np.int64)

    # Policy inputs
    X_panel = df[["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]].to_numpy(dtype=float)
    X_fuzzy = build_fuzzy_inputs_v2(df)

    # Fit logit
    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2500, solver="lbfgs", random_state=args.seed))
    logit.fit(X_panel[tr], y[tr])

    # Fit fuzzy (pick best reg from v3 table)
    fv3 = pd.read_csv(args.fuzzy_v3_final)
    best_fv3 = fv3.sort_values("mix_test_f1_10", ascending=False).iloc[0]
    reg = float(best_fv3["reg"])
    lam = float(best_fv3["mix_lambda"])
    fuzzy = fit_fuzzy_policy_v2(X_fuzzy[tr], y[tr], reg=reg, seed=args.seed + int(reg * 1e6))

    # Fit TAN (from tan_final)
    tan_final = pd.read_csv(args.tan_final).iloc[0]
    fs = str(tan_final["feature_set"])
    nb = int(tan_final["n_bins"])
    al = float(tan_final["alpha"])
    X_tan = df[FEATURE_SETS[fs]].to_numpy(dtype=float)
    tan = fit_tan_policy(X_tan[tr], y[tr], X_tan[tr], y[tr], n_bins=nb, alpha=al)

    # Single sample benchmarks
    i = int(rng.integers(0, len(df)))
    x_scalar = np.asarray([df.iloc[i]["m_neg"]], dtype=float)
    x_panel = X_panel[i : i + 1]
    x_f = X_fuzzy[i : i + 1]
    x_t = X_tan[i : i + 1]

    def scalar_fn(x):
        return float(x[0])

    def logit_fn(x):
        return float(logit.predict_proba(x)[0, 1])

    def fuzzy_fn(x):
        return float(predict_fuzzy_policy_v2(fuzzy, x)[0])

    def tan_fn(x):
        return float(predict_proba_tan(tan, x)[0])

    def soft_mix_fn(x_pack):
        x1, x2 = x_pack
        s_l = float(logit.predict_proba(x1)[0, 1])
        s_f = float(predict_fuzzy_policy_v2(fuzzy, x2)[0])
        return lam * s_l + (1.0 - lam) * s_f

    rows = []
    m, p95 = _measure(scalar_fn, x_scalar, args.iters)
    rows.append({"policy": "scalar", "mean_us": m, "p95_us": p95, "iters": args.iters})

    m, p95 = _measure(logit_fn, x_panel, args.iters)
    rows.append({"policy": "logit_panel", "mean_us": m, "p95_us": p95, "iters": args.iters})

    m, p95 = _measure(fuzzy_fn, x_f, args.iters)
    rows.append({"policy": "fuzzy_only", "mean_us": m, "p95_us": p95, "iters": args.iters})

    m, p95 = _measure(tan_fn, x_t, args.iters)
    rows.append({"policy": "tan_only", "mean_us": m, "p95_us": p95, "iters": args.iters})

    # custom measure for tuple input
    times = np.zeros(args.iters, dtype=float)
    for _ in range(200):
        soft_mix_fn((x_panel, x_f))
    for k in range(args.iters):
        t0 = time.perf_counter()
        soft_mix_fn((x_panel, x_f))
        t1 = time.perf_counter()
        times[k] = (t1 - t0) * 1e6
    rows.append(
        {
            "policy": "soft_mix_logit_fuzzy_v3",
            "mean_us": float(np.mean(times)),
            "p95_us": float(np.quantile(times, 0.95)),
            "iters": args.iters,
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()