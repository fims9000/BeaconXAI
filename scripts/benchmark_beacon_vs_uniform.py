#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "har": {"path": "data/uci_har_shifted.npz", "time_bins": 16, "model": "extratrees", "n_total": 600},
    "pamap2": {"path": "data/pamap2_acc9_w200s100_p095.npz", "time_bins": 12, "model": "extratrees", "n_total": 600},
    "wisdm": {"path": "data/wisdm_phone_accel_gyro_w200s100_p90_windowrand42.npz", "time_bins": 12, "model": "extratrees", "n_total": 600},
}

PANEL_COLS = [
    "m_neg",
    "M_B_minus",
    "r_B_minus",
    "CE_B",
    "rho_B_cost",
    "frag_drop",
    "top1_delta",
    "top3_sum_delta",
    "top3_conflict_count",
    "margin_entropy",
    "mean_conflict",
    "var_conflict_proxy",
    "frac_conflict_top3",
    "fragility_gap",
    "ce_density",
    "var_conflict",
    "conflict_connectivity",
    "delta_frag_proxy",
    "r_cf",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset BEACON vs uniform benchmark with logistic baseline")
    p.add_argument("--datasets", default="har,pamap2,wisdm")
    p.add_argument("--budgets", default="16,32,64")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-root", default="outputs_composite/v12_beacon_vs_uniform")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-total", type=int, default=-1, help="Override n_total for all datasets")
    p.add_argument("--adaptive-v2", action="store_true", help="Also evaluate adaptive_v2_preselect")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _f1_at_frac(y: np.ndarray, s: np.ndarray, frac: float) -> float:
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-s)[:k]
    pred = np.zeros(n, dtype=np.int64)
    pred[idx] = 1
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    if tp == 0:
        return 0.0
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return float(2 * p * r / max(p + r, 1e-12))


def _bootstrap_delta(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    fn,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(fn(y[idx], a[idx]) - fn(y[idx], b[idx]))
    d = float(np.mean(deltas))
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    p = float(min(1.0, 2.0 * min(np.mean(deltas <= 0.0), np.mean(deltas >= 0.0))))
    return d, lo, hi, p


def _load_split(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = json.loads((bundle_dir / "split_manifest.json").read_text(encoding="utf-8"))
    tr = np.asarray(m["train_ids"], dtype=np.int64)
    va = np.asarray(m["val_ids"], dtype=np.int64)
    te = np.asarray(m["test_ids"], dtype=np.int64)
    return tr, va, te


def _fit_score(df: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c in PANEL_COLS if c in df.columns]
    X = df.loc[:, cols].to_numpy(dtype=float)
    y = df["is_hidden_conflict"].to_numpy(dtype=np.int64)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42))
    pipe.fit(X[tr], y[tr])
    s = pipe.predict_proba(X[te])[:, 1]
    return y[te], s


def _run_part2(
    dataset_path: str,
    model: str,
    n_total: int,
    q: int,
    time_bins: int,
    seed: int,
    out_dir: Path,
    adaptive_v2: bool,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_part2_extended.py",
        "--dataset",
        dataset_path,
        "--model",
        model,
        "--n-total",
        str(n_total),
        "--q-max",
        str(q),
        "--time-bins",
        str(time_bins),
        "--neutralizer-mode",
        "interp",
        "--seed",
        str(seed),
        "--features-only",
        "--out",
        str(out_dir),
    ]
    if adaptive_v2:
        cmd.append("--adaptive-v2-preselect")
    print("$", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    budgets = [int(v.strip()) for v in args.budgets.split(",") if v.strip()]

    rows = []
    boot_rows = []

    for dname in datasets:
        if dname not in DATASETS:
            raise SystemExit(f"unknown dataset: {dname}")
        meta = DATASETS[dname]
        n_total = int(args.n_total if args.n_total > 0 else meta["n_total"])

        for q in budgets:
            bundle = f"{dname}_tb{meta['time_bins']}_q{q}_n{n_total}"
            bundle_dir = out_root / bundle
            _run_part2(
                dataset_path=meta["path"],
                model=meta["model"],
                n_total=n_total,
                q=q,
                time_bins=int(meta["time_bins"]),
                seed=args.seed,
                out_dir=bundle_dir,
                adaptive_v2=args.adaptive_v2,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                continue

            df_b = pd.read_csv(bundle_dir / "audit_features_beacon_core.csv").sort_values("sample_id")
            df_u = pd.read_csv(bundle_dir / "audit_features_uniform.csv").sort_values("sample_id")
            tr, _va, te = _load_split(bundle_dir)
            y_te_b, s_b = _fit_score(df_b, tr, te)
            y_te_u, s_u = _fit_score(df_u, tr, te)
            if not np.array_equal(y_te_b, y_te_u):
                raise RuntimeError("label mismatch between beacon and uniform test splits")
            y_te = y_te_b

            row = {
                "dataset": dname,
                "bundle": bundle,
                "q": q,
                "auroc_beacon": float(roc_auc_score(y_te, s_b)),
                "auprc_beacon": float(average_precision_score(y_te, s_b)),
                "f1_10_beacon": float(_f1_at_frac(y_te, s_b, 0.10)),
                "f1_20_beacon": float(_f1_at_frac(y_te, s_b, 0.20)),
                "auroc_uniform": float(roc_auc_score(y_te, s_u)),
                "auprc_uniform": float(average_precision_score(y_te, s_u)),
                "f1_10_uniform": float(_f1_at_frac(y_te, s_u, 0.10)),
                "f1_20_uniform": float(_f1_at_frac(y_te, s_u, 0.20)),
            }
            row["delta_auroc"] = row["auroc_beacon"] - row["auroc_uniform"]
            row["delta_auprc"] = row["auprc_beacon"] - row["auprc_uniform"]
            row["delta_f1_10"] = row["f1_10_beacon"] - row["f1_10_uniform"]
            row["delta_f1_20"] = row["f1_20_beacon"] - row["f1_20_uniform"]
            rows.append(row)

            metrics = [
                ("delta_auroc", lambda y, s: roc_auc_score(y, s)),
                ("delta_auprc", lambda y, s: average_precision_score(y, s)),
                ("delta_f1_10", lambda y, s: _f1_at_frac(y, s, 0.10)),
                ("delta_f1_20", lambda y, s: _f1_at_frac(y, s, 0.20)),
            ]
            for mi, (mname, fn) in enumerate(metrics):
                d, lo, hi, p = _bootstrap_delta(y_te, s_b, s_u, fn, n_boot=args.n_boot, seed=args.seed + q * 100 + mi)
                boot_rows.append(
                    {
                        "dataset": dname,
                        "bundle": bundle,
                        "q": q,
                        "comparison": "beacon_logit_vs_uniform_logit",
                        "metric": mname,
                        "delta": d,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_value": p,
                    }
                )

            if args.adaptive_v2 and (bundle_dir / "audit_features_adaptive_v2.csv").exists():
                df_a = pd.read_csv(bundle_dir / "audit_features_adaptive_v2.csv").sort_values("sample_id")
                y_te_a, s_a = _fit_score(df_a, tr, te)
                if not np.array_equal(y_te, y_te_a):
                    raise RuntimeError("label mismatch between adaptive_v2 and uniform")
                for mi, (mname, fn) in enumerate(metrics):
                    d, lo, hi, p = _bootstrap_delta(y_te, s_a, s_u, fn, n_boot=args.n_boot, seed=args.seed + q * 200 + mi)
                    boot_rows.append(
                        {
                            "dataset": dname,
                            "bundle": bundle,
                            "q": q,
                            "comparison": "adaptive_v2_logit_vs_uniform_logit",
                            "metric": mname,
                            "delta": d,
                            "ci_low": lo,
                            "ci_high": hi,
                            "p_value": p,
                        }
                    )

    if not args.dry_run:
        df_rows = pd.DataFrame(rows).sort_values(["dataset", "q"]).reset_index(drop=True)
        df_boot = pd.DataFrame(boot_rows).sort_values(["dataset", "q", "comparison", "metric"]).reset_index(drop=True)
        df_rows.to_csv(out_root / "beacon_vs_uniform_logit_metrics.csv", index=False)
        df_boot.to_csv(out_root / "beacon_vs_uniform_logit_bootstrap.csv", index=False)
        print(f"saved: {out_root / 'beacon_vs_uniform_logit_metrics.csv'}")
        print(f"saved: {out_root / 'beacon_vs_uniform_logit_bootstrap.csv'}")


if __name__ == "__main__":
    main()
