#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beaconxai.datasets import load_npz_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-dataset BEACON policy benchmark runner")
    p.add_argument("--config", default="configs/experiments_v11_cross_dataset.yaml")
    p.add_argument("--config-json", default="", help="JSON fallback when PyYAML unavailable")
    p.add_argument("--out-root", default="outputs_composite/v11_cross_dataset")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_cfg(path: str, json_fallback: str) -> dict[str, Any]:
    if json_fallback:
        return json.loads(json_fallback)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return dict(json.load(f))
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return dict(yaml.safe_load(f))
    except Exception as e:
        raise RuntimeError(
            f"PyYAML is required for --config YAML in this environment ({e}). "
            "Use --config-json as fallback."
        )


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _with_flags(base: list[str], params: dict[str, Any]) -> list[str]:
    out = list(base)
    for k, v in params.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
        else:
            out.extend([flag, str(v)])
    return out


def _bonferroni_claims(boot_csv: Path, model_name: str, alpha_family: float = 0.05):
    import pandas as pd

    df = pd.read_csv(boot_csv)
    df = df.copy()
    m = max(1, len(df))
    alpha_adj = alpha_family / m
    df["model"] = model_name
    df["alpha_adj"] = float(alpha_adj)
    df["supported_positive_bonf"] = (
        (df["delta"] > 0.0) & (df["ci_low"] > 0.0) & (df["p_value"] < alpha_adj)
    ).astype(int)
    return df


def _infer_n_channels(dataset_path: str) -> int:
    x_train, _y_train, _x_test, _y_test = load_npz_dataset(dataset_path)
    return int(x_train.shape[2])


def _q_from_rule(m_components: int, q_rule: dict[str, Any]) -> int:
    base_q = int(q_rule.get("base_q", 64))
    coverage = float(q_rule.get("coverage", 0.8))
    min_q = int(q_rule.get("min_q", 1))
    q = int(min(base_q, math.ceil(coverage * max(m_components, 1))))
    return max(min_q, q)


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config, args.config_json)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    n_boot = int(cfg.get("n_boot", 5000))
    alpha_family = 0.05
    q_rule = dict(cfg.get("q_rule", {"base_q": 64, "coverage": 0.8, "min_q": 1}))

    all_claims: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []

    for ds in cfg.get("datasets", []):
        ds_name = str(ds["name"])
        ds_path = str(ds["path"])
        model = str(ds.get("model", "extratrees"))
        neutralizers = list(ds.get("neutralizers", ["interp"]))
        time_bins_list = list(ds.get("time_bins", [8]))
        n_total = int(ds.get("n_total", 600))
        n_channels = _infer_n_channels(ds_path)

        for tb in time_bins_list:
            tb_i = int(tb)
            m_components = int(tb_i * n_channels)
            q_max = _q_from_rule(m_components, q_rule)
            q_ratio = float(q_max / max(m_components, 1))

            for nz in neutralizers:
                bundle_dir = out_root / f"{ds_name}_tb{tb_i}_q{q_max}_{nz}_n{n_total}"
                bundle_dir.mkdir(parents=True, exist_ok=True)

                manifest_rows.append(
                    {
                        "dataset": ds_name,
                        "path": ds_path,
                        "model": model,
                        "neutralizer": nz,
                        "time_bins": tb_i,
                        "n_channels": n_channels,
                        "n_components": m_components,
                        "q_max": q_max,
                        "q_over_m": q_ratio,
                        "n_total": n_total,
                        "bundle_dir": str(bundle_dir),
                    }
                )

                part2_cmd = [
                    sys.executable,
                    "scripts/run_part2_extended.py",
                    "--dataset",
                    ds_path,
                    "--model",
                    model,
                    "--n-total",
                    str(n_total),
                    "--q-max",
                    str(q_max),
                    "--time-bins",
                    str(tb_i),
                    "--neutralizer-mode",
                    str(nz),
                    "--seed",
                    str(seed),
                    "--out",
                    str(bundle_dir),
                ]
                _run(part2_cmd, args.dry_run)

                if bool(cfg.get("policies", {}).get("run_tan", True)):
                    tan_cfg = dict(cfg.get("policies", {}).get("tan", {}))
                    tan_base = [
                        sys.executable,
                        "scripts/train_tan_improved.py",
                        "--bundle-dir",
                        str(bundle_dir),
                        "--n-boot",
                        str(n_boot),
                        "--seed",
                        str(seed),
                    ]
                    _run(_with_flags(tan_base, tan_cfg), args.dry_run)
                    if not args.dry_run:
                        boot = bundle_dir / "tan_improved_bootstrap.csv"
                        if boot.exists():
                            all_claims.append(_bonferroni_claims(boot, "tan", alpha_family))

                if bool(cfg.get("policies", {}).get("run_fuzzy", True)):
                    fz_cfg = dict(cfg.get("policies", {}).get("fuzzy", {}))
                    fz_base = [
                        sys.executable,
                        "scripts/train_fuzzy_improved.py",
                        "--bundle-dir",
                        str(bundle_dir),
                        "--n-boot",
                        str(n_boot),
                        "--seed",
                        str(seed),
                    ]
                    _run(_with_flags(fz_base, fz_cfg), args.dry_run)
                    if not args.dry_run:
                        boot = bundle_dir / "fuzzy_improved_bootstrap.csv"
                        if boot.exists():
                            all_claims.append(_bonferroni_claims(boot, "fuzzy", alpha_family))

    manifest_path = out_root / "run_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "path",
                "model",
                "neutralizer",
                "time_bins",
                "n_channels",
                "n_components",
                "q_max",
                "q_over_m",
                "n_total",
                "bundle_dir",
            ],
        )
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)
    print(f"saved: {manifest_path}")

    if not args.dry_run and all_claims:
        import pandas as pd

        claims = pd.concat(all_claims, ignore_index=True)
        out_path = out_root / "policy_claim_registry_bonferroni.csv"
        claims.to_csv(out_path, index=False)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
