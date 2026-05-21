#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified v10 experiment runner")
    p.add_argument("--config", default="configs/experiments_v10.yaml")
    p.add_argument("--config-json", default="", help="JSON string fallback if PyYAML unavailable")
    p.add_argument("--out-root", default="outputs_composite/v10_unified")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _load_cfg(path: str, json_fallback: str) -> dict[str, Any]:
    if json_fallback:
        return json.loads(json_fallback)
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


def _bonferroni_claims(boot_csv: Path, model_name: str, alpha_family: float = 0.05) -> pd.DataFrame:
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


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.config, args.config_json)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    n_boot = int(cfg.get("n_boot", 5000))
    alpha_family = 0.05

    all_claims = []
    for ds in cfg.get("datasets", []):
        ds_name = str(ds["name"])
        ds_path = str(ds["path"])
        model = str(ds.get("model", "extratrees"))
        neutralizers = list(ds.get("neutralizers", ["interp"]))
        budgets = list(ds.get("budgets", []))

        for nz in neutralizers:
            for b in budgets:
                q_max = int(b["q_max"])
                tb = int(b["time_bins"])
                n_total = int(b.get("n_total", 300))
                bundle_dir = out_root / f"{ds_name}_tb{tb}_q{q_max}_{nz}_n{n_total}"
                bundle_dir.mkdir(parents=True, exist_ok=True)

                part2_cmd = [
                    "python",
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
                    str(tb),
                    "--neutralizer-mode",
                    nz,
                    "--seed",
                    str(seed),
                    "--out",
                    str(bundle_dir),
                ]
                _run(part2_cmd, args.dry_run)

                if bool(cfg.get("policies", {}).get("run_tan", True)):
                    tan_cfg = dict(cfg.get("policies", {}).get("tan", {}))
                    tan_base = [
                        "python",
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
                        "python",
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

    if not args.dry_run and all_claims:
        claims = pd.concat(all_claims, ignore_index=True)
        claims.to_csv(out_root / "policy_claim_registry_bonferroni.csv", index=False)
        print(f"saved: {out_root / 'policy_claim_registry_bonferroni.csv'}")


if __name__ == "__main__":
    main()
