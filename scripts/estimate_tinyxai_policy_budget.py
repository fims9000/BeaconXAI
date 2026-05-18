#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate compact policy resource profile")
    p.add_argument("--fuzzy-results", default="outputs_composite/part2_extended_v2/fuzzy_policy_results.csv")
    p.add_argument("--tan-final", default="outputs_composite/part2_extended_v2/tan_final_test.csv")
    p.add_argument("--out", default="outputs_composite/part2_extended_v2/policy_resource_profile.csv")
    return p.parse_args()


def _row(policy: str, n_features: int, n_params: int, ops: dict, requires_training: bool, interpretability: str):
    state_f32 = int(n_params * 4)
    state_i16 = int(n_params * 2)
    return {
        "policy": policy,
        "n_features": n_features,
        "n_rules": ops.get("n_rules", 0),
        "n_params": n_params,
        "state_bytes_float32": state_f32,
        "state_bytes_int16": state_i16,
        "n_comparisons": ops.get("cmp", 0),
        "n_additions": ops.get("add", 0),
        "n_multiplications": ops.get("mul", 0),
        "n_table_lookups": ops.get("lookup", 0),
        "branch_count": ops.get("branch", 0),
        "requires_training": int(requires_training),
        "interpretability": interpretability,
    }


def main() -> None:
    args = parse_args()
    fuzzy = pd.read_csv(args.fuzzy_results).iloc[0]
    tan = pd.read_csv(args.tan_final).iloc[0]

    rows = []
    rows.append(_row("scalar", 1, 1, {"cmp": 1, "add": 0, "mul": 0, "lookup": 0, "branch": 1}, False, "high"))
    rows.append(_row("logit_panel", 6, 7, {"cmp": 1, "add": 6, "mul": 6, "lookup": 0, "branch": 1}, True, "medium"))

    n_rules = int(fuzzy.get("n_rules", 27))
    # weights + rule outputs + membership params (~5*3 inputs)
    fuzzy_params = n_rules + n_rules + 15
    rows.append(_row("fuzzy_only", 3, fuzzy_params, {"n_rules": n_rules, "cmp": n_rules * 2, "add": n_rules * 3, "mul": n_rules * 4, "lookup": n_rules, "branch": n_rules}, True, "high"))

    # soft mix = logit + fuzzy + one lambda
    rows.append(_row("soft_mix", 9, 7 + fuzzy_params + 1, {"n_rules": n_rules, "cmp": n_rules * 2 + 1, "add": n_rules * 3 + 8, "mul": n_rules * 4 + 8, "lookup": n_rules, "branch": n_rules + 1}, True, "medium"))

    n_bins = int(tan.get("n_bins", 4))
    n_feat = 3 if str(tan.get("feature_set", "")).startswith("conflict_min") else 7
    # rough TAN table size: class priors + root + conditionals
    tan_params = 2 + (n_bins * 2) + max(0, n_feat - 1) * (2 * n_bins * n_bins)
    rows.append(_row("tan", n_feat, tan_params, {"cmp": n_feat, "add": n_feat * 2, "mul": 0, "lookup": n_feat * 2, "branch": n_feat}, True, "medium"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
