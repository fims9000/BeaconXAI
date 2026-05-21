#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build markdown summary + claim registry from v11 aggregates")
    p.add_argument("--input-csv", default="outputs_composite/v11_full_summary.csv")
    p.add_argument("--output", default="artifacts/v11_full_summary.md")
    p.add_argument("--claim-md", default="artifacts/claim_registry_v11.md")
    return p.parse_args()


def _fmt(x: float, n: int = 4) -> str:
    if pd.isna(x):
        return "nan"
    return f"{x:.{n}f}"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise SystemExit("input table is empty")

    # Global conservative alpha for multiple-dataset scan on F1@10 (TAN):
    alpha = 0.05 / max(1, len(df))
    tan_pos = (df["tan_d_f1_10"] > 0) & (df["tan_ci_low_f1_10"] > 0) & (df["tan_p_f1_10"] < alpha)
    fuzzy_pos = (df["fuzzy_d_f1_10"] > 0) & (df["fuzzy_ci_low_f1_10"] > 0) & (df["fuzzy_p_f1_10"] < alpha)

    out_lines = [
        "# v11 cross-dataset summary",
        "",
        f"- rows: {len(df)}",
        f"- alpha_bonf_global_f1: {alpha:.6f}",
        "",
        "| dataset | bundle | TAN dAUROC (p) | TAN dF1@10 (p) | Fuzzy dAUROC (p) | Fuzzy dF1@10 (p) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        out_lines.append(
            "| {ds} | {b} | {ta} ({tp}) | {tf} ({tfp}) | {fa} ({fp}) | {ff} ({ffp}) |".format(
                ds=r["dataset"],
                b=r["bundle"],
                ta=_fmt(r["tan_d_auroc"]),
                tp=_fmt(r["tan_p_auroc"]),
                tf=_fmt(r["tan_d_f1_10"]),
                tfp=_fmt(r["tan_p_f1_10"]),
                fa=_fmt(r["fuzzy_d_auroc"]),
                fp=_fmt(r["fuzzy_p_auroc"]),
                ff=_fmt(r["fuzzy_d_f1_10"]),
                ffp=_fmt(r["fuzzy_p_f1_10"]),
            )
        )

    claim_lines = [
        "# claim_registry_v11",
        "",
        f"- alpha_bonf_global_f1: {alpha:.6f}",
        "",
        "## Rule",
        "- supported_positive := delta_f1_10 > 0 and ci_low_f1_10 > 0 and p_f1_10 < alpha_bonf_global_f1",
        "",
        "## TAN",
        f"- supported bundles: {int(tan_pos.sum())}/{len(df)}",
    ]
    if int(tan_pos.sum()) > 0:
        bundles = ", ".join(df.loc[tan_pos, "bundle"].tolist())
        claim_lines.append(f"- bundles: {bundles}")
        claim_lines.append("- claim: TAN can improve in specific data regimes (not universal).")
    else:
        claim_lines.append("- claim: TAN has no stable quality gain over logit under v11 protocol.")

    claim_lines.extend(
        [
            "",
            "## Fuzzy",
            f"- supported bundles: {int(fuzzy_pos.sum())}/{len(df)}",
        ]
    )
    if int(fuzzy_pos.sum()) > 0:
        bundles = ", ".join(df.loc[fuzzy_pos, "bundle"].tolist())
        claim_lines.append(f"- bundles: {bundles}")
        claim_lines.append("- claim: fuzzy has isolated gain only.")
    else:
        claim_lines.append("- claim: fuzzy has no stable quality gain over logit under v11 protocol.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.claim_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    Path(args.claim_md).write_text("\n".join(claim_lines) + "\n", encoding="utf-8")
    print(f"saved: {args.output}")
    print(f"saved: {args.claim_md}")


if __name__ == "__main__":
    main()
