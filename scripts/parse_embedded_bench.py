#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parse BENCH lines from embedded UART log")
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="embedded_bench.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="ignore")
    rx = re.compile(
        r"BENCH policy=(?P<policy>\w+) (?:target_iters=(?P<target_iters>\d+) used_iters=(?P<used_iters>\d+) timeout_us=(?P<timeout>\d+) )?mean_us=(?P<mean>[\d.]+) p50_us=(?P<p50>\d+) p95_us=(?P<p95>\d+)"
    )
    rows = []
    for m in rx.finditer(text):
        rows.append(
            {
                "policy": m.group("policy"),
                "target_iters": int(m.group("target_iters") or 0),
                "used_iters": int(m.group("used_iters") or 0),
                "timeout_us": int(m.group("timeout") or 0),
                "mean_us": float(m.group("mean")),
                "p50_us": int(m.group("p50")),
                "p95_us": int(m.group("p95")),
            }
        )
    if not rows:
        raise SystemExit("No BENCH lines found in log")
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"saved: {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
