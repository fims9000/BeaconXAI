#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BEACON v6 grid (Q x neutralizer) and evaluate policies")
    p.add_argument("--dataset", default="data/uci_har_shifted.npz")
    p.add_argument("--model", default="extratrees", choices=["extratrees", "histgbt", "cnn1d"])
    p.add_argument("--n-total", type=int, default=3000)
    p.add_argument("--q-list", default="16,32,64")
    p.add_argument("--neutralizers", default="interp,zero,mean,class_mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-out", default="outputs_composite/part2_extended_v6")
    p.add_argument("--skip-feature-run", action="store_true")
    p.add_argument("--skip-policy-grid", action="store_true")
    p.add_argument("--skip-anomaly", action="store_true")
    p.add_argument("--anomaly-model", default="cnn1d", choices=["extratrees", "histgbt", "cnn1d"])
    p.add_argument("--anomaly-max-test", type=int, default=512)
    p.add_argument("--anomaly-fault-types", default="spike,drift,stuck_sensor,dropout")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    q_list = [int(v.strip()) for v in args.q_list.split(",") if v.strip()]
    modes = [v.strip() for v in args.neutralizers.split(",") if v.strip()]

    base = Path(args.base_out)
    base.mkdir(parents=True, exist_ok=True)

    if not args.skip_policy_grid:
        for q in q_list:
            for mode in modes:
                bdir = base / f"q{q}_{mode}"
                bdir.mkdir(parents=True, exist_ok=True)

                if not args.skip_feature_run:
                    _run(
                        [
                            sys.executable,
                            "scripts/run_part2_extended.py",
                            "--dataset",
                            args.dataset,
                            "--model",
                            args.model,
                            "--n-total",
                            str(args.n_total),
                            "--q-max",
                            str(q),
                            "--neutralizer-mode",
                            mode,
                            "--seed",
                            str(args.seed),
                            "--features-only",
                            "--out",
                            str(bdir),
                        ]
                    )

                _run(
                    [
                        sys.executable,
                        "scripts/evaluate_v6_policies.py",
                        "--bundle-dir",
                        str(bdir),
                        "--seed",
                        str(args.seed),
                        "--device",
                        args.device,
                        "--n-boot",
                        str(args.n_boot),
                    ]
                )

    if not args.skip_anomaly:
        for q in q_list:
            adir = base / f"anomaly_q{q}"
            adir.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    sys.executable,
                    "scripts/run_har_sensor_fault_benchmark.py",
                    "--npz-path",
                    args.dataset,
                    "--model",
                    args.anomaly_model,
                    "--q",
                    str(q),
                    "--max-test",
                    str(args.anomaly_max_test),
                    "--fault-types",
                    args.anomaly_fault_types,
                    "--n-boot",
                    str(args.n_boot),
                    "--out-summary",
                    str(adir / "sensor_anomaly_localization.csv"),
                    "--out-per-sample",
                    str(adir / "sensor_anomaly_per_sample.csv"),
                    "--out-bootstrap",
                    str(adir / "sensor_anomaly_bootstrap.csv"),
                    "--out-eval-npz",
                    str(adir / "sensor_anomaly_eval.npz"),
                ]
            )

    # aggregate
    rows = []
    brows = []
    costs = []
    anomaly = []
    anomaly_boot = []
    for q in q_list:
        for mode in modes:
            bdir = base / f"q{q}_{mode}"
            p = bdir / "v6_policy_eval.csv"
            pb = bdir / "v6_bootstrap_deltas.csv"
            pc = bdir / "tinyxai_full_audit_cost.csv"
            if p.exists():
                rows.append(p)
            if pb.exists():
                brows.append(pb)
            if pc.exists():
                costs.append(pc)
        adir = base / f"anomaly_q{q}"
        pa = adir / "sensor_anomaly_localization.csv"
        pab = adir / "sensor_anomaly_bootstrap.csv"
        if pa.exists():
            anomaly.append(pa)
        if pab.exists():
            anomaly_boot.append(pab)

    import pandas as pd

    if rows:
        pd.concat([pd.read_csv(p) for p in rows], ignore_index=True).to_csv(base / "beacon_vs_uniform_q_sweep.csv", index=False)
    if brows:
        pd.concat([pd.read_csv(p) for p in brows], ignore_index=True).to_csv(base / "bootstrap_deltas_v6.csv", index=False)
    if costs:
        pd.concat([pd.read_csv(p) for p in costs], ignore_index=True).to_csv(base / "tinyxai_full_audit_cost.csv", index=False)
    if anomaly:
        pd.concat([pd.read_csv(p) for p in anomaly], ignore_index=True).to_csv(base / "sensor_anomaly_localization.csv", index=False)
    if anomaly_boot:
        pd.concat([pd.read_csv(p) for p in anomaly_boot], ignore_index=True).to_csv(base / "sensor_anomaly_bootstrap.csv", index=False)

    claims = []
    qboot_path = base / "beacon_uniform_q_sweep_bootstrap.csv"
    if qboot_path.exists():
        qb = pd.read_csv(qboot_path)
        metrics = [
            ("delta_auroc_beacon_minus_uniform", "p_auroc", "ci_auroc_low", "ci_auroc_high"),
            ("delta_auprc_beacon_minus_uniform", "p_auprc", "ci_auprc_low", "ci_auprc_high"),
            ("delta_f1_10_beacon_minus_uniform", "p_f1_10", "ci_f1_10_low", "ci_f1_10_high"),
            ("delta_f1_20_beacon_minus_uniform", "p_f1_20", "ci_f1_20_low", "ci_f1_20_high"),
        ]
        for _, r in qb.iterrows():
            for dcol, pcol, lcol, hcol in metrics:
                claims.append(
                    {
                        "block": "q_sweep_detection",
                        "bundle": f"{r.get('neutralizer_input', r.get('neutralizer_mode', 'na'))}_q{int(r['q_max'])}",
                        "criterion": f"BEACON > uniform on {dcol}",
                        "delta": float(r[dcol]),
                        "ci_low": float(r[lcol]),
                        "ci_high": float(r[hcol]),
                        "p_value": float(r[pcol]),
                        "q1_signal": int(float(r[lcol]) > 0.0 and float(r[pcol]) < 0.05),
                    }
                )

    boot_path = base / "bootstrap_deltas_v6.csv"
    if boot_path.exists():
        b = pd.read_csv(boot_path)
        m = b[
            (b["comparison"] == "logit_beacon_vs_uniform")
            & (b["metric"].isin(["delta_auroc", "delta_auprc", "delta_f1_10", "delta_f1_20"]))
        ]
        for _, r in m.iterrows():
            claims.append(
                {
                    "block": "binary_detection",
                    "bundle": r.get("bundle", ""),
                    "criterion": f"BEACON > uniform on {r['metric']}",
                    "delta": float(r["delta"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                    "p_value": float(r["p_value"]),
                    "q1_signal": int(float(r["ci_low"]) > 0.0 and float(r["p_value"]) < 0.05),
                }
            )
    aboot_path = base / "sensor_anomaly_bootstrap.csv"
    if aboot_path.exists():
        ab = pd.read_csv(aboot_path)
        for _, r in ab.iterrows():
            claims.append(
                {
                    "block": "sensor_anomaly",
                    "bundle": "",
                    "criterion": f"beacon_xai > {r['method_b']} on {r['metric']}",
                    "delta": float(r["delta"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                    "p_value": float(r["p_value"]),
                    "q1_signal": int(float(r["ci_low"]) > 0.0 and float(r["p_value"]) < 0.05),
                }
            )
    if claims:
        pd.DataFrame(claims).to_csv(base / "manuscript_claim_registry_v6.csv", index=False)

    aliases = {
        "beacon_uniform_q_sweep.csv": "manuscript_table_detection_v6.csv",
        "sensor_anomaly_localization.csv": "manuscript_table_sensor_anomaly_v6.csv",
        "sensor_anomaly_bootstrap.csv": "manuscript_table_sensor_anomaly_bootstrap_v6.csv",
        "tinyxai_full_audit_cost.csv": "manuscript_table_tiny_full_cost.csv",
    }
    for src_name, dst_name in aliases.items():
        src = base / src_name
        if src.exists():
            src.replace(base / dst_name) if False else pd.read_csv(src).to_csv(base / dst_name, index=False)

    print(f"saved grid outputs to: {base}")


if __name__ == "__main__":
    main()
