#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset, load_uci_har
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import train_1dcnn, train_extratrees_stats, train_histgbt_stats, train_logreg
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig
from scripts.make_audit_panel_tables import (
    bootstrap_metric_delta,
    cv_logit_scores,
    evaluate_alert_policy,
    safe_auc,
)


def _parse_list(s: str, cast=int):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def _normalize_neutralizer_name(name: str) -> tuple[str, str]:
    raw = name.strip()
    key = raw.lower().replace("-", "_")
    aliases = {
        "interp": "interp",
        "zero": "zero",
        "mean": "mean",
        "channel_mean": "mean",
        "class_mean": "class_mean",
        "train_class_mean": "class_mean",
    }
    if key not in aliases:
        raise ValueError(
            f"unsupported neutralizer '{raw}'. Allowed: interp, zero, mean/channel_mean, class_mean/train_class_mean"
        )
    return raw, aliases[key]


def _build_panel_df(
    risk_df: pd.DataFrame,
    local_df: pd.DataFrame,
    q: int,
    method: str,
) -> pd.DataFrame:
    base = risk_df[(risk_df["method"] == "negative_margin") & (risk_df["q_max"] == 0)][
        ["sample_id", "is_error", "risk_score"]
    ].rename(columns={"risk_score": "m_neg"})
    loc = local_df[(local_df["method"] == method) & (local_df["q_max"] == q)][
        ["sample_id", "necessity", "counter_evidence_gain", "rho_b", "rho_b_cost", "sufficiency_margin"]
    ].copy()
    loc["frag_drop"] = np.maximum(0.0, -loc["sufficiency_margin"].astype(float))
    loc = loc.rename(
        columns={
            "counter_evidence_gain": "M_B_minus",
            "necessity": "CE_B",
            "rho_b": "r_B_minus",
            "rho_b_cost": "rho_B_cost",
        }
    )
    df = base.merge(loc, on="sample_id", how="inner")
    return df.dropna(
        subset=["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop", "is_error"]
    ).copy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q-sweep: BEACON vs uniform for conflict detection panel")
    p.add_argument("--dataset", choices=["uci_har", "npz"], default="uci_har")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--npz-path", default="")
    p.add_argument("--model", choices=["extratrees", "histgbt", "logreg", "cnn1d"], default="extratrees")
    p.add_argument("--q-values", default="16,32,64")
    p.add_argument("--neutralizers", default="interp,zero,channel_mean,train_class_mean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=1024)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--out-summary", default="outputs_composite/beacon_uniform_q_sweep.csv")
    p.add_argument("--out-bootstrap", default="outputs_composite/beacon_uniform_q_sweep_bootstrap.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    q_values = _parse_list(args.q_values, int)
    neutralizers = [_normalize_neutralizer_name(x) for x in args.neutralizers.split(",") if x.strip()]

    if args.dataset == "uci_har":
        x_train, y_train, x_test, y_test = load_uci_har(args.dataset_root)
    else:
        if not args.npz_path:
            raise ValueError("--npz-path is required for --dataset npz")
        x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)
    if args.max_test > 0:
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

    if args.model == "extratrees":
        clf = train_extratrees_stats(x_train, y_train, n_estimators=1200, max_features=0.7, min_samples_leaf=1)
    elif args.model == "histgbt":
        clf = train_histgbt_stats(x_train, y_train, max_iter=220, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=20)
    elif args.model == "logreg":
        clf = train_logreg(x_train, y_train)
    else:
        clf = train_1dcnn(x_train, y_train, epochs=8, batch_size=256, lr=1e-3, label_smoothing=0.0, use_class_weights=True, tta_shifts=(0,))

    train_margins = []
    for i in range(min(len(x_train), 2000)):
        lg = clf.logits(x_train[i])
        yh = int(np.argmax(lg))
        m = float(lg[yh] - np.max(np.delete(lg, yh)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    base_cfg = BeaconConfig(
        q_max=max(q_values),
        k0=8,
        l_min=4,
        k_pos=3,
        k_neg=3,
        q_frag_ratio=0.25,
        alpha=1.0,
        beta=0.5,
        gamma=1.0,
        tau_s=0.10,
        tau_m=tau_m,
        partition_mode="sensor_group_time",
        risk_policy="rho_only",
    )

    summary_rows = []
    boot_rows = []

    for nz_raw, nz_mode in neutralizers:
        ch_means = np.zeros(x_train.shape[-1], dtype=np.float32)
        if nz_mode in ("mean", "class_mean"):
            ch_means = x_train.mean(axis=(0, 1)).astype(np.float32)
        neutralizer = Neutralizer(mode=nz_mode, channel_means=ch_means)

        rows, local_rows, _ = evaluate_error_risk(
            x_test=x_test,
            y_test=y_test,
            predict_fn=clf.predict,
            logits_fn=clf.logits,
            neutralizer=neutralizer,
            base_cfg=base_cfg,
            q_values=q_values,
            margin_gradient_fn=getattr(clf, "margin_gradient", None),
            methods={"negative_margin", "beacon_refine", "uniform_refinement"},
        )
        risk_df = pd.DataFrame([r.__dict__ for r in rows])
        local_df = pd.DataFrame([r.__dict__ for r in local_rows])

        for q in q_values:
            bdf = _build_panel_df(risk_df, local_df, q=q, method="beacon_refine")
            udf = _build_panel_df(risk_df, local_df, q=q, method="uniform_refinement")
            key_cols = ["sample_id", "is_error", "m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]
            b = bdf[key_cols].rename(columns={c: f"{c}_b" for c in key_cols if c != "sample_id"})
            u = udf[key_cols].rename(columns={c: f"{c}_u" for c in key_cols if c != "sample_id"})
            m = b.merge(u, on="sample_id", how="inner")
            y = m["is_error_b"].to_numpy(dtype=int)
            feats = ["m_neg", "M_B_minus", "CE_B", "r_B_minus", "rho_B_cost", "frag_drop"]
            xb = np.column_stack([m[f"{f}_b"].to_numpy(dtype=float) for f in feats])
            xu = np.column_stack([m[f"{f}_u"].to_numpy(dtype=float) for f in feats])
            sb = cv_logit_scores(xb, y, seed=args.seed)
            su = cv_logit_scores(xu, y, seed=args.seed)

            f1b10 = evaluate_alert_policy(y, sb, 0.10)[2]
            f1u10 = evaluate_alert_policy(y, su, 0.10)[2]
            f1b20 = evaluate_alert_policy(y, sb, 0.20)[2]
            f1u20 = evaluate_alert_policy(y, su, 0.20)[2]
            aucb = safe_auc(y, sb)
            aucu = safe_auc(y, su)
            apb = float(average_precision_score(y, sb))
            apu = float(average_precision_score(y, su))
            summary_rows.extend(
                [
                    {
                        "neutralizer_input": nz_raw,
                        "neutralizer_mode": nz_mode,
                        "q_max": q,
                        "method": "beacon_panel",
                        "n_samples": len(y),
                        "auroc": aucb,
                        "auprc": apb,
                        "f1_at_10": f1b10,
                        "f1_at_20": f1b20,
                    },
                    {
                        "neutralizer_input": nz_raw,
                        "neutralizer_mode": nz_mode,
                        "q_max": q,
                        "method": "uniform_panel",
                        "n_samples": len(y),
                        "auroc": aucu,
                        "auprc": apu,
                        "f1_at_10": f1u10,
                        "f1_at_20": f1u20,
                    },
                ]
            )

            d_auc = bootstrap_metric_delta(safe_auc, y, sb, su, n_boot=args.n_boot)
            d_ap = bootstrap_metric_delta(lambda yy, ss: float(average_precision_score(yy, ss)), y, sb, su, n_boot=args.n_boot)
            d_f10 = bootstrap_metric_delta(lambda yy, ss: evaluate_alert_policy(yy, ss, 0.10)[2], y, sb, su, n_boot=args.n_boot)
            d_f20 = bootstrap_metric_delta(lambda yy, ss: evaluate_alert_policy(yy, ss, 0.20)[2], y, sb, su, n_boot=args.n_boot)
            boot_rows.append(
                {
                    "neutralizer_input": nz_raw,
                    "neutralizer_mode": nz_mode,
                    "q_max": q,
                    "n_samples": len(y),
                    "delta_auroc_beacon_minus_uniform": aucb - aucu,
                    "ci_auroc_low": d_auc[0],
                    "ci_auroc_high": d_auc[1],
                    "p_auroc": d_auc[2],
                    "delta_auprc_beacon_minus_uniform": apb - apu,
                    "ci_auprc_low": d_ap[0],
                    "ci_auprc_high": d_ap[1],
                    "p_auprc": d_ap[2],
                    "delta_f1_10_beacon_minus_uniform": f1b10 - f1u10,
                    "ci_f1_10_low": d_f10[0],
                    "ci_f1_10_high": d_f10[1],
                    "p_f1_10": d_f10[2],
                    "delta_f1_20_beacon_minus_uniform": f1b20 - f1u20,
                    "ci_f1_20_low": d_f20[0],
                    "ci_f1_20_high": d_f20[1],
                    "p_f1_20": d_f20[2],
                }
            )

    out_summary = Path(args.out_summary)
    out_boot = Path(args.out_bootstrap)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    pd.DataFrame(boot_rows).to_csv(out_boot, index=False)
    print(f"saved: {out_summary}")
    print(f"saved: {out_boot}")


if __name__ == "__main__":
    main()
