#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import (
    apply_standardizer,
    fit_channel_standardizer,
    load_npz_dataset,
    load_uci_har,
)
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import (
    train_1dcnn,
    train_anfis_stats,
    train_extratrees_stats,
    train_histgbt_stats,
    train_logreg,
    train_minirocket_if_available,
)
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BEACON-XAI error-risk experiments")
    p.add_argument("--dataset", choices=["uci_har", "npz"], default="uci_har")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--npz-path", default="")
    p.add_argument(
        "--model",
        choices=["anfis", "histgbt", "extratrees", "minirocket", "cnn1d", "logreg"],
        default="histgbt",
    )
    p.add_argument("--anfis-rules", type=int, default=10)
    p.add_argument("--anfis-ridge", type=float, default=0.2)
    p.add_argument("--anfis-max-samples", type=int, default=4000)
    p.add_argument("--histgbt-iters", type=int, default=220)
    p.add_argument("--histgbt-lr", type=float, default=0.08)
    p.add_argument("--histgbt-leaves", type=int, default=63)
    p.add_argument("--histgbt-min-leaf", type=int, default=20)
    p.add_argument("--cnn-epochs", type=int, default=20)
    p.add_argument("--cnn-batch-size", type=int, default=128)
    p.add_argument("--cnn-lr", type=float, default=1e-3)
    p.add_argument("--cnn-label-smoothing", type=float, default=0.0)
    p.add_argument("--cnn-no-class-weights", action="store_true")
    p.add_argument("--cnn-tta-shifts", default="0")
    p.add_argument("--neutralization", choices=["zero", "mean", "interp"], default="zero")
    p.add_argument("--q-values", default="8,16,32,64")
    p.add_argument("--k0", type=int, default=8)
    p.add_argument("--l-min", type=int, default=4)
    p.add_argument("--k-pos", type=int, default=3)
    p.add_argument("--k-neg", type=int, default=3)
    p.add_argument("--q-frag-ratio", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--tau-s", type=float, default=0.10)
    p.add_argument("--partition-mode", choices=["time_only", "time_channel"], default="time_only")
    p.add_argument("--risk-policy", choices=["rho_only", "rho_censored_boost"], default="rho_only")
    p.add_argument("--tau-m-quantile", type=float, default=0.10)
    p.add_argument("--max-test", type=int, default=0)
    p.add_argument("--out-dir", default="./outputs")
    p.add_argument(
        "--methods",
        default="confidence,entropy,negative_margin,beacon_refine,beacon_flat,uniform_refinement,budgeted_shapley_like,saliency_topk,ig_topk,simple_counterfactual,full_occlusion,beacon_composite",
    )
    p.add_argument("--enable-composite", action="store_true")
    p.add_argument("--w-beacon", type=float, default=1.6237111199364826)
    p.add_argument("--w-conf", type=float, default=1.9935299447424337)
    p.add_argument("--w-rho", type=float, default=0.9335959634678679)
    p.add_argument("--w-nec", type=float, default=0.1718514451102111)
    p.add_argument("--w-ce", type=float, default=0.7323714207898753)
    p.add_argument("--w-suff-bad", type=float, default=0.9798459952920904)
    p.add_argument("--w-censored", type=float, default=0.14685705659473536)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "uci_har":
        x_train, y_train, x_test, y_test = load_uci_har(args.dataset_root)
    else:
        if not args.npz_path:
            raise ValueError("--npz-path is required when --dataset npz")
        x_train, y_train, x_test, y_test = load_npz_dataset(args.npz_path)

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.model == "anfis":
        clf = train_anfis_stats(
            x_train,
            y_train,
            n_rules=args.anfis_rules,
            ridge=args.anfis_ridge,
            max_fit_samples=args.anfis_max_samples,
        )
    elif args.model == "histgbt":
        clf = train_histgbt_stats(
            x_train,
            y_train,
            max_iter=args.histgbt_iters,
            learning_rate=args.histgbt_lr,
            max_leaf_nodes=args.histgbt_leaves,
            min_samples_leaf=args.histgbt_min_leaf,
        )
    elif args.model == "extratrees":
        clf = train_extratrees_stats(x_train, y_train)
    elif args.model == "logreg":
        clf = train_logreg(x_train, y_train)
    elif args.model == "cnn1d":
        tta_shifts = tuple(int(v) for v in args.cnn_tta_shifts.split(",") if v.strip())
        if not tta_shifts:
            tta_shifts = (0,)
        clf = train_1dcnn(
            x_train,
            y_train,
            epochs=args.cnn_epochs,
            batch_size=args.cnn_batch_size,
            lr=args.cnn_lr,
            label_smoothing=args.cnn_label_smoothing,
            use_class_weights=not args.cnn_no_class_weights,
            tta_shifts=tta_shifts,
        )
    else:
        clf = train_minirocket_if_available(x_train, y_train)

    # tau_m: validation-like proxy from train margins
    train_margins = []
    for i in range(min(len(x_train), 2000)):
        logits = clf.logits(x_train[i])
        y_hat = int(np.argmax(logits))
        m = float(logits[y_hat] - np.max(np.delete(logits, y_hat)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, args.tau_m_quantile)) if train_margins else 0.0

    channel_means = np.zeros(x_train.shape[-1], dtype=np.float32)
    if args.neutralization == "mean":
        channel_means = x_train.mean(axis=(0, 1)).astype(np.float32)

    neutralizer = Neutralizer(mode=args.neutralization, channel_means=channel_means)

    cfg = BeaconConfig(
        q_max=max(int(v) for v in args.q_values.split(",") if v),
        k0=args.k0,
        l_min=args.l_min,
        k_pos=args.k_pos,
        k_neg=args.k_neg,
        q_frag_ratio=args.q_frag_ratio,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        tau_s=args.tau_s,
        partition_mode=args.partition_mode,
        risk_policy=args.risk_policy,
        tau_m=tau_m,
    )

    q_values = [int(v) for v in args.q_values.split(",") if v]

    if args.max_test > 0:
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

    composite_weights = None
    if args.enable_composite:
        composite_weights = {
            "beacon": args.w_beacon,
            "conf": args.w_conf,
            "rho": args.w_rho,
            "nec": args.w_nec,
            "ce": args.w_ce,
            "suff_bad": args.w_suff_bad,
            "censored": args.w_censored,
        }

    methods = {x.strip() for x in args.methods.split(",") if x.strip()}

    rows, local_rows, metrics = evaluate_error_risk(
        x_test=x_test,
        y_test=y_test,
        predict_fn=clf.predict,
        logits_fn=clf.logits,
        neutralizer=neutralizer,
        base_cfg=cfg,
        q_values=q_values,
        margin_gradient_fn=getattr(clf, "margin_gradient", None),
        composite_weights=composite_weights,
        methods=methods,
    )

    rows_out = [r.__dict__ for r in rows]
    local_rows_out = [r.__dict__ for r in local_rows]
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "risk_rows.csv", rows_out)
    _write_csv(out_dir / "local_metrics.csv", local_rows_out)
    _write_csv(out_dir / "risk_metrics.csv", metrics)

    print("Saved:")
    print(out_dir / "risk_rows.csv")
    print(out_dir / "local_metrics.csv")
    print(out_dir / "risk_metrics.csv")


if __name__ == "__main__":
    main()
