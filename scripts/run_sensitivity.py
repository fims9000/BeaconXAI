#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset, load_uci_har
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


def parse_list_floats(v: str):
    return [float(x) for x in v.split(",") if x]


def parse_list_ints(v: str):
    return [int(x) for x in v.split(",") if x]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensitivity sweeps for BEACON-XAI")
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
    p.add_argument("--q-values", default="8,16,32,64")
    p.add_argument("--k0-values", default="8,16")
    p.add_argument("--q-frag-ratios", default="0.125,0.25,0.375")
    p.add_argument("--alphas", default="0.5,1.0,1.5")
    p.add_argument("--betas", default="0.25,0.5,0.75")
    p.add_argument("--gammas", default="0.5,1.0,1.5")
    p.add_argument("--tau-s-values", default="0.05,0.10,0.20")
    p.add_argument("--tau-m-quantiles", default="0.05,0.10,0.20")
    p.add_argument("--neutralization-modes", default="zero,mean,interp")
    p.add_argument("--l-min", type=int, default=4)
    p.add_argument("--k-pos", type=int, default=3)
    p.add_argument("--k-neg", type=int, default=3)
    p.add_argument("--max-test", type=int, default=128)
    p.add_argument("--out", default="./outputs_sensitivity/sensitivity.csv")
    return p.parse_args()


def _load_data(args):
    if args.dataset == "uci_har":
        return load_uci_har(args.dataset_root)
    if not args.npz_path:
        raise ValueError("--npz-path required when --dataset npz")
    return load_npz_dataset(args.npz_path)


def _get_tau_m(train_margins, q):
    return float(np.quantile(train_margins, q)) if train_margins else 0.0


def main() -> None:
    args = parse_args()
    x_train, y_train, x_test, y_test = _load_data(args)

    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.max_test > 0:
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

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

    train_margins = []
    for i in range(min(len(x_train), 2000)):
        lg = clf.logits(x_train[i])
        y_hat = int(np.argmax(lg))
        m = float(lg[y_hat] - np.max(np.delete(lg, y_hat)))
        if m > 0:
            train_margins.append(m)

    q_values = parse_list_ints(args.q_values)
    k0_values = parse_list_ints(args.k0_values)
    q_frag_ratios = parse_list_floats(args.q_frag_ratios)
    alphas = parse_list_floats(args.alphas)
    betas = parse_list_floats(args.betas)
    gammas = parse_list_floats(args.gammas)
    tau_s_values = parse_list_floats(args.tau_s_values)
    tau_m_quantiles = parse_list_floats(args.tau_m_quantiles)
    neutralization_modes = [x.strip() for x in args.neutralization_modes.split(",") if x.strip()]

    out_rows = []

    for mode in neutralization_modes:
        ch_means = np.zeros(x_train.shape[-1], dtype=np.float32)
        if mode == "mean":
            ch_means = x_train.mean(axis=(0, 1)).astype(np.float32)
        neutralizer = Neutralizer(mode=mode, channel_means=ch_means)

        for k0 in k0_values:
            for q_frag_ratio in q_frag_ratios:
                for alpha in alphas:
                    for beta in betas:
                        for gamma in gammas:
                            for tau_s in tau_s_values:
                                for tau_m_q in tau_m_quantiles:
                                    tau_m = _get_tau_m(train_margins, tau_m_q)
                                    cfg = BeaconConfig(
                                        q_max=max(q_values),
                                        k0=k0,
                                        l_min=args.l_min,
                                        k_pos=args.k_pos,
                                        k_neg=args.k_neg,
                                        q_frag_ratio=q_frag_ratio,
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=gamma,
                                        tau_s=tau_s,
                                        tau_m=tau_m,
                                    )

                                    rows, _, metrics = evaluate_error_risk(
                                        x_test=x_test,
                                        y_test=y_test,
                                        predict_fn=clf.predict,
                                        logits_fn=clf.logits,
                                        neutralizer=neutralizer,
                                        base_cfg=cfg,
                                        q_values=q_values,
                                        margin_gradient_fn=getattr(clf, "margin_gradient", None),
                                    )

                                    for m in metrics:
                                        if m["method"] != "beacon_refine":
                                            continue
                                        out_rows.append(
                                            {
                                                "mode": mode,
                                                "k0": k0,
                                                "q_frag_ratio": q_frag_ratio,
                                                "alpha": alpha,
                                                "beta": beta,
                                                "gamma": gamma,
                                                "tau_s": tau_s,
                                                "tau_m_quantile": tau_m_q,
                                                "q_max": int(m["q_max"]),
                                                "auroc": m["auroc"],
                                                "auprc": m["auprc"],
                                                "mean_q_used": m["mean_q_used"],
                                                "censored_rate": m["censored_rate"],
                                            }
                                        )

    out = Path(args.out)
    _write_csv(out, out_rows)
    print("Saved:")
    print(out)


if __name__ == "__main__":
    main()
