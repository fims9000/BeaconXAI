#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from beaconxai.datasets import apply_standardizer, fit_channel_standardizer, load_npz_dataset, load_uci_har
from beaconxai.experiments import evaluate_error_risk
from beaconxai.models import (
    train_1dcnn,
    train_anfis_stats,
    train_extratrees_stats,
    train_logreg,
    train_minirocket_if_available,
)
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig


def parse_list(v: str, cast=float):
    return [cast(x.strip()) for x in v.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search for BEACON config")
    p.add_argument("--dataset", choices=["uci_har", "npz"], default="npz")
    p.add_argument("--dataset-root", default="./data")
    p.add_argument("--npz-path", default="./data/uci_har_shifted.npz")
    p.add_argument("--model", choices=["anfis", "extratrees", "minirocket", "cnn1d", "logreg"], default="extratrees")
    p.add_argument("--max-eval", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--k0-values", default="8,16")
    p.add_argument("--q-values", default="16,32")
    p.add_argument("--l-min-values", default="4,8")
    p.add_argument("--q-frag-ratios", default="0.25,0.375,0.5")
    p.add_argument("--alpha-values", default="0.5,1.0,1.5")
    p.add_argument("--beta-values", default="0.25,0.5,0.75")
    p.add_argument("--gamma-values", default="0.5,1.0")
    p.add_argument("--tau-s-values", default="0.05,0.10,0.20")
    p.add_argument("--partition-modes", default="time_only,time_channel")
    p.add_argument("--risk-policies", default="rho_only,rho_censored_boost")

    p.add_argument("--neutralization", choices=["zero", "mean", "interp"], default="zero")
    p.add_argument("--out", default="./outputs_composite/beacon_search.json")
    p.add_argument("--cnn-epochs", type=int, default=20)
    return p.parse_args()


def _load_data(args):
    if args.dataset == "uci_har":
        return load_uci_har(args.dataset_root)
    return load_npz_dataset(args.npz_path)


def _auc_from_metrics(metrics: list[dict], method: str, q: int) -> float:
    for m in metrics:
        if m["method"] == method and int(m["q_max"]) == int(q):
            return float(m["auroc"])
    return float("nan")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_train, y_train, x_test, y_test = _load_data(args)
    mu, sigma = fit_channel_standardizer(x_train)
    x_train = apply_standardizer(x_train, mu, sigma)
    x_test = apply_standardizer(x_test, mu, sigma)

    if args.max_eval > 0 and args.max_eval < len(x_test):
        idx = rng.choice(len(x_test), size=args.max_eval, replace=False)
        x_eval = x_test[idx]
        y_eval = y_test[idx]
    else:
        x_eval = x_test
        y_eval = y_test

    if args.model == "anfis":
        clf = train_anfis_stats(x_train, y_train)
    elif args.model == "extratrees":
        clf = train_extratrees_stats(x_train, y_train)
    elif args.model == "logreg":
        clf = train_logreg(x_train, y_train)
    elif args.model == "cnn1d":
        clf = train_1dcnn(x_train, y_train, epochs=args.cnn_epochs)
    else:
        clf = train_minirocket_if_available(x_train, y_train)

    train_margins = []
    for i in range(min(len(x_train), 2000)):
        lg = clf.logits(x_train[i])
        y_hat = int(np.argmax(lg))
        m = float(lg[y_hat] - np.max(np.delete(lg, y_hat)))
        if m > 0:
            train_margins.append(m)
    tau_m = float(np.quantile(train_margins, 0.10)) if train_margins else 0.0

    ch_means = np.zeros(x_train.shape[-1], dtype=np.float32)
    if args.neutralization == "mean":
        ch_means = x_train.mean(axis=(0, 1)).astype(np.float32)
    neutralizer = Neutralizer(mode=args.neutralization, channel_means=ch_means)

    k0_values = parse_list(args.k0_values, int)
    q_values = parse_list(args.q_values, int)
    l_min_values = parse_list(args.l_min_values, int)
    q_frag_ratios = parse_list(args.q_frag_ratios, float)
    alpha_values = parse_list(args.alpha_values, float)
    beta_values = parse_list(args.beta_values, float)
    gamma_values = parse_list(args.gamma_values, float)
    tau_s_values = parse_list(args.tau_s_values, float)
    partition_modes = [x.strip() for x in args.partition_modes.split(",") if x.strip()]
    risk_policies = [x.strip() for x in args.risk_policies.split(",") if x.strip()]

    rows = []
    best = None

    for (k0, l_min, q_frag, alpha, beta, gamma, tau_s, part_mode, risk_pol) in itertools.product(
        k0_values,
        l_min_values,
        q_frag_ratios,
        alpha_values,
        beta_values,
        gamma_values,
        tau_s_values,
        partition_modes,
        risk_policies,
    ):
        cfg = BeaconConfig(
            q_max=max(q_values),
            k0=int(k0),
            l_min=int(l_min),
            q_frag_ratio=float(q_frag),
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            tau_s=float(tau_s),
            tau_m=float(tau_m),
            partition_mode=part_mode,
            risk_policy=risk_pol,
        )

        _, _, metrics = evaluate_error_risk(
            x_test=x_eval,
            y_test=y_eval,
            predict_fn=clf.predict,
            logits_fn=clf.logits,
            neutralizer=neutralizer,
            base_cfg=cfg,
            q_values=q_values,
            margin_gradient_fn=getattr(clf, "margin_gradient", None),
        )

        q_target = max(q_values)
        a_ref = _auc_from_metrics(metrics, "beacon_refine", q_target)
        a_flat = _auc_from_metrics(metrics, "beacon_flat", q_target)
        a_conf = _auc_from_metrics(metrics, "confidence", 0)

        score = (a_ref - a_flat) + 0.75 * (a_ref - a_conf)
        row = {
            "k0": int(k0),
            "l_min": int(l_min),
            "q_frag_ratio": float(q_frag),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma),
            "tau_s": float(tau_s),
            "partition_mode": part_mode,
            "risk_policy": risk_pol,
            "auroc_ref": float(a_ref),
            "auroc_flat": float(a_flat),
            "auroc_conf": float(a_conf),
            "score": float(score),
        }
        rows.append(row)

        if best is None or row["score"] > best["score"]:
            best = row

    out = {
        "best": best,
        "top20": sorted(rows, key=lambda r: r["score"], reverse=True)[:20],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:")
    print(out_path)


if __name__ == "__main__":
    main()
