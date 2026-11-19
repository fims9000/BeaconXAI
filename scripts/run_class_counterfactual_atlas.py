#!/usr/bin/env python3
"""Per-class human-readable prototype-intervention explanations on PAMAP2.

Output:
  per_intervention.csv  every measured class-score and fixed-pair margin change
  class_atlas.csv        descriptive class-level aggregates with coverage
  pair_summary.csv       source-vs-rival descriptive contrasts
  report.md              readable class passports and local examples
  plots/*.png            class heatmaps (blank cells = not tested), signal overlays
  manifest.json          provenance, budget, model, coverage, limitations

This is NOT proof of causal features or natural class boundaries. Train-derived
means/medians can be nonphysical, and scores need not be calibrated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from beaconxai.class_counterfactual import (
    fit_prototype_bank, explain_class_pair, describe_intervention,
)
from beaconxai.contrastive import make_components
from scripts.run_q16_matched_ablation import class_cheap, best_q
from scripts.preprocess_pamap2 import ALLOWED_ACTS

PAMAP_NAMES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "Nordic walking",
    12: "ascending stairs", 13: "descending stairs",
    16: "vacuum cleaning", 17: "ironing", 24: "rope jumping",
}
NAMES = {i: PAMAP_NAMES[activity_id] for i, activity_id in enumerate(ALLOWED_ACTS)}
SENSORS = ("wrist", "chest", "ankle")


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        raise ValueError(f"Cannot determine CSV columns for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_cases(x_test: np.ndarray, y_test: np.ndarray,
                 scores: np.ndarray, bank, per_class: int, seed: int):
    """Select correct examples of every observed class, no evaluation leakage."""
    if per_class < 1:
        raise ValueError("per_class must be >=1")
    preds = np.argmax(scores, axis=1)
    rng = np.random.default_rng(seed)
    chosen = []
    counts = {}
    for c in bank.labels:
        eligible = np.flatnonzero((y_test == c) & (preds == c))
        counts[c] = {"eligible_correct_test": int(len(eligible)),
                     "total_test_labeled": int(np.sum(y_test == c))}
        if len(eligible):
            selected = rng.choice(eligible, size=min(len(eligible), per_class), replace=False)
            chosen.extend((int(j), int(c)) for j in selected)
    return chosen, counts


def gather_summary(rows: list[dict], classes: list[int], coverage: dict):
    """Descriptive averages only; source-class and target-specific effects separated."""
    atlas = []
    pairs = []
    for a in classes:
        rc = [r for r in rows if r["source"] == a]
        target = [r for r in rc if r["prototype_class"] == r["rival"]]
        mean_target = [r for r in target if r["prototype_kind"] == "mean"]
        median_target = [r for r in target if r["prototype_kind"] == "median"]
        n_cases = len({r["sample_id"] for r in rc})
        atlas.append({
            "class_id": a, "class_name": NAMES.get(a, str(a)),
            "test_labeled": coverage[a]["total_test_labeled"],
            "eligible_correct_test": coverage[a]["eligible_correct_test"],
            "audited_examples": n_cases, "interventions": len(rc),
            "target_mean_delta_margin": float(np.mean([r["delta_gap"] for r in mean_target]))
                                        if mean_target else None,
            "target_median_delta_margin": float(np.mean([r["delta_gap"] for r in median_target]))
                                          if median_target else None,
            "target_mean_delta_p_rival": float(np.mean([r["delta_p_rival"] for r in mean_target]))
                                          if mean_target else None,
            "target_median_delta_p_rival": float(np.mean([r["delta_p_rival"] for r in median_target]))
                                            if median_target else None,
            "target_prototype_flip_fraction": float(np.mean([r["flip_to_rival"] for r in target]))
                                               if target else None,
            "status": "observed" if rc else "no_correct_test_examples",
        })
    keys = sorted({(r["source"], r["rival"], r["prototype_class"], r["prototype_kind"])
                   for r in rows})
    for a, b, c, kind in keys:
        selected = [r for r in rows if (r["source"], r["rival"],
                    r["prototype_class"], r["prototype_kind"]) == (a, b, c, kind)]
        pairs.append({
            "source": a, "source_name": NAMES.get(a, str(a)),
            "rival": b, "rival_name": NAMES.get(b, str(b)),
            "prototype_class": c, "prototype_kind": kind,
            "n_interventions": len(selected),
            "n_examples": len({r["sample_id"] for r in selected}),
            "mean_delta_p_source": float(np.mean([r["delta_p_source"] for r in selected])),
            "mean_delta_p_rival": float(np.mean([r["delta_p_rival"] for r in selected])),
            "mean_delta_gap": float(np.mean([r["delta_gap"] for r in selected])),
            "fraction_flipped_to_rival": float(np.mean([r["flip_to_rival"] for r in selected])),
        })
    return atlas, pairs


def _plot_atlas(rows, bank, components, output: Path):
    """Class heatmap: only tested cells are colored; all other cells grey."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    output.mkdir(parents=True, exist_ok=True)
    n_bins = len({c.time_bin for c in components})
    n_groups = len({c.group for c in components})
    for a in bank.labels:
        group = [r for r in rows if r["source"] == a and
                 r["prototype_class"] == r["rival"] and
                 r["prototype_kind"] == "mean"]
        if not group:
            continue
        values = np.full((n_groups,n_bins), np.nan)
        counts = np.zeros((n_groups,n_bins), dtype=int)
        for row in group:
            g,b = row["group"], row["time_bin"]
            values[g,b] = (0.0 if np.isnan(values[g,b]) else values[g,b]) + row["delta_gap"]
            counts[g,b] += 1
        np.divide(values, counts, out=values, where=counts>0)
        absmax = max(float(np.nanmax(np.abs(values))), 1e-4)
        fig, ax = plt.subplots(figsize=(10, 3.8))
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#e0e0e0")
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto",
                       norm=TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax))
        ax.set_yticks(range(n_groups), SENSORS[:n_groups])
        ax.set_xticks(range(n_bins), [str(x) for x in range(n_bins)])
        ax.set_xlabel("Time bin (blank = not evaluated)")
        ax.set_title(f"{NAMES.get(a,str(a))}: change in fixed pair margin after rival MEAN replacement")
        for g in range(n_groups):
            for b in range(n_bins):
                if counts[g,b]: ax.text(b,g,str(counts[g,b]),ha="center",va="center",fontsize=7)
        fig.colorbar(im, ax=ax, label="delta log-score gap (negative = toward rival)")
        fig.tight_layout()
        fig.savefig(output/f"class_{a:02d}_heatmap.png", dpi=145)
        plt.close(fig)


def _plot_local_case(x: np.ndarray, bank, components,
                     rows: list[dict], source: int, rival: int, output: Path):
    """One representative trace vs source and rival prototypes (not model explanation by itself)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    candidates = [r for r in rows if r["source"] == source and r["rival"] == rival and
                  r["prototype_class"] == rival and r["prototype_kind"] == "mean"]
    if not candidates:
        return
    best = min(candidates, key=lambda r:r["delta_gap"])
    c = components[best["component"]]
    original = x[c.t0:c.t1,c.c0:c.c1]
    src = bank.mean[source][c.t0:c.t1,c.c0:c.c1]
    rival_mean = bank.mean[rival][c.t0:c.t1,c.c0:c.c1]
    rival_median = bank.median[rival][c.t0:c.t1,c.c0:c.c1]
    fig, axes = plt.subplots(c.c1-c.c0, 1, figsize=(9, 5.7), sharex=True)
    axes = np.atleast_1d(axes)
    tt = np.arange(c.t0,c.t1)
    for k,ax in enumerate(axes):
        ax.plot(tt,original[:,k],label="original",linewidth=2.2)
        ax.plot(tt,src[:,k],label=f"{NAMES[source]} mean",linestyle="--")
        ax.plot(tt,rival_mean[:,k],label=f"{NAMES[rival]} mean",linestyle=":")
        ax.plot(tt,rival_median[:,k],label=f"{NAMES[rival]} median",linestyle="-.")
        ax.set_ylabel(f"axis {k+1}")
        ax.grid(alpha=.2)
    axes[0].legend(ncol=2,fontsize=8)
    axes[-1].set_xlabel("sample index (not aligned physical events)")
    fig.suptitle(f"{NAMES[source]} vs {NAMES[rival]}: {SENSORS[c.group]}, bin {c.time_bin}")
    fig.tight_layout()
    fig.savefig(output/f"class_{source:02d}_example.png",dpi=145)
    plt.close(fig)


def render_report(atlas: list[dict], cases: dict[int, list[dict]], bank,
                  args, output: Path):
    lines = [
        "# Class-prototype intervention atlas (PAMAP2)",
        "",
        f"Prototype types: {args.prototype_mode}; Q={args.budget} per class pair.",
        "Q counts perturbed model inputs; baseline prediction is cached separately.",
        "For prototype_mode=both, Q=16 means FOUR components x TWO classes x TWO prototypes.",
        "",
        "**Interpretation:** reported changes describe the trained classifier's response",
        "to replacing selected time/channel fragments with class mean/median templates.",
        "**They are not causal explanations, validated natural counterfactuals, or",
        "recovered physical boundaries.** Probabilities are model outputs, not",
        "necessarily calibrated event probabilities. Typical class templates may",
        "smooth out phase-specific motion events.",
        "",
        "## Descriptive coverage by predicted and true class",
        "",
        "| Class | Correct test available | Audited | Mean delta gap to rival | Target flip rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in atlas:
        gap = ("—" if r["target_mean_delta_margin"] is None
               else f"{r['target_mean_delta_margin']:+.3f}")
        rate = ("—" if r["target_prototype_flip_fraction"] is None
                else f"{r['target_prototype_flip_fraction']:.1%}")
        lines.append(f"| {r['class_name']} | {r['eligible_correct_test']} | "
                     f"{r['audited_examples']} | {gap} | {rate} |")
    lines.extend(["", "Negative gap delta means the chosen target-template "
                  "replacement made the source less favorable relative to the rival.",
                  "The mean combines different rival classes where present. Use "
                  "`pair_summary.csv` for like-for-like pair comparisons.", "",
                  "## Local class explanations (one example per available class)", ""])
    for a in bank.labels:
        records = cases.get(a, [])
        lines.append(f"### {NAMES.get(a,str(a))}")
        if not records:
            lines.append("No correctly classified held-out example for this class. "
                         "No explanation generated.")
            lines.append("")
            continue
        rivals = sorted({r["rival"] for r in records})
        for b in rivals:
            pair = [r for r in records if r["rival"] == b]
            target_mean = [r for r in pair if r["prototype_class"] == b and
                           r["prototype_kind"] == "mean"]
            if not target_mean:
                target_mean = [r for r in pair if r["prototype_class"] == b]
            best = min(target_mean,key=lambda r:r["delta_gap"])
            lines.append(f"**Versus {NAMES.get(b,str(b))}:**")
            lines.append("")
            lines.append(describe_intervention(best,NAMES,SENSORS))
            lines.append("")
        lines.append(f"![Measured target-prototype effects](plots/class_{a:02d}_heatmap.png)")
        lines.append(f"![Representative signal fragment](plots/class_{a:02d}_example.png)")
        lines.append("")
    lines.extend([
        "## Scientific limitations",
        "",
        "Each colored heatmap cell represents a tested component; gray = not tested.",
        "Per-class results are descriptive on a small held-out set. Do not",
        "infer globally characteristic physical features from these local interventions.",
        "Replacing a fragment with a mean/median does not necessarily create a",
        "realizable signal. Source class is the correct prediction in this audit.",
        "All class templates were fitted on training participants only.",
        "No statement about explanation fidelity to humans is made without",
        "independent expert evaluation.",
    ])
    (output/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(args):
    from beaconxai.datasets import load_npz_dataset, fit_channel_standardizer, apply_standardizer
    from scripts.run_component_conflict_benchmark import _anfis_features, _train_extratrees_local

    path = Path(args.npz_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    xt, yt, xv, yv = load_npz_dataset(path)
    with np.load(path) as d:
        subjects = d["subject_test"] if "subject_test" in d else np.full(len(xv),-1,dtype=int)
    if args.max_test and len(xv)>args.max_test:
        chosen = np.random.default_rng(args.seed).choice(len(xv),args.max_test,replace=False)
        xv,yv,subjects = xv[chosen],yv[chosen],subjects[chosen]
    mu, sigma = fit_channel_standardizer(xt)
    xt = apply_standardizer(xt,mu,sigma)
    xv = apply_standardizer(xv,mu,sigma)
    t0 = time.perf_counter()
    classifier = _train_extratrees_local(xt,yt,args.trees,.7,1)
    train_seconds = time.perf_counter()-t0
    bank = fit_prototype_bank(xt,yt)
    components = make_components(xv.shape[1:],args.time_bins,"pamap3")
    n_kinds = 2 if args.prototype_mode=="both" else 1
    n_select = args.budget // (2*n_kinds)
    if n_select<1 or n_select>len(components):
        raise ValueError("Q must allow at least one component and at most M components")
    n_classes = max(len(ALLOWED_ACTS),max(bank.labels)+1)
    cols = classifier.model.classes_
    def predict_many(batch):
        probs = classifier.model.predict_proba(_anfis_features(np.asarray(batch)))
        full = np.full((len(batch),n_classes),1e-12)
        full[:,cols.astype(int)] = probs
        return np.log(np.clip(full,1e-12,1))
    base = predict_many(xv)
    chosen,coverage = select_cases(xv,yv,base,bank,args.max_examples_per_class,args.seed+11)
    if not chosen:
        raise ValueError("No correctly classified samples available on selected test fold")
    rows = []
    first_cases = {}
    tested_pairs = 0
    baseline_evals = len(xv)
    extra_evals = 0
    for sample_id,source in chosen:
        signal = xv[sample_id]
        baseline = base[sample_id]
        competing = [b for b in bank.labels if b!=source]
        competing.sort(key=lambda b:float(baseline[b]),reverse=True)
        if args.rival_mode=="runner_up":competing=competing[:1]
        elif args.max_targets:competing=competing[:args.max_targets]
        cheap = class_cheap(signal,bank.mean[source],components)[2]
        selected = best_q(cheap,n_select)
        for rival in competing:
            result = explain_class_pair(signal,predict_many,bank,components,selected,
                        source,rival,q_max=args.budget,
                        prototype_mode=args.prototype_mode,baseline=baseline,taper=args.taper)
            if result["queries_additional"]>args.budget:
                raise AssertionError("Query budget violation")
            tested_pairs += 1
            extra_evals += result["queries_additional"]
            records = []
            for r in result["records"]:
                row = dict(r)
                row.update({"sample_id":int(sample_id), "subject":int(subjects[sample_id]),
                            "true_class":int(yv[sample_id]),
                            "source_name":NAMES.get(source,str(source)),
                            "rival_name":NAMES.get(rival,str(rival))})
                rows.append(row)
                records.append(row)
            if source not in first_cases:first_cases[source] = (sample_id,records)
        print(f"{NAMES.get(source,str(source))}: sample {sample_id}, "
              f"{len(competing)} target comparisons, {extra_evals} cumulative Q",flush=True)
    atlas,pairs = gather_summary(rows,bank.labels,coverage)
    output = Path(args.out_dir)
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"per_intervention.csv",rows)
    write_csv(output/"class_atlas.csv",atlas)
    write_csv(output/"pair_summary.csv",pairs)
    if not args.no_plots:
        for a in bank.labels:
            if a not in first_cases:continue
            sample_id,_ = first_cases[a]
            _plot_local_case(xv[sample_id],bank,components,
                             [r for r in rows if r["sample_id"]==sample_id],
                             a,first_cases[a][1][0]["rival"],output/"plots")
        _plot_atlas(rows,bank,components,output/"plots")
    render_report(atlas,{a:rr for a,(i,rr) in first_cases.items()},bank,args,output)
    manifest = {
        "dataset":"PAMAP2", "npz_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source":"official dataset prepared with scripts/preprocess_pamap2.py",
        "model":"ExtraTrees on _anfis_features", "trained_seconds":train_seconds,
        "test_subjects":sorted(set(map(int,subjects.tolist()))),
        "classes_with_prototypes":bank.labels, "n_training_classes":len(bank.labels),
        "n_class_passports_observed":sum(bool(r["audited_examples"]) for r in atlas),
        "n_test_examples_sampled":len(xv), "n_explained_examples":len(chosen),
        "n_class_pair_comparisons":tested_pairs, "q_max_per_pair":args.budget,
        "q_actual_per_pair":n_select*2*n_kinds, "selected_components_per_pair":n_select,
        "mean_median":args.prototype_mode, "taper":args.taper,
        "prototype_train_bytes":bank.bytes_used,
        "baseline_model_evaluations":baseline_evals,
        "additional_model_evaluations":extra_evals,
        "rival_mode":args.rival_mode, "correct_only":True,
        "limitations":["train mean/median signal may not be physically plausible",
          "model probability estimates may be uncalibrated",
          "class atlas is descriptive for sampled held-out examples only",
          "does not establish natural class boundaries or causal features"],
    }
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print("CLASS ATLAS RESULTS",output,flush=True)
    for row in atlas:
        print(f"{row['class_name']:<20s} n={row['audited_examples']:2d} "
              f"mean rival-delta-margin={row['target_mean_delta_margin']}",flush=True)
    print("Q actual",manifest["q_actual_per_pair"],"examples audited",len(chosen),
          "total extra model evals",extra_evals,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path",type=Path,default=Path("data/pamap2_acc9_w200s100_p095.npz"))
    p.add_argument("--out-dir",type=Path,default=Path("outputs_composite/class_counterfactual_atlas"))
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--max-test",type=int,default=0,help="0: all held-out examples")
    p.add_argument("--trees",type=int,default=120)
    p.add_argument("--time-bins",type=int,default=12)
    p.add_argument("--budget",type=int,default=16)
    p.add_argument("--max-examples-per-class",type=int,default=3)
    p.add_argument("--rival-mode",choices=["runner_up","all"],default="runner_up")
    p.add_argument("--max-targets",type=int,default=0,help="0: all if rival_mode=all")
    p.add_argument("--prototype-mode",choices=["mean","median","both"],default="both")
    p.add_argument("--taper",type=int,default=2)
    p.add_argument("--no-plots",action="store_true")
    run(p.parse_args())


if __name__=="__main__":
    main()
