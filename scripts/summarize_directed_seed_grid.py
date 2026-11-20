#!/usr/bin/env python3
"""Descriptive robustness of signed BEACON under multiple train-role seeds.

Same nine held-out UCI HAR subjects, exact same 6 test windows/subject
when passed --test-sampling-seed=358, but classifier training subset,
surrogate training subset and model seed change. These are NOT
independent replications or a population significance test.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def summarize_seed_grid(root: Path, model: str, seeds: list[int]) -> dict:
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError("need a nonempty unique prespecified seed grid")
    rows=[]
    reference_indices=None
    for seed in seeds:
        folder=root/(model+"_"+str(seed))
        report=json.loads((folder/"manifest.json").read_text())
        if report["model"]!=model or not report["test_labels_never_read"]:
            raise ValueError("invalid or leaked seed result")
        with (folder/"per_sample.csv").open() as fp:
            allrows=list(csv.DictReader(fp))
        if not allrows:
            raise ValueError("empty seed result")
        indices={int(r["original_window_index"])
                 for r in allrows if r["method"]=="original_beacon_q16"}
        if reference_indices is None:
            reference_indices=indices
        elif indices!=reference_indices:
            raise ValueError("different outer test windows across seeds")
        eligible={int(r["sample"]) for r in allrows
                  if r["method"]=="original_beacon_q16" and
                  int(r["has_three_counterexamples"])}
        summary={}
        by_sub={}
        for method in ("original_beacon_q16","signed_pairwise_q16",
                       "signed_ridge_q16","cheap_pairwise_q0","cheap_original_q0"):
            subset=[r for r in allrows if r["method"]==method and
                    int(r["sample"]) in eligible]
            valid_subjects=sorted({int(r["subject"]) for r in subset})
            if set(valid_subjects)!={int(r["subject"]) for r in allrows
                                    if r["method"]=="original_beacon_q16"
                                    and int(r["sample"]) in eligible}:
                raise ValueError("unpaired subjects")
            by_sub[method]={
                s:float(np.mean([
                    float(r["counter_top3_overlap"]) for r in subset
                    if int(r["subject"])==s]))
                for s in valid_subjects}
            summary[method]=float(np.mean(list(by_sub[method].values()))) if valid_subjects else None
        b=summary["original_beacon_q16"]
        p=summary["signed_pairwise_q16"]
        diffs=np.asarray([
            by_sub["signed_pairwise_q16"][s]-by_sub["original_beacon_q16"][s]
            for s in sorted(by_sub["original_beacon_q16"])
        ])
        rows.append({
            "seed":seed,
            "test_windows":report["test_windows"],
            "eligible_windows":len(eligible),
            "eligible_subjects":len(diffs),
            "original_q16":b,"pairwise_q16":p,
            "difference_pp":float(100*(p-b)) if b is not None else None,
            "positive_subject_differences":int(np.sum(diffs>1e-12)),
            "negative_subject_differences":int(np.sum(diffs<-1e-12)),
            "equal_subject_differences":int(np.sum(abs(diffs)<=1e-12)),
            "fit_subjects":report["train_subject_groups"]["model_fit"],
            "surrogate_subjects":report["train_subject_groups"]["surrogate_fit"],
        })
    numbers=[r["difference_pp"] for r in rows if r["difference_pp"] is not None]
    return {
        "analysis":"seeds vary classifier/surrogate train roles, same fixed heldout windows",
        "model":model,"seeds":seeds,"per_seed":rows,
        "mean_difference_pp":float(np.mean(numbers)) if numbers else None,
        "min_difference_pp":float(np.min(numbers)) if numbers else None,
        "max_difference_pp":float(np.max(numbers)) if numbers else None,
        "seeds_improved":sum(v>0 for v in numbers),
        "not_independent_test_sets":True,
        "not_confirmatory_significance":True,
        "no_heldout_labels_used_for_inference":True,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--model",choices=("raw_mlp","extratrees"),required=True)
    parser.add_argument("--seeds",default="57,58,59")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    grid=[int(x) for x in args.seeds.split(",")]
    summary=summarize_seed_grid(args.root,args.model,grid)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,indent=2),encoding="utf8")
    print("SIGNED_SEED_GRID",json.dumps({
        "model":args.model,"seeds":grid,
        "differences_pp":[r["difference_pp"] for r in summary["per_seed"]],
        "mean_difference_pp":summary["mean_difference_pp"],
        "seeds_improved":summary["seeds_improved"],
        "not_independent":True
    }),flush=True)


if __name__=="__main__":
    main()
