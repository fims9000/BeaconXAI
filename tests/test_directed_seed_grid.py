import csv
import json
from pathlib import Path

import pytest

from scripts.summarize_directed_seed_grid import summarize_seed_grid


def _write(root,seed,selected=42):
    path=root/("raw_mlp_"+str(seed))
    path.mkdir(parents=True,exist_ok=True)
    meta=dict(model="raw_mlp",test_labels_never_read=True,
              test_windows=2,
              train_subject_groups=dict(
                  model_fit=[1,2],surrogate_fit=[3,4]))
    (path/"manifest.json").write_text(json.dumps(meta))
    with (path/"per_sample.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[
            "method","sample","subject","original_window_index",
            "has_three_counterexamples","counter_top3_overlap"])
        writer.writeheader()
        methods={
            "original_beacon_q16":(.5,.5),
            "signed_pairwise_q16":(.7,.6),
            "signed_ridge_q16":(.6,.5),
            "cheap_pairwise_q0":(.1,.2),
            "cheap_original_q0":(.0,.1),
        }
        for method,values in methods.items():
            for i,value in enumerate(values):
                writer.writerow(dict(method=method,sample=i,subject=i+101,
                                     original_window_index=selected+i,
                                     has_three_counterexamples=1,
                                     counter_top3_overlap=value))


def test_same_test_windows_descriptive_seed_report(tmp_path):
    _write(tmp_path,57)
    _write(tmp_path,58)
    result=summarize_seed_grid(tmp_path,"raw_mlp",[57,58])
    assert result["model"]=="raw_mlp"
    assert result["seeds_improved"]==2
    assert result["mean_difference_pp"]==pytest.approx(15.)
    assert result["not_independent_test_sets"]


def test_changed_test_windows_rejected(tmp_path):
    _write(tmp_path,57)
    _write(tmp_path,58,selected=99)
    with pytest.raises(ValueError,match="different outer"):
        summarize_seed_grid(tmp_path,"raw_mlp",[57,58])
