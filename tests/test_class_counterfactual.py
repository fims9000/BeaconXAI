"""Tests for class-prototype explanations, sign semantics, budget, atlas, and plots."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from beaconxai.class_counterfactual import (
    fit_prototype_bank, replace_component, explain_class_pair,
    describe_intervention, softmax,
)
from beaconxai.contrastive import make_components
from scripts.run_class_counterfactual_atlas import (
    gather_summary, summarize_components, select_cases, run, NAMES,
)


def data():
    rng = np.random.default_rng(5)
    x = rng.normal(scale=.12, size=(40, 96, 9)).astype(np.float32)
    y = np.r_[np.zeros(20,dtype=int),np.ones(20,dtype=int)]
    x[:20,16:32,:3] += 1.0
    x[20:,16:32,:3] -= 1.0
    return x, y, make_components((96,9), 12)


def logits(batch):
    z = batch[:,16:32,:3].mean(axis=(1,2))
    return np.column_stack((z,-z))


def test_train_only_class_mean_and_median():
    x,y,cs=data()
    b=fit_prototype_bank(x,y)
    assert b.labels==[0,1]
    assert b.counts=={0:20,1:20}
    np.testing.assert_allclose(b.mean[0],np.mean(x[:20],axis=0),atol=1e-6)
    np.testing.assert_allclose(b.median[1],np.median(x[20:],axis=0),atol=1e-6)
    assert b.bytes_used == 4*96*9*4
    assert b.get(1,"mean").shape==(96,9)
    with pytest.raises(ValueError): b.get(0,"mode")


def test_low_count_class_excluded_and_invalid_inputs():
    x,y,cs=data()
    with pytest.raises(ValueError,match="no training class"):
        fit_prototype_bank(x,y,min_count=21)

    with pytest.raises(ValueError):
        fit_prototype_bank(x[:1],y)
    invalid=x.copy(); invalid[0,0,0]=np.nan
    with pytest.raises(ValueError): fit_prototype_bank(invalid,y)


def test_patch_is_local_and_taper_does_not_modify_other_channels():
    x,y,cs=data();bank=fit_prototype_bank(x,y)
    sample=x[0];comp=cs[2]
    patched=replace_component(sample,comp,bank.mean[1],taper=0)
    sl=np.s_[comp.t0:comp.t1,comp.c0:comp.c1]
    np.testing.assert_allclose(patched[sl],bank.mean[1][sl])
    outside=np.ones_like(sample,dtype=bool);outside[sl]=False
    np.testing.assert_allclose(patched[outside],sample[outside])
    patched_taper=replace_component(sample,comp,bank.mean[1],taper=2)
    np.testing.assert_allclose(patched_taper[outside],sample[outside])
    assert np.isfinite(patched_taper).all()
    with pytest.raises(ValueError): replace_component(sample,comp,bank.mean[1],taper=-1)


def test_fixed_class_pair_signed_deltas_with_target_mean_and_median():
    x,y,cs=data();bank=fit_prototype_bank(x,y)
    sample=x[0].copy();base=logits(sample[None])[0]
    calls=[0]
    def tracked(batch):
        calls[0] += len(batch)
        return logits(batch)
    result=explain_class_pair(sample,tracked,bank,cs,[2],0,1,q_max=4,
                              prototype_mode="both",baseline=base,taper=0)
    assert result["queries_additional"]==calls[0]==4
    assert result["baseline_queries"]==0
    assert len(result["records"])==4
    recs={(r["prototype_class"],r["prototype_kind"]):r for r in result["records"]}
    assert recs[(1,"mean")]["delta_gap"]<0, "rival prototype should move margin toward rival"
    assert recs[(1,"mean")]["delta_p_rival"]>0
    assert recs[(0,"mean")]["source"]==0
    assert "оценка" in describe_intervention(recs[(1,"median")],NAMES)
    np.testing.assert_array_equal(sample,x[0])


def test_budget_and_invalid_pairs_not_silently_accepted():
    x,y,cs=data();bank=fit_prototype_bank(x,y)
    sample=x[0];base=logits(sample[None])[0]
    with pytest.raises(ValueError,match="budget"):
        explain_class_pair(sample,logits,bank,cs,[2],0,1,q_max=3,baseline=base)
    with pytest.raises(ValueError,match="distinct"):
        explain_class_pair(sample,logits,bank,cs,[2,2],0,1,q_max=8,baseline=base)
    with pytest.raises(ValueError,match="source and rival"):
        explain_class_pair(sample,logits,bank,cs,[2],0,0,q_max=4,baseline=base)
    with pytest.raises(ValueError,match="training prototypes"):
        explain_class_pair(sample,logits,bank,cs,[2],0,6,q_max=4,baseline=base)
    with pytest.raises(ValueError,match="prototype_mode"):
        explain_class_pair(sample,logits,bank,cs,[2],0,1,q_max=4,
                           baseline=base,prototype_mode="unrecognized")


def test_mean_only_allows_eight_components_for_q16():
    x,y,cs=data();bank=fit_prototype_bank(x,y)
    sample=x[0];base=logits(sample[None])[0]
    result=explain_class_pair(sample,logits,bank,cs,list(range(8)),0,1,
                              q_max=16,prototype_mode="mean",baseline=base)
    assert result["queries_additional"]==16
    assert len(result["records"])==16


def test_classes_without_examples_are_reported_as_unobserved():
    x,y,cs=data();bank=fit_prototype_bank(x,y)
    scores=logits(x[:4])
    samples,coverage=select_cases(x[:4],y[:4],scores,bank,per_class=2,seed=9)
    assert samples and set(c for i,c in samples)=={0}
    assert coverage[1]["eligible_correct_test"]==0
    rec={"source":0,"rival":1,"prototype_class":1,"prototype_kind":"mean",
         "sample_id":0,"delta_gap":-.1,"delta_p_rival":.2,
         "delta_p_source":-.2,"flip_to_rival":False}
    atlas,pairs=gather_summary([rec],bank.labels,coverage)
    assert atlas[1]["status"]=="no_correct_test_examples"
    assert atlas[1]["target_mean_delta_margin"] is None
    assert len(pairs)==1 and pairs[0]["mean_delta_gap"]==-.1


def test_real_runner_interface_on_toy_data_and_png_outputs(tmp_path,monkeypatch):
    import scripts.run_component_conflict_benchmark as src
    x,y,cs=data()
    val=x[[0,1,2,3,20,21,22,23]].copy()
    yv=y[[0,1,2,3,20,21,22,23]]
    path=tmp_path/"toy.npz"
    np.savez_compressed(path,x_train=x,y_train=y,x_test=val,y_test=yv,
                        subject_test=np.array([8]*4+[9]*4,dtype=int))

    class FakeModel:
        classes_=np.array([0,1],dtype=int)
        def predict_proba(self, f):
            z=f[:,0]
            p=1/(1+np.exp(-4*z))
            return np.stack([np.clip(p,1e-8,1-1e-8),np.clip(1-p,1e-8,1-1e-8)],axis=-1)
    class Adapter:
        model=FakeModel()
    monkeypatch.setattr(src,"_train_extratrees_local",lambda *args:Adapter())
    args=SimpleNamespace(npz_path=path,out_dir=tmp_path/"atlas",
                         seed=17,max_test=0,trees=10,time_bins=12,
                         budget=16,max_examples_per_class=2,rival_mode="runner_up",
                         max_targets=0,prototype_mode="both",taper=2,no_plots=False)
    run(args)
    out=args.out_dir
    meta=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    assert meta["q_actual_per_pair"]==16
    assert meta["n_explained_examples"]==4
    assert meta["n_class_passports_observed"]==2
    assert meta["additional_model_evaluations"]==4*16
    assert meta["prototype_train_bytes"]>0
    with (out/"per_intervention.csv").open(encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==64
    with (out/"class_atlas.csv").open(encoding="utf-8") as f:
        atlas=list(csv.DictReader(f))
    assert len(atlas)==2
    assert (out/"plots/class_00_heatmap.png").is_file()
    assert (out/"plots/class_01_example.png").is_file()
    assert (out/"report.md").is_file()


def test_directed_summary_requires_both_source_fall_and_rival_rise():
    # A smaller source score alone is NOT sufficient contrastive evidence.
    rows=[
        {"sample_id":1,"source":0,"rival":1,"prototype_class":1,
         "prototype_kind":"mean","group":2,"time_bin":3,
         "delta_p_source":-.15,"delta_p_rival":0.0,"delta_gap":-.5,
         "flip_to_rival":False},
        {"sample_id":2,"source":0,"rival":1,"prototype_class":1,
         "prototype_kind":"mean","group":2,"time_bin":3,
         "delta_p_source":-.20,"delta_p_rival":.12,"delta_gap":-.6,
         "flip_to_rival":True},
    ]
    coverage={
        0:{"total_test_labeled":2,"eligible_correct_test":2},
        1:{"total_test_labeled":0,"eligible_correct_test":0},
    }
    atlas,pairs=gather_summary(rows,[0,1],coverage)
    assert atlas[0]["target_mean_directed_fraction"]==.5
    assert pairs[0]["directed_fraction"]==.5
    comp=summarize_components(rows)
    assert len(comp)==1
    assert comp[0]["n_examples"]==2
    assert comp[0]["directed_fraction"]==.5
    assert atlas[1]["status"]=="no_correct_test_examples"


def test_user_text_must_report_other_class_switch():
    row={
        "source":0,"rival":1,"prototype_class":1,"prototype_kind":"mean",
        "group":0,"t0":10,"t1":20,
        "p_source_before":.6,"p_source_after":.2,
        "p_rival_before":.2,"p_rival_after":.1,
        "delta_p_source":-.4,"delta_p_rival":-.1,
        "delta_gap":-.3,"predicted_after":2,"flip_to_rival":False,
    }
    text=describe_intervention(row,{0:"walking",1:"running",2:"cycling"})
    assert "третий класс" in text
    assert "cycling" in text
    assert "НЕ подтверждено" in text
    row["delta_p_rival"] = .1
    row["predicted_after"] = 1
    text=describe_intervention(row,{0:"walking",1:"running",2:"cycling"})
    assert "модель переключилась" in text
    assert "обе оценки изменились" in text
