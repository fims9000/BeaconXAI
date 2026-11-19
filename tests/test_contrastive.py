import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from beaconxai.contrastive import (
    make_components, fit_class_profiles, contrastive_prior, neutralize_interp,
    selected_audit, prior_ranking, summarize_hotspots, margin,
)
from scripts.run_contrastive_q16 import (masking_coalitions, ridge_coefficients, mask_ranking, ranking_metrics)


def setup_toy():
    rng = np.random.default_rng(5)
    train = rng.normal(size=(30, 96, 9)) * 0.1
    labels = np.array([0]*15+[1]*15)
    train[:15, 32:48, :3] += 2.0
    train[15:, 32:48, :3] -= 2.0
    x = train[0].copy()
    return train, labels, x, make_components(x.shape, 12)


def toy_model(x):
    s = float(np.mean(x[32:48, :3]))
    return np.array([s, -s])


def toy_model_batch(xs):
    return np.asarray([toy_model(x) for x in xs])


def test_shapes_and_groups():
    cs = make_components((96, 9), 12)
    assert len(cs) == 36
    assert [(c.c0, c.c1) for c in cs[::12]] == [(0, 3), (3, 6), (6, 9)]
    assert [c.time_bin for c in cs[:12]] == list(range(12))
    with pytest.raises(ValueError): make_components((96, 6), 12)


def test_neutralization_only_selected_patch_changes():
    _,_,x,cs = setup_toy()
    c = cs[4]
    modified = neutralize_interp(x,c)
    mask = np.ones_like(x,dtype=bool)
    mask[c.t0:c.t1,c.c0:c.c1] = False
    np.testing.assert_allclose(modified[mask], x[mask])
    assert np.any(modified[~mask] != x[~mask])


def test_budget_pinned_classes_and_determinism():
    tr,lab,x,cs = setup_toy()
    prof = fit_class_profiles(tr,lab)
    calls = [0]
    def f(z):
        calls[0] += 1
        return toy_model(z)
    baseline = toy_model(x)
    r = selected_audit(x,f,prof,cs,budget=16,seed=123,logits0=baseline)
    assert calls[0] == 16
    assert r.calls == 16
    assert (r.source_class,r.rival_class)==(0,1)
    assert len(set(r.component_ids))==16
    assert sum(np.isfinite(r.scores))==16
    r2 = selected_audit(x,toy_model,prof,cs,budget=16,seed=123,logits0=baseline)
    assert r.component_ids == r2.component_ids
    assert r.ranked() == r2.ranked()


def test_exploration_and_no_data_leakage():
    tr,lab,x,cs = setup_toy()
    prof = fit_class_profiles(tr,lab)
    prior = contrastive_prior(x,0,1,prof,cs)
    assert prior.shape==(36,)
    assert np.isfinite(prior).all()
    assert prior_ranking(prior)[0] in [4,5]
    del tr,lab
    # Audit requires only class-conditional aggregates; no test labels or target injection index.
    result=selected_audit(x,toy_model,prof,cs,16,9,logits0=toy_model(x))
    assert isinstance(result.component_ids[0],int)


def test_constant_input_gap_and_flip_are_distinct():
    tr,lab,x,cs=setup_toy()
    prof=fit_class_profiles(tr,lab)
    r=selected_audit(x,toy_model,prof,cs,16,13,logits0=toy_model(x))
    assert np.any(np.abs(r.signed_gap_drops) > 0)
    assert margin(toy_model(x),0,1)>0
    assert len(r.target_flips)==16


def test_zero_query_and_conservative_unqueried():
    assert ranking_metrics([0,1,2],5,36)['hit3']==0
    assert ranking_metrics([0,1,2],5,36)['rank']==37
    assert ranking_metrics([0,1,2],1,36)['hit3']==1


def test_kernelshap_and_rise_fixed_queries_underdetermined():
    for name in ('rise_like','kernelshap_like'):
        masks,w=masking_coalitions(name,36,16,np.random.default_rng(9))
        assert masks.shape==(16,36)
        assert np.all((masks==0)|(masks==1))
        if name=='kernelshap_like':
            assert (masks[0]==0).all()
            assert (masks[1]==1).all()
        coeff=ridge_coefficients(masks,np.arange(16,dtype=float),w,1e-3)
        assert len(coeff)==36 and np.isfinite(coeff).all()


def test_mask_rank_wont_call_more_than_q():
    _,_,x,cs=setup_toy()
    calls=[0]
    def f(batch):
        calls[0] += len(batch)
        return toy_model_batch(batch)
    a=0;b=1;gap=margin(toy_model(x),a,b)
    for method in ('rise_like','kernelshap_like'):
        rank=mask_ranking(x,f,cs,a,b,gap,method,16,seed=11)
        assert len(set(rank))==36
    assert calls[0]==32


def test_hotspots_not_claimed_as_rules():
    tr,lab,x,cs=setup_toy()
    prof=fit_class_profiles(tr,lab)
    runs=[selected_audit(x,toy_model,prof,cs,16,s,logits0=toy_model(x)) for s in (1,2,3,4,5)]
    rows=summarize_hotspots(runs,cs,min_count=2)
    assert rows
    assert all('mean_gap_drop' in row and 'count' in row for row in rows)


def test_full_pipeline_with_fake_pamap_like_fixture(monkeypatch, tmp_path):
    """End-to-end I/O integration without claiming a real dataset experiment."""
    import sys
    import types
    from types import SimpleNamespace
    from scripts.run_contrastive_q16 import real_run
    rng = np.random.default_rng(19)
    train = rng.normal(0,.2,(40,96,9))
    train[:20,32:48,:3] += .9
    train[20:,32:48,:3] -= .9
    labels = np.array([0]*20+[1]*20)
    test = train[[1,3,5,7,9,21,23,25,27,29]].copy()
    ytest = labels[[1,3,5,7,9,21,23,25,27,29]].copy()
    ds = types.ModuleType("beaconxai.datasets")
    ds.load_npz_dataset = lambda p: (train,labels,test,ytest)
    ds.fit_channel_standardizer = lambda x: (np.zeros(9),np.ones(9))
    ds.apply_standardizer = lambda x,mu,sd: x
    monkeypatch.setitem(sys.modules,"beaconxai.datasets",ds)
    ext = types.ModuleType("scripts.run_component_conflict_benchmark")
    def inject(x,y,gs,ts,conflict_ratio,seed):
        z=x.copy();present=np.zeros(len(x),dtype=int);comp=np.full(len(x),-1,dtype=int)
        for i in range(int(round(conflict_ratio*len(x)))):
            j = next(j for j in range(len(x)) if y[j]!=y[i])
            g=i%len(gs); b=(2*i)%len(ts)
            c0,c1=gs[g];t0,t1=ts[b]
            z[i,t0:t1,c0:c1]=x[j,t0:t1,c0:c1]
            present[i]=1;comp[i]=g*len(ts)+b
        return z,present,comp
    ext._inject_component_conflicts=inject
    def feat(arr):
        return np.mean(arr[:,32:48,:3],axis=(1,2))[:,None]
    ext._anfis_features=feat
    class Model:
        def predict_proba(self, f):
            s=1/(1+np.exp(-2*f[:,0]))
            return np.c_[s,1-s]
    class Adapter:
        model=Model()
        def logits(self, x): return np.log(self.model.predict_proba(feat(x[None]))[0])
    ext._train_extratrees_local=lambda *a: Adapter()
    monkeypatch.setitem(sys.modules,"scripts.run_component_conflict_benchmark",ext)
    datapath=tmp_path/'dummy.npz';datapath.touch()
    args=SimpleNamespace(npz_path=datapath,max_test=100,seed=42,trees=120,time_bins=12,mode="both",source_class=None,target_class=None,
                         group_mode="pamap3",conflict_ratio=.5,budget=16,repeats=2,
                         explore_fraction=.25,out_dir=tmp_path/'output')
    real_run(args)
    import csv, json
    payload=json.loads((args.out_dir/'summary.json').read_text(encoding='utf-8'))
    assert payload['n_test_injected']==5
    assert len(payload['results'])==5
    assert {r['method'] for r in payload['results']}=={
        'contrastive_q16','uniform_q16','prior_q0','rise_like_q16','kernelshap_like_q16'}
    for r in payload['results']:
        assert 0 <= r['hit3'] <= 1
        assert r['M']==36
    with (args.out_dir/'per_sample.csv').open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==5*2*5
    assert (args.out_dir/"boundary_summary.json").exists()
    assert all(float(row['budget_extra']) in (0,16) for row in rows)


def test_targeted_boundary_probe_counts_all_queries_and_can_flip():
    from beaconxai.contrastive import targeted_prototype_probe
    tr,lab,x,cs=setup_toy();profiles=fit_class_profiles(tr,lab)
    calls=[0]
    def f(z):
        calls[0] += 1
        return toy_model(z)
    r=targeted_prototype_probe(x,f,profiles,cs,16,seed=11,logits0=toy_model(x),taper_points=0)
    assert calls[0]==16
    assert (r.source_class,r.target_class)==(0,1)
    assert len(set(r.checked))==16
    assert r.best_component in r.checked
    assert r.extra_model_calls==16
    # strongest toy intervention can change the pinned two-class gap
    assert r.best_target_gap_reduction > 0


def test_targeted_boundary_probe_uniform_same_operator():
    from beaconxai.contrastive import targeted_prototype_probe
    tr,lab,x,cs=setup_toy();profiles=fit_class_profiles(tr,lab)
    a=targeted_prototype_probe(x,toy_model,profiles,cs,16,seed=11,logits0=toy_model(x))
    b=targeted_prototype_probe(x,toy_model,profiles,cs,16,seed=11,selection="uniform",logits0=toy_model(x))
    assert a.extra_model_calls==b.extra_model_calls==16
    assert len(a.gap_after)==len(b.gap_after)==16
