import numpy as np

from beaconxai.baselines import run_ig_topk_risk, run_saliency_topk_risk
from beaconxai.experiments import evaluate_error_risk
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig


def logits(x: np.ndarray) -> np.ndarray:
    w = np.array([[1.0, -0.5], [0.4, 0.8]])
    s = x @ w
    v = s.sum(axis=0)
    return v.astype(np.float64)


def grad_margin(x: np.ndarray, y_hat: int) -> np.ndarray:
    # For this toy linear model with fixed competitor 1-y_hat
    w = np.array([[1.0, -0.5], [0.4, 0.8]])
    alt = 1 - y_hat
    return (w[:, y_hat] - w[:, alt])[None, :].repeat(x.shape[0], axis=0)


def test_saliency_and_ig_return_scores():
    x = np.ones((8, 2), dtype=np.float64)
    cfg = BeaconConfig(q_max=16, k0=8, l_min=2)
    n = Neutralizer("zero")

    r1, q1 = run_saliency_topk_risk(logits, x, n, cfg, margin_gradient_fn=grad_margin)
    r2, q2 = run_ig_topk_risk(logits, x, n, cfg, margin_gradient_fn=grad_margin, steps=4)

    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    assert 1 <= q1 <= cfg.q_max
    assert 1 <= q2 <= cfg.q_max


def test_evaluate_has_new_methods():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8, 2)).astype(np.float64)
    y = np.array([int(np.argmax(logits(xx))) for xx in x], dtype=np.int64)

    cfg = BeaconConfig(q_max=16, k0=8, l_min=2)
    rows, _, _ = evaluate_error_risk(
        x_test=x,
        y_test=y,
        predict_fn=lambda xx: int(np.argmax(logits(xx))),
        logits_fn=logits,
        neutralizer=Neutralizer("zero"),
        base_cfg=cfg,
        q_values=[16],
        margin_gradient_fn=grad_margin,
    )

    methods = {r.method for r in rows}
    assert "saliency_topk" in methods
    assert "ig_topk" in methods
    assert "simple_counterfactual" in methods
