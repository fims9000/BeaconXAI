import numpy as np

from beaconxai.experiments import evaluate_error_risk
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig


def _logits(x: np.ndarray) -> np.ndarray:
    s = float(x.sum())
    return np.array([s, -s], dtype=np.float64)


def _predict(x: np.ndarray) -> int:
    return int(np.argmax(_logits(x)))


def test_evaluate_error_risk_toy_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 8, 3)).astype(np.float64)
    y = np.array([0 if xx.sum() > 0 else 1 for xx in x], dtype=np.int64)

    cfg = BeaconConfig(q_max=16, k0=8, l_min=4, k_pos=3, k_neg=3)
    rows, local_rows, metrics = evaluate_error_risk(
        x_test=x,
        y_test=y,
        predict_fn=_predict,
        logits_fn=_logits,
        neutralizer=Neutralizer("zero"),
        base_cfg=cfg,
        q_values=[8, 16],
    )

    assert rows
    assert local_rows
    assert metrics
    assert any(m["method"] == "beacon_refine" for m in metrics)
