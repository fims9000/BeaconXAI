import numpy as np

from beaconxai.core import BeaconAudit
from beaconxai.neutralization import Neutralizer
from beaconxai.types import BeaconConfig


def _toy_logits(x: np.ndarray) -> np.ndarray:
    w0 = np.array([[1.0, 0.2], [0.8, 0.3], [0.5, 0.1], [0.2, 0.1]])
    w1 = -w0
    s0 = float((x * w0).sum())
    s1 = float((x * w1).sum())
    return np.array([s0, s1], dtype=np.float64)


def test_budget_and_outputs():
    x = np.array(
        [
            [1.0, 0.5],
            [0.8, 0.2],
            [0.2, 0.1],
            [0.1, 0.05],
        ],
        dtype=np.float64,
    )

    cfg = BeaconConfig(q_max=16, k0=4, l_min=2, k_pos=2, k_neg=2, q_frag_ratio=0.25)
    audit = BeaconAudit(model_logits=_toy_logits, neutralizer=Neutralizer("zero"), config=cfg)
    r = audit.audit(x)

    assert r.q_used <= cfg.q_max
    assert r.q_init == cfg.k0
    assert r.rho_b >= 1
    assert 0.0 <= r.risk_b <= 1.0
    assert isinstance(r.censored, bool)


def test_channel_time_partition_splits_channels_first():
    x = np.ones((4, 4), dtype=np.float64)
    def logits_any(z: np.ndarray) -> np.ndarray:
        s = float(z.sum())
        return np.array([s, -s], dtype=np.float64)
    cfg = BeaconConfig(
        q_max=2,
        k0=2,
        l_min=2,
        k_pos=2,
        k_neg=2,
        q_frag_ratio=0.25,
        partition_mode="channel_time",
    )
    r = BeaconAudit(model_logits=logits_any, neutralizer=Neutralizer("zero"), config=cfg).audit(x)
    leaves = r.metadata["leaf_components"]
    assert len(leaves) == 2
    # (cid, t0, t1, c0, c1): channel-first means full time range for both leaves.
    assert all(t0 == 0 and t1 == 4 for _, t0, t1, _, _ in leaves)
