import numpy as np
import pandas as pd

from scripts.make_audit_panel_tables import make_fuzzy_panel_score
from scripts.run_har_hidden_conflict_tan import TANModel


def test_fuzzy_panel_score_shape_and_finite():
    rng = np.random.default_rng(0)
    n = 64
    df = pd.DataFrame(
        {
            "m_neg": rng.normal(size=n),
            "M_B_minus": rng.normal(size=n),
            "CE_B": rng.normal(size=n),
            "r_B_minus": rng.normal(size=n),
            "rho_B_cost": rng.normal(size=n),
            "frag_drop": rng.normal(size=n),
            "is_error": rng.integers(0, 2, size=n),
        }
    )
    s = make_fuzzy_panel_score(df)
    assert s.shape == (n,)
    assert np.all(np.isfinite(s))


def test_tan_model_toy_binary():
    rng = np.random.default_rng(1)
    x0 = rng.normal(loc=-1.0, scale=0.5, size=(80, 4))
    x1 = rng.normal(loc=1.0, scale=0.5, size=(80, 4))
    X = np.vstack([x0, x1])
    y = np.array([0] * len(x0) + [1] * len(x1), dtype=np.int64)

    model = TANModel(n_bins=4, alpha=1.0).fit(X, y)
    p = model.predict_proba(X)

    assert p.shape == (len(X), 2)
    assert np.all(np.isfinite(p))
    assert np.allclose(np.sum(p, axis=1), 1.0, atol=1e-6)
