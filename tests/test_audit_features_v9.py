import numpy as np

from beaconxai.audit_features import extract_audit_vector


def test_audit_v9_feature_formulas_on_synthetic_deltas():
    deltas = np.array([-2.0, 1.0, -1.0, 0.0], dtype=np.float64)
    row = extract_audit_vector(
        beacon_result=None,
        margin=2.0,
        q_max=4,
        sample_id=0,
        label=1,
        is_hidden_conflict=1,
        method="unit",
        seed=42,
        deltas=deltas,
        rho_b_cost=0.5,
        frag_drop=0.25,
    )

    assert np.isclose(row["M_B_plus"], 1.0, atol=1e-9)
    assert np.isclose(row["M_B_minus"], 3.0, atol=1e-9)
    assert np.isclose(row["r_B_minus"], 0.75, atol=1e-9)
    assert np.isclose(row["CE_B"], 0.75, atol=1e-9)
    assert np.isclose(row["top1_delta"], 2.0, atol=1e-9)
    assert np.isclose(row["top3_sum_delta"], 4.0, atol=1e-9)
    assert np.isclose(row["top3_conflict_count"], 2.0, atol=1e-9)
    assert np.isclose(row["var_conflict"], 0.25, atol=1e-9)
    assert np.isclose(row["conflict_connectivity"], 0.0, atol=1e-9)
    assert np.isclose(row["delta_frag_proxy"], 1.5, atol=1e-9)
    assert np.isclose(row["r_cf"], 6.0, atol=1e-6)


def test_audit_v9_r_cf_is_clipped():
    row = extract_audit_vector(
        beacon_result=None,
        margin=1.0,
        q_max=2,
        sample_id=0,
        label=0,
        is_hidden_conflict=0,
        method="unit",
        seed=1,
        deltas=np.array([-100.0, 0.0], dtype=np.float64),
        rho_b_cost=1e-6,
    )
    assert np.isclose(row["r_cf"], 10.0, atol=1e-9)
