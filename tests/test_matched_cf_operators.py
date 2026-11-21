"""Small deterministic checks for matched UCI counterfactual reporting."""
import numpy as np
import pytest
from scripts.run_matched_cf_operators import macro_summary


def make(subject, op, method, success, calls, minimal=0, robust=0):
    return {
        "operator":op, "method":method, "budget":16,
        "subject":subject, "verified":int(success),
        "additional_calls":calls,
        "locally_minimal":int(minimal),
        "donor_checks":2 if success else 0,
        "donor_flips":2 if robust else 0,
        "logical_components":2 if success else None,
        "three_axis_regions":4 if success else None,
        "gravity_roughness_ratio":1.0 if success else None,
    }


def test_match_summary_does_not_weight_high_volume_subject():
    rows = ([make(1,"gravity_preserved","path",1,5)]*10 +
            [make(2,"gravity_preserved","path",0,14)] +
            [make(3,"gravity_preserved","path",1,8,1,1)])
    result = macro_summary(rows)
    assert len(result)==1
    s=result[0]
    assert s["n_windows"]==12
    assert s["n_subjects"]==3
    assert s["n_verified"]==11
    assert s["subject_macro_success"]==pytest.approx(2/3)
    assert s["n_verified_local_minimality"]==1
    assert s["n_robust_two_donors"]==1
    assert s["median_gravity_roughness_ratio_if_valid"]==1


def test_empty_is_rejected_and_zero_verified_has_no_fake_physical_metric():
    assert macro_summary([])==[]
    rows=[make(1,"independent","pair",0,14)]
    result=macro_summary(rows)[0]
    assert result["n_verified"]==0
    assert result["median_gravity_roughness_ratio_if_valid"] is None
    assert result["mean_physical_regions_if_valid"] is None


def test_operators_never_mix_in_summary():
    rows=[make(1,"independent","path",1,8),
          make(1,"gravity_preserved","path",0,14),
          make(1,"independent","pair",0,14)]
    result=macro_summary(rows)
    assert len(result)==3
    assert [(r["operator"],r["method"]) for r in result] == [
        ("gravity_preserved","path"),("independent","pair"),("independent","path")]
