import pytest

from scripts.check_esp32c3_audit_log import evaluate_uart_log,evaluate_staged_uart_log


def valid_log():
    return """BEACON_ADAPTIVE_V2_Q16_EXPERIMENTAL_ESP32C3
Q16 callback_ok=1 actual_model_calls=16 class=3 expected_class=3
SELFTEST class=PASS candidate_set=PASS top3_set=PASS
BOARD_EXPERIMENT one_window_q16_latency_us=54213
BOARD_EXPERIMENT free_heap_8bit=221421 minimum_free_heap_8bit=217121
NO_ENERGY_MEASUREMENT NO_FULL_PEAK_RAM_CERTIFICATION
DONE
"""


def test_valid_real_log_format_parses_but_does_not_claim_energy():
    record=evaluate_uart_log(valid_log())
    assert record["selftest_success"] is True
    assert record["model_evaluations"]==16
    assert record["one_window_latency_us"]==54213
    assert record["minimum_free_heap_8bit_observed"]==217121
    assert record["energy_joules"] is None
    assert record["n_independent_timing_windows"]==1


@pytest.mark.parametrize("replacement",[
    "candidate_set=FAIL",
    "top3_set=FAIL",
    "actual_model_calls=15",
    "callback_ok=0",
    "expected_class=1",
    "one_window_q16_latency_us=0",
    "minimum_free_heap_8bit=299999",
])
def test_hardware_acceptance_rejects_failed_report(replacement):
    original={
        "candidate_set=FAIL":"candidate_set=PASS",
        "top3_set=FAIL":"top3_set=PASS",
        "actual_model_calls=15":"actual_model_calls=16",
        "callback_ok=0":"callback_ok=1",
        "expected_class=1":"expected_class=3",
        "one_window_q16_latency_us=0":"one_window_q16_latency_us=54213",
        "minimum_free_heap_8bit=299999":"minimum_free_heap_8bit=217121",
    }[replacement]
    with pytest.raises(ValueError):
        evaluate_uart_log(valid_log().replace(original,replacement))


def test_truncated_log_rejected():
    with pytest.raises(ValueError,match="truncated"):
        evaluate_uart_log(valid_log().replace("DONE",""))


def test_non_board_log_rejected():
    with pytest.raises(ValueError,match="not a BEACON"):
        evaluate_uart_log("ALL PASS 16 DONE")


def staged_valid_log(q=32):
    return f"""BEACON_ADAPTIVE_V2_STAGED_EXPERIMENTAL_ESP32C3
STAGED callback_ok=1 actual_model_calls={q} expected_calls={q} class=2 expected_class=2
SELFTEST class=PASS candidate_set=PASS top3_set=PASS budget=PASS gap16=PASS
BOARD_EXPERIMENT one_window_staged_latency_us=78391
BOARD_EXPERIMENT free_heap_8bit=202123 minimum_free_heap_8bit=199001
NO_ENERGY_MEASUREMENT NO_FULL_PEAK_RAM_CERTIFICATION
DONE
"""


@pytest.mark.parametrize("q",[16,32])
def test_staged_accepts_only_valid_observed_q16_or_q32_receipt(q):
    record=evaluate_staged_uart_log(staged_valid_log(q))
    assert record["selftest_success"]
    assert record["model_evaluations"]==q
    assert record["optional_not_default_q16"]
    assert record["receipt_text_alone_does_not_prove_physical_origin"]
    assert record["energy_joules"] is None
    assert record["n_independent_timing_windows"]==1


@pytest.mark.parametrize("before,after",[
    ("callback_ok=1","callback_ok=0"),
    ("actual_model_calls=32","actual_model_calls=24"),
    ("expected_calls=32","expected_calls=24"),
    ("class=2 expected_class=2","class=1 expected_class=2"),
    ("candidate_set=PASS","candidate_set=FAIL"),
    ("top3_set=PASS","top3_set=FAIL"),
    ("budget=PASS","budget=FAIL"),
    ("gap16=PASS","gap16=FAIL"),
    ("one_window_staged_latency_us=78391","one_window_staged_latency_us=0"),
    ("minimum_free_heap_8bit=199001","minimum_free_heap_8bit=299001"),
    ("NO_ENERGY_MEASUREMENT NO_FULL_PEAK_RAM_CERTIFICATION",""),
    ("DONE",""),
])
def test_staged_rejects_missing_failed_or_inconsistent_receipt(before,after):
    with pytest.raises(ValueError):
        evaluate_staged_uart_log(staged_valid_log().replace(before,after))


def test_staged_parser_refuses_baseline_or_mixed_boots():
    with pytest.raises(ValueError):
        evaluate_staged_uart_log(valid_log())
    with pytest.raises(ValueError):
        evaluate_staged_uart_log(staged_valid_log()+staged_valid_log())
    with pytest.raises(ValueError):
        evaluate_staged_uart_log(staged_valid_log()+"DONE\n")
