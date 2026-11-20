#!/usr/bin/env python3
"""Validate *captured physical* ESP32-C3 UART self-test output.

This does NOT collect UART itself or simulate a physical measurement. The
caller must capture an actual target-board serial log separately, e.g.
pio device monitor. A synthetic test fixture must never be reported as a
hardware validation. Returns nonzero when any required check is absent.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def evaluate_uart_log(text: str) -> dict:
    if "BEACON_ADAPTIVE_V2_Q16_EXPERIMENTAL_ESP32C3" not in text:
        raise ValueError("not a BEACON ESP32-C3 original-adaptive-v2 serial log")
    m=re.search(r"Q16 callback_ok=(\d+) actual_model_calls=(\d+) class=(\d+) expected_class=(\d+)",text)
    s=re.search(r"SELFTEST class=(PASS|FAIL) candidate_set=(PASS|FAIL) top3_set=(PASS|FAIL)",text)
    timer=re.search(r"BOARD_EXPERIMENT one_window_q16_latency_us=(\d+)",text)
    memory=re.search(r"BOARD_EXPERIMENT free_heap_8bit=(\d+) minimum_free_heap_8bit=(\d+)",text)
    if not all((m,s,timer,memory)) or "DONE" not in text:
        raise ValueError("truncated or missing self-test, call count, time or heap telemetry")
    calls=int(m.group(2))
    duration=int(timer.group(1))
    free=int(memory.group(1))
    min_free=int(memory.group(2))
    if (m.group(1)!="1" or calls!=16 or int(m.group(3))!=int(m.group(4))
        or any(x!="PASS" for x in s.groups())
        or duration<1 or free<1 or min_free<1 or min_free>free):
        raise ValueError("ESP32-C3 Q16 audit failed or has invalid telemetry")
    return {
        "target":"ESP32-C3 compiled original adaptive_v2 Q16",
        "physical_board_log_required":True,
        "selftest_success":True,
        "model_evaluations":calls,
        "predicted_class":int(m.group(3)),
        "one_window_latency_us":duration,
        "free_heap_8bit_after_audit":free,
        "minimum_free_heap_8bit_observed":min_free,
        "not_peak_total_RAM":True,
        "energy_joules":None,
        "no_external_explanation_ground_truth":True,
        "n_independent_timing_windows":1,
    }


def evaluate_staged_uart_log(text: str) -> dict:
    """Validate the optional Q16/Q32 build's serial receipt format.

    A text log is caller-provided evidence, not cryptographic proof that
    physical hardware ran. A real capture is required for hardware claims.
    """
    banner="BEACON_ADAPTIVE_V2_STAGED_EXPERIMENTAL_ESP32C3"
    if banner not in text or text.count(banner)!=1:
        raise ValueError("not a unique staged ESP32-C3 serial log")
    # Only accept values after this exact boot banner, preventing accidental
    # mixture with an earlier baseline Q16 or stale staged run.
    section=text.split(banner,1)[1]
    if section.count("DONE")!=1:
        raise ValueError("truncated or mixed staged ESP32-C3 UART log")
    section=section.split("DONE",1)[0]
    m=re.search(
        r"^STAGED callback_ok=(\\d+) actual_model_calls=(\\d+) "
        r"expected_calls=(\\d+) class=(\\d+) expected_class=(\\d+)$",
        section,re.MULTILINE)
    checks=re.search(
        r"^SELFTEST class=(PASS|FAIL) candidate_set=(PASS|FAIL) "
        r"top3_set=(PASS|FAIL) budget=(PASS|FAIL) gap16=(PASS|FAIL)$",
        section,re.MULTILINE)
    timing=re.search(
        r"^BOARD_EXPERIMENT one_window_staged_latency_us=(\\d+)$",
        section,re.MULTILINE)
    memory=re.search(
        r"^BOARD_EXPERIMENT free_heap_8bit=(\\d+) "
        r"minimum_free_heap_8bit=(\\d+)$",
        section,re.MULTILINE)
    disclaimer=("NO_ENERGY_MEASUREMENT NO_FULL_PEAK_RAM_CERTIFICATION"
                in section)
    if not all((m,checks,timing,memory,disclaimer)):
        raise ValueError("truncated staged callback or missing telemetry")
    actual_q=int(m.group(2))
    expected_q=int(m.group(3))
    predicted=int(m.group(4))
    intended=int(m.group(5))
    latency=int(timing.group(1))
    heap_free=int(memory.group(1))
    heap_min=int(memory.group(2))
    if (m.group(1)!="1" or expected_q not in (16,32)
        or actual_q!=expected_q or predicted!=intended
        or any(value!="PASS" for value in checks.groups())
        or latency<=0 or heap_min<=0 or heap_free<heap_min):
        raise ValueError("staged ESP32-C3 functional self-test failed")
    return {
        "target":"ESP32-C3 optional train-calibrated Q16/Q32",
        "physical_board_log_required":True,
        "receipt_text_alone_does_not_prove_physical_origin":True,
        "selftest_success":True,
        "model_evaluations":actual_q,
        "expected_model_evaluations":expected_q,
        "predicted_class":predicted,
        "one_window_latency_us":latency,
        "free_heap_8bit_after_audit":heap_free,
        "minimum_free_heap_8bit_observed":heap_min,
        "not_peak_total_RAM":True,
        "energy_joules":None,
        "no_external_explanation_ground_truth":True,
        "n_independent_timing_windows":1,
        "optional_not_default_q16":True,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart-log",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--mode",choices=("baseline","staged"),default="baseline",
                        help="Expected compiled firmware variant (default: baseline Q16)")
    args=parser.parse_args()
    raw=args.uart_log.read_text(encoding="utf8",errors="replace")
    scorer=(evaluate_staged_uart_log if args.mode=="staged" else evaluate_uart_log)
    metrics=scorer(raw)
    result=json.dumps(metrics,ensure_ascii=False,indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(result+"\n",encoding="utf8")
    print(result)


if __name__=="__main__":
    main()
