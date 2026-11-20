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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uart-log",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    metrics=evaluate_uart_log(args.uart_log.read_text(encoding="utf8",errors="replace"))
    result=json.dumps(metrics,ensure_ascii=False,indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(result+"\n",encoding="utf8")
    print(result)


if __name__=="__main__":
    main()
