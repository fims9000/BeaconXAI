# BEACON-XAI v9 Code Audit (2026-05-20)

## Scope
- `beaconxai/core.py`
- `beaconxai/audit_features.py`
- `scripts/train_tan_improved.py`
- `scripts/train_fuzzy_improved.py`
- `scripts/run_part2_extended.py`
- tests

## Executed checks
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python -m pytest -q tests/test_beacon_core.py tests/test_fuzzy_tan.py tests/test_audit_features_v9.py`
- Re-run on existing bundle:
  - `scripts/train_tan_improved.py --bundle-dir outputs_composite/part2_q64_features_v9_n600 --compare-to logit_panel --target ce --n-boot 1000`
  - `scripts/train_fuzzy_improved.py --bundle-dir outputs_composite/part2_q64_features_v9_n600 --compare-to logit_panel --target ce --n-boot 1000`

## Findings

### 1) Bootstrap reproducibility used `hash()` (critical)
- **Where:** `scripts/train_tan_improved.py`, `scripts/train_fuzzy_improved.py`
- **Issue:** seed offset used `abs(hash(mname))`, but Python hash is process-randomized; p-values/CI were not strictly reproducible.
- **Fix:** replaced with fixed deterministic map:
  - `{"delta_auroc": 101, "delta_auprc": 211, "delta_f1_10": 307}`

### 2) Baseline logit scaling mismatch (major)
- **Where:** `scripts/train_tan_improved.py`, `scripts/train_fuzzy_improved.py`
- **Issue:** logit baseline trained on raw mixed-scale features (can bias comparison vs TAN/fuzzy preprocessing).
- **Fix:** baseline changed to `make_pipeline(StandardScaler(), LogisticRegression(...))`.

### 3) No optional normalized delta path (major vs spec)
- **Where:** `beaconxai/core.py`, `beaconxai/types.py`
- **Issue:** audit demanded optional `Δ_norm = Δ / (|m0|+eps)`. Code had only raw delta.
- **Fix:** added `BeaconConfig.normalize_delta: bool = False`; when enabled, `_delta`/`_delta_switch` return normalized delta.

### 4) `r_cf` unbounded instability (minor)
- **Where:** `beaconxai/audit_features.py`
- **Issue:** `r_cf = M_B^- / (rho + eps)` could explode on tiny `rho`.
- **Fix:** clipped to `[0, 10]` (`R_CF_MAX = 10.0`).

### 5) `conflict_connectivity` normalization (minor)
- **Where:** `beaconxai/audit_features.py`
- **Issue:** previous formula used per-node neighbor flag ratio; weakly interpretable.
- **Fix:** changed to edge-based ratio:
  - `edge_hits / max(conf_cnt - 1, 1)`.

## Remaining limitations (not fully fixable in-place)

1. **Sign convention mismatch with some docs**
   - In code: `delta = m0 - m1`, so `delta < 0` is counter-evidence/conflict.
   - Some notes/texts describe opposite sign.
   - Recommendation: unify manuscript notation or document sign map explicitly.

2. **`CE_B` semantics differ by source**
   - For BEACON rows: from `counter_evidence_gain` (top-`k_neg` mechanism).
   - For uniform/adaptive rows: proxy `M_B^- / q_max`.
   - This is consistent with current budget design, but not mathematically identical.

3. **MDLP not installed**
   - TAN falls back to quantile bins (warning shown).
   - Recommendation: install `mdlp` package for strict MDLP runs.

4. **Connectivity still linearized**
   - Uses flat index adjacency, not full time×channel graph adjacency.
   - Recommendation: pass component topology metadata to `extract_audit_vector` and compute graph adjacency explicitly.

## New tests added
- `tests/test_audit_features_v9.py`
  - synthetic checks for `M_B±`, `r_B^-`, `CE_B`, top-k stats, `var_conflict`, `delta_frag_proxy`, `r_cf` clipping.
- `tests/test_beacon_core.py`
  - new `test_delta_normalization_toggle` (raw vs normalized delta path).

All tests pass: `10 passed`.

## Post-fix Q64 rerun (existing `v9 n600` bundle)

### TAN vs logit
- `ΔAUROC = +0.0248`, CI `[-0.0262, +0.0729]`, `p=0.322`
- `ΔAUPRC = +0.0224`, CI `[-0.0480, +0.0969]`, `p=0.572`
- `ΔF1@10 = +0.0535`, CI `[-0.0556, +0.1600]`, `p=0.202`

### Fuzzy vs logit
- `ΔAUROC = +0.0581`, CI `[-0.0187, +0.1311]`, `p=0.128`
- `ΔAUPRC = +0.0337`, CI `[-0.0414, +0.1157]`, `p=0.392`
- `ΔF1@10 = +0.0504`, CI `[-0.0328, +0.1471]`, `p=0.156`

## Conclusion
- Hidden implementation/reproducibility issues were present and fixed.
- After fixes, deltas moved positive, but **still not statistically significant** at `n=600`.
- Current safe claim remains: logit is primary policy; TAN/fuzzy are promising alternatives but without confirmed superiority yet.
