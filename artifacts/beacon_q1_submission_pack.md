# BEACON-XAI v6 Q1 Submission Pack

This file is the canonical source for manuscript claims after the v6 experiments.

## Canonical Artifact Sources

- `outputs_composite/part2_extended_v6/beacon_uniform_q_sweep.csv`
- `outputs_composite/part2_extended_v6/beacon_uniform_q_sweep_bootstrap.csv`
- `outputs_composite/part2_extended_v6/sensor_anomaly_localization.csv`
- `outputs_composite/part2_extended_v6/sensor_anomaly_bootstrap.csv`
- `outputs_composite/part2_extended_v6/manuscript_claim_registry_v6.csv`
- `outputs_composite/audit_panel_vs_scalar.csv`
- `outputs_composite/audit_policy_deltas.csv`
- `outputs_composite/tinyxai_full_audit_cost.csv`

## Main Positive Claim

The strongest v6 result is obtained for the interpolation neutralizer at `Q=64`.

BEACON panel vs uniform panel:

| Setting | Metric | BEACON | Uniform | Delta | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|
| interp, Q=64 | AUROC | 0.9419 | 0.9308 | +0.0111 | [0.0049; 0.0179] | 0.002 |
| interp, Q=64 | AUPRC | 0.5906 | 0.5229 | +0.0677 | [0.0098; 0.1234] | 0.016 |
| interp, Q=64 | F1@10 | 0.5794 | 0.4953 | +0.0841 | [0.0100; 0.1461] | 0.026 |
| interp, Q=64 | F1@20 | 0.6013 | 0.6013 | +0.0000 | [-0.0196; 0.0313] | 1.000 |

Manuscript wording:

> In the v6 Q-sweep, BEACON with interpolation neutralization and `Q=64` significantly improves AUROC, AUPRC, and F1@10 over uniform occlusion in the risk-panel detection setting. The gain is concentrated in the early-alert regime; at the 20% alert budget the two methods are equivalent.

## Fuzzy / TAN Policy Claim

Source: `outputs_composite/audit_panel_vs_scalar.csv` and `outputs_composite/audit_policy_deltas.csv`.

Budget 10%:

| Policy | F1 | ECE |
|---|---:|---:|
| scalar | 0.5047 | 0.3916 |
| panel(logit) | 0.5701 | 0.0274 |
| fuzzy_policy | 0.5888 | 0.1490 |
| tan_policy | 0.5607 | 0.0189 |

Budget 20%:

| Policy | F1 | ECE |
|---|---:|---:|
| scalar | 0.5949 | 0.3916 |
| panel(logit) | 0.6013 | 0.0274 |
| fuzzy_policy | 0.5886 | 0.1490 |
| tan_policy | 0.6013 | 0.0189 |

Bootstrap deltas:

- 10%: fuzzy vs scalar, Delta F1 = +0.0841, CI [0.0331; 0.1514], p=0.00.
- 10%: fuzzy vs panel, Delta F1 = +0.0187, CI [-0.0181; 0.0907], p=0.24.
- 20%: TAN vs panel, Delta F1 = +0.0000, CI [-0.0176; 0.0313], p=1.00.
- 20%: fuzzy vs panel, Delta F1 = -0.0127, CI [-0.0195; 0.0111], p=0.96.

Manuscript wording:

> Fuzzy and TAN policies are not used as accuracy winners over the linear panel. Instead, they provide compact interpretable/probabilistic alternatives. Fuzzy significantly improves over the scalar threshold at the 10% alert budget and is statistically indistinguishable from the linear panel; TAN matches the panel at the 20% alert budget and has the best ECE among the evaluated compact policies.

## Sensor-Anomaly Boundary Condition

Source: `outputs_composite/part2_extended_v6/sensor_anomaly_localization.csv` and `sensor_anomaly_bootstrap.csv`.

The synthetic fault benchmark includes `spike`, `drift`, `stuck_sensor`, and `dropout`.

Main observation:

- BEACON does not beat the simple zero-query baselines (`variance`, `energy`, `profile_distance`) in this block.
- At `Q=16`, BEACON beats uniform only on `hit@5`, but this is not enough for the central Q1 claim.
- At `Q=64`, uniform is better than BEACON on `loc@1`.

Manuscript wording:

> The sensor-anomaly benchmark is reported as a boundary condition. For direct synthetic faults such as spikes, drift, stuck sensors, and dropout, simple zero-query statistics remain stronger than BEACON. This indicates that BEACON should not be positioned as a universal anomaly detector; its main advantage appears in budgeted counter-evidence/risk-panel detection under interpolation neutralization.

## TinyXAI / Full-Audit Cost Claim

Source: `outputs_composite/tinyxai_full_audit_cost.csv`.

Core facts:

- Full-audit cost is dominated by `(Q+1)` model calls.
- The diagnostic policy layer occupies only a tiny share of runtime.
- The reported profile is constrained CPU edge execution, not MCU deployment.

Manuscript wording:

> The policy layer is lightweight, but the full BEACON audit is not free: its cost is dominated by repeated model calls. We therefore report the full audit envelope separately from the policy-layer footprint and avoid claiming microcontroller deployment.

## Claims Allowed

1. BEACON significantly improves risk-panel detection over uniform occlusion for `interp, Q=64` on AUROC, AUPRC, and F1@10.
2. Fuzzy is significantly better than scalar at the 10% alert budget.
3. Fuzzy is statistically indistinguishable from the linear panel at the 10% alert budget.
4. TAN matches the linear panel at the 20% alert budget and has strong calibration.
5. Full-audit cost is dominated by model calls; the policy layer is small.
6. Sensor-anomaly results define a limitation: simple statistics are stronger for direct synthetic faults.

## Claims Not Allowed

1. Do not claim that BEACON is universally better than uniform.
2. Do not claim BEACON beats variance/profile-distance on synthetic sensor anomalies.
3. Do not claim TinyXAI/Arduino/MCU deployment.
4. Do not claim `hit@3`/`hit@5` localization significance unless the cited table supports it.
5. Do not present fuzzy as better calibrated than TAN or logit; its ECE is higher.
6. Do not split into two Q1 articles unless the second is framed around the v6 Q=64 result and limitations.

## Recommended One-Article Story

Title direction:

> BEACON-XAI: Budgeted Counter-Evidence Audit and Compact Diagnostic Policies for Time-Series Models

Story:

> BEACON is a budgeted local audit method for time-series classifiers. It improves early risk-panel detection over uniform occlusion when enough query budget is available (`Q=64`) and interpolation neutralization is used. Compact policies (linear, fuzzy, TAN) convert BEACON-derived audit vectors into operational alert scores. The fuzzy and TAN policies provide interpretable/probabilistic alternatives to the linear panel. The sensor-anomaly benchmark shows the boundary of the approach: simple statistics remain stronger for direct synthetic faults, so BEACON should be positioned as a counter-evidence audit method rather than a universal anomaly detector.

