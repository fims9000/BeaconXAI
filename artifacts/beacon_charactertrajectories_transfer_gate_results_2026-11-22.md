# BEACON: selective donor-transfer risk gate, independent UEA CharacterTrajectories

**Date:** 22 November 2026. **Status:** completed; negative result for the predeclared selective gate, positive *exploratory discrimination signal* for its TRAIN-fitted risk score. All code, CSV, JSON and provenance are in the research branch, not main.

## Integrity and preregistration

- [Frozen protocol before first data access](beacon_charactertrajectories_transfer_gate_preregistered_2026-11-22.md), commit a3ec537556b93fa3af8c196512ddd44b2e3d1708.
- [Pre-model technical amendment A1](beacon_charactertrajectories_amendment_A1_eq119_2026-11-22.md), commit 68a8b0b5ce1eb5ed1e5c493ec8eaa74be2b368e1: actual aeon 1.5.0 CharacterTrajectories _eq returned 1422×3×119 TRAIN and 1436×3×119 TEST, rather than anticipated 182. This discrepancy was found when the preprocessor failed, **before any classifier fitting or TEST inference**. The resampled/length-standardized archive is not the original variable-length pen trajectory.
- After successful data preprocessing, the initial full computation failed at mixed TRAIN/TEST CSV schema export, **after** the experiment had run. Only CSV column defaults were fixed. No algorithms, split, outcome labels, model, 35% threshold, query budget or gate quantile changed. Successful repeat: [GitHub Actions 45024767799](https://github.com/fims9000/BeaconXAI/actions/runs/45024767799).
- Focused CI: **34 tests passed** before the independent evaluation. All present Q≤16 checks passed, as did the complete model-input cost ledger. The workflow has four retained primary files: per_sample.csv, summary.csv, manifest.json, provenance.txt.

## Frozen model and distinct samples

UEA CharacterTrajectories _eq, 20 alphabetic-character classes, velocity/force trajectory channels represented in 119 equal-length time steps. Official data: 1422 TRAIN, 1436 TEST. The TRAIN subset was split by class and seed 57 into:

| TRAIN-only partition | Number of separate recording windows | Role |
|---|---:|---|
| Model fit and donor bank | 987 | TRAIN-only ExtraTrees, channel scaling, class profiles, TRAIN donors |
| Risk model fit | 202 | TRAIN-only supervised donor-transfer failure labels |
| Gate threshold calibration | 233 | Choose risk and matched-support gates **without** using TEST |
| Official TEST | 1436 (model-accuracy audit); 320 preselected for explanations | Final risk and donor-transfer audit |

These samples are disjoint by archive index, **not** verified independent pen-writers or sessions. One ExtraTrees, trees=120, seed=57, TRAIN-fit scaling, frozen before held-out TEST. Official full-TEST accuracy: **1397/1436 = 97.2841%**, above the preregistered 70% quality requirement.

For each x: A=argmax frozen prediction, B=runner-up, D1/D2=two closest model-confirmed TRAIN-fit donors for B; D3/D4=two further unused nearest TRAIN-fit donors. K2 compact search uses exactly the original search_sparse_robust_counterfactual core, 10 all-channel temporal blocks, taper=2, each edited donor support ≤35%, **one shared Q16 across D1+D2**. Successful mask S requires two directly measured target-class switches. D3/D4 are **not** used to choose S, rank candidate masks, compute risk features, or accept/reject an online explanation. Their observed outputs form TRAIN-only risk labels and independent TEST posthoc audit, with calls metered separately.

## Risk prediction and selection

A logistic regression (C=1, liblinear) is fit only on K2 successes of the 202 risk-fit windows: **113 certificates**, with **38 failures to transfer** to D3/D4. Nine features available *without extra online model calls*: verified edit fraction, minimum and range of measured D1/D2 target margins, selected group fraction, max source-to-donor MSE, between-donor MSE, real Q/16, original target margin, checked 1-deletion minimality. TRAIN-only StandardScaler; no feature uses D3/D4 or TEST labels. The TRAIN risk model coefficients/scaler are in manifest.json.

Of 233 separate threshold-calibration windows, K2 found **136 certificates**, and **44** of these failed D3/D4 transfer. Five candidate cutoffs were fixed as quantiles of risk-fit predicted scores; calibration chose the widest cutoff satisfying Bonferroni-adjusted one-sided Clopper–Pearson upper failure bound ≤0.5. It selected:

- risk threshold **0.9127290358700534**;
- accepted **136/136** calibration certificates, with 44 failed transfers;
- adjusted upper bound **0.4130504101969105** using alpha/5=0.02.

This satisfies the **preset, permissive 50% error bound**, but does *not* prove failure risk below an operationally attractive target such as 10–20%. The bound assumes exchangeability of calibration observations and future data, which participant information cannot establish.

The support-only control uses only a matching TRAIN-calibration acceptance quota, selecting support threshold **0.3025210084033613**. Both gates only filter an already-measured K2 certificate. They never change S, the target B, the classifier, or spend new online queries.

## Official TEST: all 320 cases in denominator

| Policy | Issued compact D1/D2 certificates | Failed on unseen D3/D4 | Conditional failure among issued | Successful same-mask transfer / all 320 |
|---|---:|---:|---:|---:|
| **Ungated K2 baseline** | 182/320 | 52 | **52/182 = 28.57%** | **130/320 = 40.625%** |
| **Frozen TRAIN risk gate** | 181/320 | 51 | **51/181 = 28.18%** | **130/320 = 40.625%** |
| Support-only gate | 182/320 | 52 | **28.57%** | **130/320 = 40.625%** |

The risk gate refused **exactly one additional mask**, which was indeed a D3/D4 failure, but **the number of successful transferred explanations remained identical: 130**. Thus the predeclared selection policy does not provide a meaningful coverage/robustness improvement on the held-out data. The full dataset failure rate must never be reported as an improvement without the denominator: conditional failure improves by only **0.39 percentage points**, and the total out-of-sample successful transfer count does not change.

Binomial 95% TEST interval for failure among issued masks (descriptive, window-level):
- K2: **[0.2213, 0.3572]**;
- risk gate: **[0.2175, 0.3533]**.

These are not person-level confidence intervals or causal guarantees.

## Explicitly exploratory posthoc discrimination diagnostics

After TEST was opened, an **unplanned descriptive** ROC-AUC analysis of the 182 K2-certified masks found:
- nine-feature TRAIN-fitted risk score vs D3/D4 failure label: **ROC-AUC = 0.7433**;
- support fraction alone vs failure label: **ROC-AUC = 0.5154**;
- risk score average precision for failures: 0.4916, support average precision: 0.2982.

Among the 182 K2-certified masks, ordered **posthoc by risk score**, score quartiles had failures:
- lowest predicted risk: 3/46 (6.5%);
- next: 10/46 (21.7%);
- next: 16/45 (35.6%);
- highest: 23/45 (51.1%).

**This trend is interesting but explicitly exploratory:** these quartile cutoffs come from examining TEST; they must not be deployed as if independently calibrated or counted as another preregistered positive result. The frozen calibration chose too wide a threshold, thus the useful score ordering was not translated into a useful policy. Any alternative risk tolerance, cutoff, score recalibration or active verification policy must be frozen on TRAIN and evaluated on **another held-out corpus**, not retested on CharacterTrajectories as independent.

## Complete cost accounting

Real model-input evaluations (not Python batch count):
- TRAIN-fit donor bank model screening: **320**;
- cached original f(x) on 202 risk-fit + 233 gate-calibration + 1436 TEST: **1871**;
- online K2 searches across risk-fit, gate-calibration and 320 TEST: **8568**, each within a *single* shared Q16 for D1/D2;
- offline TRAIN label / TEST posthoc D3/D4 checks: **862**;
- total independently reconciled: **11 621**.

By phase: risk-fit K2 online 2327, posthoc 226; calibration online 2630, posthoc 272; TEST online 3611, posthoc 364. All source/target classes chosen from original frozen classifier outputs; TEST labels used only for classifier accuracy. Every file retained in GitHub Actions. Data SHA256 **995ad72f75943d536b312e351101e84447226101b764098d461b403af4875994**; frozen numeric model SHA256 **91fc1529e230f5e847743758605ed2c7e3fcef647560254eb89a870fe6969d88**.

## Scientific conclusion and boundaries

We implemented **a real train-only risk model**, not merely a plan, and independently evaluated it at Q16. The *frozen gate* fails to produce a practically meaningful benefit; the *score* has informative risk ordering on one held-out corpus. These are distinct conclusions. Neither proves universal donor reliability, true causal effect, physically plausible pen movement, speaker/writer-independent calibration, or calibrated guarantee on other people.

Next mathematical hypothesis: learn **a better train-only decision policy for selective risk/coverage** (rather than a permissive ≤50% upper failure bound) and/or allocate some of the *same* Q16 to informative additional-donor checks instead of strictly filtering already-issued masks. The latter is qualitatively different: a gate can only remove certificates, whereas better query allocation could change the mask and improve the unconditional number of transferable explanations. Fix policy and evaluation on new data before looking at its TEST. Keep current K2 baseline and all negative results in the draft research branch.
