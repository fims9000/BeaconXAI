# BEACON: вставки в статью (каноничный пакет)

Источник чисел:  
`outputs_composite/table8_significance.csv`  
`outputs_composite/audit_panel_vs_scalar.csv`  
`outputs_composite/audit_policy_deltas.csv`  
`outputs_composite/audit_beacon_vs_uniform.csv`  
`outputs_composite/edge_resource_budget_table.csv`  
`outputs_composite/tinyxai_full_audit_cost.csv`

## 1) Hidden conflict (HAR/CNN)

- `loc@1`: 0.0703 (adaptive) vs 0.0371 (uniform), Δ=+0.0332, p=0.0132  
- `MRR`: 0.1363 (adaptive) vs 0.1015 (uniform), Δ=+0.0348, p=0.0124  
- `hit@3`: Δ=+0.0352, p=0.0776 (тенденция, без значимости 0.05)  
- `hit@5`: Δ=+0.0312, p=0.1714 (не значимо)

Формулировка:
> Adaptive BEACON improves early localization quality (loc@1, MRR) against uniform occlusion; hit@3/hit@5 remain positive but statistically non-significant at α=0.05.

## 2) Fuzzy / TAN политики (q=16)

Budget 10% (F1):
- scalar: **0.5047**
- panel(logit): **0.5701**
- fuzzy_policy: **0.5888**
- tan_policy: **0.5607**

Budget 20% (F1):
- scalar: **0.5949**
- panel(logit): **0.6013**
- fuzzy_policy: **0.5886**
- tan_policy: **0.6013**

Bootstrap deltas:
- 10%: `fuzzy - panel` Δ=+0.0187, 95% CI [-0.0181; 0.0907], p=0.24  
- 10%: `fuzzy - scalar` Δ=+0.0841, 95% CI [0.0331; 0.1514], p=0.00  
- 20%: `tan - panel` Δ=+0.0000, 95% CI [-0.0176; 0.0313], p=1.00  
- 20%: `fuzzy - panel` Δ=-0.0127, 95% CI [-0.0195; 0.0111], p=0.96

ECE:
- panel: 0.0274
- tan_policy: 0.0189
- fuzzy_policy: 0.1490
- scalar: 0.3916

## 3) BEACON vs uniform (panel/logit)

- ΔAUROC = +0.0052, 95% CI [-0.0037; 0.0173], p=0.12  
- ΔAUPRC = -0.0175, 95% CI [-0.0316; 0.0117], p=0.24  
- ΔF1@10 = +0.0280, 95% CI [-0.0298; 0.0490], p=0.84  
- ΔF1@20 = +0.0063, 95% CI [-0.0181; 0.0178], p=1.00

Формулировка:
> Для бинарной детекции в текущей постановке статистически значимого преимущества BEACON-панели над uniform-панелью не получено.

## 4) TinyXAI / edge cost (не MCU claim)

Из `edge_resource_budget_table.csv`:
- beacon_core_q8: p50=41.314 ms, p95=45.493 ms, RSS Δ=0.191 MB, state=0.250 KB
- beacon_core_q16: p50=66.908 ms, p95=70.538 ms, RSS Δ≈0 MB, state=0.531 KB
- beacon_adaptive_q8: p50=211.282 ms, p95=224.974 ms
- beacon_adaptive_q16: p50=239.807 ms, p95=255.315 ms

Из `tinyxai_full_audit_cost.csv`:
- full-audit стоимость доминируется `(Q+1)` вызовами модели;
- слой политики занимает доли процента от времени (≈0.005–0.024%).

Формулировка:
> Профиль относится к constrained CPU edge execution (~1.1 GHz, single core), не к микроконтроллерному deployment.

## 5) Что не заявлять

1. Не писать «portable = MCU/Arduino».  
2. Не заявлять значимость по `hit@3/hit@5`.  
3. Не заявлять, что BEACON лучше uniform в бинарной детекции в текущем HAR-блоке.  
4. Не путать process-level RSS baseline с инкрементальной памятью метода.

## 6) v6 Q1-gate

Для Q1-версии используем отдельный пакет:
`outputs_composite/part2_extended_v6/`.

Каноничные v6-файлы:
- `beacon_vs_uniform_q_sweep.csv`
- `bootstrap_deltas_v6.csv`
- `sensor_anomaly_localization.csv`
- `sensor_anomaly_bootstrap.csv`
- `tinyxai_full_audit_cost.csv`
- `manuscript_claim_registry_v6.csv`

Правило принятия:
> Вторая статья начинает тянуть на самостоятельную Q1-заявку только если в `manuscript_claim_registry_v6.csv` появляется хотя бы один сильный сигнал: `q1_signal=1` для BEACON против uniform в Q-sweep или для BEACON против `variance/profile_distance` в sensor-anomaly benchmark.

Текущий v6 результат:
- Q-sweep: сильный сигнал найден для `interp, Q=64`.
  - ΔAUROC = +0.0111, 95% CI [0.0049; 0.0179], p=0.002
  - ΔAUPRC = +0.0677, 95% CI [0.0098; 0.1234], p=0.016
  - ΔF1@10 = +0.0841, 95% CI [0.0100; 0.1461], p=0.026
- Sensor anomaly: сильного сигнала против `variance`, `energy`, `profile_distance` нет; эти простые baseline в текущем synthetic-fault блоке лучше BEACON.

HAR budget context (`M=72`):
- `Q=16` -> `Q/M=0.2222` (low-budget)
- `Q=32` -> `Q/M=0.4444` (medium-budget)
- `Q=64` -> `Q/M=0.8889` (**high-budget / near-full**)

Важно:
> Сигнал `interp, Q=64` корректно трактовать как результат high-budget audit regime. Его нельзя подавать как строгий ultra-low-budget claim.

CI для абсолютных метрик (`interp, Q=64`) вынесены в:
`outputs_composite/part2_extended_v6/table2_q64_metric_ci.csv`.

Sensor anomaly, `Q=64`, metric `hit@3`:
- spike: best comparator `variance_heuristic` = 0.7059, BEACON = 0.0735
- drift: best comparator `energy_heuristic` = 0.1795, BEACON = 0.0897
- stuck_sensor: best comparator `profile_distance` = 0.1273, BEACON = 0.0182
- dropout: best comparator `uniform_occlusion` = 0.0545, BEACON = 0.0182

Q64 cost envelope:
- direct constrained-CPU rows now exist for core Q=16/32/64 (`edge_resource_budget_q64_profile.csv`);
- `edge_resource_budget_q64_profile.csv` (same-profile measurement):
  - inference_only p50 = 0.757 ms;
  - core_q64 measured: p50=55.08 ms, p95=64.23 ms, mean_model_calls=60.43;
  - same-profile lower bound: `60.43 * 0.757 ≈ 45.8 ms` (consistent with measured ~55 ms).
- `tinyxai_full_audit_cost.csv` (separate simulation-based split):
  - uses inference baseline `2.897 ms` and gives conservative envelope:
  - model-call lower bound for Q=64: about 188 ms (`65 * 2.897 ms`);
  - core-style estimate: about 206 ms;
  - conservative linear upper envelope from Q16 core: about 268 ms.

Правило для статьи:
> Не смешивать в одной таблице числа из `edge_resource_budget_q64_profile.csv` и `tinyxai_full_audit_cost.csv` как будто это один и тот же runtime setup. Это два разных профиля (measured vs simulation-based envelope).

Neutralizer note:
> Проверялись `interp`, `zero`, `mean/channel_mean` и `class_mean`; статистически подтверждённый выигрыш над uniform получен только для `interp, Q=64`. Поэтому основной claim формулируется именно для этой конфигурации.

Как писать:
> In the v6 Q-sweep, BEACON with interpolation neutralization and Q=64 significantly improves AUROC, AUPRC, and F1@10 over uniform occlusion. In HAR this corresponds to a high-budget regime (Q/M=0.89 for M=72), not an ultra-low-budget setup. However, in the synthetic sensor-fault benchmark, simple zero-query statistics remain stronger than BEACON; therefore the anomaly block is reported as a boundary condition rather than as the main positive claim.

## 6.1) v7 next step (для строгого budget claim)

Минимальный перезапуск без переобучения модели:
- `time_bins=16` (`M=144`, тогда `Q=64 -> Q/M=0.4444`)
- `Q=16,32,64`
- `neutralizer=interp`
- panel policy = logit
- bootstrap = 1000

Цель:
> проверить, сохраняется ли supported-positive сигнал BEACON > uniform при более строгом бюджете (`Q/M <= 0.5`).

## 7) Part1 q-sweep guardrails (обязательно перед апдейтом Part1 текста)

Новые проверочные файлы:
- `outputs_composite/part1_localization_q_sweep/part1_best_claims_summary.csv`
- `outputs_composite/part1_localization_q_sweep/part1_allowed_claims.md`

Короткий итог:
- по `loc@1` подтверждённого универсального сигнала нет;
- по `PAMAP2` в текущем sweep supported-positive по BEACON>uniform нет;
- на `WISDM` есть локальные positive-сигналы на отдельных метриках ранжирования (`hit@3`, `hit@5`, `NRG`, `mean_rank`) и отдельных конфигурациях.

Правило для текста Part1:
1. Нельзя писать «устойчиво превосходит» в общем виде.
2. Можно писать только конфигурационно-ограниченные claims из `part1_allowed_claims.md`.
3. Обязательно указать sensitivity к `dataset / neutralizer / Q` и наличие negative cases.
