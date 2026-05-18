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

