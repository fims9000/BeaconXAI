# Handoff для научного анализатора (BEACON-XAI, draft v12)

## 1) Контекст
- Part 1 (локализация) готова.
- Part 2 (risk-panel) в ревизии: фокус на честный claim-safe вывод + дополнительные cross-dataset проверки.

## 2) Подтвержденные результаты (готовые)

### 2.1 Политики (v11, 12 бандлов)
Источник:
- `artifacts/v11_full_summary.md`
- `outputs_composite/v11_full_summary.csv`

Ключевой факт:
- Для `TAN`: **0/12 stable supported_positive** по строгому правилу.
- Для `Fuzzy`: **0/12 stable supported_positive**.
- `Fuzzy` в целом хуже `logit`.
- `TAN` даёт локальные плюсы на отдельных WISDM-конфигурациях, но без устойчивого переноса.

Итог по политикам:
- Практический baseline: `logit-panel`.
- `TAN/Fuzzy` — вторичные альтернативы (без универсального quality-claim).

### 2.2 Дополнительные датасеты
Готовы и включены в v11:
- `UWave`
- `Heartbeat`
- `SelfRegulationSCP1`

Пути:
- `outputs_composite/v11_uwave_summary.csv`
- `outputs_composite/v11_heartbeat_summary.csv`
- `outputs_composite/v11_selfregulation_summary.csv`

## 3) Текущий прогон (v12, в работе)

Цель:
- Проверка `BEACON vs uniform` на 3 датасетах (`HAR/PAMAP2/WISDM`) и бюджетах `Q=16,32,64`
- + сравнение `adaptive_v2` против `uniform`.

Скрипт:
- `scripts/benchmark_beacon_vs_uniform.py`

Выход:
- `outputs_composite/v12_beacon_vs_uniform_full/`

Мониторинг:
```bash
ps -p $(cat outputs_composite/v12_beacon_vs_uniform_full/logs/run.pid) -o pid,etime,pcpu,pmem,cmd
tail -f outputs_composite/v12_beacon_vs_uniform_full/logs/run.log
```

Промежуточно (без финального bootstrap по всем бандлам):
- HAR и PAMAP2 для `Q=16/32/64` уже рассчитаны,
- WISDM в процессе.

## 3.1) Новый перспективный протокол: early-stopping (smoke)

Добавлен скрипт:
- `scripts/run_early_stopping_equal_budget.py`

Smoke (HAR, `n_total=240`, `q_max=64`, `tol=0.005`, `min_q=10`):
- `q_mean_early = 10.83` (снижение бюджета ~83% относительно `q_max=64`)
- equal-budget baseline: `uniform` с `Q=11`
- `ΔAUROC = +0.0026`
- `ΔAUPRC = +0.0249`
- `ΔF1@10 = 0.0`
- bootstrap пока незначим (smoke-масштаб, `n_boot=200`)

Файлы:
- `outputs_composite/early_stop_har_smoke/early_stop_vs_uniform_equal_budget.csv`
- `outputs_composite/early_stop_har_smoke/early_stop_vs_uniform_equal_budget_bootstrap.csv`
- `outputs_composite/early_stop_har_smoke/early_stop_query_trace_test.csv`

## 3.2) Портативность (обновлено, v12)

Снят профиль latency (HAR, ExtraTrees, `q=64`):
- `outputs_composite/edge_portability_profile_v12.csv`
- `outputs_composite/edge_resource_budget_table_v12.csv`
- `outputs_composite/tinyxai_full_audit_cost_v12.csv`

Ключевые числа:
- `inference_only p50 = 7.30 ms`
- `beacon_core_q64 p50 = 436.07 ms`, `mean_model_calls = 57.75`
- оценка early-stop по smoke (`q_mean=10.83`):
  - optimistic: `~81.77 ms`
  - conservative: `~93.32 ms`
  - ускорение vs core_q64: `~4.67x ... 5.33x`

Отдельный файл-оценка:
- `outputs_composite/edge_portability_earlystop_estimate_v12.csv`

## 3.3) Early-stopping full-run (HAR/PAMAP2/WISDM, completed)

Запуск:
- `scripts/run_early_stop_v12_full.sh`

Итоги:
- `outputs_composite/early_stop_v12_full_summary.csv`
- `artifacts/early_stop_v12_full_summary.md`

Ключевые факты:
- средний бюджет стабильно низкий: `q_mean ≈ 10.6–11.1` (vs `q_max=64`)
- но по качеству равнобюджетный baseline (`uniform Q=11`) не проигрывает:
  - HAR: `ΔAUROC=-0.0528` (p=0.252), `ΔAUPRC=+0.0112` (p=0.93)
  - PAMAP2: `ΔAUROC=-0.0736` (p=0.20), `ΔAUPRC=-0.1108` (p=0.054)
  - WISDM: `ΔAUROC=-0.0481` (p=0.42), `ΔAUPRC=-0.0578` (p=0.422)

Вывод на текущем протоколе:
- engineering-claim по снижению бюджета подтверждён,
- quality-claim для equal-budget early-stop vs uniform **не подтверждён**.

## 4) Что нужно от научного анализатора

### A. Статистический разбор v11/v12
1. Подтвердить/опровергнуть dataset-specific эффект (особенно WISDM).
2. Оценить, достаточно ли мощности для claim по `Q=32`/`Q=64`.
3. Дать рекомендацию по формальному тексту claim (строгий/умеренный/отрицательный).

### B. Исследовательские гипотезы (next)
1. Стоит ли развивать `early-stopping` как главный вклад (реальный budget reduction)?
2. Есть ли смысл держать `TAN/Fuzzy` в основном тексте или уводить в supplementary?
3. Какие 1–2 дополнительных протокола дадут максимальный шанс на усиление статьи без раздувания scope?

Рекомендуемый next:
1. Для portability-claim использовать уже снятый latency-профиль (`edge_portability_*_v12.csv`) + early-stop estimate.
2. Дождаться финала `v12` как вспомогательного статистического подтверждения.
3. Запустить `early-stopping` full-run на HAR/PAMAP2/WISDM (equal-budget), если нужен строгий cross-dataset claim.

## 5) Рекомендуемый формат ответа анализатора
Просьба вернуть:
1. Краткий verdict (3-5 строк).
2. Таблица рисков/ограничений по силе доказательств.
3. Чёткий план следующего эксперимента (1 основной + 1 fallback).
