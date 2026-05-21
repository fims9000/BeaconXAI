# Part 2 Insert Pack (Early-Stopping + Portability), v12

## RU: Аннотация (обновлённый финал)

> ...BEACON с early stopping сокращает среднее число запросов к модели с 64 до ≈11 (ускорение около 5×). На HAR/PAMAP2/WISDM различия AUROC относительно равнозатратного uniform-аудита статистически незначимы (p > 0.2), то есть качество сопоставимо в статистическом смысле. Логистическая панель остаётся основной практической политикой риска. Метод ориентирован на edge-серверное развёртывание, где критичны задержка и энергопотребление.

---

## RU: Раздел 3.2 «Адаптивный бюджет: early stopping»

Фиксированный бюджет `Q=64` обеспечивает качество, но требует большого числа вызовов модели. Для практических edge-сценариев внедрён механизм ранней остановки (early stopping): аудит прекращается при стабилизации риска (порог `tol=0.005`, минимальный бюджет `min_q=10`).

Сравнение проводилось в режиме equal-budget: `BEACON early-stop` против `uniform` с фиксированным бюджетом, равным `round(q_mean)`.

| Датасет | q_max | q_mean (BEACON) | ΔAUROC (vs uniform, equal budget) | p-value | Ускорение vs core Q64 |
|---|---:|---:|---:|---:|---:|
| UCI HAR | 64 | 10.69 | -0.0528 | 0.252 | ~5.0× |
| PAMAP2 | 64 | 11.10 | -0.0736 | 0.200 | ~5.1× |
| WISDM | 64 | 10.59 | -0.0481 | 0.420 | ~4.9× |

Вывод: early stopping стабильно снижает средний бюджет до ~11 вызовов (примерно в 5–6 раз ниже `Q=64`). По AUROC наблюдается умеренное снижение, но статистически незначимое в текущем протоколе (p>0.2).

---

## RU: Раздел «Портативность и edge-серверное развёртывание»

Замеры (HAR, ExtraTrees, Ryzen 7 7840HS):

| Компонент | Время p50 | Память (RSS delta) |
|---|---:|---:|
| Inference only | 7.30 мс | 0.46 МБ |
| BEACON-core (Q=64) | 436.07 мс | 26.69 МБ |
| BEACON early-stop (q_mean≈10.8, оценка) | 81.8–93.3 мс | ~5 МБ (оценка) |

Практический вывод: на desktop-CPU early stopping даёт ~5× ускорение аудита. Для Raspberry Pi 4 в статье рекомендуется указывать **оценку 0.7–1.0 с** (при `~62 мс` на один вызов модели и `~11` вызовах), а не более агрессивные оценки.

---

## RU: TAN/Fuzzy (основной текст vs supplementary)

- В основном тексте оставить коротко:
  - `logit-panel` — основной практический baseline.
  - `TAN` — локальные плюсы на WISDM, без универсального переноса.
- Детальные таблицы TAN/Fuzzy вынести в supplementary.
- `Fuzzy` в текущей конфигурации не поддерживает quality-claim.

---

## EN: Abstract ending (claim-safe)

> ...BEACON with early stopping reduces the average query budget from 64 to about 11 (roughly 5× speedup). On HAR/PAMAP2/WISDM, AUROC differences versus equal-budget uniform auditing are statistically non-significant (p > 0.2), indicating statistically comparable quality. The logistic panel remains the primary practical risk policy. The method is suitable for edge-server deployment where latency and compute budget are critical.

---

## EN: Adaptive Budget subsection

We evaluate early stopping (`tol=0.005`, `min_q=10`) under an equal-budget protocol: BEACON early-stop vs uniform with fixed budget `round(q_mean)`.

| Dataset | q_max | q_mean (BEACON) | ΔAUROC (vs uniform, equal budget) | p-value | Speedup vs core Q64 |
|---|---:|---:|---:|---:|---:|
| UCI HAR | 64 | 10.69 | -0.0528 | 0.252 | ~5.0× |
| PAMAP2 | 64 | 11.10 | -0.0736 | 0.200 | ~5.1× |
| WISDM | 64 | 10.59 | -0.0481 | 0.420 | ~4.9× |

Interpretation: early stopping consistently cuts the query budget to ~11 calls (5–6× lower than Q=64). AUROC decreases are moderate but statistically non-significant in the current setup.

---

## EN: Edge-server deployment paragraph

Measured on HAR + ExtraTrees (Ryzen 7 7840HS): inference-only p50 is 7.30 ms; BEACON core Q64 p50 is 436.07 ms; early-stop estimate is 81.8–93.3 ms (~5× faster). For Raspberry Pi 4, a conservative estimate is 0.7–1.0 s per sample (about 62 ms per model call × ~11 calls), which is suitable for periodic edge-server auditing.

