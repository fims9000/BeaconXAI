# BEACON: сводка экспериментальной линии контрфактических объяснений (22.11.2026)

**Статус:** исследовательская ветка `feature/contrastive-q16-20261119`, [draft PR #1](https://github.com/fims9000/BeaconXAI/pull/1). Исходные Part 1 (локализация контрсвидетельств) и Part 2 (адаптивный бюджет раннего аудита) в README и теге `v12-final` остаются самостоятельными работами; новая линия — **условные контрфакты для фиксированного чёрного ящика**, а не замена их постановки.

**Машиночитаемая сводная таблица Q16:** [beacon_q16_counterfactual_cross_dataset_summary.csv](beacon_q16_counterfactual_cross_dataset_summary.csv). Подробные исходные записи доступны через соответствующие GitHub Actions артефакты и воспроизводимые workflows.

## Определение метрики

Фиксируем исходный предсказанный моделью класс A и B=runner-up. Поиск получает x и заранее выбранные реальные TRAIN-доноры B. Вмешательство `splice` заменяет одни и те же временные блоки S в исходном x данными одного донора с заданной стыковкой.

**J=1 только при одновременно измеренном** `argmax f(T_S(x,D1)) = argmax f(T_S(x,D2)) = B`, не более 35% фактически изменённых значений **на каждом** доноре и суммарно `Q <= 16` дополнительных входов в модель на оба донора. Кэшированные исходные предсказания и офлайновые обучающие оценки учитываются отдельно. При отсутствии подтверждения — отказ, не выдача предсказанного переключения.

Это *операторно-зависимое контрфактическое свидетельство поведения модели*, **не** доказательство причинной значимости, адекватности движения или устойчивости к произвольным донорам. Значения Q в этой линии нельзя непосредственно приравнивать к модельным вызовам классических SHAP/LIME без согласования постановок.

## Проверенные результаты по корпусам (разные модели и протоколы, нельзя усреднять без оговорок)

| Корпус и модель | Проверенный протокол | Главное сравнение при общем Q16, D1+D2 | Точный парный p | Вывод |
| --- | --- | --- | ---: | --- |
| UEA RacketSports, ExtraTrees, accuracy 82,2% | [отчёт](beacon_racketsports_holdout_results_2026-11-21.md), [run](https://github.com/fims9000/BeaconXAI/actions/runs/44936391554) | обратный 71/152 против прямого 71/152, **со встроенным требованием локальной несократимости** | 1,0 | Преимущество обратного удаления не перенеслось; маски нестабильны к D2 |
| UEA Epilepsy, ExtraTrees, accuracy 97,83% | [отчёт](beacon_epilepsy_robust_q16_results_2026-11-21.md), [run](https://github.com/fims9000/BeaconXAI/actions/runs/44944751540) | backward robust 14/138 vs posthoc D1+D2 6/138; против сильного forward robust 11/138 | 0,02148 против posthoc, **0,375** против forward | Улучшение относительно слабого контроля, не относительно сильного |
| UEA NATOPS, ExtraTrees, accuracy 83,89% | [отчёт](beacon_natops_sparse_q16_results_2026-11-21.md), [run](https://github.com/fims9000/BeaconXAI/actions/runs/44947081765) | компактный TRAIN-maximin 67/180 vs forward robust 51/180 | **0,00249** | Компактный список масок улучшил J, но 113 отказов |
| UEA UWaveGestureLibrary, ExtraTrees, accuracy **62,19%** | [отчёт](beacon_uwave_pairwise_adaptive_results_2026-11-22.md), [run](https://github.com/fims9000/BeaconXAI/actions/runs/44949994093) | pairs adaptive 160/320 vs static sparse 141/320; additive adaptive 175/320 | 0,01834 для заранее заданной пары | Недостаточное качество самой модели, значимость **условная**; пары не доминируют |
| UEA PenDigits, **raw MLP**, accuracy **94,77%** на полном 3498 TEST | [отчёт](beacon_pendigits_additive_adaptive_results_2026-11-22.md), [run](https://github.com/fims9000/BeaconXAI/actions/runs/44950732585) | additive adaptive 257/320 vs static sparse 252/320 на заранее выбранных 320 TEST | **0,125** | На качественной модели улучшение **не подтверждено** |
| UEA SpokenArabicDigits (MFCC 93-point interpolation A1), ExtraTrees, accuracy **92,72%** | [отчёт](beacon_spokenarabic_k3_transfer_results_2026-11-22.md), [run](https://github.com/fims9000/BeaconXAI/actions/runs/44953711131) | перенос на невиданные D4/D5: K3 80/320 vs K2 91/320 при общем Q16 | **0,04329** | Добавление далёкого третьего донора **снизило** безусловный перенос; amendment до inference |

У всех новых экспериментов есть отдельный фиксированный до TEST протокол и GitHub Actions с `manifest.json`, `summary.csv`, `per_sample.csv`, `provenance.txt`. Артефакты GitHub Actions могут истечь по retention; стабильные протоколы, таблицы, выводы и точные ссылки на runs сохранены в репозитории. Исторические исследования UCI HAR/PAMAP2/BasicMotions сохраняются в своих отчётах и не используются как независимый новый holdout.

**Дополнение по третьему донору (SpokenArabicDigits):** K2 сумел подтвердить 139/320 компактных объяснений на D1,D2, из них 91/320 перенеслось на D4,D5. K3 (D1,D2 плюс специально самый далёкий D3) подтвердил 122/320, из них 80/320 перенеслось на те же **невиданные** D4,D5. Условная вероятность переноса после успеха около 65,5% у обоих, но безусловный охват K3 ниже. Это не доказательство бесполезности любых K3: при фиксированном Q16 он решает более строгую онлайн-задачу. [Отчёт с A1 и журналом запросов](beacon_spokenarabic_k3_transfer_results_2026-11-22.md). Даже если TEST новой серии изначально не использовался для выбора алгоритма, preprocessing был дополнен после обнаружения реальной переменной длины: **задокументированное отклонение A1, не скрытая исходная пререгистрация**.

## Что выявили честные донорные проверки

На UWave, среди выданных компактных Q16-масок, **та же маска** сохранила целевой класс на двух ранее не использованных донорах D3+D4:
- STATIC: 73/141 = 51,8%
- PAIRS STATIC: 63/136 = 46,3%
- PAIRS ADAPTIVE: 68/160 = 42,5%
- ADDITIVE ADAPTIVE: 68/175 = 38,9%.

На PenDigits:
- STATIC: 158/252 = 62,7%
- PAIRS STATIC: 150/256 = 58,6%
- PAIRS ADAPTIVE: 149/257 = 58,0%
- ADDITIVE ADAPTIVE: 150/257 = 58,4%.

Третий и четвёртый доноры **никогда не участвовали в выборе маски**, модельные вызовы для них оплачены дополнительно **вне Q16**. Эти доли — *условные на успехах D1+D2*, а не частоты по всей выборке. Для сравнения устойчивости следует также публиковать **совместный охват всей тестовой выборки**: UWave (STATIC 73/320 vs ADDITIVE 68/320), PenDigits (STATIC 158/320 vs ADDITIVE 150/320). Тогда видно, что общая донорная переносимость нового метода не выросла, несмотря на увеличение J для двух рабочих доноров.

## Стоимость и строгие ограничения интерпретации

- UWave: 1 256 **офлайновых** model input evaluations для калибровки; 25 323 онлайн-оценки на все четыре политики при Q8/Q16 на 320 тестовых окнах; 1 224 внешние posthoc D3/D4 оценки.
- PenDigits: 5 860 офлайновых model evaluations; 20 208 онлайн-оценок на четыре политики и два бюджета; 2 044 внешние posthoc D3/D4 оценки.
- Глобальная Ridge-модель обучена только на TRAIN-калибровке; локальные posterior-коррекции после онлайн-запросов — адаптация к **текущему x**, не TRAIN leakage и не независимая гарантия.
- Pairwise-регрессия немного лучше подгоняет собственные calibration rows, но **не улучшила** Q16 относительно простой аддитивной адаптации на двух новых корпусах. Не заявлять научную победу pairwise.
- При 62,19% accuracy на UWave нельзя переносить вывод на хорошо обученные классификаторы. На PenDigits, 94,77%, заранее заданный эффект +1,56 п.п. статистически не подтверждён.
- Идентификаторы испытуемых/сеансов у этих архивных выгрузок отсутствуют, поэтому bootstrap по окнам и McNemar не служат гарантией переноса между людьми.
- Q16 описывает **стоимость онлайн-вызовов**; полная стоимость для практического развёртывания также включает офлайн-калибровку, подбор банка доноров, построение масок, CPU latency, память и энергию.

## Что развиваем дальше — без подмены отрицательных результатов

**Главный пробел**: построение маски под фиксированные два донора плохо переносится на новые источники вмешательства. Следующий проверяемый вопрос — распределять **единый Q16** между тремя донорными реализациями при поиске и затем сравнивать перенос **той же маски** на четвёртый/пятый донор, не увиденные алгоритмом. Для обеих политик одинаковы TRAIN-фильтр, порог поддержки, оператор и Q; дополнительные оценки D4/D5 используются **только для posthoc сопоставления**, оплачиваются отдельно.

Нужны два различаемых показателя: (1) частота реально подтверждённых компактных масок на используемых донорных наборах, (2) безусловная доля тестовых окон с успешным переносом на заранее отложенные доноры. При выборе трёх доноров **не считать** D3 независимой проверкой переносимости, потому что он уже участвовал в поиске.

Новую политику сначала проверить на TRAIN/внутренней валидации; новый официальный TEST зафиксировать отдельным протоколом **до открытия**. Попарную регрессию пока оставить только как исследовательскую абляцию, базовым обученным контролем считать additive-adaptive.

Основные интерфейсы: `beaconxai/donor_robust_cf.py`, `beaconxai/learned_interactions.py`, экспериментальные runners в `scripts/` и тесты в `tests/`. Не создавать дублирующие ядра поиска и не переносить исследовательскую ветку в `main` без независимой проверки.

## Решение после SpokenArabicDigits (22.11.2026)

Отдельный заранее зафиксированный контроль **K3 против K2** дал отрицательный результат по переносимости на D4/D5 (`80/320 vs 91/320`, p=0,04329). В качестве рабочих controls сохраняем K2 static maximin и аддитивный адаптивный вариант; K3-farthest считать **неприоритетной исследовательской абляцией**. Не включать его в default. Новая версия должна управлять **ценностью проверки доноров** под общий Q, оценивать риск на TRAIN и сравнивать transfer на другом holdout без подгонки по SpokenArabicDigits TEST.


## Update: train-fitted risk gate on untouched CharacterTrajectories (22.11.2026)

[New complete independent report](beacon_charactertrajectories_transfer_gate_results_2026-11-22.md), [successful CI with raw CSV/JSON](https://github.com/fims9000/BeaconXAI/actions/runs/45024767799), [original protocol](beacon_charactertrajectories_transfer_gate_preregistered_2026-11-22.md), [documented pre-model amendment A1](beacon_charactertrajectories_amendment_A1_eq119_2026-11-22.md).

The official aeon equal-length archive was **119** time points, not 182 as assumed from general dataset tables. This was discovered during preprocessor validation before any model fitting or TEST inference, and the protocol amendment only changed sample geometry while preserving split, Q, ten bins, donor selection and risk thresholds. The first completed computation subsequently failed on writing a mixed TRAIN/TEST CSV schema; fixing missing CSV field defaults did not change scientific decisions.

A TRAIN-only ExtraTrees achieved **97.284% accuracy (1397/1436)** on the official TEST. The original TRAIN was split into 987 model-fit, 202 risk-fit and 233 separate threshold-calibration windows, all disjoint by **window index**, not established independent people. Only 320 fixed TEST windows were used for explanations.

At Q16 shared across D1+D2, the existing sparse BEACON found 182/320 directly verified compact certificates, of which **130/320** transferred without changing S to unseen D3+D4, with posthoc calls outside Q16. The nine-feature logistic risk model, trained on only 113 successful K2 risk-fit windows, was calibrated on 136 successful K2 gate-calibration windows using a Bonferroni-adjusted Clopper–Pearson criterion allowing failure upper bound **0.5**. That criterion selected **the most permissive candidate cutoff** (136/136 calibration masks accepted).

- **Ungated K2**: 182 issued, 52 failed on D3/D4, 130/320 unconditional transfers.
- **TRAIN-frozen risk gate**: 181 issued, 51 failed, **same 130/320** unconditional transfers.
- **Matched TRAIN support-only gate**: 182 issued, 52 failed, **same 130/320** transfers.

**No practical positive result for the frozen risk gate**. The conditional failed-transfer fraction declined by only 0.39 percentage points (28.57% → 28.18%) because a single failing mask was rejected. The gate spent no extra online calls and could not change masks or increase unconditional successful transfers by design.

After opening TEST, *exploratory only*, the TRAIN-fitted risk score had ROC-AUC **0.7433** against observed D3/D4 failure among certified masks; edited-support fraction alone gave **0.5154**. Failure rates rose from 3/46 (6.5%) in the low-risk quartile to 23/45 (51.1%) in the highest-risk quartile. These **TEST-derived quartiles must never be used as preregistered or deployable gate thresholds**. The robust operational decision rule remains scientifically unresolved.

Full costs were independently reconciled: 320 TRAIN-donor screening, 1871 cached original f(x), 8568 online Q16 search and 862 separate D3/D4 evaluations, totalling **11621 model-input predictions**. All 34 focused tests passed, no individual search exceeded Q16. In the status CSV, the new gate policies appear as **filtered certificates from the same search**, not separate searches with three independent Q16 budgets.

**Updated priority:** a donor-risk score may meaningfully *rank* failures, but a successful ranker is not a useful selective policy without TRAIN-calibrated risk-coverage. Develop an a priori stricter risk target on future TRAIN-only data, and evaluate on an entirely new corpus. To improve unconditional transfer coverage, the algorithm must actually change candidate selection or allocate existing Q toward verification; mere refusal cannot increase successful transfers. Keep K3-farthest negative and pairwise-negative results; do not merge this draft into main.


## Update: learned donor-transfer candidate ranking and probe-feedback Q16 (Wafer / TwoPatterns, 22.11.2026)

**Полная исходная статистика, протокол, тесты и методологические ограничения:** [beacon_ucr_active_transfer_q16_results_2026-11-22.md](beacon_ucr_active_transfer_q16_results_2026-11-22.md). [Протокол до обращения к обоим TEST](beacon_wafer_twopatterns_active_transfer_preregistered_2026-11-22.md), commit e56c8d4e0b92d80f29644b48ab7e025df882e736. [Успешный GitHub Actions matrix run 45026897917](https://github.com/fims9000/BeaconXAI/actions/runs/45026897917) с двумя CSV/JSON артефактами; **365 тестов PASS, 4 warnings** в полном smoke после реализации.

Задача теперь отличается от posthoc отбрасывания ненадежного объяснения: на TRAIN вычисляются реальные метки **совместного успешного переключения класса на четырех донорах** для заранее фиксированного набора 20 масок на каждое TRAIN ranking-fit окно. Дешевый LogisticRegression score строится только из x, исходного f(x), первых D1,D2 и TRAIN class-profile gains; невидимые онлайн D3,D4 используются **только как TRAIN-метки** и posthoc TEST-аудит, никогда как признаки или online feedback. Отдельный offline ranking-calibration корпус оценивает ранжирование, но не меняет гиперпараметры. В уже существующем sparse search core реализован необязательный ranking callback; отсутствие callback сохраняет историческое поведение. Динамический вариант уточняет приоритет с учетом только измеренных D1/D2 проб, без дополнительных вызовов, всё Q16 совместное.

**Первичный Wafer, реальный датчик, ExtraTrees TEST accuracy 98.17%:** исходный STATIC SPARSE достиг 44/320 независимых D3/D4 переносов фиксированной маски; LEARNED STATIC и LEARNED ONLINE достигли **58/320** каждый. Для заранее выбранной пары ONLINE против STATIC: оба=40, online_only=18, static_only=4, двусторонний exact McNemar **p=0.0043435**, улучшение **+4.375 п.п.** в безусловном J4. Но в полной абляции **динамический feedback не доказал пользы**: LEARNED STATIC также 58/320 (57 общих, по одному индивидуальному успеху, p=1.0). K2 certificate coverage 94/320 исходный vs 148/320 у каждого обученного; **условная** доля успеха при переносе снизилась 46.8%→39.2%. Это не доказательство более надежного *каждого* выданного объяснения.

**TwoPatterns, искусственный корпус, ExtraTrees TEST accuracy 35.73%:** ONLINE 54/320 vs STATIC 49/320, **p=0.522**; нет подтвержденного улучшения, а качество самого классификатора существенно ниже заранее установленного 70%-порога. TRAIN ranking calibration AUROC у Wafer 0.831, TwoPatterns 0.518. Данные показывают зависимость успеха от того, можно ли действительно обучить дешёвый предиктор mask-transfer на другом TRAIN.

Полная научная стоимость существенна: только TRAIN label+calibration mask evaluations **7374** (Wafer) / **8832** (TwoPatterns) сверх обучения исходного классификатора. Реальные model-input totals всех этапов и сравнений: **27432 / 26677**; TEST 960 строк на корпус, Q≤16, внешние D3/D4 отдельно. Нельзя скрывать этот расход за фразой «метод стоит Q16». Нет подтвержденных IDs приборов/сеансов для статистического переноса, подлинной физической интервенции, причинности или превосходства над внешними XAI-методами.

**Решение по разработке:** сохранять TRAIN-fitted STATIC mask ranking как практически более простую исследовательскую альтернативу; ONLINE feedback — отдельная абляция, не default. Не подбирать коэффициенты по просмотренным Wafer/TwoPatterns TEST, не менять прежние отрицательные результаты и не переносить код в main до независимого повторения на хорошем классификаторе другого корпуса с предметно допустимым оператором.
