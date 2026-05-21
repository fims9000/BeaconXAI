# Глава 2: материалы и план исследований (v2)

## 1) Цель главы
Показать, в каких сценариях BEACON дает практический выигрыш, а где нет.
Фокус: `BEACON-panel/logit` как основной baseline, `TAN/Fuzzy` как альтернативные политики.

## 2) Что уже подтверждено на текущем пакете (HAR/PAMAP2/WISDM)
Источник: [final_policy_delta_summary.csv](/home/lebedeffson/Code/BeaconXAI/outputs_composite/v11_cross_dataset/final_policy_delta_summary.csv)

- `Fuzzy` стабильно хуже `logit` на всех бандлах.
- `TAN` на `HAR` и `PAMAP2` не дает прироста по `F1@10` (дельта около нуля), часто хуже по `AUROC`.
- Единственный позитивный сигнал: `WISDM`.
  - `wisdm_tb12_q58`: `TAN ΔAUROC=+0.0419`, `ΔF1@10=+0.0350`
  - `wisdm_tb8_q39`: `TAN ΔAUROC=+0.0213`, `ΔF1@10≈+0.0014`

Рабочий вывод для текста:
- `logit` — основной практический выбор.
- `TAN` может улучшать ранжирование на WISDM-подобных данных.
- `Fuzzy` в текущей конфигурации не поддерживает quality-claim.

## 3) UWave (завершено)
UWave отдельным пакетом:
- конфиг: `configs/experiments_v11_uwave_only.json`
- выход: `outputs_composite/v11_uwave_only/`
- статус: завершено (`uwave_tb16_q39_interp_n428`, `uwave_tb24_q58_interp_n428`)

Проверка:
```bash
ps -p $(cat /home/lebedeffson/Code/BeaconXAI/outputs_composite/v11_uwave_only/logs/run.pid) -o pid,etime,pcpu,pmem,cmd
tail -f /home/lebedeffson/Code/BeaconXAI/outputs_composite/v11_uwave_only/logs/run.log
```

Итог UWave:
- `uwave_tb16_q39`:
  - `TAN`: `ΔAUROC=-0.0675`, `ΔF1@10≈0`
  - `Fuzzy`: хуже `logit` по AUROC/AUPRC/F1@10
- `uwave_tb24_q58`:
  - `TAN`: слабый плюс по AUROC (`+0.0187`), но без подтверждения (`p=0.0672`, `CI_low<0`)
  - `Fuzzy`: хуже `logit` (AUROC/AUPRC значимо отрицательные дельты)

Вывод по UWave: quality-claim для `TAN/Fuzzy` не подтверждается.

## 4) Корректный протокол расширения (под наш репозиторий)
В этом репо нет `dataset_registry.json` и `beacon/run_experiment.py`.
Правильный путь:
1. Готовим `.npz` (`x_train,y_train,x_test,y_test`).
2. Добавляем датасет в `configs/experiments_v11_*.json`.
3. Запускаем `scripts/run_cross_dataset_benchmark.py`.

Базовая команда:
```bash
/home/lebedeffson/Code/venv/bin/python scripts/run_cross_dataset_benchmark.py \
  --config configs/experiments_v11_cross_dataset.json \
  --out-root outputs_composite/v11_cross_dataset
```

## 5) Следующий шаг (запущен)
`Heartbeat` завершен отдельным пакетом:
- конфиг: `configs/experiments_v11_heartbeat_only.json`
- выход: `outputs_composite/v11_heartbeat_only/`
- лог: `outputs_composite/v11_heartbeat_only/logs/run.log`
- pid: `outputs_composite/v11_heartbeat_only/logs/run.pid`

Итог Heartbeat: `TAN/Fuzzy` не дают подтвержденного quality-claim над `logit`.

`SelfRegulationSCP1` запущен:
- конфиг: `configs/experiments_v11_selfregulation_only.json`
- выход: `outputs_composite/v11_selfregulation_only/`
- лог: `outputs_composite/v11_selfregulation_only/logs/run.log`
- pid: `outputs_composite/v11_selfregulation_only/logs/run.pid`

Минимальный критерий позитивного claim:
- `ΔF1@10 > 0`
- `CI_low > 0`
- `p < 0.05` (с поправкой на множественные сравнения в итоговом реестре)

Если не выполняется:
- фиксируем ограничение и оставляем `TAN/Fuzzy` как вторичные варианты (интерпретация/калибровка), без quality-claim.
