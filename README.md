# BeaconXAI

Репозиторий для реализации и проверки **BEACON-XAI**: бюджетированный локальный аудит решений модели с
`S+ / S- / rho_B / risk` и сравнением с baseline-методами при фиксированном query budget.

## Что реализовано

- Ядро BEACON:
  - signed margin audit,
  - active leaves,
  - adaptive refinement,
  - budget split (`Q_init/Q_ref/Q_frag`),
  - `rho_B`, `rho_B^cost`, `risk`.
- Baselines:
  - `confidence`, `entropy`, `negative_margin`,
  - `beacon_flat`, `uniform_refinement`,
  - `budgeted_shapley_like`,
  - `saliency_topk`, `ig_topk`,
  - `simple_counterfactual`, `full_occlusion`.
- Composite-риск:
  - `beacon_composite` (комбинация BEACON-сигналов + uncertainty).
- Эксперименты:
  - full-run, sensitivity, claims `H1..H5`, CE-controls, `rho_B` sanity-check.
- Датасеты:
  - UCI HAR,
  - WISDM (raw UCI, phone accel+gyro preprocessing).

## Структура

- `beaconxai/` — библиотека метода и baseline'ов.
- `scripts/` — запуск экспериментов/препроцессов.
- `tests/` — smoke/unit тесты.
- `BEACON_XAI_research_document_v09_math_fixed.docx` — исходный исследовательский документ.

## Окружение

Рекомендуемое окружение (CUDA):

- `/home/lebedeffson/Code/venv_cuda`

Проверка:

```bash
/home/lebedeffson/Code/venv_cuda/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Установка зависимостей

```bash
cd /home/lebedeffson/Code/BeaconXAI
/home/lebedeffson/Code/venv_cuda/bin/pip install -r requirements.txt
/home/lebedeffson/Code/venv_cuda/bin/pip install sktime
```

## Подготовка данных

### 1) UCI HAR

Скачивается автоматически при первом запуске `run_*` скриптов.

### 2) Shifted UCI HAR (stress-test)

```bash
/home/lebedeffson/Code/venv_cuda/bin/python scripts/make_uci_har_shifted_npz.py \
  --dataset-root ./data \
  --out ./data/uci_har_shifted.npz \
  --noise-sigma 0.35 --drop-prob 0.35 --mask-prob 0.5 --mask-len 24
```

### 3) WISDM (UCI raw)

1. Скачивание архива (если нет):

```bash
cd data
curl -L --fail --output wisdm-dataset.zip \
  https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip
unzip -q -o wisdm-dataset.zip -d wisdm_raw
```

2. Препроцесс:

```bash
cd /home/lebedeffson/Code/BeaconXAI
/home/lebedeffson/Code/venv_cuda/bin/python scripts/preprocess_wisdm_uci_raw.py \
  --root ./data/wisdm_raw/wisdm-dataset/raw \
  --out ./data/wisdm_phone_accel_gyro.npz \
  --window 128 --stride 64 --train-user-frac 0.7
```

## Основной запуск claims (CUDA)

Ниже пример **tuned** конфигурации:

- `k0=8`
- `q_frag_ratio=0.5`
- `alpha=1.5`, `beta=0.25`, `gamma=0.5`
- `tau_s=0.2`
- `partition_mode=time_only`
- `risk_policy=rho_censored_boost`
- `enable_composite=true`

### Clean UCI HAR

```bash
/home/lebedeffson/Code/venv_cuda/bin/python scripts/run_claims_validation.py \
  --dataset uci_har --dataset-root ./data \
  --model cnn1d --cnn-epochs 20 --cnn-batch-size 128 --cnn-lr 0.001 \
  --neutralization zero \
  --q-values 8,16,32 --k0-values 8,16 \
  --l-min 4 --q-frag-ratio 0.5 \
  --alpha 1.5 --beta 0.25 --gamma 0.5 --tau-s 0.2 \
  --partition-mode time_only --risk-policy rho_censored_boost \
  --enable-composite \
  --out-dir ./outputs_full_clean_cuda_tuned_comp
```

### Shifted UCI HAR

```bash
/home/lebedeffson/Code/venv_cuda/bin/python scripts/run_claims_validation.py \
  --dataset npz --npz-path ./data/uci_har_shifted.npz \
  --model cnn1d --cnn-epochs 20 --cnn-batch-size 128 --cnn-lr 0.001 \
  --neutralization zero \
  --q-values 8,16,32 --k0-values 8,16 \
  --l-min 4 --q-frag-ratio 0.5 \
  --alpha 1.5 --beta 0.25 --gamma 0.5 --tau-s 0.2 \
  --partition-mode time_only --risk-policy rho_censored_boost \
  --enable-composite \
  --out-dir ./outputs_full_shifted_cuda_tuned_comp
```

### Full WISDM (core methods only)

```bash
/home/lebedeffson/Code/venv_cuda/bin/python scripts/run_claims_validation.py \
  --dataset npz --npz-path ./data/wisdm_phone_accel_gyro.npz \
  --model cnn1d --cnn-epochs 20 --cnn-batch-size 256 --cnn-lr 0.001 \
  --neutralization zero \
  --q-values 16,32 --k0-values 8 \
  --l-min 4 --q-frag-ratio 0.5 \
  --alpha 1.5 --beta 0.25 --gamma 0.5 --tau-s 0.2 \
  --partition-mode time_only --risk-policy rho_censored_boost \
  --enable-composite \
  --methods confidence,entropy,negative_margin,beacon_refine,beacon_flat,uniform_refinement,beacon_composite \
  --out-dir ./outputs_wisdm_full_core_cuda
```

## Где смотреть результаты

В каждой папке `outputs_*`:

- `claims_report.json` — итог по `H1..H5` + CE controls.
- `risk_metrics_k0_*.csv` — AUROC/AUPRC таблицы по методам и бюджету.
- `risk_rows_k0_*.csv` — построчные risk scores.
- `local_metrics_k0_*.csv` — `sufficiency/necessity/CE/rho` и т.д.

## Важные замечания

- `H4` (устойчивость по `K0`) оценивается только если в запуске есть и `K0=8`, и `K0=16`.
- Если запускается только `K0=8`, `claims_report.json` содержит `"h4_evaluable": false`.
- Для WISDM качество risk-оценки сильно зависит от базовой accuracy классификатора.

## Дополнительные скрипты

- `scripts/search_beacon_config.py` — быстрый поиск гиперпараметров BEACON.
- `scripts/tune_composite_risk.py` — подбор весов для `beacon_composite` на готовых `outputs_*`.
- `scripts/run_sensitivity.py` — sensitivity sweeps.
- `scripts/run_rho_sanity.py` — sanity-check `rho_B` vs `rho_exact/beam`.

## Тесты

```bash
/home/lebedeffson/Code/venv_cuda/bin/python -m py_compile beaconxai/*.py scripts/*.py tests/*.py
```

(В CI/локально можно также добавить `pytest`, если установлен.)
