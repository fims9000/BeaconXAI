# Data Preparation

This project uses local datasets (not committed to git).  
All commands below are run from repository root with `.venv` active.

## 1) UCI HAR (base + shifted NPZ)

Expected input folder:

- `data/UCI HAR Dataset/` (official UCI HAR archive extracted)

Generate shifted NPZ used by most HAR experiments:

```bash
.venv/bin/python scripts/make_uci_har_shifted_npz.py \
  --dataset-root data \
  --out data/uci_har_shifted.npz
```

Output:

- `data/uci_har_shifted.npz`

## 2) PAMAP2 (windowed NPZ)

PAMAP2 is downloaded automatically by preprocessing script.

```bash
.venv/bin/python scripts/preprocess_pamap2.py \
  --data-root data \
  --window-length 200 \
  --step 100 \
  --min-purity 0.95 \
  --test-subjects 8,9 \
  --out data/pamap2_acc9_w200s100_p095.npz
```

Output:

- `data/pamap2_acc9_w200s100_p095.npz`

## 3) WISDM raw -> NPZ

Expected raw folders:

- `data/wisdm_raw/wisdm-dataset/raw/phone/accel/`
- `data/wisdm_raw/wisdm-dataset/raw/phone/gyro/`

Generate NPZ:

```bash
.venv/bin/python scripts/preprocess_wisdm_uci_raw.py \
  --root data/wisdm_raw/wisdm-dataset/raw \
  --out data/wisdm_phone_accel_gyro.npz \
  --window 128 \
  --stride 64 \
  --min-purity 0.85 \
  --train-user-frac 0.7 \
  --split-mode random \
  --seed 42
```

Output:

- `data/wisdm_phone_accel_gyro.npz`

## Notes

- Keep random seed fixed (`42`) for reproducible splits.
- `data/` and `outputs*/` are ignored by git by design.
