PY ?= .venv/bin/python

.PHONY: sanity smoke data-har data-pamap2 data-wisdm hidden-conflict tan-detection audit-panel portability significance resource-budget q1-early-stop-smoke q1-early-stop-har-one q1-early-stop-har-seeds q1-early-stop-full reproduce reproduce-quick all

sanity:
	$(PY) -m py_compile beaconxai/*.py scripts/*.py tests/*.py

smoke: sanity
	$(PY) -m pytest tests -q
	$(PY) -c "import beaconxai; print('beaconxai import ok')"

data-har:
	$(PY) scripts/make_uci_har_shifted_npz.py --dataset-root data --out data/uci_har_shifted.npz

data-pamap2:
	$(PY) scripts/preprocess_pamap2.py --data-root data --out data/pamap2_acc9_w200s100_p095.npz

data-wisdm:
	$(PY) scripts/preprocess_wisdm_uci_raw.py --root data/wisdm_raw/wisdm-dataset/raw --out data/wisdm_phone_accel_gyro.npz

hidden-conflict:
	$(PY) scripts/run_har_hidden_conflict_benchmark.py \
		--out-summary outputs_composite/har_hidden_conflict_localization_table.csv \
		--out-per-sample outputs_composite/har_hidden_conflict_localization_per_sample.csv

tan-detection:
	$(PY) scripts/run_har_hidden_conflict_tan.py \
		--out-summary outputs_composite/har_hidden_conflict_detection_tan_table.csv \
		--out-per-sample outputs_composite/har_hidden_conflict_detection_tan_per_sample.csv

audit-panel:
	$(PY) scripts/make_audit_panel_tables.py --n-boot 1000

portability:
	$(PY) scripts/measure_portability.py --out outputs_composite/edge_portability_profile.csv

resource-budget:
	$(PY) scripts/estimate_resource_budget.py \
		--profile-csv outputs_composite/edge_portability_profile.csv \
		--out outputs_composite/edge_resource_budget_table.csv

q1-early-stop-smoke:
	$(PY) scripts/run_early_stopping_equal_budget.py \
		--dataset data/uci_har_shifted.npz \
		--n-total 80 \
		--time-bins 16 \
		--q-max 32 \
		--seed 7 \
		--tol 0.005 \
		--min-q 5 \
		--n-boot 20 \
		--baseline both \
		--out outputs_composite/_smoke_uniform_early_stop

q1-early-stop-har-one:
	$(PY) scripts/run_early_stopping_equal_budget.py \
		--dataset data/uci_har_shifted.npz \
		--n-total 600 \
		--time-bins 16 \
		--q-max 64 \
		--seed 1 \
		--tol 0.005 \
		--min-q 10 \
		--n-boot 1000 \
		--baseline both \
		--model-cache models/har_extratrees_seed1.pkl \
		--out outputs_composite/q1_har_seed1_early_stop

q1-early-stop-har-seeds:
	./scripts/run_q1_har_early_stop_seeds.sh

q1-early-stop-full:
	./scripts/run_early_stop_v12_full.sh

significance:
	$(PY) scripts/compute_hidden_conflict_significance.py \
		--per-sample outputs_composite/har_hidden_conflict_localization_per_sample.csv \
		--out outputs_composite/table8_significance.csv

all: hidden-conflict portability resource-budget significance
	@echo "Done. See outputs_composite/ for generated artifacts."
	@echo "make all requires locally prepared datasets (see DATA_PREPARATION.md)."

reproduce:
	$(PY) scripts/reproduce_all_figures.py --table all

reproduce-quick:
	$(PY) scripts/reproduce_all_figures.py --table all --quick
