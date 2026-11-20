# Embedded policy benchmark (ESP32 / QEMU-ready)

1. Export policy header:

```bash
cd /path/to/BeaconXAI
.venv/bin/python scripts/export_policy_to_cpp.py \
  --bundle-dir outputs_composite/part2_q64_features_v9_n1500 \
  --out-header embedded/beacon_policy.h
```

2. Build firmware (PlatformIO):

```bash
cd embedded
pio run -e esp32c3
```

Default safety limits (for small devices):
- `BENCH_ITERS=1000`
- `BENCH_WARMUP_ITERS=64`
- `BENCH_TIMEOUT_US=2000000` (2s per policy cap)

You can override them via `platformio.ini` `build_flags`.

3. Run and collect UART logs:

```bash
pio device monitor -b 115200
```

Expected lines:

```text
SCORES logit=... fuzzy=... tan=...
BENCH policy=logit iters=1000 mean_us=... p50_us=... p95_us=...
BENCH policy=fuzzy iters=1000 mean_us=... p50_us=... p95_us=...
BENCH policy=tan iters=1000 mean_us=... p50_us=... p95_us=...
DONE
```

Notes:
- This benchmark is for compact policy layer `h(a(x))` only.
- Full BEACON audit (`Q+1` model calls) is not MCU-side.


## Experimental full BEACON Q16 audit (not the policy-only benchmark)

The separate header `embedded/dual_operator_audit.hpp` is a heap-free C++11
**streaming** implementation of the fixed 8+8 dual-operator research control.
It is **not** the main BEACON algorithm and **must not** be enabled as the
default explanation method: independent UCI HAR validation showed that
combining two operators reduced full-reference agreement and its uncalibrated
"report/abstain" flag had many false positives.

It uses eight distinct train-prior-selected components, each intervened with
(a) interpolation and (b) a train-only rival-class medoid with taper=2, for
**exactly 16 separate model-example evaluations**. The real classifier is
supplied by a callback with signature:

```cpp
using MarginFn = bool (*)(const float* sample, size_t n_values,
                          void* user_ctx, float* out_top2_log_margin);
```

The callback must produce the difference between the two largest **log
class scores**, using the same definition for the cached original margin.
Passing raw probability margins would change the scientific method.

The application supplies a writable `float[T*C]` scratch buffer; the header
does not allocate a Q×window batch or heap storage. Output includes
`extra_model_evaluations`, ranked IDs and a diagnostic `reported` flag.
The latter is **not** a calibration or causality guarantee. The header does
not contain a trained classifier, inference tensor arena, DSP feature builder
or power/flash measurement.

Host C++11 acceptance check (also in CI):

```bash
g++ -std=c++11 -O2 -Wall -Wextra -Werror -pedantic \
  embedded/tests/test_dual_operator.cpp -o /tmp/beacon-q16-cpp-test
/tmp/beacon-q16-cpp-test
```

For model-dependent **host CPU** times and conservative explicit array-byte
projections, see `scripts/profile_full_audit_host.py` and
`beaconxai/resource_accounting.py`. These do not measure actual ESP32-C3
RAM, program flash or energy. A full board benchmark requires a validated
deployable model and measurement hardware; none is present in this branch.


## Actual trained raw-input MLP: C++ FP32 export and numeric parity

The reproducible host export pipeline is `scripts/export_raw_mlp_cpp.py`.
It trains `MLPClassifier(hidden=32)` on the official **training
participants** of UCI HAR, exports FP32 weights/biases and **training-only**
channel means/standard deviations, and saves test-window parity fixtures.
No giant generated header is committed to Git; GitHub Actions retains it as
a run artifact.

```bash
python scripts/preprocess_uci_har_for_q36.py \
  --root data/uci_har --out data/uci_har_official_subjects.npz
python scripts/export_raw_mlp_cpp.py \
  --npz-path data/uci_har_official_subjects.npz \
  --out-dir outputs_composite/uci_har_mlp_cpp --parity-samples 12 --seed 57
g++ -std=c++11 -O2 -Wall -Wextra -Werror -pedantic \
  -Ioutputs_composite/uci_har_mlp_cpp \
  embedded/tests/test_exported_mlp.cpp -o /tmp/beacon-raw-mlp-parity
/tmp/beacon-raw-mlp-parity outputs_composite/uci_har_mlp_cpp/model_parity_f32.bin
```

`embedded/raw_mlp_inference.hpp` standardizes **raw** sensor readings using
the exported training statistics, executes a float32 ReLU+softmax MLP and
returns the top-two pre-softmax logit difference. That difference equals
the top-two log-probability margin unless the Python implementation clips
extremely small probabilities, a discrepancy audited in the export
manifest. The test runs the exported *real classifier* through the Q16
callback with a **dummy donor** only to verify model-call integration.

The exporter and parity runner do **not** verify class correctness after
quantization (none is performed), physical RAM/Flash allocation, execution
time/energy on ESP32-C3, or causal validity of the audit. The dual-operator
abstention method is an **unpromoted negative research control**, not the
recommended default explanation.


## Baseline `adaptive_v2` for ESP32-C3 (cross-compiled, not hardware-tested)

The **original** research baseline now has a separate C++ port:
`embedded/adaptive_v2_audit.hpp`. It computes all four train-class cheap
statistics, selects Q=16, then evaluates each interpolation candidate with a
**real trained raw-input FP32 MLP** via callback. It does not include the
unsuccessful 8+8 dual-operator method as the default.

`embedded/adaptive_v2_firmware/` is an isolated PlatformIO project for
`esp32-c3-devkitm-1`. To reproduce its build:

```bash
python scripts/preprocess_uci_har_for_q36.py \
  --root data/uci_har --out data/uci_har_official_subjects.npz
python scripts/export_raw_mlp_cpp.py \
  --npz-path data/uci_har_official_subjects.npz \
  --out-dir outputs_composite/esp32c3_model \
  --adaptive-parity --parity-samples 96 --seed 57
mkdir -p embedded/adaptive_v2_firmware/include
cp embedded/adaptive_v2_audit.hpp embedded/raw_mlp_inference.hpp \
  outputs_composite/esp32c3_model/beacon_mlp_model.hpp \
  outputs_composite/esp32c3_model/beacon_adaptive_profiles.hpp \
  outputs_composite/esp32c3_model/beacon_board_reference.hpp \
  embedded/adaptive_v2_firmware/include/
pio run -d embedded/adaptive_v2_firmware -e esp32c3
```

Generated training arrays and full binary build receipts are stored as
GitHub Actions artifacts, not checked in. On **future** hardware execution
the UART self-test reports predicted class, candidate-set agreement and
top-3-set agreement against one frozen subject-held-out window, plus
a single-window timer and low-water free-heap indicator. No such board log
has yet been collected.

Build-time RAM/Flash size estimates do **not** prove successful device
execution, energy consumption, peak runtime memory or causal explanation
validity. A firmware binary is not intended for blind deployment.
