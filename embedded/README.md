# Embedded policy benchmark (ESP32 / QEMU-ready)

1. Export policy header:

```bash
cd /home/lebedeffson/Code/BeaconXAI
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
