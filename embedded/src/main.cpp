#include <Arduino.h>
#include <stdint.h>
#include <esp_timer.h>

#include "../beacon_policy.h"

namespace {

#ifndef BENCH_ITERS
#define BENCH_ITERS 1000
#endif
#ifndef BENCH_TIMEOUT_US
#define BENCH_TIMEOUT_US 2000000
#endif
#ifndef BENCH_WARMUP_ITERS
#define BENCH_WARMUP_ITERS 64
#endif

constexpr int kItersRaw = BENCH_ITERS;
constexpr int kIters = (kItersRaw < 10) ? 10 : ((kItersRaw > 20000) ? 20000 : kItersRaw);
constexpr uint64_t kBenchTimeoutUs = static_cast<uint64_t>(BENCH_TIMEOUT_US);
constexpr int kWarmupIters = (BENCH_WARMUP_ITERS < 0) ? 0 : BENCH_WARMUP_ITERS;

float sample_a[beacon_policy::kInputDim] = {
    -0.20f, 0.35f, 0.62f, 0.41f, 0.28f,
    0.19f, 0.44f, 0.57f, 2.00f, 0.12f};

inline uint64_t now_us() { return static_cast<uint64_t>(esp_timer_get_time()); }

template <typename F>
void run_bench(const char* name, F fn) {
  static uint32_t ts[20000];
  volatile float acc = 0.0f;

  for (int i = 0; i < kWarmupIters; ++i) {
    acc += fn(sample_a);
  }

  uint64_t bench_start = now_us();
  int used = 0;
  for (int i = 0; i < kIters; ++i) {
    uint64_t t0 = now_us();
    acc += fn(sample_a);
    uint64_t t1 = now_us();
    ts[used++] = static_cast<uint32_t>(t1 - t0);
    if ((t1 - bench_start) > kBenchTimeoutUs) break;
  }

  // insertion sort (kIters small)
  for (int i = 1; i < used; ++i) {
    uint32_t v = ts[i];
    int j = i - 1;
    while (j >= 0 && ts[j] > v) {
      ts[j + 1] = ts[j];
      --j;
    }
    ts[j + 1] = v;
  }

  uint64_t sum = 0;
  for (int i = 0; i < used; ++i) sum += ts[i];
  float mean = static_cast<float>(sum) / static_cast<float>(used > 0 ? used : 1);
  uint32_t p50 = ts[(used > 0 ? used : 1) / 2];
  uint32_t p95 = ts[((used > 0 ? used : 1) * 95) / 100];

  Serial.printf(
      "BENCH policy=%s target_iters=%d used_iters=%d timeout_us=%llu mean_us=%.3f p50_us=%u p95_us=%u acc=%.6f\n",
      name, kIters, used, static_cast<unsigned long long>(kBenchTimeoutUs), mean, p50, p95, static_cast<float>(acc));
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1200);

  float s1 = beacon_policy::logit_panel(sample_a);
  float s2 = beacon_policy::fuzzy_policy(sample_a);
  float s3 = beacon_policy::tan_policy(sample_a);
  Serial.printf("CONFIG bench_iters=%d warmup=%d timeout_us=%llu\n", kIters, kWarmupIters,
                static_cast<unsigned long long>(kBenchTimeoutUs));
  Serial.printf("SCORES logit=%.6f fuzzy=%.6f tan=%.6f\n", s1, s2, s3);

  run_bench("logit", [](const float* a) { return beacon_policy::logit_panel(a); });
  run_bench("fuzzy", [](const float* a) { return beacon_policy::fuzzy_policy(a); });
  run_bench("tan", [](const float* a) { return beacon_policy::tan_policy(a); });
  Serial.println("DONE");
}

void loop() { delay(1000); }
