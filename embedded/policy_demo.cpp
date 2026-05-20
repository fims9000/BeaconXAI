#include <stdint.h>
#include <stdio.h>
#ifdef ESP_PLATFORM
#include "esp_timer.h"
#endif

#include "beacon_policy.h"

static float sample_a[beacon_policy::kInputDim] = {
    -0.20f, 0.35f, 0.62f, 0.41f, 0.28f,
    0.19f, 0.44f, 0.57f, 2.00f, 0.12f};

static inline uint64_t now_us() {
#ifdef ESP_PLATFORM
  return static_cast<uint64_t>(esp_timer_get_time());
#else
  return 0;
#endif
}

int main() {
  float s1 = beacon_policy::logit_panel(sample_a);
  float s2 = beacon_policy::fuzzy_policy(sample_a);
  float s3 = beacon_policy::tan_policy(sample_a);

  printf("logit=%f fuzzy=%f tan=%f\n", s1, s2, s3);

#ifdef ESP_PLATFORM
  const int iters = 10000;
  uint64_t t0 = now_us();
  volatile float acc = 0.0f;
  for (int i = 0; i < iters; ++i) acc += beacon_policy::logit_panel(sample_a);
  uint64_t t1 = now_us();
  for (int i = 0; i < iters; ++i) acc += beacon_policy::fuzzy_policy(sample_a);
  uint64_t t2 = now_us();
  for (int i = 0; i < iters; ++i) acc += beacon_policy::tan_policy(sample_a);
  uint64_t t3 = now_us();
  printf("lat_us_logit=%f lat_us_fuzzy=%f lat_us_tan=%f\n",
         double(t1 - t0) / iters, double(t2 - t1) / iters, double(t3 - t2) / iters);
  (void)acc;
#endif
  return 0;
}
