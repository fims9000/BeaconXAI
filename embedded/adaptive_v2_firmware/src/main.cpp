// Experimental ESP32-C3 firmware build for ORIGINAL BEACON adaptive_v2 Q16.
// GENERATED HEADERS (trained model, training profiles, frozen external sample)
// are copied into include/ by CI, never committed as private training artifacts.
//
// This firmware is NOT validated on a physical board. A successful build/map
// is not a hardware time, energy, or true peak-RAM measurement.
#include <Arduino.h>
#include <esp_timer.h>
#include <esp_heap_caps.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "adaptive_v2_audit.hpp"
#include "raw_mlp_inference.hpp"
#include "beacon_mlp_model.hpp"
#include "beacon_adaptive_profiles.hpp"
#include "beacon_board_reference.hpp"
#ifdef BEACON_EXPERIMENTAL_UPGRADE
#include "staged_adaptive_upgrade.hpp"
#include "beacon_staged_policy.hpp"
#include "beacon_board_staged_reference.hpp"
#endif

namespace {
namespace model=beacon_exported_mlp;
namespace profile=beacon_exported_profiles;
namespace control=beacon_reference;
#ifdef BEACON_EXPERIMENTAL_UPGRADE
namespace stage=beacon_board_staged;
namespace threshold=beacon_exported_staged;
#endif

constexpr size_t kM=36, kQ=16;
static_assert(model::kChannels==9,"Requires the 3 x 3-channel UCI HAR layout");
static_assert(profile::kClasses==model::kOutputDim,"Model/profile class mismatch");
static_assert(profile::kWindow==model::kInputDim,"Model/profile geometry mismatch");

static float standardized[model::kInputDim];
static float scratch[model::kInputDim];
static beacon_adaptive_v2::Component components[kM];
#ifdef BEACON_EXPERIMENTAL_UPGRADE
static beacon_staged_upgrade::Result<kM> report;
#else
static beacon_adaptive_v2::Result<kM,kQ> report;
#endif

bool margin_callback(const float* window,size_t length,void*,float* out) {
  return beacon_raw_mlp::predict_top2_log_margin<
      model::kInputDim,model::kHiddenDim,model::kOutputDim>(
      window,length,model::kW1,model::kB1,model::kW2,model::kB2,out);
}

bool check_selected_set(const uint16_t* selected,
                        const uint16_t* expected,size_t q) {
  if(!selected || !expected || q<3 || q>32)return false;
  uint16_t copy[32];
  for(size_t k=0;k<q;++k)copy[k]=selected[k];
  std::sort(copy,copy+q);
  for(size_t k=0;k<q;++k)
    if(copy[k]!=expected[k])return false;
  return true;
}

bool check_rank_top3_set(const uint16_t* ranked,const uint16_t* reference) {
  if(!ranked || !reference)return false;
  uint16_t predicted[3]={ranked[0],ranked[1],ranked[2]};
  uint16_t expected[3]={reference[0],reference[1],reference[2]};
  std::sort(predicted,predicted+3);
  std::sort(expected,expected+3);
  for(size_t k=0;k<3;++k)
    if(predicted[k]!=expected[k])return false;
  return true;
}
} // namespace

void setup() {
  Serial.begin(115200);
  delay(700);
#ifdef BEACON_EXPERIMENTAL_UPGRADE
  Serial.println("BEACON_ADAPTIVE_V2_STAGED_EXPERIMENTAL_ESP32C3");
#else
  Serial.println("BEACON_ADAPTIVE_V2_Q16_EXPERIMENTAL_ESP32C3");
#endif
  const size_t n=model::kInputDim;
  if(!beacon_raw_mlp::standardize_channels<
      model::kTimeSteps,model::kChannels>(
      control::kRawInput,model::kMu,model::kSigma,standardized)) {
    Serial.println("SELFTEST FAILED normalization");
    return;
  }
  float logits[model::kOutputDim];
  if(!beacon_raw_mlp::predict_logits<
      model::kInputDim,model::kHiddenDim,model::kOutputDim>(
      standardized,n,model::kW1,model::kB1,model::kW2,model::kB2,logits)) {
    Serial.println("SELFTEST FAILED classifier");
    return;
  }
  size_t predicted=0;
  for(size_t cls=1;cls<model::kOutputDim;++cls)
    if(logits[cls]>logits[predicted])predicted=cls;
  const bool class_match=(predicted==control::kExpectedClass);
  float baseline=0.0f;
  if(!margin_callback(standardized,n,nullptr,&baseline)) {
    Serial.println("SELFTEST FAILED baseline margin");
    return;
  }
  for(size_t group=0;group<3;++group)
    for(size_t bin=0;bin<12;++bin) {
      const size_t id=group*12+bin;
      components[id]={
          static_cast<uint16_t>(bin*model::kTimeSteps/12),
          static_cast<uint16_t>((bin+1)*model::kTimeSteps/12),
          static_cast<uint16_t>(group*3),
          static_cast<uint16_t>((group+1)*3)};
    }

#ifdef BEACON_EXPERIMENTAL_UPGRADE
  const int64_t t0=esp_timer_get_time();
  const bool ok=beacon_staged_upgrade::run<kM>(
      standardized,profile::kClassMeans+predicted*n,
      model::kTimeSteps,model::kChannels,components,baseline,
      margin_callback,nullptr,scratch,threshold::kQ16GapThreshold,&report);
  const int64_t elapsed=esp_timer_get_time()-t0;
  const bool q_match=ok && report.extra_model_evaluations==stage::kExpectedQ;
  const bool set_match=q_match && check_selected_set(
      report.selected,stage::kExpectedSelected,report.ranked_count);
  const bool top3_match=q_match && check_rank_top3_set(
      report.ranked,stage::kExpectedRanked);
  const bool gap_match=ok &&
      std::fabs(report.observed_gap16-stage::kExpectedGap16)<=.002f;
  Serial.printf("STAGED callback_ok=%u actual_model_calls=%u expected_calls=%u class=%u expected_class=%u\n",
                static_cast<unsigned>(ok),report.extra_model_evaluations,
                static_cast<unsigned>(stage::kExpectedQ),
                static_cast<unsigned>(predicted),
                static_cast<unsigned>(control::kExpectedClass));
  Serial.printf("SELFTEST class=%s candidate_set=%s top3_set=%s budget=%s gap16=%s\n",
                class_match?"PASS":"FAIL",
                set_match?"PASS":"FAIL",
                top3_match?"PASS":"FAIL",
                q_match?"PASS":"FAIL",
                gap_match?"PASS":"FAIL");
  Serial.printf("BOARD_EXPERIMENT one_window_staged_latency_us=%lld\n",
                static_cast<long long>(elapsed));
#else
  const int64_t t0=esp_timer_get_time();
  const bool ok=beacon_adaptive_v2::run_q<kM,kQ>(
      standardized,profile::kClassMeans+predicted*n,
      model::kTimeSteps,model::kChannels,components,baseline,
      margin_callback,nullptr,scratch,&report);
  const int64_t elapsed=esp_timer_get_time()-t0;
  const bool set_match=ok && check_selected_set(
      report.selected,control::kExpectedSelected,kQ);
  const bool top3_match=ok && check_rank_top3_set(
      report.ranked,control::kExpectedRanked);
  Serial.printf("Q16 callback_ok=%u actual_model_calls=%u class=%u expected_class=%u\n",
                static_cast<unsigned>(ok),report.model_evaluations,
                static_cast<unsigned>(predicted),
                static_cast<unsigned>(control::kExpectedClass));
  Serial.printf("SELFTEST class=%s candidate_set=%s top3_set=%s\n",
                class_match?"PASS":"FAIL",
                set_match?"PASS":"FAIL",
                top3_match?"PASS":"FAIL");
  Serial.printf("BOARD_EXPERIMENT one_window_q16_latency_us=%lld\n",
                static_cast<long long>(elapsed));
#endif
  Serial.printf("BOARD_EXPERIMENT free_heap_8bit=%u minimum_free_heap_8bit=%u\n",
                static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
                static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT)));
  Serial.println("NO_ENERGY_MEASUREMENT NO_FULL_PEAK_RAM_CERTIFICATION");
  Serial.println("DONE");
}

void loop() {
  delay(1000);
}
