#ifndef BEACON_DUAL_OPERATOR_AUDIT_HPP
#define BEACON_DUAL_OPERATOR_AUDIT_HPP

// Experimental fixed-Q16 BEACON audit, independent of Arduino and heap allocation.
// The caller provides a *real* model callback returning the top-two LOG-SCORE
// margin, a cached baseline margin, training-prior scores, and a train-only
// rival-class medoid. This is not an MCU-validated classifier implementation.
//
// Exactly 8 interpolation + 8 medoid inferences on the SAME candidates.
// One float[T*C] scratch buffer provided by caller; model tensor arena, feature
// extraction, sample/profile storage and callback stack are NOT included here.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace beacon_full_audit {

struct Component {
  uint16_t t0, t1;
  uint16_t c0, c1;
};

constexpr unsigned kTotalQ = 16;
constexpr unsigned kPerOperator = 8;

using MarginFn = bool (*)(const float* sample, size_t n_values,
                          void* user_ctx, float* out_top2_log_margin);

struct Result {
  uint16_t selected[kPerOperator];
  uint16_t ranked[kPerOperator];
  uint8_t common_top3;
  bool reported;
  uint8_t extra_model_evaluations;
};

inline void rank_desc(const float values[kPerOperator],
                      uint8_t sorted[kPerOperator]) {
  for (unsigned i = 0; i < kPerOperator; ++i) sorted[i] = static_cast<uint8_t>(i);
  // Stable sorting: equal measured changes keep training-prior order.
  for (unsigned i = 1; i < kPerOperator; ++i) {
    const uint8_t item = sorted[i];
    unsigned j = i;
    while (j > 0 && values[sorted[j-1]] < values[item]) {
      sorted[j] = sorted[j-1];
      --j;
    }
    sorted[j] = item;
  }
}

inline bool run_q16(const float* input, const float* rival_medoid,
                    size_t time_steps, size_t channels,
                    const Component* components, size_t component_count,
                    const float* cheap_prior, float baseline_top2_log_margin,
                    MarginFn model_margin, void* model_context,
                    float* scratch_window, Result* result) {
  if (!result) return false;
  *result = Result{};
  if (!input || !rival_medoid || !components || !cheap_prior
      || !model_margin || !scratch_window
      || scratch_window == input || scratch_window == rival_medoid
      || time_steps == 0 || channels == 0 || component_count < kPerOperator
      || component_count > UINT16_MAX
      || time_steps > SIZE_MAX / channels
      || time_steps * channels > SIZE_MAX / sizeof(float)
      || !std::isfinite(baseline_top2_log_margin)) return false;

  const size_t n_values = time_steps * channels;
  for (size_t j = 0; j < n_values; ++j) {
    if (!std::isfinite(input[j]) || !std::isfinite(rival_medoid[j]))
      return false;
  }
  for (size_t j = 0; j < component_count; ++j) {
    const Component& c = components[j];
    if (!std::isfinite(cheap_prior[j]) || !(c.t0 < c.t1) || c.t1 > time_steps
        || !(c.c0 < c.c1) || c.c1 > channels) return false;
  }

  // Stable top-8 selection, O(M*8), without a heap-allocated M-length sort.
  for (unsigned k = 0; k < kPerOperator; ++k) {
    size_t best = component_count;
    for (size_t j = 0; j < component_count; ++j) {
      bool used = false;
      for (unsigned prev = 0; prev < k; ++prev)
        if (result->selected[prev] == j) used = true;
      if (!used && (best == component_count || cheap_prior[j] > cheap_prior[best]))
        best = j;
    }
    if (best == component_count) return false;
    result->selected[k] = static_cast<uint16_t>(best);
  }

  float effects[2][kPerOperator] = {{0.0f}};
  for (unsigned op = 0; op < 2; ++op) {
    for (unsigned j = 0; j < kPerOperator; ++j) {
      const Component& c = components[result->selected[j]];
      std::memcpy(scratch_window, input, n_values * sizeof(float));
      const size_t length = c.t1 - c.t0;
      for (size_t t = c.t0; t < c.t1; ++t) {
        float w = 1.0f;
        if (op == 1) {
          const size_t edge = (length / 2 < 2 ? length / 2 : 2);
          const size_t relative = t - c.t0;
          if (edge && relative < edge)
            w = static_cast<float>(relative + 1) / static_cast<float>(edge + 1);
          else if (edge && relative >= length - edge)
            w = static_cast<float>(length - relative) / static_cast<float>(edge + 1);
        }
        for (size_t channel = c.c0; channel < c.c1; ++channel) {
          const size_t index = t * channels + channel;
          if (op == 0) {
            // Python neutralize_interp: edge intervals become zero; interior
            // intervals use np.linspace(left,right,length,endpoint=False).
            if (c.t0 == 0 || c.t1 == time_steps) {
              scratch_window[index] = 0.0f;
            } else {
              const float left = input[(c.t0-1)*channels+channel];
              const float right = input[c.t1*channels+channel];
              const float frac = static_cast<float>(t-c.t0) /
                                 static_cast<float>(length);
              scratch_window[index] = left + (right-left)*frac;
            }
          } else {
            scratch_window[index] = input[index]*(1.0f-w) +
                                    rival_medoid[index]*w;
          }
        }
      }
      float updated_margin = 0.0f;
      if (!model_margin(scratch_window,n_values,model_context,&updated_margin)
          || !std::isfinite(updated_margin)) return false;
      effects[op][j] = std::fabs(baseline_top2_log_margin - updated_margin);
      result->extra_model_evaluations++;
    }
  }

  uint8_t sorted[2][kPerOperator];
  rank_desc(effects[0], sorted[0]);
  rank_desc(effects[1], sorted[1]);
  unsigned common = 0;
  for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = 0; j < 3; ++j)
      if (sorted[0][i] == sorted[1][j]) ++common;
  result->common_top3 = static_cast<uint8_t>(common);
  result->reported = common >= 2;

  uint8_t positions[2][kPerOperator];
  for (unsigned op = 0; op < 2; ++op)
    for (unsigned i = 0; i < kPerOperator; ++i)
      positions[op][sorted[op][i]] = static_cast<uint8_t>(i);

  uint8_t fused[kPerOperator];
  for (unsigned i = 0; i < kPerOperator; ++i)
    fused[i] = static_cast<uint8_t>(i);
  for (unsigned i = 1; i < kPerOperator; ++i) {
    const uint8_t item = fused[i];
    unsigned j = i;
    const unsigned item_score = positions[0][item] + positions[1][item];
    while (j > 0 &&
           positions[0][fused[j-1]] + positions[1][fused[j-1]] > item_score) {
      fused[j] = fused[j-1];
      --j;
    }
    fused[j] = item;
  }
  for (unsigned i = 0; i < kPerOperator; ++i)
    result->ranked[i] = result->selected[fused[i]];
  return true;
}

} // namespace beacon_full_audit
#endif // BEACON_DUAL_OPERATOR_AUDIT_HPP
