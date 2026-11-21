#ifndef BEACON_STAGED_ADAPTIVE_UPGRADE_HPP
#define BEACON_STAGED_ADAPTIVE_UPGRADE_HPP

// Experimental train-calibrated Q16->Q32 upgrade. NOT the default BEACON.
// The fixed Q16 baseline is always executed first. It measures the relative
// third-vs-fourth gap of the 16 actual model-response magnitudes; when below
// a training-only threshold, it queries 16 NEW components, never resending
// the first 16. No heap, oracle ranks, held-out labels or test-time training.
//
// This port shares the four-score adaptive_v2 prior and neutralization with
// the verified C++ baseline. It has no physical MCU latency or energy results.

#include "adaptive_v2_audit.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace beacon_staged_upgrade {

template <size_t M>
struct Result {
  uint16_t selected[32];
  uint16_t ranked[32];
  float effects[32];
  float observed_gap16;
  unsigned extra_model_evaluations;
  unsigned ranked_count;
  bool upgraded;
  bool ambiguous_effect_rank;
  bool valid;
};

template <size_t M>
inline bool run(
    const float* input, const float* source_class_mean,
    size_t time_steps, size_t channels,
    const beacon_adaptive_v2::Component (&components)[M],
    float baseline_top2_margin,
    beacon_adaptive_v2::MarginFn model, void* ctx,
    float* scratch, float train_selected_gap16_threshold,
    Result<M>* out) {
  if (!out) return false;
  *out=Result<M>{};
  if (M<32 || M>65535 || !std::isfinite(train_selected_gap16_threshold)
      || train_selected_gap16_threshold<0.0f) return false;
  beacon_adaptive_v2::Result<M,16> first{};
  if (!beacon_adaptive_v2::run_q<M,16>(
      input,source_class_mean,time_steps,channels,components,
      baseline_top2_margin,model,ctx,scratch,&first)) {
    out->extra_model_evaluations=first.model_evaluations;
    return false;
  }
  for (size_t k=0;k<16;++k) {
    out->selected[k]=first.selected[k];
    out->effects[k]=first.effects[k];
  }
  out->extra_model_evaluations=16;
  // Same definition as Python staged_budget.measured_top3_gap.
  float sorted[16];
  for (size_t k=0;k<16;++k)sorted[k]=first.effects[k];
  for (size_t k=1;k<16;++k) {
    const float item=sorted[k];
    size_t j=k;
    while (j>0 && sorted[j-1]>item) {
      sorted[j]=sorted[j-1];
      --j;
    }
    sorted[j]=item;
  }
  out->observed_gap16=(
      (sorted[13]-sorted[12]) /
      ((sorted[15]>1e-12f)?sorted[15]:1e-12f));
  const bool extend=(out->observed_gap16 < train_selected_gap16_threshold);
  const size_t count=extend?32:16;

  if (extend) {
    // Select actual top 32 according to the exact same training-only prior.
    // Stable component-ID tiebreak; first 16 must agree with the base stage.
    uint16_t order[M];
    for(size_t i=0;i<M;++i)order[i]=static_cast<uint16_t>(i);
    for(size_t k=0;k<32;++k) {
      size_t best=k;
      for(size_t j=k+1;j<M;++j)
        if(first.prior[order[j]]>first.prior[order[best]])best=j;
      const uint16_t tmp=order[k];order[k]=order[best];order[best]=tmp;
      if(k<16) {
        if(first.selected[k]!=order[k])return false;
      } else out->selected[k]=order[k];
    }
    const size_t n=time_steps*channels;
    for(size_t k=16;k<32;++k) {
      const auto& c=components[out->selected[k]];
      std::memcpy(scratch,input,n*sizeof(float));
      for(size_t t=c.t0;t<c.t1;++t)
        for(size_t channel=c.c0;channel<c.c1;++channel) {
          const size_t index=t*channels+channel;
          if(c.t0==0 || c.t1==time_steps) scratch[index]=0.0f;
          else {
            const float left=input[(c.t0-1)*channels+channel];
            const float right=input[c.t1*channels+channel];
            const double relative=static_cast<double>(t-c.t0)/
                static_cast<double>(c.t1-c.t0);
            scratch[index]=static_cast<float>(
                static_cast<double>(left)+
                (static_cast<double>(right)-left)*relative);
          }
        }
      float updated=0.0f;
      if(!model(scratch,n,ctx,&updated)||!std::isfinite(updated))
        return false;
      out->effects[k]=std::fabs(baseline_top2_margin-updated);
      out->extra_model_evaluations++;
    }
  }
  uint16_t sorted_ids[32];
  for(size_t k=0;k<count;++k) sorted_ids[k]=static_cast<uint16_t>(k);
  for(size_t k=1;k<count;++k) {
    const uint16_t item=sorted_ids[k];
    size_t j=k;
    while(j>0 && out->effects[sorted_ids[j-1]]<out->effects[item]) {
      sorted_ids[j]=sorted_ids[j-1];--j;
    }
    sorted_ids[j]=item;
  }
  for(size_t k=0;k<count;++k)
    out->ranked[k]=out->selected[sorted_ids[k]];
  for(size_t k=1;k<count;++k)
    if(std::fabs(out->effects[sorted_ids[k-1]]-
                 out->effects[sorted_ids[k]])<=1e-6f)
      out->ambiguous_effect_rank=true;
  out->ranked_count=static_cast<unsigned>(count);
  out->upgraded=extend;
  out->valid=true;
  return true;
}

} // namespace beacon_staged_upgrade
#endif
