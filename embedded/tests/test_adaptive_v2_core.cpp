// Allocation-free baseline adaptive_v2 Q16 unit tests without external data.
#include "../adaptive_v2_audit.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {
constexpr size_t T=24,C=9,M=36,Q=16;
struct Context { int calls; int fail_on; };

bool neutral_margin(const float* input,size_t n,void* ctx,float* out) {
  auto* c=static_cast<Context*>(ctx);
  if(c->calls==c->fail_on)return false;
  double sum=0.0;
  for(size_t i=0;i<n;++i)sum+=input[i];
  *out=2.0f-static_cast<float>(sum/1000.0);
  c->calls++;
  return true;
}
}

int main() {
  float input[T*C]={0.0f},mean[T*C]={0.0f},scratch[T*C]={0.0f};
  beacon_adaptive_v2::Component components[M];
  for(size_t group=0;group<3;++group)
    for(size_t bin=0;bin<12;++bin) {
      const size_t i=group*12+bin;
      components[i]={
          static_cast<uint16_t>(bin*2),
          static_cast<uint16_t>(bin*2+2),
          static_cast<uint16_t>(group*3),
          static_cast<uint16_t>(group*3+3)};
    }
  beacon_adaptive_v2::Result<M,Q> output{};
  Context ok{0,-1};
  bool valid=beacon_adaptive_v2::run_q<M,Q>(
      input,mean,T,C,components,2.0f,
      neutral_margin,&ok,scratch,&output);
  assert(valid&&output.valid&&ok.calls==16);
  assert(output.model_evaluations==16);
  assert(output.ambiguous_prior_cutoff);
  assert(output.ambiguous_effect_rank);
  for(size_t i=0;i<M;++i)
    assert(output.prior[i]==0.0);
  for(size_t i=0;i<Q;++i)
    assert(output.selected[i]==i && output.ranked[i]==i);

  Context failure{0,7};
  valid=beacon_adaptive_v2::run_q<M,Q>(
      input,mean,T,C,components,2.0f,
      neutral_margin,&failure,scratch,&output);
  assert(!valid);
  assert(output.model_evaluations==7);
  assert(!output.valid);

  Context never{0,-1};
  mean[0]=std::numeric_limits<float>::quiet_NaN();
  valid=beacon_adaptive_v2::run_q<M,Q>(
      input,mean,T,C,components,2.0f,
      neutral_margin,&never,scratch,&output);
  assert(!valid&&never.calls==0);
  mean[0]=0.0f;
  components[0].t1=25;
  valid=beacon_adaptive_v2::run_q<M,Q>(
      input,mean,T,C,components,2.0f,
      neutral_margin,&never,scratch,&output);
  assert(!valid&&never.calls==0);
  std::cout<<"ADAPTIVE_V2_CORE_OK true_queries=16 ties_flagged=1"<<std::endl;
  return 0;
}
