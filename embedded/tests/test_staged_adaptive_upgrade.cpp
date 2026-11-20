// Host C++11 tests: physical board behavior is NOT measured here.
#include "../staged_adaptive_upgrade.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {
constexpr size_t T=24,C=9,M=36;
struct CallState { int calls; int fail_at; };
bool model(const float*,size_t n,void* pointer,float* margin) {
  auto* state=static_cast<CallState*>(pointer);
  assert(n==T*C);
  if(state->calls==state->fail_at)return false;
  ++state->calls;
  *margin=2.0f; // tied effects; selection must remain deterministic.
  return true;
}
void setup(beacon_adaptive_v2::Component (&comps)[M]) {
  for(size_t group=0;group<3;++group)
    for(size_t bin=0;bin<12;++bin) {
      const size_t id=group*12+bin;
      comps[id]={
          static_cast<uint16_t>(bin*2),
          static_cast<uint16_t>(bin*2+2),
          static_cast<uint16_t>(group*3),
          static_cast<uint16_t>(group*3+3)};
    }
}
}

int main() {
  float window[T*C]={0.0f};
  float profile[T*C]={0.0f};
  float scratch[T*C]={0.0f};
  beacon_adaptive_v2::Component comps[M];
  setup(comps);
  beacon_staged_upgrade::Result<M> result{};
  CallState q16{0,-1};
  bool ok=beacon_staged_upgrade::run<M>(
      window,profile,T,C,comps,2.0f,
      model,&q16,scratch,0.0f,&result);
  assert(ok&&result.valid&&q16.calls==16);
  assert(result.extra_model_evaluations==16);
  assert(result.ranked_count==16);
  assert(!result.upgraded);
  assert(result.observed_gap16==0.0f);
  for(unsigned i=0;i<16;++i) {
    assert(result.selected[i]==i);
    assert(result.ranked[i]==i);
  }

  CallState q32{0,-1};
  ok=beacon_staged_upgrade::run<M>(
      window,profile,T,C,comps,2.0f,
      model,&q32,scratch,0.1f,&result);
  assert(ok&&result.valid&&q32.calls==32);
  assert(result.extra_model_evaluations==32);
  assert(result.ranked_count==32);
  assert(result.upgraded);
  for(unsigned i=0;i<32;++i) {
    assert(result.selected[i]==i);
    assert(result.ranked[i]==i);
  }

  CallState failure{0,20};
  ok=beacon_staged_upgrade::run<M>(
      window,profile,T,C,comps,2.0f,
      model,&failure,scratch,0.1f,&result);
  assert(!ok&&!result.valid);
  assert(failure.calls==20);
  assert(result.extra_model_evaluations==20);

  CallState never{0,-1};
  ok=beacon_staged_upgrade::run<M>(
      window,profile,T,C,comps,2.0f,
      model,&never,scratch,
      std::numeric_limits<float>::quiet_NaN(),&result);
  assert(!ok&&never.calls==0);
  std::cout<<"STAGED_Q16_UPGRADE_CXX_CORE_OK calls=16_or_32 no_second_first16=1"<<std::endl;
  return 0;
}
