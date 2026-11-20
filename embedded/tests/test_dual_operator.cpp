// Host-compiled C++11 acceptance tests for the heap-free streaming Q16 port.
// The callback is deliberately fake. No MCU speed, power or inference model
// correctness can be inferred from these tests.
#include "../dual_operator_audit.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>

using namespace beacon_full_audit;

namespace {
constexpr size_t kT=12;
constexpr size_t kC=9;
constexpr size_t kM=12;

struct CallbackContext {
  int calls;
  bool disagree;
  int reject_at;
};

bool model_callback(const float* sample, size_t n, void* user, float* margin) {
  auto* ctx=static_cast<CallbackContext*>(user);
  assert(n==kT*kC);
  if (ctx->calls == ctx->reject_at) return false;
  if (ctx->calls == 0) {
    // First selected component occupies times 0..2, channels 0..2;
    // interpolation on the left edge neutralizes them to zero.
    assert(sample[0] == 0.0f);
    assert(sample[3] == 2.0f);
  }
  if (ctx->calls == 8) {
    // Medoid values 6 with taper=2 and n=3 -> weights .5,1,.5.
    assert(std::fabs(sample[0] - 4.0f) < 1e-6f);
    assert(std::fabs(sample[9] - 6.0f) < 1e-6f);
    assert(std::fabs(sample[18] - 4.0f) < 1e-6f);
    assert(sample[3] == 2.0f);
  }
  const unsigned index=static_cast<unsigned>(ctx->calls % 8);
  const float impact = (ctx->calls>=8 && ctx->disagree)
      ? static_cast<float>(index+1)/10.0f
      : static_cast<float>(8-index)/10.0f;
  *margin=2.0f-impact;
  ctx->calls++;
  return true;
}

void init(Component (&comps)[kM],float (&prior)[kM]) {
  unsigned j=0;
  for (unsigned group=0;group<3;++group)
    for (unsigned bin=0;bin<4;++bin) {
      comps[j] = Component{
          static_cast<uint16_t>(bin*3),
          static_cast<uint16_t>((bin+1)*3),
          static_cast<uint16_t>(group*3),
          static_cast<uint16_t>((group+1)*3)};
      prior[j] = static_cast<float>(kM-j);
      j++;
    }
}
}  // namespace

int main() {
  float original[kT*kC];
  float medoid[kT*kC];
  float scratch[kT*kC];
  for (size_t k=0;k<kT*kC;++k) {
    original[k]=2.0f;
    medoid[k]=6.0f;
    scratch[k]=-7.0f;
  }
  Component comps[kM];
  float prior[kM];
  init(comps,prior);
  Result result{};

  CallbackContext negative{0,true,-1};
  bool ok=run_q16(original,medoid,kT,kC,comps,kM,prior,2.0f,
                  model_callback,&negative,scratch,&result);
  assert(ok);
  assert(negative.calls==16);
  assert(result.extra_model_evaluations==16);
  assert(result.common_top3==0);
  assert(!result.reported);
  for(unsigned j=0;j<8;++j)
    assert(result.selected[j]==j);

  CallbackContext positive{0,false,-1};
  ok=run_q16(original,medoid,kT,kC,comps,kM,prior,2.0f,
             model_callback,&positive,scratch,&result);
  assert(ok);
  assert(result.common_top3==3);
  assert(result.reported);
  assert(result.ranked[0]==0);
  assert(result.ranked[1]==1);
  assert(result.ranked[2]==2);

  // Callback failures must never be silently treated as valid Q16 results.
  CallbackContext failure{0,false,7};
  ok=run_q16(original,medoid,kT,kC,comps,kM,prior,2.0f,
             model_callback,&failure,scratch,&result);
  assert(!ok);
  assert(result.extra_model_evaluations==7);

  float bad_prior[kM];
  for(unsigned j=0;j<kM;++j)bad_prior[j]=prior[j];
  bad_prior[1]=std::numeric_limits<float>::quiet_NaN();
  CallbackContext never{0,false,-1};
  ok=run_q16(original,medoid,kT,kC,comps,kM,bad_prior,2.0f,
             model_callback,&never,scratch,&result);
  assert(!ok && never.calls==0);

  ok=run_q16(original,medoid,kT,kC,comps,kM,prior,2.0f,
             model_callback,&never,original,&result);
  assert(!ok && never.calls==0);

  std::cout << "Q16_STREAM_PORT_OK calls=16 taper=2 no_heap=1" << std::endl;
  return 0;
}
