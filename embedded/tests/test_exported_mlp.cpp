// Compare exported FP32 MLP with independently generated FP64 sklearn logits.
// Test data is generated from untouched subject-held-out UCI HAR windows.
// This is host numeric parity, NOT physical MCU time, memory or energy.
#include "../raw_mlp_inference.hpp"
#include "../dual_operator_audit.hpp"
#include "beacon_mlp_model.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
namespace weights = beacon_exported_mlp;

bool trained_margin(const float* input, size_t count, void*, float* gap) {
  return beacon_raw_mlp::predict_top2_log_margin<
      weights::kInputDim, weights::kHiddenDim, weights::kOutputDim>(
      input,count,weights::kW1,weights::kB1,
      weights::kW2,weights::kB2,gap);
}

bool read_exact(std::ifstream& fp, void* dest, size_t bytes) {
  return static_cast<bool>(fp.read(static_cast<char*>(dest),bytes));
}
} // namespace

int main(int argc,char** argv) {
  if(argc!=2) {
    std::cerr << "usage: mlp_parity_test fixture.bin" << std::endl;
    return 2;
  }
  std::ifstream f(argv[1],std::ios::binary);
  if(!f) return 3;
  uint32_t n=0,d=0;
  if(!read_exact(f,&n,sizeof(n)) || !read_exact(f,&d,sizeof(d))
      || n==0 || d!=weights::kInputDim) return 4;
  std::vector<float> raw(d);
  std::vector<float> input(d);
  double max_absolute_error=0.0;
  for(uint32_t i=0;i<n;++i) {
    float expected=0.0f,actual=0.0f;
    if(!read_exact(f,raw.data(),sizeof(float)*d)
        || !read_exact(f,&expected,sizeof(expected))) return 5;
    // This now checks the entire preprocessing + MLP path, not just a
    // forward pass on Python-standardized inputs.
    if(!beacon_raw_mlp::standardize_channels<
        weights::kTimeSteps,weights::kChannels>(
        raw.data(),weights::kMu,weights::kSigma,input.data())) return 11;
    if(!trained_margin(input.data(),d,nullptr,&actual)) return 6;
    const double err=std::fabs(static_cast<double>(actual)-expected);
    if(err>max_absolute_error)max_absolute_error=err;
    if(err>0.01+0.0002*std::fabs(expected)) {
      std::cerr << "margin parity mismatch sample=" << i
                << " expected=" << expected << " actual=" << actual
                << " error=" << err << std::endl;
      return 7;
    }
  }
  if(!f.good() && !f.eof()) return 8;
  {
    const float x[1]={1.0f},mu[1]={0.0f},bad_sigma[1]={0.0f};
    float dest[1]={0.0f};
    if(beacon_raw_mlp::standardize_channels<1,1>(
        x,mu,bad_sigma,dest)) return 12;
  }

  // One actual trained model callback through the experimental streaming
  // Q16 interface; this is only a structural smoke, not donor validity:
  // the same input acts as a dummy medoid in this test.
  beacon_full_audit::Component comp[36];
  float prior[36];
  unsigned k=0;
  for(unsigned group=0;group<3;++group)
    for(unsigned bin=0;bin<12;++bin) {
      const size_t start=(bin*weights::kTimeSteps)/12;
      const size_t stop=((bin+1)*weights::kTimeSteps)/12;
      comp[k]={
          static_cast<uint16_t>(start),static_cast<uint16_t>(stop),
          static_cast<uint16_t>(group*3),static_cast<uint16_t>((group+1)*3)};
      prior[k]=static_cast<float>(36-k);
      k++;
    }
  beacon_full_audit::Result outcome{};
  std::vector<float> scratch(d);
  float baseline=0.0f;
  if(!trained_margin(input.data(),d,nullptr,&baseline)) return 9;
  const bool ok=beacon_full_audit::run_q16(
      input.data(),input.data(),weights::kTimeSteps,weights::kChannels,
      comp,36,prior,baseline,trained_margin,nullptr,scratch.data(),&outcome);
  if(!ok || outcome.extra_model_evaluations!=16) return 10;
  std::cout << "MLP_FP32_PARITY_OK raw_normalization=1 samples=" << n
            << " max_abs_margin_error=" << max_absolute_error
            << " real_forward_calls=16"
            << std::endl;
  return 0;
}
