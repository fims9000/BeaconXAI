// Host-only parity for experimental train-calibrated Q16 -> optional Q32.
// This uses REAL exported FP32 classifier and inner-train-calibrated threshold.
// It does not measure energy, peak RAM or execution on a physical ESP32-C3.
#include "../staged_adaptive_upgrade.hpp"
#include "../raw_mlp_inference.hpp"
#include "beacon_mlp_model.hpp"
#include "beacon_adaptive_profiles.hpp"
#include "beacon_staged_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
namespace model=beacon_exported_mlp;
namespace profiles=beacon_exported_profiles;
namespace policy=beacon_exported_staged;
constexpr size_t kM=36,kQMax=32;

bool predict_margin(const float* x,size_t n,void*,float* out) {
  return beacon_raw_mlp::predict_top2_log_margin<
      model::kInputDim,model::kHiddenDim,model::kOutputDim>(
      x,n,model::kW1,model::kB1,model::kW2,model::kB2,out);
}
bool read_exact(std::ifstream& fp,void* data,size_t count) {
  return static_cast<bool>(fp.read(static_cast<char*>(data),count));
}
}

int main(int argc,char**argv) {
  if(argc!=2) {
    std::cerr<<"usage: staged_cpp_parity staged_fixture.bin"<<std::endl;
    return 2;
  }
  std::ifstream fp(argv[1],std::ios::binary);
  if(!fp)return 3;
  uint32_t count=0,dimension=0,components=0,max_q=0;
  if(!read_exact(fp,&count,4)||!read_exact(fp,&dimension,4)
      ||!read_exact(fp,&components,4)||!read_exact(fp,&max_q,4)
      ||count==0||dimension!=model::kInputDim||components!=kM
      ||max_q!=kQMax||profiles::kWindow!=model::kInputDim
      ||profiles::kClasses!=model::kOutputDim)return 4;

  beacon_adaptive_v2::Component parts[kM];
  for(size_t group=0;group<3;++group)
    for(size_t bin=0;bin<12;++bin) {
      const size_t i=group*12+bin;
      parts[i]={
        static_cast<uint16_t>(bin*model::kTimeSteps/12),
        static_cast<uint16_t>((bin+1)*model::kTimeSteps/12),
        static_cast<uint16_t>(group*3),
        static_cast<uint16_t>((group+1)*3)};
    }
  std::vector<float> raw(dimension),standardized(dimension),scratch(dimension);
  unsigned exact_choices=0,ambiguous_choice=0,matching_sets=0;
  unsigned exact_ranks=0,ambiguous_ranks=0,chosen_q32=0;
  unsigned total_calls=0;
  double largest_gap_error=0.;
  for(uint32_t sample=0;sample<count;++sample) {
    uint32_t expected_class=0,expected_q=0;
    double expected_gap=0.,expected_min_sep=0.;
    uint16_t selected[kQMax],ranked[kQMax];
    if(!read_exact(fp,raw.data(),dimension*sizeof(float))
        ||!read_exact(fp,&expected_class,4)
        ||!read_exact(fp,&expected_q,4)
        ||!read_exact(fp,&expected_gap,sizeof(double))
        ||!read_exact(fp,&expected_min_sep,sizeof(double))
        ||!read_exact(fp,selected,sizeof(selected))
        ||!read_exact(fp,ranked,sizeof(ranked)))return 5;
    if(expected_q!=16&&expected_q!=32)return 6;
    if(!beacon_raw_mlp::standardize_channels<
         model::kTimeSteps,model::kChannels>(
         raw.data(),model::kMu,model::kSigma,standardized.data()))return 7;

    float logits[model::kOutputDim];
    if(!beacon_raw_mlp::predict_logits<
        model::kInputDim,model::kHiddenDim,model::kOutputDim>(
        standardized.data(),dimension,model::kW1,model::kB1,
        model::kW2,model::kB2,logits))return 8;
    unsigned predicted=0;
    for(unsigned j=1;j<model::kOutputDim;++j)
      if(logits[j]>logits[predicted])predicted=j;
    if(predicted!=expected_class) {
      std::cerr<<"predicted class mismatch sample="<<sample<<std::endl;
      return 9;
    }
    float baseline=0.f;
    if(!predict_margin(standardized.data(),dimension,nullptr,&baseline))return 10;
    beacon_staged_upgrade::Result<kM> result{};
    if(!beacon_staged_upgrade::run<kM>(
        standardized.data(),profiles::kClassMeans+predicted*dimension,
        model::kTimeSteps,model::kChannels,parts,baseline,
        predict_margin,nullptr,scratch.data(),
        policy::kQ16GapThreshold,&result)||!result.valid)return 11;
    const unsigned actual_q=result.extra_model_evaluations;
    if(actual_q!=result.ranked_count || (actual_q!=16&&actual_q!=32))return 12;
    total_calls+=actual_q;
    chosen_q32+=(actual_q==32);
    const double gapdiff=std::fabs(result.observed_gap16-expected_gap);
    largest_gap_error=std::max(largest_gap_error,gapdiff);
    if(gapdiff>2e-3) {
      std::cerr<<"measured gap mismatch sample="<<sample
               <<" absolute_error="<<gapdiff<<std::endl;
      return 13;
    }
    if(actual_q!=expected_q) {
      if(std::fabs(expected_gap-policy::kQ16GapThreshold)>2e-3) {
        std::cerr<<"nonambiguous budget mismatch sample="<<sample
                 <<" expected_Q="<<expected_q<<" actual_Q="<<actual_q
                 <<" gap="<<expected_gap<<std::endl;
        return 14;
      }
      ambiguous_choice++;
      continue;
    }
    exact_choices++;
    uint16_t actual_set[kQMax];
    for(size_t j=0;j<actual_q;++j)actual_set[j]=result.selected[j];
    std::sort(actual_set,actual_set+actual_q);
    for(size_t j=0;j<actual_q;++j)
      if(actual_set[j]!=selected[j]) {
        std::cerr<<"candidate set mismatch sample="<<sample
                 <<" Q="<<actual_q<<std::endl;
        return 15;
      }
    matching_sets++;
    bool same_rank=true;
    for(size_t j=0;j<actual_q;++j)
      if(result.ranked[j]!=ranked[j])same_rank=false;
    if(!same_rank) {
      if(expected_min_sep>1e-4 && !result.ambiguous_effect_rank) {
        std::cerr<<"nonambiguous rank mismatch sample="<<sample
                 <<" gap="<<expected_min_sep<<std::endl;
        return 16;
      }
      ambiguous_ranks++;
    } else exact_ranks++;
  }
  if(!fp.good()&&!fp.eof())return 17;
  if(exact_choices==0 || matching_sets==0)return 18;
  std::cout<<"STAGED_Q16_Q32_CPP_PARITY_OK samples="<<count
           <<" true_model_calls="<<total_calls
           <<" Q32_windows="<<chosen_q32
           <<" matching_budget_choices="<<exact_choices
           <<" boundary_ambiguous_budget_choices="<<ambiguous_choice
           <<" matching_candidate_sets="<<matching_sets
           <<" matching_full_rankings="<<exact_ranks
           <<" near_tie_rankings="<<ambiguous_ranks
           <<" max_gap_abs_error="<<largest_gap_error
           <<" learned_gap_threshold="<<policy::kQ16GapThreshold
           <<std::endl;
  return 0;
}
