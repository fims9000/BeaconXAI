// Full end-to-end parity for BASELINE adaptive_v2 Q16, not dual operator.
// Raw UCI HAR -> C++ train normalization -> trained FP32 MLP -> predicted
// class -> four cheap prior components -> top-Q set -> 16 interventions
// -> moving-margin ranks. Compare to independent original Python baselines.
// Flag cutoff/response ties; do not pretend NumPy argpartition has an order.
#include "../adaptive_v2_audit.hpp"
#include "../raw_mlp_inference.hpp"
#include "beacon_mlp_model.hpp"
#include "beacon_adaptive_profiles.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

namespace {
namespace weights=beacon_exported_mlp;
namespace profiles=beacon_exported_profiles;
constexpr size_t kGroups=3,kBins=12,kM=kGroups*kBins,kQ=16;

bool trained_margin(const float* x,size_t n,void*,float* margin) {
  return beacon_raw_mlp::predict_top2_log_margin<
      weights::kInputDim,weights::kHiddenDim,weights::kOutputDim>(
      x,n,weights::kW1,weights::kB1,weights::kW2,weights::kB2,margin);
}

bool read_exact(std::ifstream& fp,void* to,size_t length) {
  return static_cast<bool>(fp.read(static_cast<char*>(to),length));
}
}

int main(int argc,char**argv) {
  if(argc!=2) {
    std::cerr<<"usage: cpp_baseline_adaptive_v2 fixture.bin"<<std::endl;
    return 2;
  }
  std::ifstream fp(argv[1],std::ios::binary);
  if(!fp)return 3;
  uint32_t n=0,d=0,m=0,q=0;
  if(!read_exact(fp,&n,4)||!read_exact(fp,&d,4)||
     !read_exact(fp,&m,4)||!read_exact(fp,&q,4)||
     n==0||d!=weights::kInputDim||m!=kM||q!=kQ||
     profiles::kClasses!=weights::kOutputDim||
     profiles::kWindow!=weights::kInputDim)return 4;

  beacon_adaptive_v2::Component comps[kM];
  for(size_t group=0;group<kGroups;++group)
    for(size_t bin=0;bin<kBins;++bin) {
      const size_t index=group*kBins+bin;
      comps[index]={
          static_cast<uint16_t>(bin*weights::kTimeSteps/kBins),
          static_cast<uint16_t>((bin+1)*weights::kTimeSteps/kBins),
          static_cast<uint16_t>(group*3),
          static_cast<uint16_t>((group+1)*3)};
    }
  std::vector<float> raw(d),standardized(d),scratch(d);
  double max_prior_diff=0.0;
  unsigned candidate_set_mismatch=0,rank_mismatch=0;
  unsigned unambiguous_sets=0,unambiguous_ranks=0;
  unsigned ambiguous_prior=0,ambiguous_effect=0;

  for(uint32_t i=0;i<n;++i) {
    uint32_t expected_class=0;
    double cutoff_gap=0,effect_min_sep=0;
    double expected_prior[kM];
    uint16_t expected_selected[kQ],expected_ranked[kQ];
    if(!read_exact(fp,raw.data(),d*sizeof(float))||
       !read_exact(fp,&expected_class,4)||
       !read_exact(fp,&cutoff_gap,sizeof(double))||
       !read_exact(fp,&effect_min_sep,sizeof(double))||
       !read_exact(fp,expected_prior,sizeof(expected_prior))||
       !read_exact(fp,expected_selected,sizeof(expected_selected))||
       !read_exact(fp,expected_ranked,sizeof(expected_ranked)))return 5;
    if(!beacon_raw_mlp::standardize_channels<
          weights::kTimeSteps,weights::kChannels>(
          raw.data(),weights::kMu,weights::kSigma,standardized.data()))
      return 6;

    float logits[weights::kOutputDim];
    if(!beacon_raw_mlp::predict_logits<
          weights::kInputDim,weights::kHiddenDim,weights::kOutputDim>(
          standardized.data(),d,weights::kW1,weights::kB1,
          weights::kW2,weights::kB2,logits))return 7;
    size_t predicted=0;
    for(size_t k=1;k<weights::kOutputDim;++k)
      if(logits[k]>logits[predicted])predicted=k;
    if(predicted!=expected_class) {
      std::cerr<<"class mismatch window="<<i
               <<" expected="<<expected_class
               <<" C++="<<predicted<<std::endl;
      return 8;
    }
    float original_margin=0.0f;
    if(!trained_margin(standardized.data(),d,nullptr,&original_margin))
      return 9;

    const float* class_profile=profiles::kClassMeans+predicted*d;
    beacon_adaptive_v2::Result<kM,kQ> result{};
    if(!beacon_adaptive_v2::run_q<kM,kQ>(
        standardized.data(),class_profile,weights::kTimeSteps,
        weights::kChannels,comps,original_margin,
        trained_margin,nullptr,scratch.data(),&result)||
       !result.valid||result.model_evaluations!=kQ)return 10;
    for(size_t j=0;j<kM;++j) {
      const double diff=std::fabs(result.prior[j]-expected_prior[j]);
      max_prior_diff=std::max(max_prior_diff,diff);
      if(diff>0.002) {
        std::cerr<<"prior mismatch window="<<i<<" component="<<j
                 <<" difference="<<diff<<std::endl;
        return 11;
      }
    }

    uint16_t actual_selected[kQ];
    for(size_t k=0;k<kQ;++k)actual_selected[k]=result.selected[k];
    std::sort(actual_selected,actual_selected+kQ);
    bool same_set=true;
    for(size_t k=0;k<kQ;++k)
      if(actual_selected[k]!=expected_selected[k])same_set=false;
    if(!same_set) {
      candidate_set_mismatch++;
      if(cutoff_gap>1e-5 && !result.ambiguous_prior_cutoff) {
        std::cerr<<"nonambiguous Q16 selected-set mismatch on sample="<<i
                 <<" cutoff_gap="<<cutoff_gap<<std::endl;
        return 12;
      }
    } else if(cutoff_gap>1e-5 && !result.ambiguous_prior_cutoff)
      unambiguous_sets++;

    bool same_rank=true;
    for(size_t k=0;k<kQ;++k)
      if(result.ranked[k]!=expected_ranked[k])same_rank=false;
    if(!same_rank) {
      rank_mismatch++;
      // NumPy argpartition order is not specified, so even with a strict
      // measured-effect rank C++ cannot assert order when the candidate SET
      // changed at an ambiguous prior cutoff.
      if(same_set && effect_min_sep>1e-4 &&
         !result.ambiguous_effect_rank) {
        std::cerr<<"nonambiguous full Q16 rank mismatch on sample="<<i
                 <<" min_effect_gap="<<effect_min_sep<<std::endl;
        return 13;
      }
    } else if(effect_min_sep>1e-4 && same_set)
      unambiguous_ranks++;

    ambiguous_prior+=(cutoff_gap<=1e-5||result.ambiguous_prior_cutoff);
    ambiguous_effect+=(effect_min_sep<=1e-4||result.ambiguous_effect_rank);
  }
  if(!fp.good()&&!fp.eof())return 14;
  if(unambiguous_sets==0) {
    std::cerr<<"No strict-cutoff samples: parity is inconclusive"<<std::endl;
    return 15;
  }
  std::cout<<"ADAPTIVE_V2_Q16_CPP_PARITY_OK"
           <<" samples="<<n
           <<" true_model_calls="<<n*kQ
           <<" max_prior_diff="<<max_prior_diff
           <<" set_mismatches="<<candidate_set_mismatch
           <<" rank_mismatches="<<rank_mismatch
           <<" strict_set_matches="<<unambiguous_sets
           <<" strict_rank_matches="<<unambiguous_ranks
           <<" ambiguous_prior="<<ambiguous_prior
           <<" ambiguous_effect="<<ambiguous_effect<<std::endl;
  return 0;
}
