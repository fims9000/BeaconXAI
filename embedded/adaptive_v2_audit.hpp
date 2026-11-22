#ifndef BEACON_ADAPTIVE_V2_AUDIT_HPP
#define BEACON_ADAPTIVE_V2_AUDIT_HPP

// Experimental allocation-free port of BEACON adaptive_v2_original_q16.
// Matches the *four-term class-mean cheap score* and moving top-two margin
// intervention. Uses deterministic descending-score shortlist ties rather
// than NumPy's implementation-dependent np.argpartition ordering. Evaluate
// selected SET and rank separately in parity tests; tied cutoffs are flagged.
// No classifier parameters, training, or MCU timer in this standalone header.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace beacon_adaptive_v2 {

struct Component {
  uint16_t t0, t1, c0, c1;
};

using MarginFn = bool (*)(const float* standardized_window, size_t values,
                          void* context, float* out_top2_log_margin);

template <size_t M, size_t Q>
struct Result {
  uint16_t selected[Q];
  uint16_t ranked[Q];
  double prior[M];
  double cutoff_gap;
  float effects[Q];
  unsigned model_evaluations;
  bool ambiguous_prior_cutoff;
  bool ambiguous_effect_rank;
  bool valid;
};

template <size_t M>
inline void zscore(const double (&raw)[M], double (&normalized)[M]) {
  double sum = 0.0;
  for (size_t i=0; i<M; ++i) sum += raw[i];
  const double mu=sum/static_cast<double>(M);
  double variance=0.0;
  for (size_t i=0; i<M; ++i) {
    const double delta=raw[i]-mu;
    variance+=delta*delta;
  }
  const double sd=std::sqrt(variance/static_cast<double>(M));
  for (size_t i=0; i<M; ++i)
    normalized[i]=(sd<1e-12)?0.0:(raw[i]-mu)/sd;
}

template <size_t M, size_t Q>
inline bool cheap_prior(
    const float* x, const float* source_class_mean,
    size_t time_steps, size_t channels,
    const Component (&components)[M],
    double (&combined)[M]) {
  if (!x || !source_class_mean || !time_steps || !channels ||
      M<3 || Q<1 || Q>M || M>65535 ||
      time_steps>SIZE_MAX/channels ||
      time_steps*channels>SIZE_MAX/sizeof(float)) return false;
  const size_t n=time_steps*channels;
  for (size_t i=0;i<n;++i)
    if (!std::isfinite(x[i]) || !std::isfinite(source_class_mean[i]))
      return false;

  double energy[M], variance[M], profile_dist[M], mean_dev[M];
  for (size_t k=0;k<M;++k) {
    const Component& c=components[k];
    if (!(c.t0<c.t1) || c.t1>time_steps ||
        !(c.c0<c.c1) || c.c1>channels) return false;
    const double area=static_cast<double>(
        (c.t1-c.t0)*(c.c1-c.c0));
    double s=0.0, ss=0.0, sp=0.0, sm=0.0;
    for (size_t t=c.t0;t<c.t1;++t)
      for (size_t ch=c.c0;ch<c.c1;++ch) {
        const size_t index=t*channels+ch;
        const double v=x[index], m=source_class_mean[index];
        s+=v;
        ss+=v*v;
        sp+=std::fabs(v-m);
        sm+=m;
      }
    const double mean=s/area;
    const double profmean=sm/area;
    energy[k]=ss/area;
    // population variance, identical definition to numpy.var(patch).
    variance[k]=ss/area-mean*mean;
    if (variance[k]<0.0 && variance[k]>-1e-12) variance[k]=0.0;
    profile_dist[k]=sp/area;
    mean_dev[k]=std::fabs(mean-profmean);
    if (!std::isfinite(variance[k])) return false;
  }
  double ze[M],zv[M],zp[M],zm[M];
  zscore(energy,ze);
  zscore(variance,zv);
  zscore(profile_dist,zp);
  zscore(mean_dev,zm);
  for(size_t k=0;k<M;++k) {
    combined[k]=ze[k]+zv[k]+zp[k]+zm[k];
    if(!std::isfinite(combined[k])) return false;
  }
  return true;
}

template <size_t M, size_t Q>
inline bool run_q(
    const float* standardized_input, const float* source_class_mean,
    size_t time_steps, size_t channels,
    const Component (&components)[M],
    float original_top2_log_margin,
    MarginFn model, void* context,
    float* scratch,
    Result<M,Q>* out) {
  if (!out) return false;
  *out=Result<M,Q>{};
  if (!model || !scratch || !std::isfinite(original_top2_log_margin) ||
      scratch==standardized_input || scratch==source_class_mean)
    return false;
  if (!cheap_prior<M,Q>(
      standardized_input,source_class_mean,time_steps,channels,
      components,out->prior)) return false;

  // Top Q by cheap scores, stable by component ID. np.argpartition returns
  // the same top-Q SET when the cutoff is strict, but order is unspecified.
  uint16_t order[M];
  for(size_t i=0;i<M;++i)order[i]=static_cast<uint16_t>(i);
  for(size_t k=0;k<Q;++k) {
    size_t best=k;
    for(size_t j=k+1;j<M;++j)
      if(out->prior[order[j]]>out->prior[order[best]])best=j;
    const uint16_t tmp=order[k];
    order[k]=order[best];
    order[best]=tmp;
    out->selected[k]=order[k];
  }
  double next_max=-std::numeric_limits<double>::infinity();
  if(Q<M)for(size_t i=Q;i<M;++i)
    if(out->prior[order[i]]>next_max)next_max=out->prior[order[i]];
  out->cutoff_gap=(Q<M)?out->prior[out->selected[Q-1]]-next_max:
                         std::numeric_limits<double>::infinity();
  out->ambiguous_prior_cutoff=(Q<M && out->cutoff_gap<=1e-6);

  const size_t n=time_steps*channels;
  for(size_t k=0;k<Q;++k) {
    const Component& c=components[out->selected[k]];
    std::memcpy(scratch,standardized_input,n*sizeof(float));
    for(size_t t=c.t0;t<c.t1;++t)
      for(size_t ch=c.c0;ch<c.c1;++ch) {
        const size_t index=t*channels+ch;
        if (c.t0==0 || c.t1==time_steps) scratch[index]=0.0f;
        else {
          const float left=standardized_input[(c.t0-1)*channels+ch];
          const float right=standardized_input[c.t1*channels+ch];
          const double frac=static_cast<double>(t-c.t0)/
                            static_cast<double>(c.t1-c.t0);
          // Corresponds to NumPy linspace(left, right, n, endpoint=False)
          // to FP32 rounding; slight machine differences checked by parity.
          scratch[index]=static_cast<float>(
              static_cast<double>(left)+
              (static_cast<double>(right)-left)*frac);
        }
      }
    float updated_margin=0.0f;
    if(!model(scratch,n,context,&updated_margin) ||
       !std::isfinite(updated_margin))return false;
    out->effects[k]=std::fabs(original_top2_log_margin-updated_margin);
    out->model_evaluations++;
  }
  uint16_t effect_order[Q];
  for(size_t k=0;k<Q;++k)effect_order[k]=static_cast<uint16_t>(k);
  for(size_t k=1;k<Q;++k) {
    const uint16_t item=effect_order[k];
    size_t j=k;
    while(j>0 && out->effects[effect_order[j-1]]<out->effects[item]) {
      effect_order[j]=effect_order[j-1];
      --j;
    }
    effect_order[j]=item;
  }
  for(size_t k=0;k<Q;++k)
    out->ranked[k]=out->selected[effect_order[k]];
  for(size_t k=1;k<Q;++k)
    if(std::fabs(out->effects[effect_order[k-1]]-
                 out->effects[effect_order[k]])<=1e-6f)
      out->ambiguous_effect_rank=true;
  out->valid=true;
  return true;
}

} // namespace beacon_adaptive_v2
#endif // BEACON_ADAPTIVE_V2_AUDIT_HPP
