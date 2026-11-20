#ifndef BEACON_RAW_MLP_INFERENCE_HPP
#define BEACON_RAW_MLP_INFERENCE_HPP

// Minimal, allocation-free FP32 forward pass for a one-hidden-layer
// scikit-learn MLP (ReLU hidden, softmax output). A model exporter must provide
// weights/biases, input geometry and class mapping; no firmware, tensor arena,
// quantization, hardware timing or energy claims are made by this header.
//
// For a multi-class softmax MLP, the difference between top-two LOG
// probabilities equals the difference between top-two pre-softmax logits.
// This is the moving-margin quantity consumed by the BEACON Q16 callback.
#include <cmath>
#include <cstddef>

namespace beacon_raw_mlp {

// Same training-only channel normalization as Python apply_standardizer:
// (raw[t,c] - train_mean[c]) / (train_sigma[c]).
// Works in-place, uses no heap. Call before predict_top2_log_margin.
template <size_t TimeSteps, size_t Channels>
inline bool standardize_channels(
    const float* raw, const float* training_mu, const float* training_sigma,
    float* standardized) {
  if (!raw || !training_mu || !training_sigma || !standardized
      || TimeSteps<1 || Channels<1) return false;
  for (size_t c=0; c<Channels; ++c) {
    if (!std::isfinite(training_mu[c]) ||
        !std::isfinite(training_sigma[c]) ||
        training_sigma[c] <= 0.0f) return false;
  }
  for (size_t t=0; t<TimeSteps; ++t)
    for (size_t c=0; c<Channels; ++c) {
      const size_t index=t*Channels+c;
      if (!std::isfinite(raw[index])) return false;
      standardized[index] =
          (raw[index]-training_mu[c])/training_sigma[c];
      if (!std::isfinite(standardized[index])) return false;
    }
  return true;
}

template <size_t InputDim, size_t HiddenDim, size_t OutputDim>
inline bool predict_top2_log_margin(
    const float* standardized_input, size_t length,
    const float* w1, const float* b1,
    const float* w2, const float* b2,
    float* out_margin) {
  if (!standardized_input || !w1 || !b1 || !w2 || !b2 || !out_margin
      || InputDim < 1 || HiddenDim < 1 || OutputDim < 2
      || length != InputDim) return false;

  float hidden[HiddenDim];
  for (size_t j=0; j<HiddenDim; ++j) {
    float accum = b1[j];
    for (size_t i=0; i<InputDim; ++i)
      accum += standardized_input[i]*w1[i*HiddenDim+j];
    if (!std::isfinite(accum)) return false;
    hidden[j] = accum > 0.0f ? accum : 0.0f;
  }

  float largest=-INFINITY;
  float second=-INFINITY;
  for (size_t k=0; k<OutputDim; ++k) {
    float accum=b2[k];
    for (size_t j=0; j<HiddenDim; ++j)
      accum += hidden[j]*w2[j*OutputDim+k];
    if (!std::isfinite(accum)) return false;
    if (accum >= largest) {
      second=largest;
      largest=accum;
    } else if (accum > second) {
      second=accum;
    }
  }
  if (!std::isfinite(largest) || !std::isfinite(second)) return false;
  *out_margin = largest-second;
  return std::isfinite(*out_margin);
}

} // namespace beacon_raw_mlp
#endif // BEACON_RAW_MLP_INFERENCE_HPP
