// lucid/_C/backend/cpu/Norm.h
//
// CPU implementations of LayerNorm and RMSNorm forward and backward passes.
// Both norms iterate over `outer` independent rows of length N (the normalized
// size) and compute per-row statistics.
//
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta.
//   The f32 forward path uses vDSP (vmean_f32, vdotpr_f32, etc.) for SIMD
//   throughput; the f64 forward path uses a generic template loop.
//   The backward pass computes dx, dgamma, and dbeta using the standard
//   LayerNorm gradient formula: dx_i = r * (N * dxn_i - sum_dxn - xn_i * sum_dxn_xn) / N.
//
// RMSNorm: y = gamma * x / sqrt(mean(x^2) + eps).  No mean subtraction; only
//   the root-mean-square scale is normalised.  The backward pass returns dx
//   and dgamma.
//
// Both norms save per-row reciprocal standard deviation (rstd = 1/std) during
// the forward pass so the backward pass can avoid a redundant division.

#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// LayerNorm forward: normalises each of `outer` rows of length N.
// Writes normalised output to y, per-row mean to saved_mean, and per-row
// reciprocal standard deviation to saved_rstd (for use by the backward pass).
LUCID_INTERNAL void layer_norm_forward_f32(const float* x,
                                           const float* gamma,
                                           const float* beta,
                                           float* y,
                                           float* saved_mean,
                                           float* saved_rstd,
                                           std::size_t outer,
                                           std::size_t N,
                                           double eps);
LUCID_INTERNAL void layer_norm_forward_f64(const double* x,
                                           const double* gamma,
                                           const double* beta,
                                           double* y,
                                           double* saved_mean,
                                           double* saved_rstd,
                                           std::size_t outer,
                                           std::size_t N,
                                           double eps);

// LayerNorm backward: given upstream gradient g, computes dx, dgamma, dbeta.
// saved_mean and saved_rstd must be the values produced by the corresponding
// forward call.  dgamma and dbeta are accumulated (not zeroed) so the caller
// must zero them first when computing over multiple batches.
LUCID_INTERNAL void layer_norm_backward_f32(const float* x,
                                            const float* gamma,
                                            const float* saved_mean,
                                            const float* saved_rstd,
                                            const float* g,
                                            float* dx,
                                            float* dgamma,
                                            float* dbeta,
                                            std::size_t outer,
                                            std::size_t N);
LUCID_INTERNAL void layer_norm_backward_f64(const double* x,
                                            const double* gamma,
                                            const double* saved_mean,
                                            const double* saved_rstd,
                                            const double* g,
                                            double* dx,
                                            double* dgamma,
                                            double* dbeta,
                                            std::size_t outer,
                                            std::size_t N);

// RMSNorm forward: normalises each of `outer` rows of length N by their
// root-mean-square.  Writes per-row rstd to saved_rstd for the backward pass.
LUCID_INTERNAL void rms_norm_forward_f32(const float* x,
                                         const float* gamma,
                                         float* y,
                                         float* saved_rstd,
                                         std::size_t outer,
                                         std::size_t N,
                                         double eps);
LUCID_INTERNAL void rms_norm_forward_f64(const double* x,
                                         const double* gamma,
                                         double* y,
                                         double* saved_rstd,
                                         std::size_t outer,
                                         std::size_t N,
                                         double eps);

// RMSNorm backward: computes dx and dgamma given upstream gradient g.
// dgamma is accumulated; caller must zero it before the first call.
LUCID_INTERNAL void rms_norm_backward_f32(const float* x,
                                          const float* gamma,
                                          const float* saved_rstd,
                                          const float* g,
                                          float* dx,
                                          float* dgamma,
                                          std::size_t outer,
                                          std::size_t N);
LUCID_INTERNAL void rms_norm_backward_f64(const double* x,
                                          const double* gamma,
                                          const double* saved_rstd,
                                          const double* g,
                                          double* dx,
                                          double* dgamma,
                                          std::size_t outer,
                                          std::size_t N);

}  // namespace lucid::backend::cpu
