#pragma once

// =====================================================================
// Lucid C++ engine — normalization kernels.
// =====================================================================
//
// All kernels operate on a (outer, N) flattened layout where:
//   outer = product of dims NOT being normalized
//   N     = product of dims BEING normalized (the "feature" axes)
//
// LayerNorm normalizes per (outer-row), computing one mean+var per row.
// RMSNorm   normalizes per (outer-row), computing one rms per row.
//
// γ and β have shape `(N,)` and broadcast across rows.
//
// Layer: backend/cpu/. F32 + F64 only (Phase 3.6).

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// LayerNorm forward.
//   y[o, i]          = γ[i] * (x[o, i] - mean[o]) / sqrt(var[o] + ε) + β[i]
//   saved_mean[o]    = mean over feature axis
//   saved_rstd[o]    = 1 / sqrt(var + ε)
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

// LayerNorm backward (combined dx + dγ + dβ).
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

// RMSNorm forward.
//   rms[o]      = sqrt(mean(x²) + ε)
//   y[o, i]     = γ[i] * x[o, i] / rms[o]
//   saved_rstd[o] = 1 / rms[o]
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

// RMSNorm backward (dx + dγ).
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
