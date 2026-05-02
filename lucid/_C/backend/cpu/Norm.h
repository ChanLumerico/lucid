#pragma once

#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

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
