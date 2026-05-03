// lucid/_C/backend/cpu/Norm.cpp
//
// Implements the LayerNorm and RMSNorm forward and backward passes declared in
// Norm.h.
//
// LayerNorm f32 forward uses a fast path (layer_norm_forward_f32_fast) that
// calls vDSP helpers (vmean_f32, vdotpr_f32, vsadd_f32, vsmul_f32, vmadd_f32)
// to exploit Apple Silicon NEON; the f64 forward uses a generic template.
//
// The LayerNorm backward uses the standard closed-form gradient:
//   dx_i = rstd * (N * dxn_i - sum(dxn) - xn_i * sum(dxn * xn)) / N
// where xn_i = (x_i - mean) * rstd is the normalised activation and dxn_i =
// gamma_i * g_i is the pre-affine upstream gradient.
//
// The RMSNorm backward uses:
//   dx_i = rstd * (gamma_i * g_i - x_i * rstd^2 * cross / N)
// where cross = sum_j(gamma_j * g_j * x_j).

#include "Norm.h"

#include <cmath>
#include <vector>

#include <Accelerate/Accelerate.h>

#include "Vdsp.h"
#include "Vforce.h"

namespace lucid::backend::cpu {

namespace {

// Fast vDSP-accelerated f32 forward path.  Uses a temporary `centered` buffer
// for (x - mean) to allow reuse with vdotpr_f32 for the variance computation.
void layer_norm_forward_f32_fast(const float* x,
                                 const float* gamma,
                                 const float* beta,
                                 float* y,
                                 float* saved_mean,
                                 float* saved_rstd,
                                 std::size_t outer,
                                 std::size_t N,
                                 double eps) {
    const float inv_N = 1.0f / static_cast<float>(N);

    std::vector<float> centered(N);
    for (std::size_t o = 0; o < outer; ++o) {
        const float* xb = x + o * N;
        float* yb = y + o * N;

        const float mean = vmean_f32(xb, N);
        saved_mean[o] = mean;

        const float neg_mean = -mean;
        vsadd_f32(xb, neg_mean, centered.data(), N);

        const float var = vdotpr_f32(centered.data(), centered.data(), N) * inv_N;
        const float rstd = 1.0f / std::sqrt(var + static_cast<float>(eps));
        saved_rstd[o] = rstd;

        vsmul_f32(centered.data(), rstd, yb, N);

        vmadd_f32(yb, gamma, beta, yb, N);
    }
}

// Generic typed LayerNorm forward; used for f64 and any future non-f32 path.
template <typename T>
void layer_norm_forward(const T* x,
                        const T* gamma,
                        const T* beta,
                        T* y,
                        T* saved_mean,
                        T* saved_rstd,
                        std::size_t outer,
                        std::size_t N,
                        double eps) {
    const T inv_N = T{1} / static_cast<T>(N);
    for (std::size_t o = 0; o < outer; ++o) {
        const T* xb = x + o * N;
        T mean = T{};
        for (std::size_t i = 0; i < N; ++i)
            mean += xb[i];
        mean *= inv_N;

        T var = T{};
        for (std::size_t i = 0; i < N; ++i) {
            const T d = xb[i] - mean;
            var += d * d;
        }
        var *= inv_N;
        const T rstd = T{1} / std::sqrt(var + static_cast<T>(eps));

        T* yb = y + o * N;
        for (std::size_t i = 0; i < N; ++i) {
            const T xn = (xb[i] - mean) * rstd;
            yb[i] = gamma[i] * xn + beta[i];
        }
        saved_mean[o] = mean;
        saved_rstd[o] = rstd;
    }
}

// Generic typed LayerNorm backward.  Accumulates dgamma and dbeta across all
// outer rows; recomputes xn_i = (x_i - mean)*rstd inline to avoid an extra
// buffer allocation.
template <typename T>
void layer_norm_backward(const T* x,
                         const T* gamma,
                         const T* saved_mean,
                         const T* saved_rstd,
                         const T* g,
                         T* dx,
                         T* dgamma,
                         T* dbeta,
                         std::size_t outer,
                         std::size_t N) {
    const T inv_N = T{1} / static_cast<T>(N);

    for (std::size_t i = 0; i < N; ++i) {
        dgamma[i] = T{};
        dbeta[i] = T{};
    }

    for (std::size_t o = 0; o < outer; ++o) {
        const T* xb = x + o * N;
        const T* gb = g + o * N;
        const T m = saved_mean[o];
        const T r = saved_rstd[o];

        T sum_dxn = T{};
        T sum_dxn_xn = T{};
        for (std::size_t i = 0; i < N; ++i) {
            const T xn_i = (xb[i] - m) * r;
            const T dxn_i = gamma[i] * gb[i];
            sum_dxn += dxn_i;
            sum_dxn_xn += dxn_i * xn_i;
        }

        T* dxb = dx + o * N;
        for (std::size_t i = 0; i < N; ++i) {
            const T xn_i = (xb[i] - m) * r;
            const T dxn_i = gamma[i] * gb[i];
            dxb[i] = inv_N * r * (static_cast<T>(N) * dxn_i - sum_dxn - xn_i * sum_dxn_xn);

            dgamma[i] += gb[i] * xn_i;
            dbeta[i] += gb[i];
        }
    }
}

// Generic typed RMSNorm forward: rstd = 1 / sqrt(mean(x^2) + eps).
template <typename T>
void rms_norm_forward(
    const T* x, const T* gamma, T* y, T* saved_rstd, std::size_t outer, std::size_t N, double eps) {
    const T inv_N = T{1} / static_cast<T>(N);
    for (std::size_t o = 0; o < outer; ++o) {
        const T* xb = x + o * N;
        T sumsq = T{};
        for (std::size_t i = 0; i < N; ++i)
            sumsq += xb[i] * xb[i];
        const T meansq = sumsq * inv_N;
        const T rstd = T{1} / std::sqrt(meansq + static_cast<T>(eps));

        T* yb = y + o * N;
        for (std::size_t i = 0; i < N; ++i) {
            yb[i] = gamma[i] * xb[i] * rstd;
        }
        saved_rstd[o] = rstd;
    }
}

// Generic typed RMSNorm backward: computes dx and accumulates dgamma.
// cross = sum_j(gamma_j * g_j * x_j) is the per-row contraction needed to
// back-propagate through the normalisation denominator.
template <typename T>
void rms_norm_backward(const T* x,
                       const T* gamma,
                       const T* saved_rstd,
                       const T* g,
                       T* dx,
                       T* dgamma,
                       std::size_t outer,
                       std::size_t N) {
    const T inv_N = T{1} / static_cast<T>(N);
    for (std::size_t i = 0; i < N; ++i)
        dgamma[i] = T{};

    for (std::size_t o = 0; o < outer; ++o) {
        const T* xb = x + o * N;
        const T* gb = g + o * N;
        const T r = saved_rstd[o];

        T cross = T{};
        for (std::size_t i = 0; i < N; ++i) {
            const T dxn_i = gamma[i] * gb[i];
            cross += dxn_i * xb[i];
        }
        const T scaled = inv_N * r * r * cross;

        T* dxb = dx + o * N;
        for (std::size_t i = 0; i < N; ++i) {
            const T dxn_i = gamma[i] * gb[i];
            dxb[i] = r * (dxn_i - xb[i] * scaled);
            dgamma[i] += gb[i] * xb[i] * r;
        }
    }
}

}  // namespace

void layer_norm_forward_f32(const float* x,
                            const float* gamma,
                            const float* beta,
                            float* y,
                            float* saved_mean,
                            float* saved_rstd,
                            std::size_t outer,
                            std::size_t N,
                            double eps) {
    layer_norm_forward_f32_fast(x, gamma, beta, y, saved_mean, saved_rstd, outer, N, eps);
}

void layer_norm_forward_f64(const double* x,
                            const double* gamma,
                            const double* beta,
                            double* y,
                            double* saved_mean,
                            double* saved_rstd,
                            std::size_t outer,
                            std::size_t N,
                            double eps) {
    layer_norm_forward<double>(x, gamma, beta, y, saved_mean, saved_rstd, outer, N, eps);
}

void layer_norm_backward_f32(const float* x,
                             const float* gamma,
                             const float* saved_mean,
                             const float* saved_rstd,
                             const float* g,
                             float* dx,
                             float* dgamma,
                             float* dbeta,
                             std::size_t outer,
                             std::size_t N) {
    layer_norm_backward<float>(x, gamma, saved_mean, saved_rstd, g, dx, dgamma, dbeta, outer, N);
}

void layer_norm_backward_f64(const double* x,
                             const double* gamma,
                             const double* saved_mean,
                             const double* saved_rstd,
                             const double* g,
                             double* dx,
                             double* dgamma,
                             double* dbeta,
                             std::size_t outer,
                             std::size_t N) {
    layer_norm_backward<double>(x, gamma, saved_mean, saved_rstd, g, dx, dgamma, dbeta, outer, N);
}

void rms_norm_forward_f32(const float* x,
                          const float* gamma,
                          float* y,
                          float* saved_rstd,
                          std::size_t outer,
                          std::size_t N,
                          double eps) {
    rms_norm_forward<float>(x, gamma, y, saved_rstd, outer, N, eps);
}

void rms_norm_forward_f64(const double* x,
                          const double* gamma,
                          double* y,
                          double* saved_rstd,
                          std::size_t outer,
                          std::size_t N,
                          double eps) {
    rms_norm_forward<double>(x, gamma, y, saved_rstd, outer, N, eps);
}

void rms_norm_backward_f32(const float* x,
                           const float* gamma,
                           const float* saved_rstd,
                           const float* g,
                           float* dx,
                           float* dgamma,
                           std::size_t outer,
                           std::size_t N) {
    rms_norm_backward<float>(x, gamma, saved_rstd, g, dx, dgamma, outer, N);
}

void rms_norm_backward_f64(const double* x,
                           const double* gamma,
                           const double* saved_rstd,
                           const double* g,
                           double* dx,
                           double* dgamma,
                           std::size_t outer,
                           std::size_t N) {
    rms_norm_backward<double>(x, gamma, saved_rstd, g, dx, dgamma, outer, N);
}

}  // namespace lucid::backend::cpu
