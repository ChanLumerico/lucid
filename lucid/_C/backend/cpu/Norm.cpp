#include "Norm.h"

#include <cmath>
#include <vector>

#include <Accelerate/Accelerate.h>

#include "Vdsp.h"
#include "Vforce.h"

namespace lucid::backend::cpu {

namespace {

// F32 fast path using vDSP primitives.
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
    // Reuse a per-row scratch buffer for the centered values.
    std::vector<float> centered(N);
    for (std::size_t o = 0; o < outer; ++o) {
        const float* xb = x + o * N;
        float* yb = y + o * N;
        // 1. mean via vDSP_meanv
        const float mean = vmean_f32(xb, N);
        saved_mean[o] = mean;
        // 2. centered = x - mean
        const float neg_mean = -mean;
        vsadd_f32(xb, neg_mean, centered.data(), N);
        // 3. variance = dot(centered, centered) / N
        const float var = vdotpr_f32(centered.data(), centered.data(), N) * inv_N;
        const float rstd = 1.0f / std::sqrt(var + static_cast<float>(eps));
        saved_rstd[o] = rstd;
        // 4. xnorm = centered * rstd  (in-place on yb as scratch)
        vsmul_f32(centered.data(), rstd, yb, N);
        // 5. y = gamma * xnorm + beta  (vDSP_vma: A*B + C)
        vmadd_f32(yb, gamma, beta, yb, N);
    }
}

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
    // dx per row uses the standard combined formula:
    //   dx_i = (1/N) · rstd · [N · dxn_i - sum(dxn) - xn_i · sum(dxn · xn)]
    // where dxn = γ · g, xn = (x - μ) · rstd.
    const T inv_N = T{1} / static_cast<T>(N);

    // Initialize dγ, dβ accumulators (caller passes already-zeroed buffers,
    // but we zero defensively in case).
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
            // Accumulate dγ_i, dβ_i across rows.
            dgamma[i] += gb[i] * xn_i;
            dbeta[i] += gb[i];
        }
    }
}

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

template <typename T>
void rms_norm_backward(const T* x,
                       const T* gamma,
                       const T* saved_rstd,
                       const T* g,
                       T* dx,
                       T* dgamma,
                       std::size_t outer,
                       std::size_t N) {
    // dx_i = (1/rstd) ... wait, derive directly:
    //
    //   y_i  = γ_i x_i r            (r = 1/rms)
    //   y_i  depends on x_j through r as well.
    //   ∂L/∂x_i = γ_i g_i r - x_i · (1/(N·rms²)) · sum_j(γ_j g_j x_j) · r
    //          = r · (γ_i g_i - x_i · r² · (1/N) · sum_j (γ_j g_j x_j))
    //
    // Equivalent form using dxn = γ · g:
    //   dx_i = r · (dxn_i - x_i · r² · (1/N) · sum_j (dxn_j · x_j))
    //
    // dγ_i = sum_o (g_o,i · x_o,i · r_o)
    const T inv_N = T{1} / static_cast<T>(N);
    for (std::size_t i = 0; i < N; ++i)
        dgamma[i] = T{};

    for (std::size_t o = 0; o < outer; ++o) {
        const T* xb = x + o * N;
        const T* gb = g + o * N;
        const T r = saved_rstd[o];

        // Pre-compute cross-sum: sum_j γ_j g_j x_j
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
