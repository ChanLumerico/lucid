#include "Reduce.h"

#include <algorithm>
#include <limits>

#include <Accelerate/Accelerate.h>

namespace lucid::backend::cpu {

namespace {

template <typename T, typename Op>
void axis_reduce(const T* in,
                 T* out,
                 std::size_t outer,
                 std::size_t reduce_dim,
                 std::size_t inner,
                 T identity,
                 Op op) {
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < inner; ++i) {
            T acc = identity;
            const T* base = in + o * reduce_dim * inner + i;
            for (std::size_t r = 0; r < reduce_dim; ++r) {
                acc = op(acc, base[r * inner]);
            }
            out[o * inner + i] = acc;
        }
    }
}

}  // namespace

void sum_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o) {
            float acc = 0.f;
            vDSP_sve(in + o * reduce_dim, 1, &acc, static_cast<vDSP_Length>(reduce_dim));
            out[o] = acc;
        }
        return;
    }
    axis_reduce<float>(in, out, outer, reduce_dim, inner, 0.f,
                       [](float a, float b) { return a + b; });
}

void sum_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    if (inner == 1) {
        for (std::size_t o = 0; o < outer; ++o) {
            double acc = 0.0;
            vDSP_sveD(in + o * reduce_dim, 1, &acc, static_cast<vDSP_Length>(reduce_dim));
            out[o] = acc;
        }
        return;
    }
    axis_reduce<double>(in, out, outer, reduce_dim, inner, 0.0,
                        [](double a, double b) { return a + b; });
}

void max_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
    axis_reduce<float>(in, out, outer, reduce_dim, inner, NEG_INF,
                       [](float a, float b) { return a > b ? a : b; });
}

void max_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    axis_reduce<double>(in, out, outer, reduce_dim, inner, NEG_INF,
                        [](double a, double b) { return a > b ? a : b; });
}

void min_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    constexpr float POS_INF = std::numeric_limits<float>::infinity();
    axis_reduce<float>(in, out, outer, reduce_dim, inner, POS_INF,
                       [](float a, float b) { return a < b ? a : b; });
}

void min_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    constexpr double POS_INF = std::numeric_limits<double>::infinity();
    axis_reduce<double>(in, out, outer, reduce_dim, inner, POS_INF,
                        [](double a, double b) { return a < b ? a : b; });
}

void prod_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    axis_reduce<float>(in, out, outer, reduce_dim, inner, 1.f,
                       [](float a, float b) { return a * b; });
}

void prod_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner) {
    axis_reduce<double>(in, out, outer, reduce_dim, inner, 1.0,
                        [](double a, double b) { return a * b; });
}

}  // namespace lucid::backend::cpu
