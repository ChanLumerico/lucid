#pragma once

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

LUCID_INTERNAL void sum_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void sum_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

LUCID_INTERNAL void max_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void max_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

LUCID_INTERNAL void min_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void min_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

LUCID_INTERNAL void prod_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);
LUCID_INTERNAL void prod_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

}  // namespace lucid::backend::cpu
