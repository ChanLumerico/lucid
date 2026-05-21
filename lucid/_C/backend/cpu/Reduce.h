// lucid/_C/backend/cpu/Reduce.h
//
// Single-axis reduction primitives used by the CPU backend.  Each function
// reduces one axis of a tensor described by three extent values:
//   outer       — product of all dimensions before the reduce axis
//   reduce_dim  — size of the dimension being reduced
//   inner       — product of all dimensions after the reduce axis
//
// The input layout follows row-major order: element [o, r, i] lives at
// in[o * reduce_dim * inner + r * inner + i].  The output has shape
// [outer, inner] and is written row-major.
//
// sum_axis_f32 uses vDSP_sve for the inner == 1 (contiguous) case because
// vDSP provides a numerically stable single-pass summation that is faster
// than the scalar loop.  All other operations use the generic axis_reduce
// template defined in Reduce.cpp.

#pragma once

#include <cstddef>
#include <cstdint>

#include "../../api.h"

namespace lucid::backend::cpu {

// Reduces a single axis by summation into a packed ``(outer, inner)`` output.
//
// For each ``(o, i)`` pair the kernel walks the reduce dimension contiguously
// when ``inner == 1`` and dispatches to Accelerate's ``vDSP_sve`` for a
// vectorised compensated-sum implementation; in the strided case it falls
// back to the scalar ``axis_reduce`` template with identity ``0``.
//
// Parameters
// ----------
// in : const float*
//     Source buffer laid out as ``(outer, reduce_dim, inner)`` in row-major
//     order.
// out : float*
//     Destination buffer of shape ``(outer, inner)``; written densely.
// outer : std::size_t
//     Product of all dimensions strictly before the reduce axis.  May be ``0``.
// reduce_dim : std::size_t
//     Size of the axis being summed.  A value of ``0`` yields ``0`` per cell
//     (the additive identity).
// inner : std::size_t
//     Product of all dimensions strictly after the reduce axis.
//
// Math
// ----
// $$\text{out}[o, i] = \sum_{r=0}^{\text{reduce\_dim}-1} \text{in}[o, r, i].$$
//
// Notes
// -----
// The contiguous fast path uses ``vDSP_sve`` which performs a vectorised
// single-pass summation on Apple Silicon NEON.  Numerical results may differ
// from the scalar loop by a few ULP for very long reductions but are typically
// stabler.
//
// See Also
// --------
// max_axis_f32, min_axis_f32, prod_axis_f32 : Companion reductions sharing the
//     same ``(outer, reduce_dim, inner)`` layout convention.
LUCID_INTERNAL void sum_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Double-precision counterpart to :cpp:func:`sum_axis_f32`.
//
// Uses ``vDSP_sveD`` on the contiguous (``inner == 1``) path and the generic
// ``axis_reduce`` template otherwise.
//
// Parameters
// ----------
// in : const double*
//     Source buffer of shape ``(outer, reduce_dim, inner)``.
// out : double*
//     Destination buffer of shape ``(outer, inner)``.
// outer, reduce_dim, inner : std::size_t
//     Layout extents; see :cpp:func:`sum_axis_f32`.
//
// Math
// ----
// $$\text{out}[o, i] = \sum_{r=0}^{\text{reduce\_dim}-1} \text{in}[o, r, i].$$
LUCID_INTERNAL void sum_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Reduces a single axis by elementwise maximum into a packed ``(outer, inner)``
// output.
//
// The kernel always uses the generic ``axis_reduce`` template — Accelerate
// does not expose a stride-aware vDSP equivalent for ``max`` with the same
// semantics, so vectorisation is left to the compiler.  Initial accumulator
// is ``-inf`` which means an empty reduce dimension yields ``-inf`` per cell.
//
// Parameters
// ----------
// in : const float*
//     Source buffer of shape ``(outer, reduce_dim, inner)``.
// out : float*
//     Destination buffer of shape ``(outer, inner)``.
// outer, reduce_dim, inner : std::size_t
//     Layout extents; see :cpp:func:`sum_axis_f32`.
//
// Math
// ----
// $$\text{out}[o, i] = \max_{r} \text{in}[o, r, i].$$
//
// Notes
// -----
// NaN propagation follows the C++ ``a > b`` semantics: a NaN on either side
// of the comparison loses, so a NaN element will not necessarily survive the
// reduction.  Downstream code that requires NaN-poisoning must guard explicitly.
LUCID_INTERNAL void max_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Double-precision counterpart to :cpp:func:`max_axis_f32`.
//
// Parameters
// ----------
// in : const double*
//     Source buffer of shape ``(outer, reduce_dim, inner)``.
// out : double*
//     Destination buffer of shape ``(outer, inner)``.
// outer, reduce_dim, inner : std::size_t
//     Layout extents; see :cpp:func:`sum_axis_f32`.
//
// Math
// ----
// $$\text{out}[o, i] = \max_{r} \text{in}[o, r, i].$$
LUCID_INTERNAL void max_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Reduces a single axis by elementwise minimum into a packed ``(outer, inner)``
// output.
//
// Identity element is ``+inf``; an empty reduce dimension yields ``+inf`` per
// output cell.  Uses the generic ``axis_reduce`` template.
//
// Parameters
// ----------
// in : const float*
//     Source buffer of shape ``(outer, reduce_dim, inner)``.
// out : float*
//     Destination buffer of shape ``(outer, inner)``.
// outer, reduce_dim, inner : std::size_t
//     Layout extents; see :cpp:func:`sum_axis_f32`.
//
// Math
// ----
// $$\text{out}[o, i] = \min_{r} \text{in}[o, r, i].$$
LUCID_INTERNAL void min_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Double-precision counterpart to :cpp:func:`min_axis_f32`.
//
// Parameters
// ----------
// in : const double*
//     Source buffer of shape ``(outer, reduce_dim, inner)``.
// out : double*
//     Destination buffer of shape ``(outer, inner)``.
// outer, reduce_dim, inner : std::size_t
//     Layout extents; see :cpp:func:`sum_axis_f32`.
//
// Math
// ----
// $$\text{out}[o, i] = \min_{r} \text{in}[o, r, i].$$
LUCID_INTERNAL void min_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Reduces a single axis by multiplicative product into a packed
// ``(outer, inner)`` output.
//
// Identity element is ``1``; an empty reduce dimension yields ``1`` per cell.
// Uses the generic ``axis_reduce`` template — there is no vDSP equivalent.
// Note that floating-point products are not associative, so the result is
// deterministic only for a fixed loop order (the kernel iterates ``r`` from
// ``0`` to ``reduce_dim - 1``).
//
// Parameters
// ----------
// in : const float*
//     Source buffer of shape ``(outer, reduce_dim, inner)``.
// out : float*
//     Destination buffer of shape ``(outer, inner)``.
// outer, reduce_dim, inner : std::size_t
//     Layout extents; see :cpp:func:`sum_axis_f32`.
//
// Math
// ----
// $$\text{out}[o, i] = \prod_{r=0}^{\text{reduce\_dim}-1} \text{in}[o, r, i].$$
//
// Notes
// -----
// Overflow and underflow are unchecked.  Callers that require ``log-sum-exp``
// style stability must compute the product in log space themselves.
LUCID_INTERNAL void prod_axis_f32(
    const float* in, float* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

// Double-precision counterpart to :cpp:func:`prod_axis_f32`.
//
// Parameters
// ----------
// in : const double*
//     Source buffer of shape ``(outer, reduce_dim, inner)``.
// out : double*
//     Destination buffer of shape ``(outer, inner)``.
// outer, reduce_dim, inner : std::size_t
//     Layout extents; see :cpp:func:`sum_axis_f32`.
//
// Math
// ----
// $$\text{out}[o, i] = \prod_{r=0}^{\text{reduce\_dim}-1} \text{in}[o, r, i].$$
LUCID_INTERNAL void prod_axis_f64(
    const double* in, double* out, std::size_t outer, std::size_t reduce_dim, std::size_t inner);

}  // namespace lucid::backend::cpu
