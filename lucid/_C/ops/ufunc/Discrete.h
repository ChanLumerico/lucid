// lucid/_C/ops/ufunc/Discrete.h
//
// Backward nodes for discontinuous (piecewise-constant) unary operations:
// round, floor, ceil, invert.  These functions have zero derivative almost
// everywhere (the derivative is undefined at integer boundaries) so all four
// set kHasGradient = false.  UnaryKernel::forward will skip autograd wiring
// entirely; grad_formula is provided only for completeness and returns an
// empty CpuStorage as a zero-gradient sentinel.  This matches reference framework's
// behaviour for the same ops.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for element-wise round-to-nearest-even: $y = \operatorname{round}(x)$.
//
// Piecewise-constant; derivative is zero almost everywhere and undefined
// at half-integer boundaries.  ``kHasGradient = false`` causes
// ``UnaryKernel::forward`` to skip autograd wiring entirely.
//
// Attributes
// ----------
// kSavesInput : bool
//     ``false`` — no tensor needs to be saved across the backward pass.
// kHasGradient : bool
//     ``false`` — predicate-style op; the backward node is never invoked
//     in practice.
// schema_v1 : OpSchema
//     Op name ``"round"`` with ``AmpPolicy::KeepInput`` (integer inputs
//     pass through unchanged).
//
// Math
// ----
// $$y_i = \operatorname{round}(x_i)$$
//
// Ties round to even ("banker's rounding"), matching IEEE-754.
//
// Notes
// -----
// CPU dispatch routes through ``backend::IBackend::round``; GPU uses
// MLX's ``round`` primitive.  Integer dtypes are a no-op.
//
// See Also
// --------
// :class:`FloorBackward`, :class:`CeilBackward`.
class LUCID_API RoundBackward : public UnaryOp<RoundBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::round`` to compute $y = \operatorname{round}(x)$
    // with banker's rounding.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.round(a, s, dt);
    }
    // Zero-gradient sentinel — returns an empty ``CpuStorage`` because
    // ``kHasGradient = false`` guarantees this is never called.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Incoming upstream gradient (ignored).
    //
    // Returns
    // -------
    // Storage
    //     Empty ``CpuStorage`` placeholder.
    Storage grad_formula(const Storage& g);
};

// Element-wise round-to-nearest-even — returns a new tensor whose values
// are ``round(a)`` with the same shape and dtype as ``a``.
//
// Not differentiable: ``a`` is detached from the autograd graph.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  Integer dtypes are returned unchanged.
//
// Returns
// -------
// TensorImplPtr
//     New tensor with rounded values; same shape, dtype, and device as ``a``.
//
// Math
// ----
// $$y_i = \operatorname{round}(x_i)$$  (round-half-to-even / banker's rounding).
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// See Also
// --------
// :func:`floor_op`, :func:`ceil_op`.
LUCID_API TensorImplPtr round_op(const TensorImplPtr& a);

// Autograd node for element-wise floor: $y = \lfloor x \rfloor$.
//
// Piecewise-constant; derivative is zero almost everywhere and undefined
// at integer boundaries.  ``kHasGradient = false``; the backward node is
// registered for schema completeness only.
//
// Attributes
// ----------
// kSavesInput : bool
//     ``false`` — nothing to save.
// kHasGradient : bool
//     ``false`` — autograd wiring is skipped.
// schema_v1 : OpSchema
//     Op name ``"floor"`` with ``AmpPolicy::KeepInput``.
//
// Math
// ----
// $$y_i = \lfloor x_i \rfloor$$  (largest integer not greater than $x_i$).
//
// Notes
// -----
// Integer dtypes are a no-op.  Negative values round toward $-\infty$
// (NOT toward zero — that is ``trunc``).
class LUCID_API FloorBackward : public UnaryOp<FloorBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::floor`` to compute $y = \lfloor x \rfloor$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.floor(a, s, dt);
    }
    // Zero-gradient sentinel; never invoked since ``kHasGradient = false``.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Incoming gradient (ignored).
    //
    // Returns
    // -------
    // Storage
    //     Empty ``CpuStorage`` placeholder.
    Storage grad_formula(const Storage& g);
};

// Element-wise floor — returns the largest integer-valued tensor not
// greater than ``a``, with the same dtype as ``a``.
//
// Not differentiable: output is detached from the autograd graph.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  Integer dtypes are returned unchanged.
//
// Returns
// -------
// TensorImplPtr
//     New tensor with floored values; same shape, dtype, and device as ``a``.
//
// Math
// ----
// $$y_i = \lfloor x_i \rfloor$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Examples
// --------
// ``floor_op(-1.2) == -2.0`` (not ``-1.0`` — that would be ``trunc``).
//
// See Also
// --------
// :func:`ceil_op`, :func:`round_op`.
LUCID_API TensorImplPtr floor_op(const TensorImplPtr& a);

// Autograd node for element-wise ceiling: $y = \lceil x \rceil$.
//
// Piecewise-constant; derivative is zero almost everywhere and undefined
// at integer boundaries.  ``kHasGradient = false``.
//
// Attributes
// ----------
// kSavesInput : bool
//     ``false``.
// kHasGradient : bool
//     ``false``.
// schema_v1 : OpSchema
//     Op name ``"ceil"`` with ``AmpPolicy::KeepInput``.
//
// Math
// ----
// $$y_i = \lceil x_i \rceil$$  (smallest integer not less than $x_i$).
//
// Notes
// -----
// Integer dtypes are a no-op.  Negative values round toward zero (e.g.
// ``ceil(-1.2) = -1``).
class LUCID_API CeilBackward : public UnaryOp<CeilBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::ceil`` to compute $y = \lceil x \rceil$.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.ceil(a, s, dt);
    }
    // Zero-gradient sentinel; never invoked since ``kHasGradient = false``.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Incoming gradient (ignored).
    //
    // Returns
    // -------
    // Storage
    //     Empty ``CpuStorage`` placeholder.
    Storage grad_formula(const Storage& g);
};

// Element-wise ceiling — returns the smallest integer-valued tensor not
// less than ``a``, with the same dtype as ``a``.
//
// Not differentiable: output is detached from the autograd graph.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.  Integer dtypes are returned unchanged.
//
// Returns
// -------
// TensorImplPtr
//     New tensor with ceiled values; same shape, dtype, and device as ``a``.
//
// Math
// ----
// $$y_i = \lceil x_i \rceil$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// See Also
// --------
// :func:`floor_op`, :func:`round_op`.
LUCID_API TensorImplPtr ceil_op(const TensorImplPtr& a);

// Autograd node for element-wise bitwise NOT: $y = \mathtt{\sim}x$.
//
// Defined only for integer dtypes (including ``Bool``); floating-point
// inputs are rejected upstream by ``AmpPolicy::KeepInput``.  Bitwise
// operations have no meaningful gradient — ``kHasGradient = false``.
//
// Attributes
// ----------
// kSavesInput : bool
//     ``false`` — nothing saved.
// kHasGradient : bool
//     ``false`` — autograd wiring skipped.
// schema_v1 : OpSchema
//     Op name ``"invert"`` with ``AmpPolicy::KeepInput`` (prevents float
//     promotion so the integer dtype is preserved).
//
// Math
// ----
// $$y_i = \mathtt{\sim} x_i$$
//
// For ``Bool`` inputs this is logical negation; for unsigned integers
// it produces the bitwise complement modulo $2^{\text{bits}}$.
class LUCID_API InvertBackward : public UnaryOp<InvertBackward> {
public:
    static constexpr bool kSavesInput = false;
    static constexpr bool kHasGradient = false;
    static const OpSchema schema_v1;
    // Forward — calls ``IBackend::invert`` to compute $y = \mathtt{\sim} x$
    // (bitwise NOT) on integer dtypes.
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.invert(a, s, dt);
    }
    // Zero-gradient sentinel; never invoked since ``kHasGradient = false``.
    //
    // Parameters
    // ----------
    // g : const Storage&
    //     Incoming gradient (ignored).
    //
    // Returns
    // -------
    // Storage
    //     Empty ``CpuStorage`` placeholder.
    Storage grad_formula(const Storage& g);
};

// Element-wise bitwise NOT — returns the bit-flipped tensor with the
// same integer dtype as ``a``.
//
// Not differentiable: output is detached from the autograd graph.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of integer dtype (``Bool``, ``Int8``, ``Int16``,
//     ``Int32``, ``Int64``, or unsigned variants).  Floating-point dtypes
//     raise an error.
//
// Returns
// -------
// TensorImplPtr
//     New tensor with the bitwise complement; same shape, dtype, and
//     device as ``a``.
//
// Math
// ----
// $$y_i = \mathtt{\sim} x_i$$
//
// Shape
// -----
// Output shape equals input shape (elementwise).
//
// Raises
// ------
// Errors if ``a`` has a floating-point dtype — bitwise NOT is only
// defined on integers.
//
// Notes
// -----
// Exposed in Python as both ``invert`` and ``bitwise_not`` (reference
// framework alias).
LUCID_API TensorImplPtr invert_op(const TensorImplPtr& a);

}  // namespace lucid
