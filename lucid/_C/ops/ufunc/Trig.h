// lucid/_C/ops/ufunc/Trig.h
//
// Autograd backward nodes and entry points for trigonometric operations:
// sin, cos, tan, arcsin, arccos, arctan.  On CPU, the backend delegates to
// vForce (vvsinf, vvcosf, …) for vectorised throughput.  All ops use
// AmpPolicy::Promote so that float64 inputs retain their precision.

#pragma once

#include "../../api.h"
#include "../../backend/IBackend.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_UnaryOp.h"

namespace lucid {

// Autograd node for element-wise sine $y = \sin(x)$.
//
// Saves the input ``x`` so the backward pass can evaluate $\cos(x)$ and
// scale the upstream gradient by it.  The input is interpreted in radians.
//
// Math
// ----
// $$y = \sin(x), \qquad
// \frac{\partial y}{\partial x} = \cos(x), \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \cos(x)\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"sin"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvsinf``/``vvsin`` (CPU) / MLX ``sin`` (GPU).
class LUCID_API SinBackward : public UnaryOp<SinBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.sin(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Compute $y = \sin(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`SinBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape, values in radians.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values in $[-1, 1]$.
//
// See Also
// --------
// :class:`SinBackward` — backward node.
LUCID_API TensorImplPtr sin_op(const TensorImplPtr& a);

// Autograd node for element-wise cosine $y = \cos(x)$.
//
// Saves the input ``x`` so the backward pass can compute $-\sin(x)$ and
// scale the upstream gradient by it.  The input is interpreted in radians.
//
// Math
// ----
// $$y = \cos(x), \qquad
// \frac{\partial y}{\partial x} = -\sin(x), \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// -\sin(x)\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"cos"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvcosf``/``vvcos`` (CPU) / MLX ``cos`` (GPU).
class LUCID_API CosBackward : public UnaryOp<CosBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.cos(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Compute $y = \cos(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`CosBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape, values in radians.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values in $[-1, 1]$.
//
// See Also
// --------
// :class:`CosBackward` — backward node.
LUCID_API TensorImplPtr cos_op(const TensorImplPtr& a);

// Autograd node for element-wise tangent $y = \tan(x)$.
//
// Saves the input ``x`` so the backward pass can build $\cos^2(x)$ in the
// denominator.  Defined for all real $x$ except the singular points
// $x = (k + \tfrac{1}{2})\pi$ where $\cos x = 0$; behaviour at those
// points is undefined.
//
// Math
// ----
// $$y = \tan(x), \qquad
// \frac{\partial y}{\partial x} = \sec^2(x) = \frac{1}{\cos^2(x)}, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{1}{\cos^2(x)}\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"tan"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvtanf``/``vvtan`` (CPU) / MLX ``tan`` (GPU).
class LUCID_API TanBackward : public UnaryOp<TanBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.tan(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Compute $y = \tan(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`TanBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape, values in radians.  Singular at
//     $x = (k + \tfrac{1}{2})\pi$.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype.
//
// See Also
// --------
// :class:`TanBackward` — backward node.
LUCID_API TensorImplPtr tan_op(const TensorImplPtr& a);

// Autograd node for element-wise arcsine $y = \arcsin(x)$.
//
// Saves the input ``x`` so the backward pass can form the radicand
// $1 - x^2$.  Defined on the closed input domain $[-1, 1]$ with range
// $[-\tfrac{\pi}{2}, \tfrac{\pi}{2}]$; the gradient is unbounded at the
// endpoints $x = \pm 1$.
//
// Math
// ----
// $$y = \arcsin(x), \qquad
// \frac{\partial y}{\partial x} = \frac{1}{\sqrt{1 - x^2}}, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{1}{\sqrt{1 - x^2}}\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"arcsin"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvasinf``/``vvasin`` (CPU) / MLX ``arcsin`` (GPU).
class LUCID_API AsinBackward : public UnaryOp<AsinBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.asin(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Compute $y = \arcsin(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`AsinBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor with values in $[-1, 1]$.  Out-of-domain values
//     produce NaN, matching the reference framework's behaviour.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values in
//     $[-\tfrac{\pi}{2}, \tfrac{\pi}{2}]$.
//
// See Also
// --------
// :class:`AsinBackward` — backward node.
LUCID_API TensorImplPtr arcsin_op(const TensorImplPtr& a);

// Autograd node for element-wise arccosine $y = \arccos(x)$.
//
// Shares the radicand $1 - x^2$ with :class:`AsinBackward` but with a
// negated sign in the derivative.  Defined on $[-1, 1]$ with range
// $[0, \pi]$; the gradient is unbounded at $x = \pm 1$.
//
// Math
// ----
// $$y = \arccos(x), \qquad
// \frac{\partial y}{\partial x} = -\frac{1}{\sqrt{1 - x^2}}, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// -\frac{1}{\sqrt{1 - x^2}}\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"arccos"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvacosf``/``vvacos`` (CPU) / MLX ``arccos`` (GPU).
class LUCID_API AcosBackward : public UnaryOp<AcosBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.acos(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Compute $y = \arccos(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`AcosBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor with values in $[-1, 1]$.  Out-of-domain values
//     produce NaN.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values in $[0, \pi]$.
//
// See Also
// --------
// :class:`AcosBackward` — backward node.
LUCID_API TensorImplPtr arccos_op(const TensorImplPtr& a);

// Autograd node for element-wise arctangent $y = \arctan(x)$.
//
// Saves the input ``x`` so the backward pass can compute the denominator
// $1 + x^2$.  Defined on all of $\mathbb{R}$ with range
// $(-\tfrac{\pi}{2}, \tfrac{\pi}{2})$ and a smooth, bounded gradient.
//
// Math
// ----
// $$y = \arctan(x), \qquad
// \frac{\partial y}{\partial x} = \frac{1}{1 + x^2}, \qquad
// \frac{\partial \mathcal{L}}{\partial x} =
// \frac{1}{1 + x^2}\,\frac{\partial \mathcal{L}}{\partial y}.$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     Registered as ``"arctan"`` with ``AmpPolicy::Promote``.
//
// Notes
// -----
// Dispatch: Accelerate ``vvatanf``/``vvatan`` (CPU) / MLX ``arctan`` (GPU).
class LUCID_API AtanBackward : public UnaryOp<AtanBackward> {
public:
    static const OpSchema schema_v1;
    static Storage dispatch(backend::IBackend& be, const Storage& a, const Shape& s, Dtype dt) {
        return be.atan(a, s, dt);
    }
    Storage grad_formula(const Storage& g);
};

// Compute $y = \arctan(x)$ element-wise.  Allocates a fresh output of the
// same shape and dtype as ``a`` and delegates to
// :class:`AtanBackward::forward`.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any shape.  Domain is all of $\mathbb{R}$.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype, values in
//     $(-\tfrac{\pi}{2}, \tfrac{\pi}{2})$.
//
// See Also
// --------
// :class:`AtanBackward` — backward node.
LUCID_API TensorImplPtr arctan_op(const TensorImplPtr& a);

}  // namespace lucid
