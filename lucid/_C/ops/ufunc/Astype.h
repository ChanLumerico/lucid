// lucid/_C/ops/ufunc/Astype.h
// Element-wise dtype cast: same shape, different element type.
//
// Autograd:
//   astype IS differentiable.  When the input has requires_grad and grad
//   mode is enabled, ``astype_op`` wires an ``AstypeBackward`` node that
//   casts the incoming gradient back to the source dtype.  This makes
//   ``logits.float()`` mid-graph behave correctly under autograd, and
//   lets AMP-aware ops (Linear / Conv / Matmul / BatchNorm) cast their
//   inputs to the effective compute dtype while keeping the gradient
//   chain intact — the backward pass of the kernel produces a gradient
//   at the cast dtype, then AstypeBackward casts it back to the
//   original Parameter / activation dtype before it accumulates.
//
//   Without this wiring (Lucid ≤ 3.3.0-rc) every AMP cast silently
//   produced a ``requires_grad=False`` tensor, which broke the
//   ``wire_autograd`` any_grad check in NaryKernel and dropped the
//   entire backward chain at every cast boundary.
#pragma once
#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/Dtype.h"
#include "../../core/OpSchema.h"
#include "../../core/fwd.h"
#include "../../kernel/IKernel.h"
namespace lucid {

// Autograd node for ``astype``: casts the incoming gradient back to the
// source dtype so it can accumulate into the original Parameter or
// activation's gradient slot.
//
// The forward cast is value-preserving — integer-to-float and
// float-to-integer paths route through the backend's ``astype`` kernel,
// which clamps / rounds as appropriate rather than reinterpreting bits.
// The backward applies the inverse cast (also value-preserving), so a
// no-op forward cast (matching dtypes) becomes a no-op backward.  This
// is the hook that lets AMP-driven casts inside Linear / Conv / Matmul /
// BatchNorm participate in the autograd graph instead of silently
// dropping ``requires_grad`` at every boundary.
//
// Math
// ----
// $$y = \mathrm{cast}_{D_{\text{dst}}}(x),
//   \qquad \frac{\partial L}{\partial x}
//   = \mathrm{cast}_{D_{\text{src}}}\!\Bigl(\frac{\partial L}{\partial y}\Bigr)$$
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"astype"``, ``AmpPolicy::KeepInput`` — the AMP layer does not
//     promote the input because ``astype`` itself is the dtype-control
//     primitive that AMP composes with.
// src_dtype_ : Dtype
//     Dtype of the original input; the gradient is cast back to this
//     dtype during backward.  Default ``Dtype::F32``.
// dtype_ : Dtype
//     Inherited from ``AutogradNode``; carries the cast / output dtype.
//
// Notes
// -----
// Complex dtypes are not currently routed through this op; AMP and the
// public ``Tensor.astype`` API only target the real-floating and integer
// dtype families.  Same-dtype calls short-circuit before reaching this
// node — see ``astype_op``.
class LUCID_API AstypeBackward : public FuncOp<AstypeBackward, 1>, public kernel::IKernel {
public:
    static const OpSchema schema_v1;

    // Source dtype to cast the gradient back to during backward.
    Dtype src_dtype_ = Dtype::F32;

    // Schema-driven op name; participates in the profiler / op registry.
    std::string_view name() const noexcept override { return schema_v1.name; }

    // Node-name override used by the autograd graph dumper.
    std::string node_name() const override { return std::string(schema_v1.name); }

    // Eager-mode backward.  Casts ``grad_out`` from ``dtype_`` (the cast
    // / output dtype) back to ``src_dtype_`` so it can accumulate into
    // the original input's gradient slot.  If the two dtypes match (the
    // forward cast was a no-op), passes the gradient through unchanged.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient at the cast dtype.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Single-element vector holding the gradient cast back to
    //     ``src_dtype_``.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward used when ``create_graph=True``.  Recursively
    // calls ``astype_op`` so the inverse cast itself remains
    // differentiable, enabling higher-order gradients through AMP
    // boundaries.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Cast ``a`` to ``dst_dtype``, preserving shape and (when grad mode is
// enabled) wiring an ``AstypeBackward`` node that casts the gradient back
// during backward.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// dst_dtype : Dtype
//     Target dtype for the output.  Floating <-> integer casts are
//     value-preserving (clamp / round semantics live in the backend's
//     ``astype`` kernel, not in a raw reinterpret).
//
// Returns
// -------
// TensorImplPtr
//     New tensor with dtype ``dst_dtype`` and the same shape as ``a``.
//     If ``a.dtype() == dst_dtype`` the function returns ``a`` itself so
//     the existing ``grad_fn`` chain is preserved — earlier versions
//     allocated a fresh ``TensorImpl`` here which silently broke autograd
//     routing under AMP.
//
// Notes
// -----
// Used internally by the AMP-aware kernels (Linear / Conv / Matmul /
// BatchNorm) to cast inputs to the effective compute dtype while keeping
// the backward chain intact.  Argmin / argmax reductions also call this
// to materialise their integer outputs; those callers do not enable
// gradients, so the autograd wiring path is skipped and the cost stays
// identical to a raw backend ``astype``.
LUCID_API TensorImplPtr astype_op(const TensorImplPtr& a, Dtype dst_dtype);
}  // namespace lucid
