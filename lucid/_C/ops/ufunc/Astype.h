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

// Backward node for ``astype``: casts the incoming gradient (at the cast
// dtype) back to the source dtype before propagating it upstream.
//
// Saved state:
//   src_dtype_ — the dtype of the original input.  ``dtype_`` (inherited
//                from AutogradNode) carries the cast/output dtype.
//   device_    — the device on which to perform the inverse cast.
class LUCID_API AstypeBackward : public FuncOp<AstypeBackward, 1>, public kernel::IKernel {
public:
    static const OpSchema schema_v1;

    // Source dtype to cast the gradient back to during backward.
    Dtype src_dtype_ = Dtype::F32;

    std::string_view name() const noexcept override { return schema_v1.name; }
    std::string node_name() const override { return std::string(schema_v1.name); }

    // Cast ``grad_out`` from ``dtype_`` (= dst_dtype) back to ``src_dtype_``.
    // If they match (no-op cast), passes the gradient through unchanged.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward for create_graph=True.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

LUCID_API TensorImplPtr astype_op(const TensorImplPtr& a, Dtype dst_dtype);
}  // namespace lucid
