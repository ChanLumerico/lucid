// lucid/_C/ops/diffeq/Operand.h
//
// Shared operand checking for the diffeq primitives.
//
// Every op here takes one tensor that fixes the dtype, device, and shape of
// the call -- the state, or the residual -- and some number of operands that
// have to agree with it.  Each op had grown its own spelling of that check:
// an inline loop in one, a local lambda in another, a straight-line sequence
// in the third.  Three copies of the same four comparisons is where they stop
// being incidental, and where a divergence between them stops being visible.

#pragma once

#include <string>
#include <string_view>

#include "../../core/Error.h"
#include "../../core/Shape.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"

namespace lucid::diffeq {

// How strictly an operand's shape has to match the reference's.
enum class ShapeRule {
    // Identical shape.  What a stagewise combination needs: the arithmetic is
    // elementwise, so anything else is a bug in the caller.
    Exact,
    // Identical element count, any shape.  What a reduction needs: a linear
    // solve hands back an (n, 1) column where the state is flat, and forcing
    // a reshape would allocate on every iteration to satisfy a check rather
    // than the arithmetic.
    SameCount,
};

// Reference metadata every operand of one call is measured against.
//
// Constructed once from the op's defining tensor and passed to each
// :func:`check_operand`, so the four comparisons stay identical across
// operands and across ops.
struct OperandSpec {
    Dtype dtype;
    Device device;
    Shape shape;
    // Op name, used only to build error messages.
    std::string_view op;

    // Build a spec from the tensor that defines the call.
    static OperandSpec from(const TensorImplPtr& reference, std::string_view op) {
        return OperandSpec{reference->dtype(), reference->device(), reference->shape(), op};
    }
};

// Reject an operand that disagrees with the call's reference tensor.
//
// Parameters
// ----------
// t : TensorImplPtr
//     The operand to check.
// label : std::string_view
//     Argument name, as it should appear in a null-check message.
// spec : OperandSpec
//     Reference metadata for this call.
// rule : ShapeRule
//     Whether the shape must match exactly or only in element count.
//
// Raises
// ------
// DtypeMismatch, DeviceMismatch, ShapeMismatch
//     On the first disagreement found, naming the op from ``spec``.
inline void check_operand(const TensorImplPtr& t,
                          std::string_view label,
                          const OperandSpec& spec,
                          ShapeRule rule = ShapeRule::Exact) {
    Validator::input(t, std::string(label)).non_null();
    if (t->dtype() != spec.dtype)
        throw DtypeMismatch(std::string(dtype_name(spec.dtype)),
                            std::string(dtype_name(t->dtype())), std::string(spec.op));
    if (t->device() != spec.device)
        throw DeviceMismatch(std::string(device_name(spec.device)),
                             std::string(device_name(t->device())), std::string(spec.op));

    const bool ok = rule == ShapeRule::Exact
                        ? t->shape() == spec.shape
                        : shape_numel(t->shape()) == shape_numel(spec.shape);
    if (!ok)
        throw ShapeMismatch(spec.shape, t->shape(), std::string(spec.op));
}

}  // namespace lucid::diffeq
