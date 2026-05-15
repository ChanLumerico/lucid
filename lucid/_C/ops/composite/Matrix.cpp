// lucid/_C/ops/composite/Matrix.cpp
//
// ``mm`` and ``bmm`` validate the rank up front then defer to ``matmul_op``;
// the rank check makes their error messages crisper than letting matmul's
// internal planner reject the input.
//
// ``kron`` builds the block structure with the standard outer-product trick:
// expand each input to a 2·ndim-rank tensor where every dimension of ``a`` is
// followed by a size-1 placeholder for ``b`` and vice versa, multiply with
// broadcasting, then collapse the interleaved pairs back into the result
// shape (sa[i] · sb[i]).

#include "Matrix.h"

#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/Matmul.h"
#include "../bfunc/Mul.h"
#include "../utils/View.h"

namespace lucid {

TensorImplPtr mm_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b)
        ErrorBuilder("mm").fail("null input");
    if (a->shape().size() != 2 || b->shape().size() != 2)
        ErrorBuilder("mm").fail("both operands must be 2-D");
    return matmul_op(a, b);
}

TensorImplPtr bmm_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b)
        ErrorBuilder("bmm").fail("null input");
    if (a->shape().size() != 3 || b->shape().size() != 3)
        ErrorBuilder("bmm").fail("both operands must be 3-D");
    if (a->shape()[0] != b->shape()[0])
        ErrorBuilder("bmm").fail("batch dimensions must match");
    return matmul_op(a, b);
}

TensorImplPtr kron_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    if (!a || !b)
        ErrorBuilder("kron").fail("null input");
    const auto& sa = a->shape();
    const auto& sb = b->shape();
    if (sa.size() != sb.size())
        ErrorBuilder("kron").fail("operands must share rank");

    // Interleave size-1 placeholders so the multiply broadcasts to the full
    // 2·ndim-rank block tensor.
    const std::size_t ndim = sa.size();
    Shape a_expanded;
    Shape b_expanded;
    a_expanded.reserve(2 * ndim);
    b_expanded.reserve(2 * ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        a_expanded.push_back(sa[i]);
        a_expanded.push_back(1);
        b_expanded.push_back(1);
        b_expanded.push_back(sb[i]);
    }
    auto a_r = reshape_op(a, a_expanded);
    auto b_r = reshape_op(b, b_expanded);
    auto product = mul_op(a_r, b_r);

    // Collapse each (sa[i], sb[i]) pair into a single sa[i]·sb[i] axis.
    Shape out_shape;
    out_shape.reserve(ndim);
    for (std::size_t i = 0; i < ndim; ++i)
        out_shape.push_back(sa[i] * sb[i]);
    return reshape_op(product, out_shape);
}

}  // namespace lucid
