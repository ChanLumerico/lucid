// lucid/_C/ops/bfunc/_Extremum.h
//
// The gradient split shared by MaximumBackward and MinimumBackward.  Both
// helpers return ``{first, second}``: ``first`` is zero where ``p < q``
// and ``second`` where ``q < p``.  Maximum passes ``(a, b)``: ``a`` loses
// where ``a < b``.  Minimum passes ``(b, a)``: ``a`` loses where ``b < a``.
// Either way the result is ``{da, db}``.
//
// The rule is the reference's, read off its vectorised kernels:
//
//   - the winner takes the whole gradient and the loser none;
//   - a tie (``-0`` against ``+0`` and ``inf`` against ``inf`` included)
//     splits it in half;
//   - an unordered pair (a NaN on either side) gives *both* operands the
//     whole gradient, since neither "loses" a comparison that is false.
//
// Sending a tie to the first operand instead made ``hypot`` and the
// complex ``abs`` — both built on maximum and minimum of the same pair —
// double the gradient on ``|a| == |b|``.

#pragma once

#include <utility>

#include "../../autograd/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Select.h"
#include "Compare.h"
#include "Mul.h"

namespace lucid::bfunc_detail {

// Eager weights at the output's shape.  ``first`` keeps
// $1 - \tfrac12(\mathbb 1\{p < q\} + \mathbb 1\{q \ge p\})$ of the
// gradient: 0 where ``p < q``, ½ on a tie, 1 where ``p > q`` or the pair
// is unordered; ``second`` the same with ``p`` and ``q`` exchanged.
// Every term is 0, ½ or 1, so the weight is exact in any floating dtype.
inline std::pair<Storage, Storage> extremum_grads(const Storage& grad_out,
                                                  const Storage& p,
                                                  const Storage& q,
                                                  std::size_t n,
                                                  Dtype dt,
                                                  Device device) {
    auto share = [&](const Storage& self, const Storage& other) {
        Storage lost = lt_mask_storage(self, other, n, dt, device);
        Storage not_won = ge_mask_storage(other, self, n, dt, device);
        Storage halves = add_storages(lost, not_won, n, dt, device);
        Storage weight =
            add_scalar_storage(mul_scalar_storage(halves, -0.5, n, dt, device), 1.0, n, dt, device);
        return multiply_storages(grad_out, weight, n, dt, device);
    };
    return {share(p, q), share(q, p)};
}

// Graph-mode form of :func:`extremum_grads`, the reference's own
// ``where(p == q, g / 2, g).masked_fill(p < q, 0)``.  The masks are
// constants of the inputs, so the result depends on ``grad_out`` alone
// and stays detached when it is.
inline std::pair<TensorImplPtr, TensorImplPtr> extremum_grads_graph(const TensorImplPtr& grad_out,
                                                                    const TensorImplPtr& p,
                                                                    const TensorImplPtr& q) {
    auto zero = zeros_like_op(grad_out);
    auto shared = where_op(equal_op(p, q), mul_op(grad_out, full_like_op(grad_out, 0.5)), grad_out);
    return {where_op(less_op(p, q), zero, shared), where_op(less_op(q, p), zero, shared)};
}

}  // namespace lucid::bfunc_detail
