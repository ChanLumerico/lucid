// lucid/_C/ops/composite/Indexing.h
//
// Indexing convenience ops layered on top of ``gather``, ``scatter_add``,
// ``split_at``, and ``sort``.  Each entry function below is a thin shape
// shim — the underlying primitives carry the gradient.
//
//   take(a, indices)            — gather over a flattened ``a``
//   index_select(a, dim, idx)   — gather with a 1-D index broadcast to ``a``'s rank
//   narrow(a, dim, start, len)  — slice a contiguous window via ``split_at``
//   scatter(base, dim, idx, src)— overwrite-semantics scatter via ``scatter_add``
//   kthvalue(a, k, dim, keepdim)— sort + gather to pluck the k-th element

#pragma once

#include <cstdint>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr take_op(const TensorImplPtr& a, const TensorImplPtr& indices);
LUCID_API TensorImplPtr
index_select_op(const TensorImplPtr& a, int dim, const TensorImplPtr& indices);

LUCID_API TensorImplPtr
narrow_op(const TensorImplPtr& a, int dim, std::int64_t start, std::int64_t length);

LUCID_API TensorImplPtr scatter_op(const TensorImplPtr& base,
                                   int dim,
                                   const TensorImplPtr& indices,
                                   const TensorImplPtr& src);
LUCID_API TensorImplPtr
kthvalue_op(const TensorImplPtr& a, std::int64_t k, int dim, bool keepdim);

}  // namespace lucid
