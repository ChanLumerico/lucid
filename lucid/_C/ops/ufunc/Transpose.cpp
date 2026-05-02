#include "Transpose.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

const OpSchema PermuteBackward::schema_v1{"permute", 1, AmpPolicy::KeepInput, true};

namespace {

// Validate / normalize a permutation. Returns sorted-ascending check vector.
std::vector<int> validate_perm(const std::vector<int>& perm, int ndim) {
    if (static_cast<int>(perm.size()) != ndim) {
        ErrorBuilder("permute").fail("perm length must equal tensor ndim");
    }
    std::vector<int> normalized;
    normalized.reserve(ndim);
    for (int p : perm) {
        const int wrapped = p < 0 ? p + ndim : p;
        if (wrapped < 0 || wrapped >= ndim) {
            ErrorBuilder("permute").index_error("axis out of range");
        }
        normalized.push_back(wrapped);
    }
    std::vector<int> sorted = normalized;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < ndim; ++i) {
        if (sorted[i] != i) {
            ErrorBuilder("permute").fail("perm must be a permutation of 0..ndim-1");
        }
    }
    return normalized;
}

std::vector<int> inverse_perm(const std::vector<int>& perm) {
    std::vector<int> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        inv[static_cast<std::size_t>(perm[i])] = static_cast<int>(i);
    }
    return inv;
}

}  // namespace

TensorImplPtr PermuteBackward::forward(const TensorImplPtr& a, const std::vector<int>& perm_user) {
    Validator::input(a, "permute.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    const auto perm = validate_perm(perm_user, ndim);

    // Compute output shape and adjusted strides (metadata-only view).
    Shape out_shape;
    Stride out_stride;
    out_shape.reserve(ndim);
    out_stride.reserve(ndim);
    for (int p : perm) {
        out_shape.push_back(a->shape()[static_cast<std::size_t>(p)]);
        out_stride.push_back(a->stride()[static_cast<std::size_t>(p)]);
    }

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), out_shape};

    // Both paths physically materialise the permuted data so that downstream
    // ops (matmul, conv, ...) can always assume contiguous row-major layout.
    // GPU: MLX contiguous() materialises the lazy transpose in-place.
    // CPU: permute_copy_<dtype> produces a fresh contiguous CpuStorage.
    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .permute(a->storage(), a->shape(), perm, a->dtype());
    TensorImplPtr out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(),
                                                     a->device(), false);

    auto bwd = std::make_shared<PermuteBackward>();
    bwd->perm_ = perm;
    kernel::NaryKernel<PermuteBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                          /*save_ins=*/false);
    return out;
}

std::vector<Storage> PermuteBackward::apply(Storage grad_out) {
    // dx = permute(g, inverse_perm). The gradient arrives in `out_shape_`;
    // applying inverse_perm produces a buffer in `input_shapes_[0]` layout.
    const auto inv = inverse_perm(perm_);
    Storage dx =
        backend::Dispatcher::for_device(device_).permute(grad_out, out_shape_, inv, dtype_);
    return {std::move(dx)};
}

TensorImplPtr permute_op(const TensorImplPtr& a, const std::vector<int>& perm) {
    return PermuteBackward::forward(a, perm);
}

// transpose(t) / _T(t) — reverse all axes.
TensorImplPtr transpose_op(const TensorImplPtr& a) {
    Validator::input(a, "transpose.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    std::vector<int> perm(ndim);
    for (int i = 0; i < ndim; ++i)
        perm[i] = ndim - 1 - i;
    return PermuteBackward::forward(a, perm);
}

TensorImplPtr T_op(const TensorImplPtr& a) {
    return transpose_op(a);
}

// _mT(t) — swap last two axes. Requires ndim >= 2.
TensorImplPtr mT_op(const TensorImplPtr& a) {
    Validator::input(a, "mT.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    if (ndim < 2)
        ErrorBuilder("mT").fail("requires ndim >= 2");
    std::vector<int> perm(ndim);
    for (int i = 0; i < ndim; ++i)
        perm[i] = i;
    std::swap(perm[ndim - 1], perm[ndim - 2]);
    return PermuteBackward::forward(a, perm);
}

// swapaxes(t, a1, a2) — swap two specific axes.
TensorImplPtr swapaxes_op(const TensorImplPtr& a, int axis1, int axis2) {
    Validator::input(a, "swapaxes.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    auto wrap = [&](int x) {
        const int w = x < 0 ? x + ndim : x;
        if (w < 0 || w >= ndim)
            ErrorBuilder("swapaxes").index_error("axis out of range");
        return w;
    };
    const int w1 = wrap(axis1);
    const int w2 = wrap(axis2);
    std::vector<int> perm(ndim);
    for (int i = 0; i < ndim; ++i)
        perm[i] = i;
    std::swap(perm[w1], perm[w2]);
    return PermuteBackward::forward(a, perm);
}

LUCID_REGISTER_OP(PermuteBackward)

}  // namespace lucid
