#include "Transpose.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/cpu/Shape.h"
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

CpuStorage allocate_like(const Shape& shape, Dtype dt) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = shape_numel(shape) * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    return out;
}

CpuStorage permute_copy(const CpuStorage& in,
                        const Shape& in_shape,
                        const std::vector<int>& perm,
                        Dtype dt) {
    Shape out_shape;
    out_shape.reserve(perm.size());
    for (int p : perm)
        out_shape.push_back(in_shape[static_cast<std::size_t>(p)]);

    auto out = allocate_like(out_shape, dt);
    if (out.nbytes == 0)
        return out;

    switch (dt) {
        case Dtype::F32:
            backend::cpu::permute_copy_f32(reinterpret_cast<const float*>(in.ptr.get()),
                                           reinterpret_cast<float*>(out.ptr.get()), in_shape, perm);
            break;
        case Dtype::F64:
            backend::cpu::permute_copy_f64(reinterpret_cast<const double*>(in.ptr.get()),
                                           reinterpret_cast<double*>(out.ptr.get()), in_shape,
                                           perm);
            break;
        case Dtype::I32:
            backend::cpu::permute_copy_i32(reinterpret_cast<const std::int32_t*>(in.ptr.get()),
                                           reinterpret_cast<std::int32_t*>(out.ptr.get()), in_shape,
                                           perm);
            break;
        case Dtype::I64:
            backend::cpu::permute_copy_i64(reinterpret_cast<const std::int64_t*>(in.ptr.get()),
                                           reinterpret_cast<std::int64_t*>(out.ptr.get()), in_shape,
                                           perm);
            break;
        default:
            ErrorBuilder("permute").not_implemented("dtype not supported (F32/F64/I32/I64)");
    }
    return out;
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

    // CPU path: metadata-only view — no copy, O(0) allocations.
    // GPU path: MLX lazy transpose — also avoids materialisation.
    TensorImplPtr out;
    if (a->device() == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        if (!ga.arr)
            ErrorBuilder("permute").fail("null GPU array");
        // MLX transpose: contiguous-materialize so downstream GPU kernels
        // receive a layout-contiguous GpuStorage (MLX's own view semantics
        // are internal; our GpuStorage wrapper always holds a contiguous array).
        auto raw = ::mlx::core::transpose(*ga.arr, perm);
        raw = ::mlx::core::contiguous(raw);
        auto out_storage = Storage{gpu::wrap_mlx_array(std::move(raw), a->dtype())};
        out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(),
                                           a->device(), false);
    } else {
        out = TensorImpl::make_view(a, out_shape, out_stride, /*offset_bytes=*/0);
    }

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
    Storage dx;
    if (device_ == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad_out);
        if (!gg.arr)
            ErrorBuilder("permute backward").fail("null GPU array");
        auto raw = ::mlx::core::transpose(*gg.arr, inv);
        raw = ::mlx::core::contiguous(raw);
        dx = Storage{gpu::wrap_mlx_array(std::move(raw), dtype_)};
    } else {
        const auto& g_cpu = std::get<CpuStorage>(grad_out);
        dx = Storage{permute_copy(g_cpu, out_shape_, inv, dtype_)};
    }
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
