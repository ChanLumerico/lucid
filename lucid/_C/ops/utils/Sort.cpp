#include "Sort.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/FuncOp.h"
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
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::wrap_axis;

bool differentiable_dtype(Dtype dt) {
    return dt == Dtype::F32 || dt == Dtype::F64;
}

Storage scatter_axis_add_storage(const Storage& grad,
                                 const Storage& indices,
                                 const Shape& input_shape,
                                 const Shape& grad_shape,
                                 int axis,
                                 Dtype dt,
                                 Device device) {
    return backend::Dispatcher::for_device(device)
        .scatter_add_axis(grad, indices, input_shape, grad_shape, axis, dt);
}

class IndexScatterBackward : public FuncOp<IndexScatterBackward, 1> {
public:
    static const OpSchema schema_v1;

    Storage indices_;
    Shape grad_shape_;
    int axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        return {scatter_axis_add_storage(grad_out, indices_, input_shapes_[0], grad_shape_, axis_,
                                         dtype_, device_)};
    }
};

const OpSchema IndexScatterBackward::schema_v1{"index_scatter", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, /*internal=*/true};

TensorImplPtr attach_index_scatter_grad(const TensorImplPtr& a,
                                        TensorImplPtr out,
                                        Storage indices,
                                        int axis) {
    if (!differentiable_dtype(a->dtype()))
        return out;
    auto bwd = std::make_shared<IndexScatterBackward>();
    bwd->grad_shape_ = out->shape();
    bwd->indices_ = std::move(indices);
    bwd->axis_ = axis;
    kernel::NaryKernel<IndexScatterBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                               /*save_ins=*/false);
    return out;
}

}  // namespace

TensorImplPtr sort_op(const TensorImplPtr& a, int axis) {
    Validator::input(a, "sort.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"sort", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Shape out_shape = a->shape();
    auto [values, indices] = backend::Dispatcher::for_device(device)
                                 .sort_select(a->storage(), a->shape(), out_shape, ax, dt,
                                              /*descending=*/false);
    auto out = fresh(std::move(values), std::move(out_shape), dt, device);
    return attach_index_scatter_grad(a, std::move(out), std::move(indices), ax);
}

TensorImplPtr argsort_op(const TensorImplPtr& a, int axis) {
    Validator::input(a, "argsort.a").non_null();
    const Device device = a->device();
    OpScopeFull scope{"argsort", device, a->dtype(), a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Storage out = backend::Dispatcher::for_device(device).argsort(a->storage(), a->shape(), ax,
                                                                  a->dtype());
    return fresh(std::move(out), a->shape(), Dtype::I32, device);
}

namespace {
TensorImplPtr argext_dispatch(
    const TensorImplPtr& a, int axis, bool keepdims, bool is_min, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Device device = a->device();
    OpScopeFull scope{name, device, a->dtype(), a->shape()};
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        auto out = is_min ? ::mlx::core::argmin(*ga.arr, axis, keepdims)
                          : ::mlx::core::argmax(*ga.arr, axis, keepdims);
        Shape sh = mlx_shape_to_lucid(out.shape());
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out), Dtype::I32)}, std::move(sh),
                     Dtype::I32, device);
    }
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Shape out_shape = a->shape();
    if (keepdims)
        out_shape[ax] = 1;
    else
        out_shape.erase(out_shape.begin() + ax);
    auto out_cpu = allocate_cpu(out_shape, Dtype::I64);
    const auto& ca = std::get<CpuStorage>(a->storage());
    std::size_t outer = 1;
    for (int d = 0; d < ax; ++d)
        outer *= static_cast<std::size_t>(a->shape()[d]);
    std::size_t inner = 1;
    for (std::size_t d = ax + 1; d < a->shape().size(); ++d)
        inner *= static_cast<std::size_t>(a->shape()[d]);
    const std::size_t L = static_cast<std::size_t>(a->shape()[ax]);
    auto run = [&](const auto* src) {
        auto* dst = reinterpret_cast<std::int64_t*>(out_cpu.ptr.get());
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < inner; ++j) {
                std::int64_t best = 0;
                auto best_v = src[o * L * inner + j];
                for (std::size_t k = 1; k < L; ++k) {
                    const auto v = src[(o * L + k) * inner + j];
                    if (is_min ? (v < best_v) : (v > best_v)) {
                        best_v = v;
                        best = static_cast<std::int64_t>(k);
                    }
                }
                dst[o * inner + j] = best;
            }
    };
    if (a->dtype() == Dtype::F32)
        run(reinterpret_cast<const float*>(ca.ptr.get()));
    else if (a->dtype() == Dtype::F64)
        run(reinterpret_cast<const double*>(ca.ptr.get()));
    else if (a->dtype() == Dtype::I32)
        run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()));
    else if (a->dtype() == Dtype::I64)
        run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()));
    else
        ErrorBuilder(name).not_implemented("dtype not supported");
    return fresh(Storage{std::move(out_cpu)}, std::move(out_shape), Dtype::I64, device);
}
}  // namespace

TensorImplPtr argmax_op(const TensorImplPtr& a, int axis, bool keepdims) {
    return argext_dispatch(a, axis, keepdims, /*is_min=*/false, "argmax");
}
TensorImplPtr argmin_op(const TensorImplPtr& a, int axis, bool keepdims) {
    return argext_dispatch(a, axis, keepdims, /*is_min=*/true, "argmin");
}

TensorImplPtr nonzero_op(const TensorImplPtr& a) {
    // Returns a 2-D tensor of shape (N, ndim) where N = number of non-zero
    // elements. Each row is a multi-index into `a`. Matches numpy
    // `np.nonzero` flattened-into-2D semantics.
    //
    // CPU-only op (data-dependent output length forces GPU sync; MLX has
    // no equivalent primitive). Per convention, GPU input is accepted but
    // the result lives on CPU — caller must `.gpu()` if they need it back.
    Validator::input(a, "nonzero.a").non_null();
    const std::size_t ndim = a->shape().size();
    OpScopeFull scope{"nonzero", a->device(), a->dtype(), a->shape()};

    CpuStorage cpu = (a->device() == Device::GPU)
                         ? gpu::download_gpu_to_cpu(std::get<GpuStorage>(a->storage()), a->shape())
                         : std::get<CpuStorage>(a->storage());

    const std::size_t n = shape_numel(a->shape());
    std::vector<bool> mask(n, false);
    auto check_nonzero = [&](const auto* p) {
        for (std::size_t i = 0; i < n; ++i)
            mask[i] = static_cast<double>(p[i]) != 0.0;
    };
    switch (a->dtype()) {
        case Dtype::F32:
            check_nonzero(reinterpret_cast<const float*>(cpu.ptr.get()));
            break;
        case Dtype::F64:
            check_nonzero(reinterpret_cast<const double*>(cpu.ptr.get()));
            break;
        case Dtype::I32:
            check_nonzero(reinterpret_cast<const std::int32_t*>(cpu.ptr.get()));
            break;
        case Dtype::I64:
            check_nonzero(reinterpret_cast<const std::int64_t*>(cpu.ptr.get()));
            break;
        case Dtype::Bool:
            check_nonzero(reinterpret_cast<const std::uint8_t*>(cpu.ptr.get()));
            break;
        default:
            ErrorBuilder("nonzero").not_implemented("dtype not supported");
    }
    // Count non-zeros and allocate (N, ndim) output.
    std::size_t count = 0;
    for (auto m : mask)
        if (m)
            ++count;
    Shape out_shape{static_cast<std::int64_t>(count), static_cast<std::int64_t>(ndim)};
    auto out = allocate_cpu(out_shape, Dtype::I64);
    auto* dst = reinterpret_cast<std::int64_t*>(out.ptr.get());

    // Build strides for unraveling flat index into multi-index.
    Stride stride(ndim);
    if (ndim > 0) {
        stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)ndim - 2; d >= 0; --d)
            stride[d] = stride[d + 1] * a->shape()[d + 1];
    }
    std::size_t row = 0;
    for (std::size_t flat = 0; flat < n; ++flat) {
        if (!mask[flat])
            continue;
        std::size_t rem = flat;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::int64_t coord = rem / static_cast<std::size_t>(stride[d]);
            rem %= static_cast<std::size_t>(stride[d]);
            dst[row * ndim + d] = coord;
        }
        ++row;
    }
    return fresh(Storage{std::move(out)}, std::move(out_shape), Dtype::I64, /*device=*/Device::CPU);
}

TensorImplPtr unique_op(const TensorImplPtr& a) {
    // Sorted unique values across the flattened input.
    //
    // CPU-only op (variable-length output; MLX has no equivalent). Per
    // convention, GPU input is accepted but result lives on CPU.
    Validator::input(a, "unique.a").non_null();
    const Dtype dt = a->dtype();
    OpScopeFull scope{"unique", a->device(), dt, a->shape()};

    CpuStorage cpu = (a->device() == Device::GPU)
                         ? gpu::download_gpu_to_cpu(std::get<GpuStorage>(a->storage()), a->shape())
                         : std::get<CpuStorage>(a->storage());
    const std::size_t n = shape_numel(a->shape());

    auto run = [&](const auto* src) -> CpuStorage {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(src)>>;
        std::vector<T> vals(src, src + n);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
        CpuStorage out;
        out.dtype = dt;
        out.nbytes = vals.size() * sizeof(T);
        out.ptr = allocate_aligned_bytes(out.nbytes);
        std::memcpy(out.ptr.get(), vals.data(), out.nbytes);
        return out;
    };

    CpuStorage out_cpu;
    Shape out_shape;
    auto wrap = [&](auto&& s) {
        out_shape = {static_cast<std::int64_t>(s.nbytes / dtype_size(dt))};
        out_cpu = std::move(s);
    };
    if (dt == Dtype::F32)
        wrap(run(reinterpret_cast<const float*>(cpu.ptr.get())));
    else if (dt == Dtype::F64)
        wrap(run(reinterpret_cast<const double*>(cpu.ptr.get())));
    else if (dt == Dtype::I32)
        wrap(run(reinterpret_cast<const std::int32_t*>(cpu.ptr.get())));
    else if (dt == Dtype::I64)
        wrap(run(reinterpret_cast<const std::int64_t*>(cpu.ptr.get())));
    else
        ErrorBuilder("unique").not_implemented("dtype not supported");
    return fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, /*device=*/Device::CPU);
}

std::vector<TensorImplPtr> topk_op(const TensorImplPtr& a, std::int64_t k, int axis) {
    Validator::input(a, "topk.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"topk", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    if (k <= 0 || k > a->shape()[static_cast<std::size_t>(ax)])
        ErrorBuilder("topk").fail("k must be in (0, axis_size]");
    Shape out_shape = a->shape();
    out_shape[static_cast<std::size_t>(ax)] = k;
    auto [values, indices] = backend::Dispatcher::for_device(device)
                                 .sort_select(a->storage(), a->shape(), out_shape, ax, dt,
                                              /*descending=*/true);
    auto indices_out = fresh(std::move(indices), out_shape, Dtype::I32, device);
    auto values_out = fresh(std::move(values), out_shape, dt, device);
    values_out = attach_index_scatter_grad(a, std::move(values_out), indices_out->storage(), ax);
    return {std::move(values_out), std::move(indices_out)};
}

LUCID_REGISTER_OP(IndexScatterBackward)

}  // namespace lucid
