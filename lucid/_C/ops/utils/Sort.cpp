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
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
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

template <typename T>
std::pair<CpuStorage, CpuStorage> sort_select_cpu(const CpuStorage& input,
                                                  const Shape& input_shape,
                                                  const Shape& output_shape,
                                                  int axis,
                                                  Dtype dt,
                                                  bool descending) {
    auto values = allocate_cpu(output_shape, dt);
    auto indices = allocate_cpu(output_shape, Dtype::I32);
    const auto* src = reinterpret_cast<const T*>(input.ptr.get());
    auto* dst = reinterpret_cast<T*>(values.ptr.get());
    auto* idx_dst = reinterpret_cast<std::int32_t*>(indices.ptr.get());

    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(input_shape[d]);
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < input_shape.size(); ++d)
        inner *= static_cast<std::size_t>(input_shape[d]);
    const std::size_t L = static_cast<std::size_t>(input_shape[static_cast<std::size_t>(axis)]);
    const std::size_t K = static_cast<std::size_t>(output_shape[static_cast<std::size_t>(axis)]);

    std::vector<std::int32_t> order(L);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            std::iota(order.begin(), order.end(), 0);
            auto cmp = [&](std::int32_t lhs, std::int32_t rhs) {
                const T lv = src[(o * L + static_cast<std::size_t>(lhs)) * inner + j];
                const T rv = src[(o * L + static_cast<std::size_t>(rhs)) * inner + j];
                return descending ? (lv > rv) : (lv < rv);
            };
            if (K == L) {
                std::sort(order.begin(), order.end(), cmp);
            } else {
                std::partial_sort(order.begin(), order.begin() + K, order.end(), cmp);
            }
            for (std::size_t k = 0; k < K; ++k) {
                const std::int32_t src_k = order[k];
                const std::size_t out_flat = (o * K + k) * inner + j;
                const std::size_t src_flat = (o * L + static_cast<std::size_t>(src_k)) * inner + j;
                dst[out_flat] = src[src_flat];
                idx_dst[out_flat] = src_k;
            }
        }
    }
    return {std::move(values), std::move(indices)};
}

template <typename T>
CpuStorage scatter_axis_add_cpu(const CpuStorage& grad,
                                const CpuStorage& indices,
                                const Shape& input_shape,
                                const Shape& grad_shape,
                                int axis,
                                Dtype dt) {
    auto out = allocate_cpu(input_shape, dt);
    const auto* g = reinterpret_cast<const T*>(grad.ptr.get());
    const auto* idx = reinterpret_cast<const std::int32_t*>(indices.ptr.get());
    auto* dst = reinterpret_cast<T*>(out.ptr.get());

    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(input_shape[d]);
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < input_shape.size(); ++d)
        inner *= static_cast<std::size_t>(input_shape[d]);
    const std::size_t L = static_cast<std::size_t>(input_shape[static_cast<std::size_t>(axis)]);
    const std::size_t K = static_cast<std::size_t>(grad_shape[static_cast<std::size_t>(axis)]);

    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t grad_flat = (o * K + k) * inner + j;
                std::int32_t src_k = idx[grad_flat];
                if (src_k < 0)
                    src_k += static_cast<std::int32_t>(L);
                const std::size_t dst_flat = (o * L + static_cast<std::size_t>(src_k)) * inner + j;
                dst[dst_flat] += g[grad_flat];
            }
        }
    }
    return out;
}

Storage scatter_axis_add_storage(const Storage& grad,
                                 const Storage& indices,
                                 const Shape& input_shape,
                                 const Shape& grad_shape,
                                 int axis,
                                 Dtype dt,
                                 Device device) {
    if (device == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad);
        const auto& gi = std::get<GpuStorage>(indices);
        auto base = ::mlx::core::zeros(gpu::to_mlx_shape(input_shape), gpu::to_mlx_dtype(dt));
        auto out = ::mlx::core::scatter_add_axis(base, *gi.arr, *gg.arr, axis);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    const auto& g = std::get<CpuStorage>(grad);
    const auto& idx = std::get<CpuStorage>(indices);
    switch (dt) {
        case Dtype::F32:
            return Storage{scatter_axis_add_cpu<float>(g, idx, input_shape, grad_shape, axis, dt)};
        case Dtype::F64:
            return Storage{scatter_axis_add_cpu<double>(g, idx, input_shape, grad_shape, axis, dt)};
        default:
            throw NotImplementedError("sort/topk backward: dtype not supported");
    }
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

const OpSchema IndexScatterBackward::schema_v1{"index_scatter", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_index_scatter_grad(const TensorImplPtr& a,
                                        TensorImplPtr out,
                                        Storage indices,
                                        int axis) {
    if (!GradMode::is_enabled() || !a->requires_grad_ || !differentiable_dtype(a->dtype_)) {
        return out;
    }

    auto bwd = std::make_shared<IndexScatterBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->grad_shape_ = out->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->indices_ = std::move(indices);
    bwd->axis_ = axis;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
    bwd->set_saved_versions({a->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

::mlx::core::array take_descending_top_indices(const ::mlx::core::array& idx,
                                               int axis,
                                               std::int64_t k) {
    const auto& full_shape = idx.shape();
    const std::int64_t L = full_shape[axis];
    std::vector<std::int32_t> selector(static_cast<std::size_t>(k));
    for (std::int64_t i = 0; i < k; ++i)
        selector[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(L - 1 - i);
    ::mlx::core::Shape selector_shape(full_shape.size(), 1);
    selector_shape[axis] = static_cast<int>(k);
    ::mlx::core::array selector_arr(selector.data(), selector_shape, ::mlx::core::int32);
    ::mlx::core::Shape out_shape = full_shape;
    out_shape[axis] = static_cast<int>(k);
    selector_arr = ::mlx::core::broadcast_to(selector_arr, out_shape);
    return ::mlx::core::take_along_axis(idx, selector_arr, axis);
}

}  // namespace

TensorImplPtr sort_op(const TensorImplPtr& a, int axis) {
    if (!a)
        throw LucidError("sort: null input");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"sort", device, dt, a->shape_};
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto idx = ::mlx::core::argsort(*ga.arr, ax);
        auto out_raw = ::mlx::core::take_along_axis(*ga.arr, idx, ax);
        auto out =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out_raw), dt)}, a->shape_, dt, device);
        return attach_index_scatter_grad(
            a, std::move(out), Storage{gpu::wrap_mlx_array(std::move(idx), Dtype::I32)}, ax);
    }
    Shape out_shape = a->shape_;
    const auto& ca = std::get<CpuStorage>(a->storage_);
    CpuStorage out_cpu;
    CpuStorage idx_cpu;
    if (dt == Dtype::F32)
        std::tie(out_cpu, idx_cpu) =
            sort_select_cpu<float>(ca, a->shape_, out_shape, ax, dt, false);
    else if (dt == Dtype::F64)
        std::tie(out_cpu, idx_cpu) =
            sort_select_cpu<double>(ca, a->shape_, out_shape, ax, dt, false);
    else if (dt == Dtype::I32)
        std::tie(out_cpu, idx_cpu) =
            sort_select_cpu<std::int32_t>(ca, a->shape_, out_shape, ax, dt, false);
    else if (dt == Dtype::I64)
        std::tie(out_cpu, idx_cpu) =
            sort_select_cpu<std::int64_t>(ca, a->shape_, out_shape, ax, dt, false);
    else
        throw NotImplementedError("sort: dtype not supported");
    auto out = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    return attach_index_scatter_grad(a, std::move(out), Storage{std::move(idx_cpu)}, ax);
}

TensorImplPtr argsort_op(const TensorImplPtr& a, int axis) {
    if (!a)
        throw LucidError("argsort: null input");
    const Device device = a->device_;
    OpScope scope{"argsort", device, a->dtype_, a->shape_};
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto out = ::mlx::core::argsort(*ga.arr, axis);
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out), Dtype::I32)}, a->shape_,
                     Dtype::I32, device);
    }
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    Shape out_shape = a->shape_;
    auto out_cpu = allocate_cpu(out_shape, Dtype::I32);
    const auto& ca = std::get<CpuStorage>(a->storage_);
    std::size_t outer = 1, inner = 1;
    for (int d = 0; d < ax; ++d)
        outer *= static_cast<std::size_t>(a->shape_[d]);
    for (std::size_t d = ax + 1; d < a->shape_.size(); ++d)
        inner *= static_cast<std::size_t>(a->shape_[d]);
    const std::size_t L = static_cast<std::size_t>(a->shape_[ax]);
    auto* dst = reinterpret_cast<std::int32_t*>(out_cpu.ptr.get());

    auto run = [&](const auto* src) {
        std::vector<std::int32_t> idx(L);
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < inner; ++j) {
                for (std::size_t k = 0; k < L; ++k)
                    idx[k] = static_cast<std::int32_t>(k);
                std::sort(idx.begin(), idx.end(), [&](std::int32_t x, std::int32_t y) {
                    return src[(o * L + x) * inner + j] < src[(o * L + y) * inner + j];
                });
                for (std::size_t k = 0; k < L; ++k)
                    dst[(o * L + k) * inner + j] = idx[k];
            }
    };
    if (a->dtype_ == Dtype::F32)
        run(reinterpret_cast<const float*>(ca.ptr.get()));
    else if (a->dtype_ == Dtype::F64)
        run(reinterpret_cast<const double*>(ca.ptr.get()));
    else if (a->dtype_ == Dtype::I32)
        run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()));
    else if (a->dtype_ == Dtype::I64)
        run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()));
    else
        throw NotImplementedError("argsort: dtype not supported");
    return fresh(Storage{std::move(out_cpu)}, std::move(out_shape), Dtype::I32, device);
}

namespace {
TensorImplPtr argext_dispatch(
    const TensorImplPtr& a, int axis, bool keepdims, bool is_min, const char* name) {
    if (!a)
        throw LucidError(std::string(name) + ": null input");
    const Device device = a->device_;
    OpScope scope{name, device, a->dtype_, a->shape_};
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto out = is_min ? ::mlx::core::argmin(*ga.arr, axis, keepdims)
                          : ::mlx::core::argmax(*ga.arr, axis, keepdims);
        Shape sh = mlx_shape_to_lucid(out.shape());
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out), Dtype::I32)}, std::move(sh),
                     Dtype::I32, device);
    }
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    Shape out_shape = a->shape_;
    if (keepdims)
        out_shape[ax] = 1;
    else
        out_shape.erase(out_shape.begin() + ax);
    auto out_cpu = allocate_cpu(out_shape, Dtype::I64);
    const auto& ca = std::get<CpuStorage>(a->storage_);
    std::size_t outer = 1;
    for (int d = 0; d < ax; ++d)
        outer *= static_cast<std::size_t>(a->shape_[d]);
    std::size_t inner = 1;
    for (std::size_t d = ax + 1; d < a->shape_.size(); ++d)
        inner *= static_cast<std::size_t>(a->shape_[d]);
    const std::size_t L = static_cast<std::size_t>(a->shape_[ax]);
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
    if (a->dtype_ == Dtype::F32)
        run(reinterpret_cast<const float*>(ca.ptr.get()));
    else if (a->dtype_ == Dtype::F64)
        run(reinterpret_cast<const double*>(ca.ptr.get()));
    else if (a->dtype_ == Dtype::I32)
        run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()));
    else if (a->dtype_ == Dtype::I64)
        run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()));
    else
        throw NotImplementedError(std::string(name) + ": dtype not supported");
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
    if (!a)
        throw LucidError("nonzero: null input");
    const std::size_t ndim = a->shape_.size();
    OpScope scope{"nonzero", a->device_, a->dtype_, a->shape_};

    CpuStorage cpu = (a->device_ == Device::GPU)
                         ? gpu::download_gpu_to_cpu(std::get<GpuStorage>(a->storage_), a->shape_)
                         : std::get<CpuStorage>(a->storage_);

    const std::size_t n = shape_numel(a->shape_);
    std::vector<bool> mask(n, false);
    auto check_nonzero = [&](const auto* p) {
        for (std::size_t i = 0; i < n; ++i)
            mask[i] = static_cast<double>(p[i]) != 0.0;
    };
    switch (a->dtype_) {
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
            throw NotImplementedError("nonzero: dtype not supported");
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
            stride[d] = stride[d + 1] * a->shape_[d + 1];
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
    if (!a)
        throw LucidError("unique: null input");
    const Dtype dt = a->dtype_;
    OpScope scope{"unique", a->device_, dt, a->shape_};

    CpuStorage cpu = (a->device_ == Device::GPU)
                         ? gpu::download_gpu_to_cpu(std::get<GpuStorage>(a->storage_), a->shape_)
                         : std::get<CpuStorage>(a->storage_);
    const std::size_t n = shape_numel(a->shape_);

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
        throw NotImplementedError("unique: dtype not supported");
    return fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, /*device=*/Device::CPU);
}

TensorImplPtr topk_op(const TensorImplPtr& a, std::int64_t k, int axis) {
    if (!a)
        throw LucidError("topk: null input");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"topk", device, dt, a->shape_};
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    if (k <= 0 || k > a->shape_[static_cast<std::size_t>(ax)])
        throw LucidError("topk: k must be in (0, axis_size]");
    Shape out_shape = a->shape_;
    out_shape[static_cast<std::size_t>(ax)] = k;
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto full_idx = ::mlx::core::argsort(*ga.arr, ax);
        auto idx = take_descending_top_indices(full_idx, ax, k);
        auto values = ::mlx::core::take_along_axis(*ga.arr, idx, ax);
        values = ::mlx::core::contiguous(values);
        Shape sh = mlx_shape_to_lucid(values.shape());
        auto out =
            fresh(Storage{gpu::wrap_mlx_array(std::move(values), dt)}, std::move(sh), dt, device);
        return attach_index_scatter_grad(
            a, std::move(out), Storage{gpu::wrap_mlx_array(std::move(idx), Dtype::I32)}, ax);
    }
    const auto& ca = std::get<CpuStorage>(a->storage_);
    CpuStorage out_cpu;
    CpuStorage idx_cpu;
    if (dt == Dtype::F32)
        std::tie(out_cpu, idx_cpu) = sort_select_cpu<float>(ca, a->shape_, out_shape, ax, dt, true);
    else if (dt == Dtype::F64)
        std::tie(out_cpu, idx_cpu) =
            sort_select_cpu<double>(ca, a->shape_, out_shape, ax, dt, true);
    else if (dt == Dtype::I32)
        std::tie(out_cpu, idx_cpu) =
            sort_select_cpu<std::int32_t>(ca, a->shape_, out_shape, ax, dt, true);
    else if (dt == Dtype::I64)
        std::tie(out_cpu, idx_cpu) =
            sort_select_cpu<std::int64_t>(ca, a->shape_, out_shape, ax, dt, true);
    else
        throw NotImplementedError("topk: dtype not supported");
    auto out = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    return attach_index_scatter_grad(a, std::move(out), Storage{std::move(idx_cpu)}, ax);
}

LUCID_REGISTER_OP(IndexScatterBackward)

}  // namespace lucid
