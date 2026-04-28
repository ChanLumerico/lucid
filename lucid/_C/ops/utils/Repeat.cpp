#include "Repeat.h"

#include <algorithm>
#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
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
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::numel;
using utils_detail::wrap_axis;

template <typename T>
void repeat_backward_typed(const std::byte* grad,
                           std::byte* dst,
                           const Shape& input_shape,
                           const Shape& output_shape,
                           int axis,
                           std::int64_t repeats) {
    auto* out = reinterpret_cast<T*>(dst);
    const auto* g = reinterpret_cast<const T*>(grad);
    std::fill_n(out, shape_numel(input_shape), T{});

    const std::size_t ndim = output_shape.size();
    Stride out_stride(ndim), in_stride(ndim);
    if (ndim > 0) {
        out_stride.back() = 1;
        in_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            out_stride[static_cast<std::size_t>(d)] = out_stride[static_cast<std::size_t>(d) + 1] *
                                                      output_shape[static_cast<std::size_t>(d) + 1];
            in_stride[static_cast<std::size_t>(d)] = in_stride[static_cast<std::size_t>(d) + 1] *
                                                     input_shape[static_cast<std::size_t>(d) + 1];
        }
    }

    const std::size_t total = shape_numel(output_shape);
    for (std::size_t flat = 0; flat < total; ++flat) {
        std::size_t rem = flat;
        std::size_t in_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
            rem -= coord * static_cast<std::size_t>(out_stride[d]);
            if (static_cast<int>(d) == axis) {
                coord /= static_cast<std::size_t>(repeats);
            }
            in_flat += coord * static_cast<std::size_t>(in_stride[d]);
        }
        out[in_flat] += g[flat];
    }
}

template <typename T>
void tile_backward_typed(const std::byte* grad,
                         std::byte* dst,
                         const Shape& input_shape,
                         const Shape& padded_shape,
                         const Shape& output_shape) {
    auto* out = reinterpret_cast<T*>(dst);
    const auto* g = reinterpret_cast<const T*>(grad);
    std::fill_n(out, shape_numel(input_shape), T{});

    const std::size_t ndim = output_shape.size();
    Stride out_stride(ndim), padded_stride(ndim);
    if (ndim > 0) {
        out_stride.back() = 1;
        padded_stride.back() = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
            out_stride[static_cast<std::size_t>(d)] = out_stride[static_cast<std::size_t>(d) + 1] *
                                                      output_shape[static_cast<std::size_t>(d) + 1];
            padded_stride[static_cast<std::size_t>(d)] =
                padded_stride[static_cast<std::size_t>(d) + 1] *
                padded_shape[static_cast<std::size_t>(d) + 1];
        }
    }

    const std::size_t total = shape_numel(output_shape);
    for (std::size_t flat = 0; flat < total; ++flat) {
        std::size_t rem = flat;
        std::size_t in_flat = 0;
        for (std::size_t d = 0; d < ndim; ++d) {
            const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
            rem -= coord * static_cast<std::size_t>(out_stride[d]);
            const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
            in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
        }
        out[in_flat] += g[flat];
    }
}

class RepeatBackward : public FuncOp<RepeatBackward, 1> {
public:
    static const OpSchema schema_v1;

    int axis_ = 0;
    std::int64_t repeats_ = 1;

    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::GPU) {
            const auto& g = std::get<GpuStorage>(grad_out);
            ::mlx::core::Shape reshape_shape;
            reshape_shape.reserve(out_shape_.size() + 1);
            for (std::size_t d = 0; d < out_shape_.size(); ++d) {
                if (static_cast<int>(d) == axis_) {
                    reshape_shape.push_back(
                        static_cast<::mlx::core::ShapeElem>(input_shapes_[0][d]));
                    reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(repeats_));
                } else {
                    reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(out_shape_[d]));
                }
            }
            auto reshaped = ::mlx::core::reshape(*g.arr, reshape_shape);
            auto summed = ::mlx::core::sum(reshaped, std::vector<int>{axis_ + 1},
                                           /*keepdims=*/false);
            summed = ::mlx::core::reshape(summed, gpu::to_mlx_shape(input_shapes_[0]));
            return {Storage{gpu::wrap_mlx_array(std::move(summed), dtype_)}};
        }

        auto out = allocate_cpu(input_shapes_[0], dtype_);
        const auto& g = std::get<CpuStorage>(grad_out);
        switch (dtype_) {
            case Dtype::F32:
                repeat_backward_typed<float>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                             out_shape_, axis_, repeats_);
                break;
            case Dtype::F64:
                repeat_backward_typed<double>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                              out_shape_, axis_, repeats_);
                break;
            case Dtype::I32:
                repeat_backward_typed<std::int32_t>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                                    out_shape_, axis_, repeats_);
                break;
            case Dtype::I64:
                repeat_backward_typed<std::int64_t>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                                    out_shape_, axis_, repeats_);
                break;
            default:
                ErrorBuilder("repeat backward").not_implemented("dtype not supported");
        }
        return {Storage{std::move(out)}};
    }
};

const OpSchema RepeatBackward::schema_v1{"repeat", 1, AmpPolicy::KeepInput, true};

class TileBackward : public FuncOp<TileBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape padded_shape_;
    std::vector<std::int64_t> reps_;

    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::GPU) {
            const auto& g = std::get<GpuStorage>(grad_out);
            ::mlx::core::Shape reshape_shape;
            reshape_shape.reserve(reps_.size() * 2);
            std::vector<int> sum_axes;
            sum_axes.reserve(reps_.size());
            for (std::size_t d = 0; d < reps_.size(); ++d) {
                sum_axes.push_back(static_cast<int>(reshape_shape.size()));
                reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(reps_[d]));
                reshape_shape.push_back(static_cast<::mlx::core::ShapeElem>(padded_shape_[d]));
            }
            auto reshaped = ::mlx::core::reshape(*g.arr, reshape_shape);
            auto summed = sum_axes.empty() ? reshaped
                                           : ::mlx::core::sum(reshaped, sum_axes,
                                                              /*keepdims=*/false);
            summed = ::mlx::core::reshape(summed, gpu::to_mlx_shape(input_shapes_[0]));
            return {Storage{gpu::wrap_mlx_array(std::move(summed), dtype_)}};
        }

        auto out = allocate_cpu(input_shapes_[0], dtype_);
        const auto& g = std::get<CpuStorage>(grad_out);
        switch (dtype_) {
            case Dtype::F32:
                tile_backward_typed<float>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                           padded_shape_, out_shape_);
                break;
            case Dtype::F64:
                tile_backward_typed<double>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                            padded_shape_, out_shape_);
                break;
            case Dtype::I32:
                tile_backward_typed<std::int32_t>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                                  padded_shape_, out_shape_);
                break;
            case Dtype::I64:
                tile_backward_typed<std::int64_t>(g.ptr.get(), out.ptr.get(), input_shapes_[0],
                                                  padded_shape_, out_shape_);
                break;
            default:
                ErrorBuilder("tile backward").not_implemented("dtype not supported");
        }
        return {Storage{std::move(out)}};
    }
};

const OpSchema TileBackward::schema_v1{"tile", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_repeat_grad(const TensorImplPtr& a,
                                 TensorImplPtr out,
                                 int axis,
                                 std::int64_t repeats) {
    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;

    auto bwd = std::make_shared<RepeatBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->axis_ = axis;
    bwd->repeats_ = repeats;
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
    bwd->set_saved_versions({a->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

TensorImplPtr attach_tile_grad(const TensorImplPtr& a,
                               TensorImplPtr out,
                               Shape padded_shape,
                               std::vector<std::int64_t> reps) {
    if (!GradMode::is_enabled() || !a->requires_grad_)
        return out;

    auto bwd = std::make_shared<TileBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->padded_shape_ = std::move(padded_shape);
    bwd->reps_ = std::move(reps);
    bwd->set_next_edges(std::vector<Edge>{Edge(detail::ensure_grad_fn(a), 0)});
    bwd->set_saved_versions({a->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

LUCID_REGISTER_OP(RepeatBackward)
LUCID_REGISTER_OP(TileBackward)

}  // namespace

TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis) {
    Validator::input(a, "repeat.a").non_null();
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScopeFull scope{"repeat", device, dt, a->shape_};
    int ax = wrap_axis(axis, static_cast<int>(a->shape_.size()));
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto out = ::mlx::core::repeat(*ga.arr, static_cast<int>(repeats), ax);
        Shape sh = mlx_shape_to_lucid(out.shape());
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, std::move(sh), dt, device);
        return attach_repeat_grad(a, std::move(result), ax, repeats);
    }
    Shape out_shape = a->shape_;
    out_shape[ax] *= repeats;
    auto out_cpu = allocate_cpu(out_shape, dt);
    const auto& ca = std::get<CpuStorage>(a->storage_);
    const std::size_t elem = dtype_size(dt);
    std::size_t outer = 1;
    for (int d = 0; d < ax; ++d)
        outer *= static_cast<std::size_t>(a->shape_[d]);
    std::size_t inner = elem;
    for (std::size_t d = ax + 1; d < a->shape_.size(); ++d)
        inner *= static_cast<std::size_t>(a->shape_[d]);
    const std::size_t L = static_cast<std::size_t>(a->shape_[ax]);
    auto* dst = out_cpu.ptr.get();
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t k = 0; k < L; ++k) {
            const auto* src = ca.ptr.get() + (o * L + k) * inner;
            for (std::int64_t r = 0; r < repeats; ++r) {
                std::memcpy(dst, src, inner);
                dst += inner;
            }
        }
    }
    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    return attach_repeat_grad(a, std::move(result), ax, repeats);
}

TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps) {
    Validator::input(a, "tile.a").non_null();
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScopeFull scope{"tile", device, dt, a->shape_};
    const std::size_t nout = reps.size();
    if (nout < a->shape_.size())
        ErrorBuilder("tile").fail("reps must be at least as long as ndim");

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        std::vector<int> reps_int(reps.begin(), reps.end());
        auto out = ::mlx::core::tile(*ga.arr, std::move(reps_int));
        Shape sh = mlx_shape_to_lucid(out.shape());
        Shape padded(nout, 1);
        const std::size_t lead = nout - a->shape_.size();
        for (std::size_t d = 0; d < a->shape_.size(); ++d)
            padded[lead + d] = a->shape_[d];
        auto result =
            fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, std::move(sh), dt, device);
        return attach_tile_grad(a, std::move(result), std::move(padded), std::move(reps));
    }
    Shape padded(nout, 1);
    const std::size_t lead = nout - a->shape_.size();
    for (std::size_t d = 0; d < a->shape_.size(); ++d)
        padded[lead + d] = a->shape_[d];
    Shape out_shape(nout);
    for (std::size_t d = 0; d < nout; ++d)
        out_shape[d] = padded[d] * reps[d];

    auto out_cpu = allocate_cpu(out_shape, dt);
    const auto& ca = std::get<CpuStorage>(a->storage_);
    const std::size_t elem = dtype_size(dt);
    Stride in_stride(nout);
    if (nout > 0) {
        in_stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 2; d >= 0; --d)
            in_stride[d] = in_stride[d + 1] * padded[d + 1];
    }
    const std::size_t total = numel(out_shape);
    std::vector<std::int64_t> coord(nout, 0);
    for (std::size_t f = 0; f < total; ++f) {
        std::size_t in_flat = 0;
        for (std::size_t d = 0; d < nout; ++d)
            in_flat += static_cast<std::size_t>(coord[d] % padded[d]) *
                       static_cast<std::size_t>(in_stride[d]);
        std::memcpy(out_cpu.ptr.get() + f * elem, ca.ptr.get() + in_flat * elem, elem);
        for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 1; d >= 0; --d) {
            if (++coord[d] < out_shape[d])
                break;
            coord[d] = 0;
        }
    }
    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    return attach_tile_grad(a, std::move(result), std::move(padded), std::move(reps));
}

}  // namespace lucid
