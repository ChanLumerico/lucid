#include "Repeat.h"

#include <algorithm>
#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/FuncOp.h"
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
    auto bwd = std::make_shared<RepeatBackward>();
    bwd->axis_ = axis;
    bwd->repeats_ = repeats;
    kernel::NaryKernel<RepeatBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                         /*save_ins=*/false);
    return out;
}

TensorImplPtr attach_tile_grad(const TensorImplPtr& a,
                               TensorImplPtr out,
                               Shape padded_shape,
                               std::vector<std::int64_t> reps) {
    auto bwd = std::make_shared<TileBackward>();
    bwd->padded_shape_ = std::move(padded_shape);
    bwd->reps_ = std::move(reps);
    kernel::NaryKernel<TileBackward, 1>::wire_autograd(std::move(bwd), {a}, out,
                                                       /*save_ins=*/false);
    return out;
}

LUCID_REGISTER_OP(RepeatBackward)
LUCID_REGISTER_OP(TileBackward)

}  // namespace

TensorImplPtr repeat_op(const TensorImplPtr& a, std::int64_t repeats, int axis) {
    Validator::input(a, "repeat.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"repeat", device, dt, a->shape()};
    int ax = wrap_axis(axis, static_cast<int>(a->shape().size()));
    Shape out_shape = a->shape();
    out_shape[ax] *= repeats;
    Storage out_storage =
        backend::Dispatcher::for_device(device).repeat(a->storage(), a->shape(), dt, repeats, ax);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_repeat_grad(a, std::move(result), ax, repeats);
}

TensorImplPtr tile_op(const TensorImplPtr& a, std::vector<std::int64_t> reps) {
    Validator::input(a, "tile.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"tile", device, dt, a->shape()};
    const std::size_t nout = reps.size();
    if (nout < a->shape().size())
        ErrorBuilder("tile").fail("reps must be at least as long as ndim");

    Shape padded(nout, 1);
    const std::size_t lead = nout - a->shape().size();
    for (std::size_t d = 0; d < a->shape().size(); ++d)
        padded[lead + d] = a->shape()[d];
    Shape out_shape(nout);
    for (std::size_t d = 0; d < nout; ++d)
        out_shape[d] = padded[d] * reps[d];

    Storage out_storage =
        backend::Dispatcher::for_device(device).tile(a->storage(), a->shape(), dt, reps);
    auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
    return attach_tile_grad(a, std::move(result), std::move(padded), std::move(reps));
}

}  // namespace lucid
