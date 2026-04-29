#include "Meshgrid.h"

#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::check_dtype_device_match;
using utils_detail::fresh;
using utils_detail::mlx_shape_to_lucid;
using utils_detail::numel;

template <typename T>
CpuStorage meshgrid_backward_cpu(const CpuStorage& grad,
                                 const Shape& output_shape,
                                 int carry_axis,
                                 Dtype dt) {
    Shape input_shape{output_shape[static_cast<std::size_t>(carry_axis)]};
    auto out = allocate_cpu(input_shape, dt);
    const auto* g = reinterpret_cast<const T*>(grad.ptr.get());
    auto* dst = reinterpret_cast<T*>(out.ptr.get());
    const std::size_t total = numel(output_shape);
    std::vector<std::int64_t> coord(output_shape.size(), 0);
    for (std::size_t flat = 0; flat < total; ++flat) {
        dst[static_cast<std::size_t>(coord[carry_axis])] += g[flat];
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(output_shape.size()) - 1; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < output_shape[static_cast<std::size_t>(d)]) {
                break;
            }
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
    return out;
}

Storage meshgrid_backward_storage(const Storage& grad,
                                  const Shape& input_shape,
                                  const Shape& output_shape,
                                  int carry_axis,
                                  Dtype dt,
                                  Device device) {
    if (device == Device::GPU) {
        std::vector<int> axes;
        axes.reserve(output_shape.size() > 0 ? output_shape.size() - 1 : 0);
        for (int d = 0; d < static_cast<int>(output_shape.size()); ++d) {
            if (d != carry_axis)
                axes.push_back(d);
        }
        if (axes.empty())
            return clone_storage(grad, shape_numel(input_shape), dt, device);
        const auto& gg = std::get<GpuStorage>(grad);
        auto out = ::mlx::core::sum(*gg.arr, axes, false);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    const auto& g = std::get<CpuStorage>(grad);
    switch (dt) {
        case Dtype::F32:
            return Storage{meshgrid_backward_cpu<float>(g, output_shape, carry_axis, dt)};
        case Dtype::F64:
            return Storage{meshgrid_backward_cpu<double>(g, output_shape, carry_axis, dt)};
        default:
            ErrorBuilder("meshgrid backward").not_implemented("dtype not supported");
    }
}

class MeshgridBackward : public FuncOp<MeshgridBackward, 1> {
public:
    static const OpSchema schema_v1;

    int carry_axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        return {meshgrid_backward_storage(grad_out, input_shapes_[0], out_shape_, carry_axis_,
                                          dtype_, device_)};
    }
};

const OpSchema MeshgridBackward::schema_v1{"meshgrid", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_meshgrid_grad(const TensorImplPtr& input,
                                   TensorImplPtr output,
                                   int carry_axis) {
    auto bwd = std::make_shared<MeshgridBackward>();
    bwd->carry_axis_ = carry_axis;
    kernel::NaryKernel<MeshgridBackward, 1>::wire_autograd(std::move(bwd), {input}, output,
                                                           /*save_ins=*/false);
    return output;
}

}  // namespace

std::vector<TensorImplPtr> meshgrid_op(const std::vector<TensorImplPtr>& xs, bool indexing_xy) {
    check_dtype_device_match(xs, "meshgrid");
    const Dtype dt = xs[0]->dtype();
    const Device device = xs[0]->device();
    OpScopeFull scope{"meshgrid", device, dt, Shape{}};
    if (device == Device::GPU) {
        std::vector<::mlx::core::array> arrays;
        arrays.reserve(xs.size());
        for (auto& t : xs)
            arrays.push_back(*std::get<GpuStorage>(t->storage()).arr);
        auto out = ::mlx::core::meshgrid(arrays, false, indexing_xy ? "xy" : "ij");
        std::vector<TensorImplPtr> result;
        result.reserve(out.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            // MLX meshgrid returns broadcast lazy views — materialize.
            auto ac = ::mlx::core::contiguous(out[i]);
            Shape sh = mlx_shape_to_lucid(ac.shape());
            auto t =
                fresh(Storage{gpu::wrap_mlx_array(std::move(ac), dt)}, std::move(sh), dt, device);
            std::size_t carry_axis = i;
            if (indexing_xy && out.size() >= 2 && i < 2)
                carry_axis = 1 - i;
            result.push_back(
                attach_meshgrid_grad(xs[i], std::move(t), static_cast<int>(carry_axis)));
        }
        return result;
    }
    const std::size_t N = xs.size();
    std::vector<std::int64_t> dims(N);
    for (std::size_t i = 0; i < N; ++i) {
        if (xs[i]->shape().size() != 1)
            ErrorBuilder("meshgrid").fail("each input must be 1-D");
        dims[i] = xs[i]->shape()[0];
    }
    std::vector<std::int64_t> out_dims = dims;
    if (indexing_xy && N >= 2)
        std::swap(out_dims[0], out_dims[1]);
    Shape out_shape(out_dims.begin(), out_dims.end());
    const std::size_t total = numel(out_shape);
    const std::size_t elem = dtype_size(dt);

    Stride out_stride(N);
    if (N > 0) {
        out_stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)N - 2; d >= 0; --d)
            out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
    }

    std::vector<TensorImplPtr> result;
    result.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        auto out_cpu = allocate_cpu(out_shape, dt);
        const auto& cv = std::get<CpuStorage>(xs[i]->storage());
        std::size_t carry_axis = i;
        if (indexing_xy && N >= 2 && i < 2)
            carry_axis = 1 - i;

        std::vector<std::int64_t> coord(N, 0);
        for (std::size_t f = 0; f < total; ++f) {
            const std::int64_t k = coord[carry_axis];
            std::memcpy(out_cpu.ptr.get() + f * elem,
                        cv.ptr.get() + static_cast<std::size_t>(k) * elem, elem);
            for (std::ptrdiff_t d = (std::ptrdiff_t)N - 1; d >= 0; --d) {
                if (++coord[d] < out_shape[d])
                    break;
                coord[d] = 0;
            }
        }
        auto out = fresh(Storage{std::move(out_cpu)}, out_shape, dt, device);
        result.push_back(attach_meshgrid_grad(xs[i], std::move(out), static_cast<int>(carry_axis)));
    }
    return result;
}

LUCID_REGISTER_OP(MeshgridBackward)

}  // namespace lucid
