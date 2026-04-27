#include "Layout.h"

#include <algorithm>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "View.h"  // reshape_op / ViewBackward
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;
using utils_detail::numel;

}  // namespace

TensorImplPtr flatten_op(const TensorImplPtr& a, int start_axis, int end_axis) {
    if (!a) throw LucidError("flatten: null input");
    const int ndim = static_cast<int>(a->shape_.size());
    int s = start_axis < 0 ? start_axis + ndim : start_axis;
    int e = end_axis < 0 ? end_axis + ndim : end_axis;
    if (s < 0 || e >= ndim || s > e)
        throw LucidError("flatten: invalid axis range");

    // Delegate to reshape_op so we inherit the ViewBackward autograd wiring.
    std::vector<std::int64_t> new_shape;
    for (int d = 0; d < s; ++d) new_shape.push_back(a->shape_[d]);
    std::int64_t flat = 1;
    for (int d = s; d <= e; ++d) flat *= a->shape_[d];
    new_shape.push_back(flat);
    for (int d = e + 1; d < ndim; ++d) new_shape.push_back(a->shape_[d]);
    return reshape_op(a, new_shape);
}

const OpSchema BroadcastBackward::schema_v1{
    "broadcast_to", 1, AmpPolicy::KeepInput, true};

namespace {

// Sum `grad` (shape = output_shape) down to `input_shape` by reducing along
// the broadcast axes. Handles right-aligned shape padding (PyTorch / NumPy
// semantics).
Storage reduce_broadcast(const Storage& grad, const Shape& input_shape,
                          const Shape& output_shape, Dtype dt, Device device) {
    const std::size_t nout = output_shape.size();
    const std::size_t nin  = input_shape.size();
    Shape padded(nout, 1);
    std::copy(input_shape.begin(), input_shape.end(),
              padded.begin() + (nout - nin));

    if (device == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad);
        ::mlx::core::array out = *gg.arr;
        // Sum along axes where padded[d] == 1 but output_shape[d] != 1.
        std::vector<int> sum_axes;
        for (std::size_t d = 0; d < nout; ++d) {
            if (padded[d] == 1 && output_shape[d] != 1) {
                sum_axes.push_back(static_cast<int>(d));
            }
        }
        if (!sum_axes.empty())
            out = ::mlx::core::sum(out, sum_axes, /*keepdims=*/true);
        // Drop the leading singleton axes added by right-alignment.
        if (nout != nin) {
            out = ::mlx::core::reshape(out, gpu::to_mlx_shape(input_shape));
        }
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }

    // CPU: reduce by accumulating into an `input_shape`-sized buffer.
    const auto& gc = std::get<CpuStorage>(grad);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = shape_numel(input_shape) * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);
    if (out.nbytes > 0) std::memset(out.ptr.get(), 0, out.nbytes);

    // Strides for `output_shape` (row-major) and for `padded` (used to map
    // each output element back to an input element).
    std::vector<std::size_t> in_str(nout, 0);
    std::size_t s = 1;
    for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 1; d >= 0; --d) {
        in_str[d] = (padded[d] == 1) ? 0 : s;
        s *= static_cast<std::size_t>(padded[d]);
    }
    const std::size_t out_numel = shape_numel(output_shape);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* gp = reinterpret_cast<const T*>(gc.ptr.get());
        T* dp = reinterpret_cast<T*>(out.ptr.get());
        std::vector<std::size_t> coord(nout, 0);
        for (std::size_t f = 0; f < out_numel; ++f) {
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < nout; ++d) in_flat += coord[d] * in_str[d];
            dp[in_flat] += gp[f];
            for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 1; d >= 0; --d) {
                if (++coord[d] < static_cast<std::size_t>(output_shape[d])) break;
                coord[d] = 0;
            }
        }
    };
    switch (dt) {
        case Dtype::F32: run(float{}); break;
        case Dtype::F64: run(double{}); break;
        case Dtype::I32: run(std::int32_t{}); break;
        case Dtype::I64: run(std::int64_t{}); break;
        default:
            throw NotImplementedError("broadcast backward: dtype not supported");
    }
    return Storage{std::move(out)};
}

}  // namespace

std::vector<Storage> BroadcastBackward::apply(Storage grad_out) {
    return {reduce_broadcast(grad_out, input_shape_, output_shape_,
                              dtype_, device_)};
}

LUCID_REGISTER_OP(BroadcastBackward)

TensorImplPtr broadcast_to_op(const TensorImplPtr& a, const Shape& shape) {
    if (!a) throw LucidError("broadcast_to: null input");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"broadcast_to", device, dt, shape};

    auto build_with_grad = [&](Storage&& out_storage) {
        auto out = std::make_shared<TensorImpl>(std::move(out_storage),
                                                 shape, dt, device,
                                                 /*requires_grad=*/false);
        if (GradMode::is_enabled() && a->requires_grad_) {
            auto a_edge = detail::ensure_grad_fn(a);
            auto bwd = std::make_shared<BroadcastBackward>();
            bwd->input_shapes_  = {a->shape_};
            bwd->out_shape_     = shape;
            bwd->dtype_         = dt;
            bwd->device_        = device;
            bwd->input_tensors_ = {a};
            bwd->input_shape_   = a->shape_;
            bwd->output_shape_  = shape;
            bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, 0)});
            bwd->set_saved_versions({a->version_});
            out->grad_fn_ = std::move(bwd);
            out->is_leaf_ = false;
            out->requires_grad_ = true;
        }
        return out;
    };

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto out = ::mlx::core::broadcast_to(*ga.arr, gpu::to_mlx_shape(shape));
        // broadcast_to returns a lazy view — materialize so downstream
        // download/copy sees the actual broadcast values.
        out = ::mlx::core::contiguous(out);
        return build_with_grad(Storage{gpu::wrap_mlx_array(std::move(out), dt)});
    }
    const std::size_t nin = a->shape_.size();
    const std::size_t nout = shape.size();
    if (nin > nout) throw ShapeMismatch(shape, a->shape_, "broadcast_to");
    Shape padded(nout, 1);
    std::copy(a->shape_.begin(), a->shape_.end(),
              padded.begin() + (nout - nin));
    for (std::size_t d = 0; d < nout; ++d) {
        if (padded[d] != shape[d] && padded[d] != 1)
            throw ShapeMismatch(shape, a->shape_, "broadcast_to");
    }
    auto out_cpu = allocate_cpu(shape, dt);
    const auto& ca = std::get<CpuStorage>(a->storage_);

    std::vector<std::size_t> in_str(nout, 0);
    std::size_t s = 1;
    for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 1; d >= 0; --d) {
        in_str[d] = (padded[d] == 1) ? 0 : s;
        s *= static_cast<std::size_t>(padded[d]);
    }
    const std::size_t out_numel = numel(shape);
    auto run = [&](auto* dst, const auto* src) {
        std::vector<std::size_t> coord(nout, 0);
        for (std::size_t f = 0; f < out_numel; ++f) {
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < nout; ++d)
                in_flat += coord[d] * in_str[d];
            dst[f] = src[in_flat];
            for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 1; d >= 0; --d) {
                if (++coord[d] < static_cast<std::size_t>(shape[d])) break;
                coord[d] = 0;
            }
        }
    };
    switch (dt) {
        case Dtype::F32:
            run(reinterpret_cast<float*>(out_cpu.ptr.get()),
                reinterpret_cast<const float*>(ca.ptr.get())); break;
        case Dtype::F64:
            run(reinterpret_cast<double*>(out_cpu.ptr.get()),
                reinterpret_cast<const double*>(ca.ptr.get())); break;
        case Dtype::I32:
            run(reinterpret_cast<std::int32_t*>(out_cpu.ptr.get()),
                reinterpret_cast<const std::int32_t*>(ca.ptr.get())); break;
        case Dtype::I64:
            run(reinterpret_cast<std::int64_t*>(out_cpu.ptr.get()),
                reinterpret_cast<const std::int64_t*>(ca.ptr.get())); break;
        case Dtype::Bool:
        case Dtype::I8:
            run(reinterpret_cast<std::uint8_t*>(out_cpu.ptr.get()),
                reinterpret_cast<const std::uint8_t*>(ca.ptr.get())); break;
        default:
            throw NotImplementedError("broadcast_to: dtype not supported");
    }
    return build_with_grad(Storage{std::move(out_cpu)});
}

TensorImplPtr expand_op(const TensorImplPtr& a, const Shape& shape) {
    return broadcast_to_op(a, shape);
}

}  // namespace lucid
