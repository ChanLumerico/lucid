#include "Softmax.h"

#include <cmath>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema SoftmaxBackward::schema_v1{"softmax", 1, AmpPolicy::ForceFP32, true};

namespace {

// Decompose `shape` into (outer, axis_dim, inner) where `axis` is the reduce
// axis. Same convention as backend/cpu/Reduce.h.
struct OIR {
    std::size_t outer;
    std::size_t axis_dim;
    std::size_t inner;
};

OIR oir_for_axis(const Shape& shape, int axis) {
    OIR r{1, static_cast<std::size_t>(shape[axis]), 1};
    for (int d = 0; d < axis; ++d)
        r.outer *= static_cast<std::size_t>(shape[d]);
    for (std::size_t d = axis + 1; d < shape.size(); ++d)
        r.inner *= static_cast<std::size_t>(shape[d]);
    return r;
}

template <typename T>
void softmax_forward_typed(
    const T* in, T* out, std::size_t outer, std::size_t axis_dim, std::size_t inner) {
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < inner; ++i) {
            // Pass 1: find max.
            const T* base = in + o * axis_dim * inner + i;
            T m = base[0];
            for (std::size_t r = 1; r < axis_dim; ++r) {
                const T v = base[r * inner];
                if (v > m)
                    m = v;
            }
            // Pass 2: exp(x - m), accumulate sum.
            T* obase = out + o * axis_dim * inner + i;
            T s = T{};
            for (std::size_t r = 0; r < axis_dim; ++r) {
                const T e = std::exp(base[r * inner] - m);
                obase[r * inner] = e;
                s += e;
            }
            // Pass 3: divide.
            const T inv = T{1} / s;
            for (std::size_t r = 0; r < axis_dim; ++r) {
                obase[r * inner] *= inv;
            }
        }
    }
}

template <typename T>
void softmax_backward_typed(
    const T* z, const T* g, T* dx, std::size_t outer, std::size_t axis_dim, std::size_t inner) {
    // dx = z (g - sum(g z, axis))
    // Per (outer, inner) slice: compute sum_r (g[r] * z[r]) along axis once,
    // then dx[r] = z[r] * (g[r] - sum).
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < inner; ++i) {
            const T* zb = z + o * axis_dim * inner + i;
            const T* gb = g + o * axis_dim * inner + i;
            T sum = T{};
            for (std::size_t r = 0; r < axis_dim; ++r) {
                sum += gb[r * inner] * zb[r * inner];
            }
            T* xb = dx + o * axis_dim * inner + i;
            for (std::size_t r = 0; r < axis_dim; ++r) {
                xb[r * inner] = zb[r * inner] * (gb[r * inner] - sum);
            }
        }
    }
}

}  // namespace

TensorImplPtr SoftmaxBackward::forward(const TensorImplPtr& a, int axis) {
    Validator::input(a, "softmax.a").non_null();
    if (a->device_ == Device::CPU && !a->is_contiguous())
        ErrorBuilder("softmax").not_implemented(
            "non-contiguous input not supported (call .contiguous() first)");

    const int ndim = static_cast<int>(a->shape_.size());
    const int wrapped = axis < 0 ? axis + ndim : axis;
    if (wrapped < 0 || wrapped >= ndim)
        ErrorBuilder("softmax").index_error("axis out of range");

    OpScopeFull scope{schema_v1.name, a->device_, a->dtype_, a->shape_};

    Storage out_storage;
    if (a->device_ == Device::GPU) {
        const auto& g = std::get<GpuStorage>(a->storage_);
        if (!g.arr)
            ErrorBuilder("softmax").fail("null GPU input");
        // MLX softmax(arr, axis, precise=true) — `precise=true` mirrors the
        // CPU path's two-pass max-subtract scheme (numerical stability).
        auto out = ::mlx::core::softmax(*g.arr, wrapped, /*precise=*/true);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), a->dtype_)};
    } else {
        const auto oir = oir_for_axis(a->shape_, wrapped);
        const auto& a_cpu = std::get<CpuStorage>(a->storage_);
        CpuStorage out;
        out.dtype = a->dtype_;
        out.nbytes = a_cpu.nbytes;
        out.ptr = allocate_aligned_bytes(out.nbytes);
        if (a->numel() > 0) {
            switch (a->dtype_) {
                case Dtype::F32:
                    softmax_forward_typed<float>(reinterpret_cast<const float*>(a_cpu.ptr.get()),
                                                 reinterpret_cast<float*>(out.ptr.get()), oir.outer,
                                                 oir.axis_dim, oir.inner);
                    break;
                case Dtype::F64:
                    softmax_forward_typed<double>(reinterpret_cast<const double*>(a_cpu.ptr.get()),
                                                  reinterpret_cast<double*>(out.ptr.get()),
                                                  oir.outer, oir.axis_dim, oir.inner);
                    break;
                default:
                    ErrorBuilder("softmax").not_implemented("dtype not supported");
            }
        }
        out_storage = Storage{std::move(out)};
    }

    auto result = std::make_shared<TensorImpl>(std::move(out_storage), a->shape_, a->dtype_,
                                               a->device_, false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()) * 5);

    if (!GradMode::is_enabled() || !a->requires_grad_)
        return result;

    auto a_edge = detail::ensure_grad_fn(a);
    auto bwd = std::make_shared<SoftmaxBackward>();
    bwd->input_shapes_ = {a->shape_};
    bwd->out_shape_ = a->shape_;
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a};
    bwd->saved_output_ = result->storage_;
    bwd->axis_ = wrapped;
    bwd->set_next_edges(std::vector<Edge>{Edge(a_edge, /*input_nr=*/0)});
    bwd->set_saved_versions({a->version_});

    result->grad_fn_ = std::move(bwd);
    result->is_leaf_ = false;
    result->requires_grad_ = true;
    return result;
}

std::vector<Storage> SoftmaxBackward::apply(Storage grad_out) {
    if (device_ == Device::GPU) {
        const auto& z = std::get<GpuStorage>(saved_output_);
        const auto& g = std::get<GpuStorage>(grad_out);
        if (!z.arr || !g.arr) {
            ErrorBuilder("softmax backward").fail("null GPU array");
        }
        // dx = z · (g − sum(g·z, axis, keepdims))
        auto gz = ::mlx::core::multiply(*g.arr, *z.arr);
        auto sum_gz = ::mlx::core::sum(gz, std::vector<int>{axis_},
                                       /*keepdims=*/true);
        auto diff = ::mlx::core::subtract(*g.arr, sum_gz);
        auto dx = ::mlx::core::multiply(*z.arr, diff);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)}};
    }
    const auto& z_cpu = std::get<CpuStorage>(saved_output_);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);

    const auto oir = oir_for_axis(input_shapes_[0], axis_);
    CpuStorage dx;
    dx.dtype = dtype_;
    dx.nbytes = z_cpu.nbytes;
    dx.ptr = allocate_aligned_bytes(dx.nbytes);

    if (shape_numel(input_shapes_[0]) > 0) {
        switch (dtype_) {
            case Dtype::F32:
                softmax_backward_typed<float>(reinterpret_cast<const float*>(z_cpu.ptr.get()),
                                              reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                              reinterpret_cast<float*>(dx.ptr.get()), oir.outer,
                                              oir.axis_dim, oir.inner);
                break;
            case Dtype::F64:
                softmax_backward_typed<double>(reinterpret_cast<const double*>(z_cpu.ptr.get()),
                                               reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                               reinterpret_cast<double*>(dx.ptr.get()), oir.outer,
                                               oir.axis_dim, oir.inner);
                break;
            default:
                ErrorBuilder("softmax backward").not_implemented("dtype not supported");
        }
    }
    return {Storage{std::move(dx)}};
}

TensorImplPtr softmax_op(const TensorImplPtr& a, int axis) {
    return SoftmaxBackward::forward(a, axis);
}
LUCID_REGISTER_OP(SoftmaxBackward)

}  // namespace lucid
