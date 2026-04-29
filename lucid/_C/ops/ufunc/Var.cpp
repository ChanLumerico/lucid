#include "Var.h"

#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::allocate_cpu;
using ufunc_detail::fresh;

// VarBackward: dx = (2/N) * (x - mean) * broadcast(grad)
class VarBackward : public AutogradNode<VarBackward, 1> {
public:
    static const OpSchema schema_v1;

    Shape input_shape_;
    std::vector<int> axes_;
    bool keepdims_;
    std::int64_t count_;
    Storage saved_input_;
    Storage saved_mean_;  // broadcast to input_shape_

    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t n = shape_numel(input_shape_);
        Storage centered = subtract_storages(saved_input_, saved_mean_, n, dtype_, device_);
        Storage g_b = broadcast_back_for_reduce(grad_out, out_shape_, input_shape_, axes_,
                                                keepdims_, dtype_, device_);
        Storage scaled = mul_scalar_storage(g_b, 2.0 / (double)count_, n, dtype_, device_);
        Storage dx = multiply_storages(centered, scaled, n, dtype_, device_);
        return {std::move(dx)};
    }
};

const OpSchema VarBackward::schema_v1{"var", 1, AmpPolicy::KeepInput, true};

}  // namespace

TensorImplPtr var_op(const TensorImplPtr& a, const std::vector<int>& axes_user, bool keepdims) {
    Validator::input(a, "var.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    const auto axes = normalize_axes(axes_user, static_cast<int>(a->shape().size()));
    const Shape out_shape = reduce_output_shape(a->shape(), axes, keepdims);
    OpScopeFull scope{"var", device, dt, out_shape};

    TensorImplPtr out;  // declared early so the grad block below can attach.

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage());
        std::vector<int> ax(axes.begin(), axes.end());
        auto out_mlx = ::mlx::core::var(*ga.arr, ax, keepdims, /*ddof=*/0);
        out = fresh(Storage{gpu::wrap_mlx_array(std::move(out_mlx), dt)}, out_shape, dt, device);
        // Fall through to grad-wiring at the bottom of this function.
        // We still need `reduced` for the backward; compute it now.
        std::size_t reduced_gpu = 1;
        for (auto ax_i : axes)
            reduced_gpu *= static_cast<std::size_t>(a->shape()[ax_i]);
        if (reduced_gpu == 0)
            reduced_gpu = 1;
        if (GradMode::is_enabled() && a->requires_grad()) {
            // Build saved_mean storage broadcast to a's shape on GPU.
            auto m = ::mlx::core::mean(*ga.arr, ax, /*keepdims=*/true);
            auto m_b = ::mlx::core::broadcast_to(m, gpu::to_mlx_shape(a->shape()));
            m_b = ::mlx::core::contiguous(m_b);
            Storage mean_storage{gpu::wrap_mlx_array(std::move(m_b), dt)};
            auto bwd = std::make_shared<VarBackward>();
            bwd->input_shape_ = a->shape();
            bwd->out_shape_ = out_shape;
            bwd->dtype_ = dt;
            bwd->device_ = device;
            bwd->axes_ = axes;
            bwd->keepdims_ = keepdims;
            bwd->count_ = static_cast<std::int64_t>(reduced_gpu);
            bwd->saved_input_ = a->storage();
            bwd->saved_mean_ = std::move(mean_storage);
            auto a_edge = detail::ensure_grad_fn(a);
            std::vector<Edge> edges;
            edges.emplace_back(a_edge, 0);
            bwd->set_next_edges(std::move(edges));
            bwd->set_saved_versions({a->version()});
            out->set_grad_fn(std::move(bwd));
            out->set_leaf(false);
            out->set_requires_grad(true);
        }
        return out;
    }

    const auto& ca = std::get<CpuStorage>(a->storage());
    const std::size_t in_numel = shape_numel(a->shape());
    const std::size_t out_numel = shape_numel(out_shape);
    std::size_t reduced = 1;
    for (auto ax : axes)
        reduced *= static_cast<std::size_t>(a->shape()[ax]);
    if (reduced == 0)
        reduced = 1;

    auto compute = [&](auto* out_p, const auto* in_p) {
        using T = std::remove_pointer_t<decltype(out_p)>;
        std::vector<double> means(out_numel, 0.0);
        std::vector<double> coords(a->shape().size(), 0);
        const auto& sh = a->shape();

        Shape kd_shape = sh;
        for (auto ax : axes)
            kd_shape[ax] = 1;

        Stride kd_stride(kd_shape.size());
        if (!kd_shape.empty()) {
            kd_stride.back() = 1;
            for (std::ptrdiff_t i = (std::ptrdiff_t)kd_shape.size() - 2; i >= 0; --i)
                kd_stride[i] = kd_stride[i + 1] * kd_shape[i + 1];
        }

        Stride in_stride(sh.size());
        if (!sh.empty()) {
            in_stride.back() = 1;
            for (std::ptrdiff_t i = (std::ptrdiff_t)sh.size() - 2; i >= 0; --i)
                in_stride[i] = in_stride[i + 1] * sh[i + 1];
        }

        auto flat_to_coord = [&](std::size_t flat) {
            for (std::size_t d = 0; d < sh.size(); ++d) {
                coords[d] = flat / static_cast<std::size_t>(in_stride[d]);
                flat %= static_cast<std::size_t>(in_stride[d]);
            }
        };

        auto kd_flat = [&]() {
            std::size_t f = 0;
            for (std::size_t d = 0; d < sh.size(); ++d) {
                std::int64_t c = coords[d];
                if (kd_shape[d] == 1)
                    c = 0;
                f += c * static_cast<std::size_t>(kd_stride[d]);
            }
            return f;
        };

        for (std::size_t i = 0; i < in_numel; ++i) {
            flat_to_coord(i);
            means[kd_flat()] += static_cast<double>(in_p[i]);
        }
        for (auto& m : means)
            m /= static_cast<double>(reduced);

        std::vector<double> vars(out_numel, 0.0);
        for (std::size_t i = 0; i < in_numel; ++i) {
            flat_to_coord(i);
            const auto kf = kd_flat();
            const double d = static_cast<double>(in_p[i]) - means[kf];
            vars[kf] += d * d;
        }
        for (std::size_t k = 0; k < out_numel; ++k) {
            out_p[k] = static_cast<T>(vars[k] / static_cast<double>(reduced));
        }
    };

    auto out_cpu = allocate_cpu(out_shape, dt);
    if (dt == Dtype::F32)
        compute(reinterpret_cast<float*>(out_cpu.ptr.get()),
                reinterpret_cast<const float*>(ca.ptr.get()));
    else if (dt == Dtype::F64)
        compute(reinterpret_cast<double*>(out_cpu.ptr.get()),
                reinterpret_cast<const double*>(ca.ptr.get()));
    else
        ErrorBuilder("var").not_implemented("dtype not supported");

    out = fresh(Storage{std::move(out_cpu)}, out_shape, dt, device);

    if (GradMode::is_enabled() && a->requires_grad()) {
        Storage mean_storage;
        if (device == Device::GPU) {
            const auto& ga = std::get<GpuStorage>(a->storage());
            std::vector<int> ax(axes.begin(), axes.end());
            auto m = ::mlx::core::mean(*ga.arr, ax, /*keepdims=*/true);
            auto m_b = ::mlx::core::broadcast_to(m, gpu::to_mlx_shape(a->shape()));
            mean_storage = Storage{gpu::wrap_mlx_array(std::move(m_b), dt)};
        } else {
            CpuStorage m;
            m.dtype = dt;
            const std::size_t n_in = shape_numel(a->shape());
            m.nbytes = n_in * dtype_size(dt);
            m.ptr = allocate_aligned_bytes(m.nbytes);
            std::vector<double> means(out_numel, 0.0);
            std::vector<std::int64_t> coords(a->shape().size(), 0);
            const auto& sh = a->shape();
            Shape kd_shape = sh;
            for (auto ax : axes)
                kd_shape[ax] = 1;
            Stride kd_stride(kd_shape.size());
            kd_stride.back() = 1;
            for (std::ptrdiff_t i = (std::ptrdiff_t)kd_shape.size() - 2; i >= 0; --i)
                kd_stride[i] = kd_stride[i + 1] * kd_shape[i + 1];
            Stride in_stride(sh.size());
            in_stride.back() = 1;
            for (std::ptrdiff_t i = (std::ptrdiff_t)sh.size() - 2; i >= 0; --i)
                in_stride[i] = in_stride[i + 1] * sh[i + 1];

            auto compute_means = [&](const auto* in_p) {
                for (std::size_t i = 0; i < n_in; ++i) {
                    std::size_t flat = i;
                    for (std::size_t d = 0; d < sh.size(); ++d) {
                        coords[d] = flat / static_cast<std::size_t>(in_stride[d]);
                        flat %= static_cast<std::size_t>(in_stride[d]);
                    }
                    std::size_t kf = 0;
                    for (std::size_t d = 0; d < sh.size(); ++d) {
                        std::int64_t c = (kd_shape[d] == 1) ? 0 : coords[d];
                        kf += c * static_cast<std::size_t>(kd_stride[d]);
                    }
                    means[kf] += static_cast<double>(in_p[i]);
                }
                for (auto& v : means)
                    v /= static_cast<double>(reduced);
            };
            if (dt == Dtype::F32)
                compute_means(reinterpret_cast<const float*>(ca.ptr.get()));
            else
                compute_means(reinterpret_cast<const double*>(ca.ptr.get()));
            auto fill_broadcast = [&](auto* dst) {
                using T = std::remove_pointer_t<decltype(dst)>;
                for (std::size_t i = 0; i < n_in; ++i) {
                    std::size_t flat = i;
                    for (std::size_t d = 0; d < sh.size(); ++d) {
                        coords[d] = flat / static_cast<std::size_t>(in_stride[d]);
                        flat %= static_cast<std::size_t>(in_stride[d]);
                    }
                    std::size_t kf = 0;
                    for (std::size_t d = 0; d < sh.size(); ++d) {
                        std::int64_t c = (kd_shape[d] == 1) ? 0 : coords[d];
                        kf += c * static_cast<std::size_t>(kd_stride[d]);
                    }
                    dst[i] = static_cast<T>(means[kf]);
                }
            };
            if (dt == Dtype::F32)
                fill_broadcast(reinterpret_cast<float*>(m.ptr.get()));
            else
                fill_broadcast(reinterpret_cast<double*>(m.ptr.get()));
            mean_storage = Storage{std::move(m)};
        }

        auto bwd = std::make_shared<VarBackward>();
        bwd->input_shape_ = a->shape();
        bwd->out_shape_ = out_shape;
        bwd->dtype_ = dt;
        bwd->device_ = device;
        bwd->axes_ = axes;
        bwd->keepdims_ = keepdims;
        bwd->count_ = static_cast<std::int64_t>(reduced);
        bwd->saved_input_ = a->storage();
        bwd->saved_mean_ = std::move(mean_storage);

        auto a_edge = detail::ensure_grad_fn(a);
        std::vector<Edge> edges;
        edges.emplace_back(a_edge, 0);
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version()});
        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
    }
    return out;
}

}  // namespace lucid
