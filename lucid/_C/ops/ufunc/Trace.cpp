#include "Trace.h"

#include <algorithm>
#include <cstring>
#include <variant>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::allocate_cpu;
using ufunc_detail::fresh;

class TraceBackward : public Node {
public:
    Shape input_shape_;  // (M, N)
    Dtype dtype_;
    Device device_;

    std::vector<Storage> apply(Storage grad_out) override {
        const std::int64_t M = input_shape_[0];
        const std::int64_t N = input_shape_[1];
        const std::int64_t L = std::min(M, N);
        const std::size_t total = static_cast<std::size_t>(M * N);

        if (device_ == Device::GPU) {
            const auto& gg = std::get<GpuStorage>(grad_out);
            auto eye = ::mlx::core::eye(static_cast<int>(M), static_cast<int>(N), 0,
                                        gpu::to_mlx_dtype(dtype_));
            auto out = ::mlx::core::multiply(eye, *gg.arr);
            return {Storage{gpu::wrap_mlx_array(std::move(out), dtype_)}};
        }

        const auto& cg = std::get<CpuStorage>(grad_out);
        CpuStorage dx;
        dx.dtype = dtype_;
        dx.nbytes = total * dtype_size(dtype_);
        dx.ptr = allocate_aligned_bytes(dx.nbytes);
        std::memset(dx.ptr.get(), 0, dx.nbytes);
        auto fill = [&](auto* dst, const auto* gp) {
            for (std::int64_t i = 0; i < L; ++i)
                dst[i * N + i] = *gp;
        };
        if (dtype_ == Dtype::F32)
            fill(reinterpret_cast<float*>(dx.ptr.get()),
                 reinterpret_cast<const float*>(cg.ptr.get()));
        else if (dtype_ == Dtype::F64)
            fill(reinterpret_cast<double*>(dx.ptr.get()),
                 reinterpret_cast<const double*>(cg.ptr.get()));
        else
            throw NotImplementedError("trace backward: dtype not supported");
        return {Storage{std::move(dx)}};
    }
};

}  // namespace

TensorImplPtr trace_op(const TensorImplPtr& a) {
    if (!a)
        throw LucidError("trace: null input");
    if (a->shape_.size() < 2)
        throw LucidError("trace: input must have ndim >= 2");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    const auto& sh = a->shape_;
    OpScope scope{"trace", device, dt, Shape{}};

    Shape out_shape(sh.begin() + 2, sh.end());

    TensorImplPtr out;
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        auto raw = ::mlx::core::trace(*ga.arr, /*offset=*/0,
                                      /*axis1=*/0, /*axis2=*/1);
        out = fresh(Storage{gpu::wrap_mlx_array(std::move(raw), dt)}, out_shape, dt, device);
    } else {
        const auto& ca = std::get<CpuStorage>(a->storage_);
        const std::int64_t M = sh[0], N = sh[1];
        const std::int64_t L = std::min(M, N);
        const std::size_t out_numel = shape_numel(out_shape);
        auto out_cpu = allocate_cpu(out_shape, dt);
        auto run = [&](auto* out_p, const auto* in_p) {
            using T = std::remove_pointer_t<decltype(out_p)>;
            for (std::size_t k = 0; k < out_numel; ++k) {
                T sum{};
                for (std::int64_t i = 0; i < L; ++i) {
                    const std::size_t idx = (i * N + i) * out_numel + k;
                    sum = static_cast<T>(static_cast<double>(sum) + static_cast<double>(in_p[idx]));
                }
                out_p[k] = sum;
            }
        };
        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(out_cpu.ptr.get()),
                reinterpret_cast<const float*>(ca.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(out_cpu.ptr.get()),
                reinterpret_cast<const double*>(ca.ptr.get()));
        else
            throw NotImplementedError("trace: dtype not supported");
        out = fresh(Storage{std::move(out_cpu)}, out_shape, dt, device);
    }

    if (GradMode::is_enabled() && a->requires_grad_ && a->shape_.size() == 2) {
        auto bwd = std::make_shared<TraceBackward>();
        bwd->input_shape_ = a->shape_;
        bwd->dtype_ = dt;
        bwd->device_ = device;
        auto a_edge = detail::ensure_grad_fn(a);
        std::vector<Edge> edges;
        edges.emplace_back(a_edge, 0);
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version_});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
    }
    return out;
}

}  // namespace lucid
