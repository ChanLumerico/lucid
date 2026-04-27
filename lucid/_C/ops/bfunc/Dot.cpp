#include "Dot.h"

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
#include "_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Backward for 1-D × 1-D dot: da = b * grad, db = a * grad (grad is scalar).
class Dot1DBackward : public Node {
public:
    Storage saved_a_, saved_b_;
    std::size_t numel_;
    Dtype dtype_;
    Device device_;

    std::vector<Storage> apply(Storage grad_out) override {
        double g = 0.0;
        if (device_ == Device::CPU) {
            const auto& cg = std::get<CpuStorage>(grad_out);
            if (dtype_ == Dtype::F32)
                g = *reinterpret_cast<const float*>(cg.ptr.get());
            else
                g = *reinterpret_cast<const double*>(cg.ptr.get());
        } else {
            const auto& gg = std::get<GpuStorage>(grad_out);
            auto cpu = gpu::download_gpu_to_cpu(gg, Shape{});
            if (dtype_ == Dtype::F32)
                g = *reinterpret_cast<const float*>(cpu.ptr.get());
            else
                g = *reinterpret_cast<const double*>(cpu.ptr.get());
        }
        Storage da = mul_scalar_storage(saved_b_, g, numel_, dtype_, device_);
        Storage db = mul_scalar_storage(saved_a_, g, numel_, dtype_, device_);
        return {std::move(da), std::move(db)};
    }
};

// Backward for 2-D × 2-D dot: da = grad @ b.T, db = a.T @ grad.
class Dot2DBackward : public Node {
public:
    Storage saved_a_, saved_b_;
    Shape a_shape_, b_shape_;
    Dtype dtype_;
    Device device_;

    std::vector<Storage> apply(Storage grad_out) override {
        const std::int64_t M = a_shape_[0], K = a_shape_[1], N = b_shape_[1];
        if (device_ == Device::GPU) {
            const auto& gg = std::get<GpuStorage>(grad_out);
            const auto& ga = std::get<GpuStorage>(saved_a_);
            const auto& gb = std::get<GpuStorage>(saved_b_);
            auto bT = ::mlx::core::transpose(*gb.arr, {1, 0});
            auto aT = ::mlx::core::transpose(*ga.arr, {1, 0});
            auto da = ::mlx::core::matmul(*gg.arr, bT);
            auto db = ::mlx::core::matmul(aT, *gg.arr);
            return {Storage{gpu::wrap_mlx_array(std::move(da), dtype_)},
                    Storage{gpu::wrap_mlx_array(std::move(db), dtype_)}};
        }
        const auto& cg = std::get<CpuStorage>(grad_out);
        const auto& ca = std::get<CpuStorage>(saved_a_);
        const auto& cb = std::get<CpuStorage>(saved_b_);
        CpuStorage da, db;
        da.dtype = db.dtype = dtype_;
        da.nbytes = static_cast<std::size_t>(M * K) * dtype_size(dtype_);
        db.nbytes = static_cast<std::size_t>(K * N) * dtype_size(dtype_);
        da.ptr = allocate_aligned_bytes(da.nbytes);
        db.ptr = allocate_aligned_bytes(db.nbytes);
        auto run = [&](auto* dap, auto* dbp,
                       const auto* gp, const auto* ap, const auto* bp) {
            using T = std::remove_pointer_t<decltype(dap)>;
            for (std::int64_t i = 0; i < M; ++i)
                for (std::int64_t k = 0; k < K; ++k) {
                    T s{};
                    for (std::int64_t j = 0; j < N; ++j)
                        s = s + gp[i * N + j] * bp[k * N + j];
                    dap[i * K + k] = s;
                }
            for (std::int64_t k = 0; k < K; ++k)
                for (std::int64_t j = 0; j < N; ++j) {
                    T s{};
                    for (std::int64_t i = 0; i < M; ++i)
                        s = s + ap[i * K + k] * gp[i * N + j];
                    dbp[k * N + j] = s;
                }
        };
        if (dtype_ == Dtype::F32)
            run(reinterpret_cast<float*>(da.ptr.get()),
                reinterpret_cast<float*>(db.ptr.get()),
                reinterpret_cast<const float*>(cg.ptr.get()),
                reinterpret_cast<const float*>(ca.ptr.get()),
                reinterpret_cast<const float*>(cb.ptr.get()));
        else if (dtype_ == Dtype::F64)
            run(reinterpret_cast<double*>(da.ptr.get()),
                reinterpret_cast<double*>(db.ptr.get()),
                reinterpret_cast<const double*>(cg.ptr.get()),
                reinterpret_cast<const double*>(ca.ptr.get()),
                reinterpret_cast<const double*>(cb.ptr.get()));
        else
            throw NotImplementedError("dot backward: dtype not supported");
        return {Storage{std::move(da)}, Storage{std::move(db)}};
    }
};

}  // namespace

TensorImplPtr dot_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "dot");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"dot", device, dt, Shape{}};

    auto wire_grad = [&](const TensorImplPtr& out) {
        if (!(GradMode::is_enabled() &&
              (a->requires_grad_ || b->requires_grad_)))
            return;
        std::shared_ptr<Node> bwd;
        if (a->shape_.size() == 1 && b->shape_.size() == 1) {
            auto n = std::make_shared<Dot1DBackward>();
            n->saved_a_ = a->storage_;
            n->saved_b_ = b->storage_;
            n->numel_   = static_cast<std::size_t>(a->shape_[0]);
            n->dtype_   = dt;
            n->device_  = device;
            bwd = std::move(n);
        } else if (a->shape_.size() == 2 && b->shape_.size() == 2) {
            auto n = std::make_shared<Dot2DBackward>();
            n->saved_a_ = a->storage_;
            n->saved_b_ = b->storage_;
            n->a_shape_ = a->shape_;
            n->b_shape_ = b->shape_;
            n->dtype_   = dt;
            n->device_  = device;
            bwd = std::move(n);
        } else {
            return;
        }
        auto a_edge = detail::ensure_grad_fn(a);
        auto b_edge = detail::ensure_grad_fn(b);
        std::vector<Edge> edges;
        edges.emplace_back(a_edge, 0);
        edges.emplace_back(b_edge, 0);
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version_, b->version_});
        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
    };

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        ::mlx::core::array out = (a->shape_.size() == 1 && b->shape_.size() == 1)
            ? ::mlx::core::sum(::mlx::core::multiply(*ga.arr, *gb.arr))
            : ::mlx::core::matmul(*ga.arr, *gb.arr);
        Shape out_shape;
        for (auto d : out.shape())
            out_shape.push_back(static_cast<std::int64_t>(d));
        auto t = fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)},
                       std::move(out_shape), dt, device);
        wire_grad(t);
        return t;
    }

    const auto& ca = std::get<CpuStorage>(a->storage_);
    const auto& cb = std::get<CpuStorage>(b->storage_);

    if (a->shape_.size() == 1 && b->shape_.size() == 1) {
        if (a->shape_[0] != b->shape_[0])
            throw ShapeMismatch(a->shape_, b->shape_, "dot");
        Shape out_shape{};
        auto out_cpu = allocate_cpu(out_shape, dt);
        const std::size_t n = static_cast<std::size_t>(a->shape_[0]);
        if (dt == Dtype::F32) {
            const auto* p = reinterpret_cast<const float*>(ca.ptr.get());
            const auto* q = reinterpret_cast<const float*>(cb.ptr.get());
            float s = 0.f;
            for (std::size_t i = 0; i < n; ++i) s += p[i] * q[i];
            *reinterpret_cast<float*>(out_cpu.ptr.get()) = s;
        } else if (dt == Dtype::F64) {
            const auto* p = reinterpret_cast<const double*>(ca.ptr.get());
            const auto* q = reinterpret_cast<const double*>(cb.ptr.get());
            double s = 0.0;
            for (std::size_t i = 0; i < n; ++i) s += p[i] * q[i];
            *reinterpret_cast<double*>(out_cpu.ptr.get()) = s;
        } else {
            throw NotImplementedError("dot: dtype not supported (CPU)");
        }
        auto t = fresh(Storage{std::move(out_cpu)}, out_shape, dt, device);
        wire_grad(t);
        return t;
    }

    if (a->shape_.size() == 2 && b->shape_.size() == 2) {
        const std::int64_t M = a->shape_[0], K = a->shape_[1];
        const std::int64_t Kb = b->shape_[0], N = b->shape_[1];
        if (K != Kb) throw ShapeMismatch(a->shape_, b->shape_, "dot");
        Shape out_shape{M, N};
        auto out_cpu = allocate_cpu(out_shape, dt);
        auto run = [&](auto* op, const auto* ap, const auto* bp) {
            using T = std::remove_pointer_t<decltype(op)>;
            for (std::int64_t i = 0; i < M; ++i)
                for (std::int64_t j = 0; j < N; ++j) {
                    T s{};
                    for (std::int64_t k = 0; k < K; ++k)
                        s = s + ap[i * K + k] * bp[k * N + j];
                    op[i * N + j] = s;
                }
        };
        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(out_cpu.ptr.get()),
                reinterpret_cast<const float*>(ca.ptr.get()),
                reinterpret_cast<const float*>(cb.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(out_cpu.ptr.get()),
                reinterpret_cast<const double*>(ca.ptr.get()),
                reinterpret_cast<const double*>(cb.ptr.get()));
        else
            throw NotImplementedError("dot: dtype not supported (CPU)");
        auto t = fresh(Storage{std::move(out_cpu)}, out_shape, dt, device);
        wire_grad(t);
        return t;
    }

    throw NotImplementedError("dot: CPU supports only 1-D × 1-D and 2-D × 2-D");
}

}  // namespace lucid
