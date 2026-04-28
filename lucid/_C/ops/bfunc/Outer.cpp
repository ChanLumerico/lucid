#include "Outer.h"

#include <variant>

#include <mlx/ops.h>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "_BinaryOp.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

class OuterBackward : public Node {
public:
    Storage saved_a_, saved_b_;
    std::int64_t M_, N_;
    Dtype dtype_;
    Device device_;

    std::vector<Storage> apply(Storage grad_out) override {
        if (device_ == Device::GPU) {
            const auto& gg = std::get<GpuStorage>(grad_out);
            const auto& ga = std::get<GpuStorage>(saved_a_);
            const auto& gb = std::get<GpuStorage>(saved_b_);
            auto da = ::mlx::core::matmul(
                *gg.arr,
                ::mlx::core::reshape(*gb.arr, ::mlx::core::Shape{static_cast<int>(N_), 1}));
            da = ::mlx::core::squeeze(da, 1);
            auto db = ::mlx::core::matmul(
                ::mlx::core::reshape(*ga.arr, ::mlx::core::Shape{1, static_cast<int>(M_)}),
                *gg.arr);
            db = ::mlx::core::squeeze(db, 0);
            return {Storage{gpu::wrap_mlx_array(std::move(da), dtype_)},
                    Storage{gpu::wrap_mlx_array(std::move(db), dtype_)}};
        }
        const auto& cg = std::get<CpuStorage>(grad_out);
        const auto& ca = std::get<CpuStorage>(saved_a_);
        const auto& cb = std::get<CpuStorage>(saved_b_);
        CpuStorage da, db;
        da.dtype = db.dtype = dtype_;
        da.nbytes = static_cast<std::size_t>(M_) * dtype_size(dtype_);
        db.nbytes = static_cast<std::size_t>(N_) * dtype_size(dtype_);
        da.ptr = allocate_aligned_bytes(da.nbytes);
        db.ptr = allocate_aligned_bytes(db.nbytes);
        auto run = [&](auto* dap, auto* dbp, const auto* gp, const auto* ap, const auto* bp) {
            using T = std::remove_pointer_t<decltype(dap)>;
            for (std::int64_t i = 0; i < M_; ++i) {
                T s{};
                for (std::int64_t j = 0; j < N_; ++j)
                    s = s + gp[i * N_ + j] * bp[j];
                dap[i] = s;
            }
            for (std::int64_t j = 0; j < N_; ++j) {
                T s{};
                for (std::int64_t i = 0; i < M_; ++i)
                    s = s + ap[i] * gp[i * N_ + j];
                dbp[j] = s;
            }
        };
        if (dtype_ == Dtype::F32)
            run(reinterpret_cast<float*>(da.ptr.get()), reinterpret_cast<float*>(db.ptr.get()),
                reinterpret_cast<const float*>(cg.ptr.get()),
                reinterpret_cast<const float*>(ca.ptr.get()),
                reinterpret_cast<const float*>(cb.ptr.get()));
        else if (dtype_ == Dtype::F64)
            run(reinterpret_cast<double*>(da.ptr.get()), reinterpret_cast<double*>(db.ptr.get()),
                reinterpret_cast<const double*>(cg.ptr.get()),
                reinterpret_cast<const double*>(ca.ptr.get()),
                reinterpret_cast<const double*>(cb.ptr.get()));
        else
            ErrorBuilder("outer backward").not_implemented("dtype not supported");
        return {Storage{std::move(da)}, Storage{std::move(db)}};
    }
};

}  // namespace

TensorImplPtr outer_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "outer");
    if (a->shape_.size() != 1 || b->shape_.size() != 1)
        ErrorBuilder("outer").fail("requires 1-D inputs");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScopeFull scope{"outer", device, dt, Shape{}};

    auto wire_grad = [&](const TensorImplPtr& out) {
        if (!(GradMode::is_enabled() && (a->requires_grad_ || b->requires_grad_)))
            return;
        auto bwd = std::make_shared<OuterBackward>();
        bwd->saved_a_ = a->storage_;
        bwd->saved_b_ = b->storage_;
        bwd->M_ = a->shape_[0];
        bwd->N_ = b->shape_[0];
        bwd->dtype_ = dt;
        bwd->device_ = device;
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
        auto out = ::mlx::core::outer(*ga.arr, *gb.arr);
        Shape out_shape;
        for (auto d : out.shape())
            out_shape.push_back(static_cast<std::int64_t>(d));
        auto t = fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)}, std::move(out_shape), dt,
                       device);
        wire_grad(t);
        return t;
    }

    const std::int64_t M = a->shape_[0];
    const std::int64_t N = b->shape_[0];
    Shape out_shape{M, N};
    auto out_cpu = allocate_cpu(out_shape, dt);
    const auto& ca = std::get<CpuStorage>(a->storage_);
    const auto& cb = std::get<CpuStorage>(b->storage_);
    auto run = [&](auto* op, const auto* ap, const auto* bp) {
        for (std::int64_t i = 0; i < M; ++i)
            for (std::int64_t j = 0; j < N; ++j)
                op[i * N + j] = ap[i] * bp[j];
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
        ErrorBuilder("outer").not_implemented("dtype not supported");
    auto t = fresh(Storage{std::move(out_cpu)}, out_shape, dt, device);
    wire_grad(t);
    return t;
}

}  // namespace lucid
