#include "Linear.h"

#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/cpu/Blas.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../ops/bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

const OpSchema LinearBackward::schema_v1{"linear", 1, AmpPolicy::Promote, true};

namespace {

// Flatten leading dims of x into a single batch axis. Returns (M, K) where
// M = product of all but last dim, K = last dim.
struct FlatX {
    std::size_t M;
    std::size_t K;
};

FlatX flatten_x(const Shape& x_shape) {
    if (x_shape.empty()) {
        throw LucidError("linear: x must be at least 1-D");
    }
    std::size_t m = 1;
    for (std::size_t d = 0; d + 1 < x_shape.size(); ++d) {
        m *= static_cast<std::size_t>(x_shape[d]);
    }
    return {m, static_cast<std::size_t>(x_shape.back())};
}

CpuStorage allocate_2d_like(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

// y[m, n] = bias[n] for m in 0..M, n in 0..N. Broadcasts a 1-D bias over rows.
template <typename T>
void add_bias_typed(T* y, const T* b, std::size_t M, std::size_t N) {
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            y[m * N + n] += b[n];
        }
    }
}

void add_bias(CpuStorage& y, const CpuStorage& b, std::size_t M, std::size_t N, Dtype dt) {
    switch (dt) {
        case Dtype::F32:
            add_bias_typed<float>(reinterpret_cast<float*>(y.ptr.get()),
                                  reinterpret_cast<const float*>(b.ptr.get()), M, N);
            break;
        case Dtype::F64:
            add_bias_typed<double>(reinterpret_cast<double*>(y.ptr.get()),
                                   reinterpret_cast<const double*>(b.ptr.get()), M, N);
            break;
        default:
            throw NotImplementedError("linear: bias dtype not supported");
    }
}

// db = sum(grad, axis=0..M-1) — collapse all leading dims to 1, leaving (N,).
template <typename T>
void sum_rows_typed(const T* g, T* db, std::size_t M, std::size_t N) {
    for (std::size_t n = 0; n < N; ++n)
        db[n] = T{};
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            db[n] += g[m * N + n];
        }
    }
}

}  // namespace

TensorImplPtr LinearBackward::forward(const TensorImplPtr& x,
                                      const TensorImplPtr& W,
                                      const TensorImplPtr& b) {
    if (!x || !W || !b)
        throw LucidError("linear: null input");
    if (x->dtype_ != W->dtype_ || x->dtype_ != b->dtype_)
        throw DtypeMismatch(std::string(dtype_name(x->dtype_)), std::string(dtype_name(W->dtype_)),
                            "linear");
    if (x->device_ != W->device_ || x->device_ != b->device_)
        throw DeviceMismatch(std::string(device_name(x->device_)),
                             std::string(device_name(W->device_)), "linear");
    if (x->device_ == Device::CPU &&
        (!x->is_contiguous() || !W->is_contiguous() || !b->is_contiguous()))
        throw NotImplementedError(
            "linear: non-contiguous input not supported (call .contiguous() first)");
    if (W->shape_.size() != 2)
        throw ShapeMismatch(W->shape_, Shape{}, "linear: W must be 2-D");
    if (b->shape_.size() != 1)
        throw ShapeMismatch(b->shape_, Shape{}, "linear: b must be 1-D");

    const auto fx = flatten_x(x->shape_);
    const std::size_t M = fx.M;                                    // batch product
    const std::size_t K = fx.K;                                    // in_features
    const std::size_t N = static_cast<std::size_t>(W->shape_[0]);  // out_features

    if (W->shape_[1] != static_cast<std::int64_t>(K))
        throw ShapeMismatch(W->shape_, x->shape_, "linear: W.shape[1] != x.last_dim");
    if (b->shape_[0] != static_cast<std::int64_t>(N))
        throw ShapeMismatch(b->shape_, W->shape_, "linear: b.shape[0] != W.shape[0]");

    Shape out_shape = x->shape_;
    out_shape.back() = static_cast<std::int64_t>(N);

    OpScope scope{schema_v1.name, x->device_, x->dtype_, out_shape};
    scope.set_flops(static_cast<std::int64_t>(2 * M * N * K));

    Storage out_storage;
    if (x->device_ == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage_);
        const auto& gW = std::get<GpuStorage>(W->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        if (!gx.arr || !gW.arr || !gb.arr) {
            throw LucidError("linear: null GPU input");
        }
        ::mlx::core::array out_arr =
            (M == 0 || N == 0 || K == 0)
                ? ::mlx::core::zeros(gpu::to_mlx_shape(out_shape), gpu::to_mlx_dtype(x->dtype_))
                : ::mlx::core::add(::mlx::core::matmul(*gx.arr, ::mlx::core::transpose(*gW.arr)),
                                   *gb.arr);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out_arr), x->dtype_)};
    } else {
        auto out_cpu = allocate_2d_like(M * N, x->dtype_);
        // y = x @ W^T  (M,K) @ (K,N transposed-from-(N,K)) = (M,N)
        if (M > 0 && N > 0 && K > 0) {
            switch (x->dtype_) {
                case Dtype::F32:
                    backend::cpu::sgemm(
                        /*transA=*/false, /*transB=*/true, M, N, K, 1.0f,
                        reinterpret_cast<const float*>(std::get<CpuStorage>(x->storage_).ptr.get()),
                        K,
                        reinterpret_cast<const float*>(std::get<CpuStorage>(W->storage_).ptr.get()),
                        K, 0.0f, reinterpret_cast<float*>(out_cpu.ptr.get()), N);
                    break;
                case Dtype::F64:
                    backend::cpu::dgemm(
                        /*transA=*/false, /*transB=*/true, M, N, K, 1.0,
                        reinterpret_cast<const double*>(
                            std::get<CpuStorage>(x->storage_).ptr.get()),
                        K,
                        reinterpret_cast<const double*>(
                            std::get<CpuStorage>(W->storage_).ptr.get()),
                        K, 0.0, reinterpret_cast<double*>(out_cpu.ptr.get()), N);
                    break;
                default:
                    throw NotImplementedError("linear: dtype not supported (F32/F64)");
            }
        } else if (M * N > 0) {
            std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);
        }
        // y += bias broadcast
        if (M > 0 && N > 0) {
            add_bias(out_cpu, std::get<CpuStorage>(b->storage_), M, N, x->dtype_);
        }
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), x->dtype_,
                                            x->device_, false);

    if (!GradMode::is_enabled() || !(x->requires_grad_ || W->requires_grad_ || b->requires_grad_)) {
        return out;
    }

    auto x_edge = detail::ensure_grad_fn(x);
    auto W_edge = detail::ensure_grad_fn(W);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<LinearBackward>();
    bwd->input_shapes_ = {x->shape_, W->shape_, b->shape_};
    bwd->out_shape_ = out->shape_;
    bwd->dtype_ = x->dtype_;
    bwd->device_ = x->device_;
    bwd->input_tensors_ = {x, W, b};
    bwd->saved_inputs_ = {x->storage_, W->storage_, b->storage_};
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0), Edge(W_edge, 0), Edge(b_edge, 0)});
    bwd->set_saved_versions({x->version_, W->version_, b->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
    return out;
}

std::vector<Storage> LinearBackward::apply(Storage grad_out) {
    const auto fx = flatten_x(input_shapes_[0]);
    const std::size_t M = fx.M;
    const std::size_t K = fx.K;
    const std::size_t N = static_cast<std::size_t>(input_shapes_[1][0]);

    if (device_ == Device::GPU) {
        const auto& gG = std::get<GpuStorage>(grad_out);
        const auto& gX = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gW = std::get<GpuStorage>(saved_inputs_[1]);
        if (!gG.arr || !gX.arr || !gW.arr) {
            throw LucidError("linear backward: null GPU array");
        }
        // Reshape g and x to 2-D (M,N) / (M,K) so the matmul has the right
        // rank; reshape result back to original shapes for dx.
        Shape flatX{static_cast<std::int64_t>(M), static_cast<std::int64_t>(K)};
        Shape flatG{static_cast<std::int64_t>(M), static_cast<std::int64_t>(N)};
        auto g_2d = ::mlx::core::reshape(*gG.arr, gpu::to_mlx_shape(flatG));
        auto x_2d = ::mlx::core::reshape(*gX.arr, gpu::to_mlx_shape(flatX));
        if (M == 0 || N == 0 || K == 0) {
            auto zx =
                ::mlx::core::zeros(gpu::to_mlx_shape(input_shapes_[0]), gpu::to_mlx_dtype(dtype_));
            auto zw =
                ::mlx::core::zeros(gpu::to_mlx_shape(input_shapes_[1]), gpu::to_mlx_dtype(dtype_));
            auto zb =
                ::mlx::core::zeros(gpu::to_mlx_shape(input_shapes_[2]), gpu::to_mlx_dtype(dtype_));
            return {Storage{gpu::wrap_mlx_array(std::move(zx), dtype_)},
                    Storage{gpu::wrap_mlx_array(std::move(zw), dtype_)},
                    Storage{gpu::wrap_mlx_array(std::move(zb), dtype_)}};
        }
        // dx (flat) = g @ W ; reshape back to input_shapes_[0]
        auto dx_flat = ::mlx::core::matmul(g_2d, *gW.arr);
        auto dx = ::mlx::core::reshape(dx_flat, gpu::to_mlx_shape(input_shapes_[0]));
        // dW = g^T @ x : (N, M) @ (M, K) -> (N, K)
        auto dW = ::mlx::core::matmul(::mlx::core::transpose(g_2d), x_2d);
        // db = sum(g, axis=0) : (N,)
        auto db = ::mlx::core::sum(g_2d, std::vector<int>{0},
                                   /*keepdims=*/false);
        return {Storage{gpu::wrap_mlx_array(std::move(dx), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dW), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(db), dtype_)}};
    }

    const auto& g_cpu = std::get<CpuStorage>(grad_out);
    const auto& x_cpu = std::get<CpuStorage>(saved_inputs_[0]);
    const auto& W_cpu = std::get<CpuStorage>(saved_inputs_[1]);

    // dx = grad @ W : (M,N) @ (N,K) -> (M,K)
    CpuStorage dx_cpu = allocate_2d_like(M * K, dtype_);
    // dW = grad^T @ x : (N,M) @ (M,K) -> (N,K)
    CpuStorage dW_cpu = allocate_2d_like(N * K, dtype_);
    // db = sum(grad, axis=0..M-1) : (N,)
    CpuStorage db_cpu = allocate_2d_like(N, dtype_);

    if (M > 0 && N > 0 && K > 0) {
        switch (dtype_) {
            case Dtype::F32: {
                const auto* gp = reinterpret_cast<const float*>(g_cpu.ptr.get());
                const auto* xp = reinterpret_cast<const float*>(x_cpu.ptr.get());
                const auto* wp = reinterpret_cast<const float*>(W_cpu.ptr.get());
                backend::cpu::sgemm(false, false, M, K, N, 1.0f, gp, N, wp, K, 0.0f,
                                    reinterpret_cast<float*>(dx_cpu.ptr.get()), K);
                backend::cpu::sgemm(true, false, N, K, M, 1.0f, gp, N, xp, K, 0.0f,
                                    reinterpret_cast<float*>(dW_cpu.ptr.get()), K);
                sum_rows_typed<float>(gp, reinterpret_cast<float*>(db_cpu.ptr.get()), M, N);
                break;
            }
            case Dtype::F64: {
                const auto* gp = reinterpret_cast<const double*>(g_cpu.ptr.get());
                const auto* xp = reinterpret_cast<const double*>(x_cpu.ptr.get());
                const auto* wp = reinterpret_cast<const double*>(W_cpu.ptr.get());
                backend::cpu::dgemm(false, false, M, K, N, 1.0, gp, N, wp, K, 0.0,
                                    reinterpret_cast<double*>(dx_cpu.ptr.get()), K);
                backend::cpu::dgemm(true, false, N, K, M, 1.0, gp, N, xp, K, 0.0,
                                    reinterpret_cast<double*>(dW_cpu.ptr.get()), K);
                sum_rows_typed<double>(gp, reinterpret_cast<double*>(db_cpu.ptr.get()), M, N);
                break;
            }
            default:
                throw NotImplementedError("linear backward: dtype not supported");
        }
    } else {
        // Empty case — zero-fill all grads.
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        if (dW_cpu.nbytes)
            std::memset(dW_cpu.ptr.get(), 0, dW_cpu.nbytes);
        if (db_cpu.nbytes)
            std::memset(db_cpu.ptr.get(), 0, db_cpu.nbytes);
    }

    return {Storage{std::move(dx_cpu)}, Storage{std::move(dW_cpu)}, Storage{std::move(db_cpu)}};
}

TensorImplPtr linear_op(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b) {
    return LinearBackward::forward(x, W, b);
}

LUCID_REGISTER_OP(LinearBackward)

}  // namespace lucid
