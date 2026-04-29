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
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../kernel/primitives/BatchedMatmul.h"
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
        ErrorBuilder("linear").fail("x must be at least 1-D");
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
            ErrorBuilder("linear").not_implemented("bias dtype not supported");
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
    Validator::input(x, "linear.x").non_null();
    Validator::input(W, "linear.W").non_null().ndim(2);
    Validator::input(b, "linear.b").non_null().ndim(1);
    Validator::pair(x, W, "linear").same_dtype().same_device();
    Validator::pair(x, b, "linear").same_dtype().same_device();

    const auto fx = flatten_x(x->shape());
    const std::size_t M = fx.M;                                     // batch product
    const std::size_t K = fx.K;                                     // in_features
    const std::size_t N = static_cast<std::size_t>(W->shape()[0]);  // out_features

    if (W->shape()[1] != static_cast<std::int64_t>(K))
        throw ShapeMismatch(W->shape(), x->shape(), "linear: W.shape[1] != x.last_dim");
    if (b->shape()[0] != static_cast<std::int64_t>(N))
        throw ShapeMismatch(b->shape(), W->shape(), "linear: b.shape[0] != W.shape[0]");

    Shape out_shape = x->shape();
    out_shape.back() = static_cast<std::int64_t>(N);

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), out_shape};
    scope.set_flops(static_cast<std::int64_t>(2 * M * N * K));

    Storage out_storage;
    if (x->device() == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(x->storage());
        const auto& gW = std::get<GpuStorage>(W->storage());
        const auto& gb = std::get<GpuStorage>(b->storage());
        if (!gx.arr || !gW.arr || !gb.arr) {
            ErrorBuilder("linear").fail("null GPU input");
        }
        ::mlx::core::array out_arr =
            (M == 0 || N == 0 || K == 0)
                ? ::mlx::core::zeros(gpu::to_mlx_shape(out_shape), gpu::to_mlx_dtype(x->dtype()))
                : ::mlx::core::add(::mlx::core::matmul(*gx.arr, ::mlx::core::transpose(*gW.arr)),
                                   *gb.arr);
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out_arr), x->dtype())};
    } else {
        auto out_cpu = allocate_2d_like(M * N, x->dtype());
        // y = x @ W^T  (M,K) @ (K,N transposed-from-(N,K)) = (M,N)
        if (M > 0 && N > 0 && K > 0) {
            switch (x->dtype()) {
                case Dtype::F32:
                    backend::cpu::sgemm(
                        /*transA=*/false, /*transB=*/true, M, N, K, 1.0f,
                        reinterpret_cast<const float*>(
                            std::get<CpuStorage>(x->storage()).ptr.get()),
                        K,
                        reinterpret_cast<const float*>(
                            std::get<CpuStorage>(W->storage()).ptr.get()),
                        K, 0.0f, reinterpret_cast<float*>(out_cpu.ptr.get()), N);
                    break;
                case Dtype::F64:
                    backend::cpu::dgemm(
                        /*transA=*/false, /*transB=*/true, M, N, K, 1.0,
                        reinterpret_cast<const double*>(
                            std::get<CpuStorage>(x->storage()).ptr.get()),
                        K,
                        reinterpret_cast<const double*>(
                            std::get<CpuStorage>(W->storage()).ptr.get()),
                        K, 0.0, reinterpret_cast<double*>(out_cpu.ptr.get()), N);
                    break;
                default:
                    ErrorBuilder("linear").not_implemented("dtype not supported (F32/F64)");
            }
        } else if (M * N > 0) {
            std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);
        }
        // y += bias broadcast
        if (M > 0 && N > 0) {
            add_bias(out_cpu, std::get<CpuStorage>(b->storage()), M, N, x->dtype());
        }
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);

    kernel::NaryKernel<LinearBackward, 3>::wire_autograd({x, W, b}, out);
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
            ErrorBuilder("linear backward").fail("null GPU array");
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

    // db = sum(grad, axis=0..M-1) : (N,)  — always done directly.
    CpuStorage db_cpu = allocate_2d_like(N, dtype_);

    if (M > 0 && N > 0 && K > 0) {
        // Use cpu_matmul_nd for dx and dW via the BatchedMatmul primitive.
        // Linear always presents a 2-D (M, K)/(M, N) layout (batch=1).
        // cpu_matmul_nd handles the general case cleanly and is the canonical
        // path for any future N-D batched extension.

        // dx = grad @ W  : (M, N) @ (N, K) -> (M, K)
        //   a = grad [M, N], b = W [N, K], no transpose needed
        kernel::primitives::NdMatmulInfo dx_info;
        dx_info.M = static_cast<int>(M);
        dx_info.K = static_cast<int>(N);
        dx_info.N = static_cast<int>(K);
        dx_info.batch = 1;
        dx_info.out_shape = {static_cast<std::int64_t>(M), static_cast<std::int64_t>(K)};
        CpuStorage dx_cpu =
            kernel::primitives::cpu_matmul_nd(g_cpu, W_cpu, dx_info, false, false, dtype_);

        // dW = grad^T @ x : (N, M) @ (M, K) -> (N, K)
        //   a = grad [M, N] transposed (transA=true), b = x [M, K]
        kernel::primitives::NdMatmulInfo dW_info;
        dW_info.M = static_cast<int>(N);
        dW_info.K = static_cast<int>(M);
        dW_info.N = static_cast<int>(K);
        dW_info.batch = 1;
        dW_info.out_shape = {static_cast<std::int64_t>(N), static_cast<std::int64_t>(K)};
        CpuStorage dW_cpu =
            kernel::primitives::cpu_matmul_nd(g_cpu, x_cpu, dW_info, true, false, dtype_);

        switch (dtype_) {
            case Dtype::F32:
                sum_rows_typed<float>(reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                      reinterpret_cast<float*>(db_cpu.ptr.get()), M, N);
                break;
            case Dtype::F64:
                sum_rows_typed<double>(reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                       reinterpret_cast<double*>(db_cpu.ptr.get()), M, N);
                break;
            default:
                ErrorBuilder("linear backward").not_implemented("dtype not supported");
        }

        return {Storage{std::move(dx_cpu)}, Storage{std::move(dW_cpu)}, Storage{std::move(db_cpu)}};
    }

    // Empty case — zero-fill all grads.
    CpuStorage dx_cpu = allocate_2d_like(M * K, dtype_);
    CpuStorage dW_cpu = allocate_2d_like(N * K, dtype_);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    if (dW_cpu.nbytes)
        std::memset(dW_cpu.ptr.get(), 0, dW_cpu.nbytes);
    if (db_cpu.nbytes)
        std::memset(db_cpu.ptr.get(), 0, db_cpu.nbytes);

    return {Storage{std::move(dx_cpu)}, Storage{std::move(dW_cpu)}, Storage{std::move(db_cpu)}};
}

TensorImplPtr linear_op(const TensorImplPtr& x, const TensorImplPtr& W, const TensorImplPtr& b) {
    return LinearBackward::forward(x, W, b);
}

LUCID_REGISTER_OP(LinearBackward)

}  // namespace lucid
