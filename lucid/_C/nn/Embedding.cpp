#include "Embedding.h"

#include <cmath>
#include <cstring>
#include <vector>

#include <mlx/ops.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
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
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr = allocate_aligned_bytes(s.nbytes);
    return s;
}

// Read an integer index from any int dtype at flat offset i.
inline std::int64_t read_index(const CpuStorage& ts, std::size_t i) {
    switch (ts.dtype) {
        case Dtype::I8:
            return reinterpret_cast<const std::int8_t*>(ts.ptr.get())[i];
        case Dtype::I16:
            return reinterpret_cast<const std::int16_t*>(ts.ptr.get())[i];
        case Dtype::I32:
            return reinterpret_cast<const std::int32_t*>(ts.ptr.get())[i];
        case Dtype::I64:
            return reinterpret_cast<const std::int64_t*>(ts.ptr.get())[i];
        case Dtype::Bool:
            return reinterpret_cast<const std::uint8_t*>(ts.ptr.get())[i];
        default:
            ErrorBuilder("embedding").not_implemented("index dtype must be integer");
    }
}

}  // namespace

// =====================================================================
// Embedding
// =====================================================================

const OpSchema EmbeddingBackward::schema_v1{"embedding", 1, AmpPolicy::Promote, true};

TensorImplPtr EmbeddingBackward::forward(const TensorImplPtr& weight,
                                         const TensorImplPtr& indices,
                                         int padding_idx) {
    if (!weight || !indices)
        ErrorBuilder("embedding").fail("null input");
    if (weight->device() != indices->device())
        throw DeviceMismatch(std::string(device_name(weight->device())),
                             std::string(device_name(indices->device())),
                             "embedding: weight/indices");
    if (weight->shape().size() != 2)
        throw ShapeMismatch(weight->shape(), Shape{},
                            "embedding: weight must be 2-D (num_embeddings, dim)");

    const std::int64_t N = weight->shape()[0];
    const std::int64_t D = weight->shape()[1];
    const std::size_t M = indices->numel();

    Shape out_shape = indices->shape();
    out_shape.push_back(D);
    OpScopeFull scope{schema_v1.name, weight->device(), weight->dtype(), out_shape};

    Storage out_storage;
    if (weight->device() == Device::GPU) {
        const auto& gw = std::get<GpuStorage>(weight->storage());
        const auto& gi = std::get<GpuStorage>(indices->storage());
        // mlx::take(weight[N, D], indices) — flatten then take, reshape back.
        auto idx = ::mlx::core::astype(*gi.arr, ::mlx::core::int64);
        auto out = ::mlx::core::take(*gw.arr, idx, /*axis=*/0);
        if (padding_idx >= 0) {
            // Zero rows where indices == padding_idx. Use broadcast multiply.
            auto pad = ::mlx::core::astype(::mlx::core::array(padding_idx), ::mlx::core::int64);
            auto mask = ::mlx::core::not_equal(idx, pad);  // shape: indices.shape
            auto mask_dt = ::mlx::core::astype(mask, gpu::to_mlx_dtype(weight->dtype()));
            // Insert trailing dim to broadcast over D.
            auto mask_shape = mask_dt.shape();
            mask_shape.push_back(1);
            mask_dt = ::mlx::core::reshape(mask_dt, mask_shape);
            out = ::mlx::core::multiply(out, mask_dt);
        }
        out_storage = Storage{gpu::wrap_mlx_array(std::move(out), weight->dtype())};
    } else {
        auto out_cpu = allocate_size(M * static_cast<std::size_t>(D), weight->dtype());
        const auto& ws = std::get<CpuStorage>(weight->storage());
        const auto& is = std::get<CpuStorage>(indices->storage());
        const std::size_t row_bytes = static_cast<std::size_t>(D) * dtype_size(weight->dtype());

        for (std::size_t i = 0; i < M; ++i) {
            const std::int64_t id = read_index(is, i);
            if (id < 0 || id >= N) {
                ErrorBuilder("embedding").index_error("index out of range");
            }
            std::byte* dst = out_cpu.ptr.get() + i * row_bytes;
            if (padding_idx >= 0 && id == padding_idx) {
                std::memset(dst, 0, row_bytes);
            } else {
                const std::byte* src = ws.ptr.get() + static_cast<std::size_t>(id) * row_bytes;
                std::memcpy(dst, src, row_bytes);
            }
        }
        out_storage = Storage{std::move(out_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, weight->dtype(),
                                            weight->device(), false);

    if (!GradMode::is_enabled() || !weight->requires_grad())
        return out;

    auto w_edge = detail::ensure_grad_fn(weight);
    auto bwd = std::make_shared<EmbeddingBackward>();
    bwd->input_shapes_ = {weight->shape()};
    bwd->out_shape_ = out_shape;
    bwd->dtype_ = weight->dtype();
    bwd->device_ = weight->device();
    bwd->input_tensors_ = {weight};
    bwd->saved_inputs_ = {weight->storage()};
    bwd->saved_indices_ = indices->storage();
    bwd->saved_indices_shape_ = indices->shape();
    bwd->saved_indices_dtype_ = indices->dtype();
    bwd->padding_idx_ = padding_idx;
    bwd->weight_shape_ = weight->shape();
    bwd->set_next_edges(std::vector<Edge>{Edge(w_edge, 0)});
    bwd->set_saved_versions(
        std::vector<std::int64_t>{static_cast<std::int64_t>(weight->version())});
    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

std::vector<Storage> EmbeddingBackward::apply(Storage grad_out) {
    const std::int64_t N = weight_shape_[0];
    const std::int64_t D = weight_shape_[1];
    const std::size_t M = shape_numel(saved_indices_shape_);
    const std::size_t row_floats = static_cast<std::size_t>(D);

    if (device_ == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto& gi = std::get<GpuStorage>(saved_indices_);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        // Flatten indices and grad rows: indices [..,] → [M], grad [..,D] → [M, D].
        auto idx_flat = ::mlx::core::reshape(::mlx::core::astype(*gi.arr, ::mlx::core::int64),
                                             {static_cast<int>(M)});
        auto grad_flat = ::mlx::core::reshape(*gg.arr, {static_cast<int>(M), static_cast<int>(D)});
        if (padding_idx_ >= 0) {
            auto pad = ::mlx::core::astype(::mlx::core::array(padding_idx_), ::mlx::core::int64);
            auto mask = ::mlx::core::not_equal(idx_flat, pad);
            auto mask_dt = ::mlx::core::astype(mask, mlx_dt);
            auto mask_b = ::mlx::core::reshape(mask_dt, {static_cast<int>(M), 1});
            grad_flat = ::mlx::core::multiply(grad_flat, mask_b);
        }
        // dW[i, :] = sum_j (idx[j] == i) * grad[j, :].  Implemented as
        //   onehot = (arange(N)[None, :] == idx[:, None])     # [M, N]
        //   dW = onehot^T @ grad                              # [N, D]
        // This sidesteps MLX's awkward scatter_add-with-1d-indices shape rules.
        auto idx_col = ::mlx::core::reshape(idx_flat, {static_cast<int>(M), 1});
        auto arange_n = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64),
            {1, static_cast<int>(N)});
        auto onehot = ::mlx::core::astype(::mlx::core::equal(arange_n, idx_col), mlx_dt);
        auto onehot_t = ::mlx::core::transpose(onehot);  // [N, M]
        auto dW = ::mlx::core::matmul(onehot_t, grad_flat);
        return {Storage{gpu::wrap_mlx_array(std::move(dW), dtype_)}};
    }

    auto dW = allocate_size(static_cast<std::size_t>(N) * row_floats, dtype_);
    std::memset(dW.ptr.get(), 0, dW.nbytes);

    const auto& gs = std::get<CpuStorage>(grad_out);
    const auto& is = std::get<CpuStorage>(saved_indices_);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        T* wp = reinterpret_cast<T*>(dW.ptr.get());
        for (std::size_t i = 0; i < M; ++i) {
            const std::int64_t id = read_index(is, i);
            if (padding_idx_ >= 0 && id == padding_idx_)
                continue;
            T* row = wp + static_cast<std::size_t>(id) * row_floats;
            const T* src = gp + i * row_floats;
            for (std::size_t k = 0; k < row_floats; ++k)
                row[k] += src[k];
        }
    };
    if (dtype_ == Dtype::F32)
        run(float{});
    else if (dtype_ == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("embedding backward").not_implemented("dtype not supported");

    return {Storage{std::move(dW)}};
}

TensorImplPtr embedding_op(const TensorImplPtr& weight,
                           const TensorImplPtr& indices,
                           int padding_idx) {
    return EmbeddingBackward::forward(weight, indices, padding_idx);
}
LUCID_REGISTER_OP(EmbeddingBackward)

// =====================================================================
// Sinusoidal positional embedding — pure forward, no grad.
// =====================================================================

TensorImplPtr sinusoidal_pos_embedding_op(std::int64_t seq_len,
                                          std::int64_t embed_dim,
                                          Dtype dtype,
                                          Device device) {
    if (seq_len < 0)
        ErrorBuilder("sinusoidal_pos_embedding").fail("seq_len < 0");
    if (embed_dim <= 0)
        ErrorBuilder("sinusoidal_pos_embedding").fail("embed_dim must be > 0");

    const std::size_t L = static_cast<std::size_t>(seq_len);
    const std::size_t D = static_cast<std::size_t>(embed_dim);
    const std::size_t Dh = D / 2;

    Shape out_shape{seq_len, embed_dim};
    OpScopeFull scope{"sinusoidal_pos_embedding", device, dtype, out_shape};

    if (device == Device::GPU) {
        if (L * D == 0) {
            auto z = ::mlx::core::zeros(gpu::to_mlx_shape(out_shape), gpu::to_mlx_dtype(dtype));
            return std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(z), dtype)},
                                                out_shape, dtype, device, false);
        }
        const auto mlx_dt = gpu::to_mlx_dtype(dtype);
        // Build pos[L, 1] and theta[1, Dh].
        auto pos = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(L), 1), mlx_dt),
            {static_cast<int>(L), 1});
        auto k = ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(Dh), 1), mlx_dt);
        const double inv_d = -std::log(10000.0) / static_cast<double>(D);
        auto two_inv_d =
            ::mlx::core::astype(::mlx::core::array(static_cast<float>(2.0 * inv_d)), mlx_dt);
        auto theta = ::mlx::core::exp(::mlx::core::multiply(two_inv_d, k));
        auto theta_row = ::mlx::core::reshape(theta, {1, static_cast<int>(Dh)});
        auto angle = ::mlx::core::matmul(pos, theta_row);  // [L, Dh]
        auto sin_t = ::mlx::core::sin(angle);
        auto cos_t = ::mlx::core::cos(angle);
        // Interleave: [L, D] where D is 2*Dh (even case). For odd D, append zero col.
        auto sin_e = ::mlx::core::expand_dims(sin_t, /*axis=*/-1);  // [L, Dh, 1]
        auto cos_e = ::mlx::core::expand_dims(cos_t, /*axis=*/-1);  // [L, Dh, 1]
        auto stacked = ::mlx::core::concatenate(std::vector<::mlx::core::array>{sin_e, cos_e},
                                                /*axis=*/-1);  // [L, Dh, 2]
        auto out = ::mlx::core::reshape(
            stacked, {static_cast<int>(L), static_cast<int>(2 * Dh)});  // [L, 2*Dh]
        if (D % 2 != 0) {
            // Append a zero column.
            auto pad = ::mlx::core::zeros({static_cast<int>(L), 1}, mlx_dt);
            out = ::mlx::core::concatenate(std::vector<::mlx::core::array>{out, pad}, -1);
        }
        return std::make_shared<TensorImpl>(Storage{gpu::wrap_mlx_array(std::move(out), dtype)},
                                            out_shape, dtype, device, false);
    }

    auto out_cpu = allocate_size(L * D, dtype);
    if (L * D == 0) {
        return std::make_shared<TensorImpl>(Storage{std::move(out_cpu)}, out_shape, dtype, device,
                                            false);
    }
    const double inv_d = -std::log(10000.0) / static_cast<double>(D);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
        for (std::size_t i = 0; i < L; ++i) {
            for (std::size_t k = 0; k < Dh; ++k) {
                const double theta = std::exp(2.0 * static_cast<double>(k) * inv_d);
                const double angle = static_cast<double>(i) * theta;
                op[i * D + 2 * k] = static_cast<T>(std::sin(angle));
                if (2 * k + 1 < D)
                    op[i * D + 2 * k + 1] = static_cast<T>(std::cos(angle));
            }
        }
    };
    if (dtype == Dtype::F32)
        run(float{});
    else if (dtype == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("sinusoidal_pos_embedding").not_implemented("dtype must be F32 or F64");

    return std::make_shared<TensorImpl>(Storage{std::move(out_cpu)}, out_shape, dtype, device,
                                        false);
}

// =====================================================================
// Rotary positional embedding (RoPE).
// =====================================================================

const OpSchema RotaryPosEmbeddingBackward::schema_v1{"rotary_pos_embedding", 1,
                                                     AmpPolicy::ForceFP32, true};

namespace {

// Build cos/sin tables of shape (L, D/2) from positions and theta_k.
// pos: [L] doubles (caller's responsibility), theta: [D/2] doubles.
template <typename T>
void build_cos_sin_tables(
    const double* pos, const double* theta, std::size_t L, std::size_t Dh, T* cos_out, T* sin_out) {
    for (std::size_t i = 0; i < L; ++i) {
        for (std::size_t k = 0; k < Dh; ++k) {
            const double angle = pos[i] * theta[k];
            cos_out[i * Dh + k] = static_cast<T>(std::cos(angle));
            sin_out[i * Dh + k] = static_cast<T>(std::sin(angle));
        }
    }
}

// out = x · cos + rotate(x) · sin, where rotate depends on `interleaved`.
template <typename T>
void rope_forward(const T* xp,
                  const T* cosp,
                  const T* sinp,
                  T* op,
                  std::size_t batch,
                  std::size_t L,
                  std::size_t D,
                  bool interleaved) {
    const std::size_t Dh = D / 2;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < L; ++i) {
            const T* xrow = xp + (b * L + i) * D;
            T* orow = op + (b * L + i) * D;
            const T* crow = cosp + i * Dh;
            const T* srow = sinp + i * Dh;
            if (interleaved) {
                for (std::size_t k = 0; k < Dh; ++k) {
                    const T xe = xrow[2 * k];
                    const T xo = xrow[2 * k + 1];
                    const T c = crow[k];
                    const T s = srow[k];
                    orow[2 * k] = xe * c - xo * s;
                    orow[2 * k + 1] = xo * c + xe * s;
                }
            } else {
                for (std::size_t k = 0; k < Dh; ++k) {
                    const T xa = xrow[k];
                    const T xb = xrow[k + Dh];
                    const T c = crow[k];
                    const T s = srow[k];
                    orow[k] = xa * c - xb * s;
                    orow[k + Dh] = xb * c + xa * s;
                }
            }
        }
    }
}

// dx = transpose(R) · dout. With R = [[cos,-sin],[sin,cos]],
//   dx_e =  cos·dout_e + sin·dout_o
//   dx_o = -sin·dout_e + cos·dout_o
template <typename T>
void rope_backward(const T* gp,
                   const T* cosp,
                   const T* sinp,
                   T* dxp,
                   std::size_t batch,
                   std::size_t L,
                   std::size_t D,
                   bool interleaved) {
    const std::size_t Dh = D / 2;
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < L; ++i) {
            const T* gr = gp + (b * L + i) * D;
            T* dxr = dxp + (b * L + i) * D;
            const T* cr = cosp + i * Dh;
            const T* sr = sinp + i * Dh;
            if (interleaved) {
                for (std::size_t k = 0; k < Dh; ++k) {
                    const T ge = gr[2 * k];
                    const T go = gr[2 * k + 1];
                    const T c = cr[k];
                    const T s = sr[k];
                    dxr[2 * k] = c * ge + s * go;
                    dxr[2 * k + 1] = -s * ge + c * go;
                }
            } else {
                for (std::size_t k = 0; k < Dh; ++k) {
                    const T ga = gr[k];
                    const T gb = gr[k + Dh];
                    const T c = cr[k];
                    const T s = sr[k];
                    dxr[k] = c * ga + s * gb;
                    dxr[k + Dh] = -s * ga + c * gb;
                }
            }
        }
    }
}

}  // namespace

TensorImplPtr RotaryPosEmbeddingBackward::forward(const TensorImplPtr& input,
                                                  const TensorImplPtr& position_ids_or_null,
                                                  bool interleaved) {
    Validator::input(input, "rotary_pos_embedding.input").non_null();
    if (position_ids_or_null && position_ids_or_null->device() != input->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(position_ids_or_null->device())),
                             "rotary_pos_embedding: input/position_ids");
    if (input->shape().size() < 2)
        ErrorBuilder("rotary_pos_embedding").fail("input must be at least 2-D ([..., L, D])");

    const std::size_t ndim = input->shape().size();
    const std::size_t L = static_cast<std::size_t>(input->shape()[ndim - 2]);
    const std::size_t D = static_cast<std::size_t>(input->shape()[ndim - 1]);
    if (D % 2 != 0)
        ErrorBuilder("rotary_pos_embedding").fail("embed_dim must be even");
    const std::size_t Dh = D / 2;
    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < ndim; ++i)
        batch *= static_cast<std::size_t>(input->shape()[i]);

    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), input->shape()};

    if (input->device() == Device::GPU) {
        const auto& gx = std::get<GpuStorage>(input->storage());
        const auto mlx_dt = gpu::to_mlx_dtype(input->dtype());

        // pos: [L, 1] in input dtype.
        ::mlx::core::array pos_arr{0};
        if (position_ids_or_null) {
            if (position_ids_or_null->shape().size() != 1 ||
                static_cast<std::size_t>(position_ids_or_null->shape()[0]) != L)
                throw ShapeMismatch(position_ids_or_null->shape(),
                                    Shape{static_cast<std::int64_t>(L)},
                                    "rotary_pos_embedding: position_ids");
            const auto& gp = std::get<GpuStorage>(position_ids_or_null->storage());
            pos_arr = ::mlx::core::astype(*gp.arr, mlx_dt);
        } else {
            pos_arr = ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(L), 1), mlx_dt);
        }
        pos_arr = ::mlx::core::reshape(pos_arr, {static_cast<int>(L), 1});

        // theta[k] = exp(-2k log(10000) / D), shape [1, Dh].
        const double base = std::log(10000.0);
        auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(Dh), 1), mlx_dt);
        auto coef = ::mlx::core::astype(
            ::mlx::core::array(static_cast<float>(-2.0 * base / static_cast<double>(D))), mlx_dt);
        auto theta = ::mlx::core::exp(::mlx::core::multiply(coef, k_arr));
        auto theta_row = ::mlx::core::reshape(theta, {1, static_cast<int>(Dh)});
        auto angle = ::mlx::core::matmul(pos_arr, theta_row);  // [L, Dh]
        auto cos_t = ::mlx::core::cos(angle);
        auto sin_t = ::mlx::core::sin(angle);

        // Reshape input [..., L, D] → [..., L, Dh, 2] for interleaved or
        // [..., L, 2, Dh] for split.
        auto in_shape = gpu::to_mlx_shape(input->shape());
        ::mlx::core::Shape pair_shape = in_shape;
        // Replace last dim with [Dh, 2] (interleaved) or [2, Dh] (split).
        pair_shape.pop_back();
        ::mlx::core::array out_arr{0};
        // Reshape cos/sin to broadcast: shape (..., L, Dh) → broadcast over batch.
        // We need cos_b shape compatible with (batch, L, Dh).
        // cos_t / sin_t are (L, Dh); insert leading 1s for batch dims.
        ::mlx::core::Shape cos_shape(in_shape.size() - 1, 1);
        cos_shape[in_shape.size() - 2] = static_cast<int>(L);
        // Wait, the reshape just inserts singleton batch dims and keeps L.
        // cos_shape now has ndim-1 dims; replace last with Dh.
        cos_shape[in_shape.size() - 2] = static_cast<int>(L);
        // Need ndim-1 dims total. Last dim is Dh.
        cos_shape.back() = static_cast<int>(L);
        // Actually: out_shape is (..., L, D). cos shape needs to be broadcast-able with
        // (..., L, Dh) (then expanded to (..., L, Dh, 2) for interleaved). Simplest:
        // cos_b shape = (1,)*nbatch + (L, Dh).
        ::mlx::core::Shape cos_b_shape;
        for (std::size_t i = 0; i + 2 < ndim; ++i)
            cos_b_shape.push_back(1);
        cos_b_shape.push_back(static_cast<int>(L));
        cos_b_shape.push_back(static_cast<int>(Dh));
        auto cos_b = ::mlx::core::reshape(cos_t, cos_b_shape);
        auto sin_b = ::mlx::core::reshape(sin_t, cos_b_shape);

        if (interleaved) {
            // x reshape [..., L, Dh, 2]
            ::mlx::core::Shape x_shape = in_shape;
            x_shape.pop_back();
            x_shape.push_back(static_cast<int>(Dh));
            x_shape.push_back(2);
            auto x_re = ::mlx::core::reshape(*gx.arr, x_shape);
            // Slice even/odd via stride or take_along_axis. Use take_along_axis with
            // [...,Dh,1] indices, simpler to split via narrow contiguous reshape and slicing.
            // Easier: arange(0,2) as last-axis index, take with 0 then 1.
            auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
            auto xe = ::mlx::core::take(x_re, idx0, /*axis=*/-1);  // [..., L, Dh]
            auto xo = ::mlx::core::take(x_re, idx1, /*axis=*/-1);
            auto out_e = ::mlx::core::subtract(::mlx::core::multiply(xe, cos_b),
                                               ::mlx::core::multiply(xo, sin_b));
            auto out_o = ::mlx::core::add(::mlx::core::multiply(xo, cos_b),
                                          ::mlx::core::multiply(xe, sin_b));
            // Stack back interleaved.
            auto out_e_e = ::mlx::core::expand_dims(out_e, /*axis=*/-1);
            auto out_o_e = ::mlx::core::expand_dims(out_o, /*axis=*/-1);
            auto stacked =
                ::mlx::core::concatenate(std::vector<::mlx::core::array>{out_e_e, out_o_e}, -1);
            out_arr = ::mlx::core::reshape(stacked, in_shape);
        } else {
            // Split: x[..., :Dh] and x[..., Dh:]. Use take with index ranges.
            auto idx_first = ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(Dh), 1),
                                                 ::mlx::core::int64);
            auto idx_second = ::mlx::core::astype(
                ::mlx::core::arange(static_cast<int>(Dh), static_cast<int>(D), 1),
                ::mlx::core::int64);
            auto xa = ::mlx::core::take(*gx.arr, idx_first, /*axis=*/-1);
            auto xb = ::mlx::core::take(*gx.arr, idx_second, /*axis=*/-1);
            auto out_a = ::mlx::core::subtract(::mlx::core::multiply(xa, cos_b),
                                               ::mlx::core::multiply(xb, sin_b));
            auto out_b = ::mlx::core::add(::mlx::core::multiply(xb, cos_b),
                                          ::mlx::core::multiply(xa, sin_b));
            out_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{out_a, out_b}, -1);
        }

        auto out = std::make_shared<TensorImpl>(
            Storage{gpu::wrap_mlx_array(std::move(out_arr), input->dtype())}, input->shape(),
            input->dtype(), input->device(), false);

        if (!GradMode::is_enabled() || !input->requires_grad())
            return out;
        auto x_edge = detail::ensure_grad_fn(input);
        auto bwd = std::make_shared<RotaryPosEmbeddingBackward>();
        bwd->input_shapes_ = {input->shape()};
        bwd->out_shape_ = input->shape();
        bwd->dtype_ = input->dtype();
        bwd->device_ = input->device();
        bwd->input_tensors_ = {input};
        bwd->saved_inputs_ = {input->storage()};
        bwd->saved_cos_ = Storage{gpu::wrap_mlx_array(std::move(cos_t), input->dtype())};
        bwd->saved_sin_ = Storage{gpu::wrap_mlx_array(std::move(sin_t), input->dtype())};
        bwd->interleaved_ = interleaved;
        bwd->orig_shape_ = input->shape();
        bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
        bwd->set_saved_versions(
            std::vector<std::int64_t>{static_cast<std::int64_t>(input->version())});
        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return out;
    }

    // Build pos[i] and theta[k] in double.
    std::vector<double> pos(L);
    if (position_ids_or_null) {
        if (position_ids_or_null->shape().size() != 1 ||
            static_cast<std::size_t>(position_ids_or_null->shape()[0]) != L) {
            throw ShapeMismatch(position_ids_or_null->shape(), Shape{static_cast<std::int64_t>(L)},
                                "rotary_pos_embedding: position_ids must be 1-D, length L");
        }
        const auto& ps = std::get<CpuStorage>(position_ids_or_null->storage());
        for (std::size_t i = 0; i < L; ++i) {
            switch (position_ids_or_null->dtype()) {
                case Dtype::I8:
                    pos[i] = reinterpret_cast<const std::int8_t*>(ps.ptr.get())[i];
                    break;
                case Dtype::I16:
                    pos[i] = reinterpret_cast<const std::int16_t*>(ps.ptr.get())[i];
                    break;
                case Dtype::I32:
                    pos[i] = reinterpret_cast<const std::int32_t*>(ps.ptr.get())[i];
                    break;
                case Dtype::I64:
                    pos[i] = reinterpret_cast<const std::int64_t*>(ps.ptr.get())[i];
                    break;
                case Dtype::F32:
                    pos[i] = reinterpret_cast<const float*>(ps.ptr.get())[i];
                    break;
                case Dtype::F64:
                    pos[i] = reinterpret_cast<const double*>(ps.ptr.get())[i];
                    break;
                default:
                    ErrorBuilder("rotary_pos_embedding")
                        .not_implemented("position_ids dtype not supported");
            }
        }
    } else {
        for (std::size_t i = 0; i < L; ++i)
            pos[i] = static_cast<double>(i);
    }
    std::vector<double> theta(Dh);
    const double base = std::log(10000.0);
    for (std::size_t k = 0; k < Dh; ++k) {
        theta[k] = std::exp(-2.0 * static_cast<double>(k) * base / static_cast<double>(D));
    }

    auto cos_cpu = allocate_size(L * Dh, input->dtype());
    auto sin_cpu = allocate_size(L * Dh, input->dtype());
    auto out_cpu = allocate_size(batch * L * D, input->dtype());

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        build_cos_sin_tables<T>(pos.data(), theta.data(), L, Dh,
                                reinterpret_cast<T*>(cos_cpu.ptr.get()),
                                reinterpret_cast<T*>(sin_cpu.ptr.get()));
        const auto& xs = std::get<CpuStorage>(input->storage());
        rope_forward<T>(reinterpret_cast<const T*>(xs.ptr.get()),
                        reinterpret_cast<const T*>(cos_cpu.ptr.get()),
                        reinterpret_cast<const T*>(sin_cpu.ptr.get()),
                        reinterpret_cast<T*>(out_cpu.ptr.get()), batch, L, D, interleaved);
    };
    if (input->dtype() == Dtype::F32)
        run(float{});
    else if (input->dtype() == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("rotary_pos_embedding").not_implemented("dtype must be F32 or F64");

    auto out = std::make_shared<TensorImpl>(Storage{std::move(out_cpu)}, input->shape(),
                                            input->dtype(), input->device(), false);

    if (!GradMode::is_enabled() || !input->requires_grad())
        return out;

    auto x_edge = detail::ensure_grad_fn(input);
    auto bwd = std::make_shared<RotaryPosEmbeddingBackward>();
    bwd->input_shapes_ = {input->shape()};
    bwd->out_shape_ = input->shape();
    bwd->dtype_ = input->dtype();
    bwd->device_ = input->device();
    bwd->input_tensors_ = {input};
    bwd->saved_inputs_ = {input->storage()};
    bwd->saved_cos_ = Storage{std::move(cos_cpu)};
    bwd->saved_sin_ = Storage{std::move(sin_cpu)};
    bwd->interleaved_ = interleaved;
    bwd->orig_shape_ = input->shape();
    bwd->set_next_edges(std::vector<Edge>{Edge(x_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{static_cast<std::int64_t>(input->version())});
    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

std::vector<Storage> RotaryPosEmbeddingBackward::apply(Storage grad_out) {
    const std::size_t ndim = orig_shape_.size();
    const std::size_t L = static_cast<std::size_t>(orig_shape_[ndim - 2]);
    const std::size_t D = static_cast<std::size_t>(orig_shape_[ndim - 1]);
    const std::size_t Dh = D / 2;
    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < ndim; ++i)
        batch *= static_cast<std::size_t>(orig_shape_[i]);

    if (device_ == Device::GPU) {
        const auto& gg = std::get<GpuStorage>(grad_out);
        const auto& cs = std::get<GpuStorage>(saved_cos_);
        const auto& ss = std::get<GpuStorage>(saved_sin_);

        ::mlx::core::Shape cos_b_shape;
        for (std::size_t i = 0; i + 2 < ndim; ++i)
            cos_b_shape.push_back(1);
        cos_b_shape.push_back(static_cast<int>(L));
        cos_b_shape.push_back(static_cast<int>(Dh));
        auto cos_b = ::mlx::core::reshape(*cs.arr, cos_b_shape);
        auto sin_b = ::mlx::core::reshape(*ss.arr, cos_b_shape);

        auto in_shape = gpu::to_mlx_shape(orig_shape_);
        ::mlx::core::array out_arr{0};
        if (interleaved_) {
            ::mlx::core::Shape g_pair_shape = in_shape;
            g_pair_shape.pop_back();
            g_pair_shape.push_back(static_cast<int>(Dh));
            g_pair_shape.push_back(2);
            auto g_re = ::mlx::core::reshape(*gg.arr, g_pair_shape);
            auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
            auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
            auto ge = ::mlx::core::take(g_re, idx0, /*axis=*/-1);
            auto go = ::mlx::core::take(g_re, idx1, /*axis=*/-1);
            // dx_e = cos*ge + sin*go ; dx_o = -sin*ge + cos*go
            auto dxe = ::mlx::core::add(::mlx::core::multiply(cos_b, ge),
                                        ::mlx::core::multiply(sin_b, go));
            auto dxo = ::mlx::core::subtract(::mlx::core::multiply(cos_b, go),
                                             ::mlx::core::multiply(sin_b, ge));
            auto dxe_e = ::mlx::core::expand_dims(dxe, -1);
            auto dxo_e = ::mlx::core::expand_dims(dxo, -1);
            auto stacked =
                ::mlx::core::concatenate(std::vector<::mlx::core::array>{dxe_e, dxo_e}, -1);
            out_arr = ::mlx::core::reshape(stacked, in_shape);
        } else {
            auto idx_first = ::mlx::core::astype(::mlx::core::arange(0, static_cast<int>(Dh), 1),
                                                 ::mlx::core::int64);
            auto idx_second = ::mlx::core::astype(
                ::mlx::core::arange(static_cast<int>(Dh), static_cast<int>(D), 1),
                ::mlx::core::int64);
            auto ga = ::mlx::core::take(*gg.arr, idx_first, /*axis=*/-1);
            auto gb = ::mlx::core::take(*gg.arr, idx_second, /*axis=*/-1);
            auto dxa = ::mlx::core::add(::mlx::core::multiply(cos_b, ga),
                                        ::mlx::core::multiply(sin_b, gb));
            auto dxb = ::mlx::core::subtract(::mlx::core::multiply(cos_b, gb),
                                             ::mlx::core::multiply(sin_b, ga));
            out_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{dxa, dxb}, -1);
        }
        return {Storage{gpu::wrap_mlx_array(std::move(out_arr), dtype_)}};
    }

    auto dx = allocate_size(batch * L * D, dtype_);
    const auto& gs = std::get<CpuStorage>(grad_out);
    const auto& cs = std::get<CpuStorage>(saved_cos_);
    const auto& ss = std::get<CpuStorage>(saved_sin_);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        rope_backward<T>(reinterpret_cast<const T*>(gs.ptr.get()),
                         reinterpret_cast<const T*>(cs.ptr.get()),
                         reinterpret_cast<const T*>(ss.ptr.get()),
                         reinterpret_cast<T*>(dx.ptr.get()), batch, L, D, interleaved_);
    };
    if (dtype_ == Dtype::F32)
        run(float{});
    else if (dtype_ == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("rotary_pos_embedding backward").not_implemented("dtype not supported");
    return {Storage{std::move(dx)}};
}

TensorImplPtr rotary_pos_embedding_op(const TensorImplPtr& input,
                                      const TensorImplPtr& position_ids_or_null,
                                      bool interleaved) {
    return RotaryPosEmbeddingBackward::forward(input, position_ids_or_null, interleaved);
}
LUCID_REGISTER_OP(RotaryPosEmbeddingBackward)

}  // namespace lucid
