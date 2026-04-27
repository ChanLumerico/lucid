#include "Attention.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <mlx/ops.h>

#include "../backend/cpu/Blas.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/Exceptions.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/TensorImpl.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema ScaledDotProductAttentionBackward::schema_v1{
    "scaled_dot_product_attention", 1, AmpPolicy::ForceFP32, true};

namespace {

CpuStorage allocate_size(std::size_t numel, Dtype dt) {
    CpuStorage s;
    s.dtype  = dt;
    s.nbytes = numel * dtype_size(dt);
    s.ptr    = allocate_aligned_bytes(s.nbytes);
    return s;
}

// Flatten leading dims of an N-D tensor with shape [..., L, d] into
// (B = product of leading dims, L, d). Requires N >= 2.
struct Flat3 {
    std::size_t B;
    std::size_t L;
    std::size_t D;
};

Flat3 flatten_qkv(const Shape& s, const char* name) {
    if (s.size() < 2) {
        throw LucidError(std::string("attention: ") + name +
                          " must be at least 2-D ([..., L, d])");
    }
    std::size_t b = 1;
    for (std::size_t i = 0; i + 2 < s.size(); ++i) {
        b *= static_cast<std::size_t>(s[i]);
    }
    return {b,
            static_cast<std::size_t>(s[s.size() - 2]),
            static_cast<std::size_t>(s.back())};
}

template <typename T>
void gemm_typed(bool transA, bool transB, int M, int N, int K, T alpha,
                const T* A, int lda, const T* B, int ldb, T beta,
                T* C, int ldc);

template <>
void gemm_typed<float>(bool transA, bool transB, int M, int N, int K,
                       float alpha, const float* A, int lda,
                       const float* B, int ldb, float beta,
                       float* C, int ldc) {
    backend::cpu::sgemm(transA, transB, M, N, K, alpha, A, lda,
                        B, ldb, beta, C, ldc);
}

template <>
void gemm_typed<double>(bool transA, bool transB, int M, int N, int K,
                        double alpha, const double* A, int lda,
                        const double* B, int ldb, double beta,
                        double* C, int ldc) {
    backend::cpu::dgemm(transA, transB, M, N, K, alpha, A, lda,
                        B, ldb, beta, C, ldc);
}

// scores[b, i, j] += additive_mask[..., i, j], broadcasting per-batch when
// mask numel == L_q*L_k (one mask shared across batches) or per element when
// mask numel == B*L_q*L_k.
template <typename T>
void apply_additive_mask(T* scores, const T* mask,
                         std::size_t B, std::size_t Lq, std::size_t Lk,
                         std::size_t mask_numel) {
    const std::size_t per_batch = Lq * Lk;
    if (mask_numel == per_batch) {
        for (std::size_t b = 0; b < B; ++b) {
            T* s = scores + b * per_batch;
            for (std::size_t k = 0; k < per_batch; ++k) s[k] += mask[k];
        }
    } else if (mask_numel == B * per_batch) {
        const std::size_t total = B * per_batch;
        for (std::size_t k = 0; k < total; ++k) scores[k] += mask[k];
    } else {
        throw ShapeMismatch(Shape{static_cast<std::int64_t>(mask_numel)},
                            Shape{static_cast<std::int64_t>(B),
                                  static_cast<std::int64_t>(Lq),
                                  static_cast<std::int64_t>(Lk)},
                            "attention attn_mask");
    }
}

template <typename T>
void apply_bool_mask(T* scores, const std::uint8_t* mask,
                     std::size_t B, std::size_t Lq, std::size_t Lk,
                     std::size_t mask_numel) {
    const T neg_inf = -std::numeric_limits<T>::infinity();
    const std::size_t per_batch = Lq * Lk;
    if (mask_numel == per_batch) {
        for (std::size_t b = 0; b < B; ++b) {
            T* s = scores + b * per_batch;
            for (std::size_t k = 0; k < per_batch; ++k)
                if (mask[k]) s[k] = neg_inf;
        }
    } else if (mask_numel == B * per_batch) {
        const std::size_t total = B * per_batch;
        for (std::size_t k = 0; k < total; ++k)
            if (mask[k]) scores[k] = neg_inf;
    } else {
        throw ShapeMismatch(Shape{static_cast<std::int64_t>(mask_numel)},
                            Shape{static_cast<std::int64_t>(B),
                                  static_cast<std::int64_t>(Lq),
                                  static_cast<std::int64_t>(Lk)},
                            "attention attn_mask");
    }
}

template <typename T>
void apply_causal_mask(T* scores, std::size_t B,
                       std::size_t Lq, std::size_t Lk) {
    const T neg_inf = -std::numeric_limits<T>::infinity();
    for (std::size_t b = 0; b < B; ++b) {
        for (std::size_t i = 0; i < Lq; ++i) {
            T* row = scores + (b * Lq + i) * Lk;
            for (std::size_t j = i + 1; j < Lk; ++j) row[j] = neg_inf;
        }
    }
}

// In-place row-wise softmax over the last axis. When an entire row is -inf
// (e.g., a fully-masked query), define the output as 0 (rather than NaN).
template <typename T>
void softmax_rows(T* x, std::size_t rows, std::size_t cols) {
    for (std::size_t r = 0; r < rows; ++r) {
        T* row = x + r * cols;
        T m = row[0];
        for (std::size_t j = 1; j < cols; ++j) if (row[j] > m) m = row[j];
        if (!std::isfinite(m)) {
            for (std::size_t j = 0; j < cols; ++j) row[j] = T{0};
            continue;
        }
        T s = T{0};
        for (std::size_t j = 0; j < cols; ++j) {
            row[j] = std::exp(row[j] - m);
            s += row[j];
        }
        const T inv = (s > T{0}) ? T{1} / s : T{0};
        for (std::size_t j = 0; j < cols; ++j) row[j] *= inv;
    }
}

// dscores[r, j] = w[r, j] * (dw[r, j] - sum_l w[r, l] * dw[r, l]).
template <typename T>
void softmax_backward_rows(const T* w, const T* dw, T* dscores,
                           std::size_t rows, std::size_t cols) {
    for (std::size_t r = 0; r < rows; ++r) {
        const T* wr  = w  + r * cols;
        const T* dwr = dw + r * cols;
        T sum = T{0};
        for (std::size_t j = 0; j < cols; ++j) sum += wr[j] * dwr[j];
        T* dr = dscores + r * cols;
        for (std::size_t j = 0; j < cols; ++j) dr[j] = wr[j] * (dwr[j] - sum);
    }
}

// Build the output shape: leading dims from Q, then (L_q, d_v).
Shape build_output_shape(const Shape& q_shape, const Shape& v_shape) {
    Shape out;
    out.reserve(q_shape.size());
    for (std::size_t i = 0; i + 2 < q_shape.size(); ++i) out.push_back(q_shape[i]);
    out.push_back(q_shape[q_shape.size() - 2]);   // L_q
    out.push_back(v_shape.back());                // d_v
    return out;
}

template <typename T>
struct AttentionForwardCtx {
    std::size_t B;
    std::size_t Lq;
    std::size_t Lk;
    std::size_t Dk;
    std::size_t Dv;
    T scale;
};

template <typename T>
void compute_attention_forward(const T* Qp, const T* Kp, const T* Vp,
                               const TensorImplPtr& attn_mask, bool is_causal,
                               T* weights, T* output,
                               const AttentionForwardCtx<T>& ctx) {
    // Step 1: scores[b] = Q[b] @ K[b]^T * scale. Reuse `weights` as the
    // scratch buffer; softmax overwrites in place.
    for (std::size_t b = 0; b < ctx.B; ++b) {
        const T* Qb = Qp + b * ctx.Lq * ctx.Dk;
        const T* Kb = Kp + b * ctx.Lk * ctx.Dk;
        T* sb = weights + b * ctx.Lq * ctx.Lk;
        gemm_typed<T>(/*transA=*/false, /*transB=*/true,
                       static_cast<int>(ctx.Lq), static_cast<int>(ctx.Lk),
                       static_cast<int>(ctx.Dk),
                       ctx.scale, Qb, static_cast<int>(ctx.Dk),
                       Kb, static_cast<int>(ctx.Dk),
                       T{0}, sb, static_cast<int>(ctx.Lk));
    }

    // Step 2: optional masks.
    if (attn_mask) {
        const auto& ms = std::get<CpuStorage>(attn_mask->storage_);
        const std::size_t mn = attn_mask->numel();
        if (attn_mask->dtype_ == Dtype::Bool) {
            apply_bool_mask<T>(weights,
                               reinterpret_cast<const std::uint8_t*>(ms.ptr.get()),
                               ctx.B, ctx.Lq, ctx.Lk, mn);
        } else if (attn_mask->dtype_ == Dtype::F32 && std::is_same_v<T, float>) {
            apply_additive_mask<T>(weights,
                                    reinterpret_cast<const T*>(ms.ptr.get()),
                                    ctx.B, ctx.Lq, ctx.Lk, mn);
        } else if (attn_mask->dtype_ == Dtype::F64 && std::is_same_v<T, double>) {
            apply_additive_mask<T>(weights,
                                    reinterpret_cast<const T*>(ms.ptr.get()),
                                    ctx.B, ctx.Lq, ctx.Lk, mn);
        } else {
            throw NotImplementedError(
                "attention: attn_mask must be Bool or match input dtype");
        }
    }
    if (is_causal) apply_causal_mask<T>(weights, ctx.B, ctx.Lq, ctx.Lk);

    // Step 3: row-wise softmax over the L_k axis (in-place on `weights`).
    softmax_rows<T>(weights, ctx.B * ctx.Lq, ctx.Lk);

    // Step 4: output[b] = weights[b] @ V[b].
    for (std::size_t b = 0; b < ctx.B; ++b) {
        const T* Wb = weights + b * ctx.Lq * ctx.Lk;
        const T* Vb = Vp + b * ctx.Lk * ctx.Dv;
        T* Ob = output + b * ctx.Lq * ctx.Dv;
        gemm_typed<T>(false, false,
                       static_cast<int>(ctx.Lq), static_cast<int>(ctx.Dv),
                       static_cast<int>(ctx.Lk),
                       T{1}, Wb, static_cast<int>(ctx.Lk),
                       Vb, static_cast<int>(ctx.Dv),
                       T{0}, Ob, static_cast<int>(ctx.Dv));
    }
}

}  // namespace

namespace {

struct ForwardCore {
    TensorImplPtr output;
    Storage weights_storage;
    std::size_t B;
    std::size_t Lq;
    std::size_t Lk;
    std::size_t Dk;
    std::size_t Dv;
    Shape out_shape;
    Shape weights_shape;
};

ForwardCore run_forward(const TensorImplPtr& q, const TensorImplPtr& k,
                        const TensorImplPtr& v,
                        const TensorImplPtr& attn_mask, double scale,
                        bool is_causal) {
    if (!q || !k || !v) throw LucidError("attention: null input");
    if (q->device_ != k->device_ || q->device_ != v->device_)
        throw DeviceMismatch(std::string(device_name(q->device_)),
                              std::string(device_name(k->device_)),
                              "attention: Q/K/V device mismatch");
    if (q->dtype_ != k->dtype_ || q->dtype_ != v->dtype_)
        throw DtypeMismatch(std::string(dtype_name(q->dtype_)),
                            std::string(dtype_name(k->dtype_)),
                            "attention: Q/K/V dtype mismatch");
    if (q->shape_.size() < 2 || k->shape_.size() < 2 || v->shape_.size() < 2)
        throw LucidError("attention: Q/K/V must be at least 2-D");

    const auto fq = flatten_qkv(q->shape_, "Q");
    const auto fk = flatten_qkv(k->shape_, "K");
    const auto fv = flatten_qkv(v->shape_, "V");
    if (fq.B != fk.B || fq.B != fv.B)
        throw ShapeMismatch(q->shape_, k->shape_,
                             "attention: leading dims of Q/K/V must be equal");
    if (fq.D != fk.D)
        throw ShapeMismatch(q->shape_, k->shape_,
                             "attention: Q.last_dim must equal K.last_dim");
    if (fk.L != fv.L)
        throw ShapeMismatch(k->shape_, v->shape_,
                             "attention: K.L_k must equal V.L_k");

    OpScope scope{ScaledDotProductAttentionBackward::schema_v1.name,
                  q->device_, q->dtype_,
                  build_output_shape(q->shape_, v->shape_)};

    Shape out_shape = build_output_shape(q->shape_, v->shape_);
    Shape weights_shape;
    weights_shape.reserve(q->shape_.size());
    for (std::size_t i = 0; i + 2 < q->shape_.size(); ++i)
        weights_shape.push_back(q->shape_[i]);
    weights_shape.push_back(q->shape_[q->shape_.size() - 2]);  // L_q
    weights_shape.push_back(k->shape_[k->shape_.size() - 2]);  // L_k

    Storage weights_storage;
    Storage output_storage;

    if (q->device_ == Device::GPU) {
        const auto& gQ = std::get<GpuStorage>(q->storage_);
        const auto& gK = std::get<GpuStorage>(k->storage_);
        const auto& gV = std::get<GpuStorage>(v->storage_);
        const auto mlx_dt = gpu::to_mlx_dtype(q->dtype_);
        // K^T along last 2 dims; mlx::matmul supports batched N-D.
        auto k_t = ::mlx::core::swapaxes(*gK.arr, -2, -1);
        auto scores = ::mlx::core::matmul(*gQ.arr, k_t);
        auto scale_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(scale)), mlx_dt);
        scores = ::mlx::core::multiply(scores, scale_arr);

        auto neg_inf = ::mlx::core::astype(
            ::mlx::core::array(-std::numeric_limits<float>::infinity()), mlx_dt);
        if (attn_mask) {
            const auto& gM = std::get<GpuStorage>(attn_mask->storage_);
            if (attn_mask->dtype_ == Dtype::Bool) {
                scores = ::mlx::core::where(*gM.arr, neg_inf, scores);
            } else {
                scores = ::mlx::core::add(scores, *gM.arr);
            }
        }
        if (is_causal) {
            // Build [Lq, Lk] upper-tri mask (k>q diag=1) and apply.
            auto mask = ::mlx::core::triu(
                ::mlx::core::ones({static_cast<int>(fq.L), static_cast<int>(fk.L)},
                                   ::mlx::core::bool_), /*k=*/1);
            scores = ::mlx::core::where(mask, neg_inf, scores);
        }
        auto weights = ::mlx::core::softmax(scores, std::vector<int>{-1}, /*precise=*/true);
        auto output = ::mlx::core::matmul(weights, *gV.arr);
        weights_storage = Storage{gpu::wrap_mlx_array(std::move(weights), q->dtype_)};
        output_storage  = Storage{gpu::wrap_mlx_array(std::move(output),  q->dtype_)};
    } else {
        const std::size_t weights_numel = fq.B * fq.L * fk.L;
        const std::size_t output_numel  = fq.B * fq.L * fv.D;
        auto weights_cpu = allocate_size(weights_numel, q->dtype_);
        auto output_cpu  = allocate_size(output_numel,  q->dtype_);

        auto run = [&](auto type_tag) {
            using T = decltype(type_tag);
            AttentionForwardCtx<T> ctx{fq.B, fq.L, fk.L, fq.D, fv.D, static_cast<T>(scale)};
            const auto& qs = std::get<CpuStorage>(q->storage_);
            const auto& ks = std::get<CpuStorage>(k->storage_);
            const auto& vs = std::get<CpuStorage>(v->storage_);
            compute_attention_forward<T>(
                reinterpret_cast<const T*>(qs.ptr.get()),
                reinterpret_cast<const T*>(ks.ptr.get()),
                reinterpret_cast<const T*>(vs.ptr.get()),
                attn_mask, is_causal,
                reinterpret_cast<T*>(weights_cpu.ptr.get()),
                reinterpret_cast<T*>(output_cpu.ptr.get()), ctx);
        };
        if (q->dtype_ == Dtype::F32) run(float{});
        else if (q->dtype_ == Dtype::F64) run(double{});
        else throw NotImplementedError("attention: dtype not supported (F32/F64 only)");

        weights_storage = Storage{std::move(weights_cpu)};
        output_storage  = Storage{std::move(output_cpu)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(output_storage),
                                             out_shape, q->dtype_,
                                             q->device_, false);

    scope.set_flops(static_cast<std::int64_t>(2) *
                    static_cast<std::int64_t>(fq.B) *
                    static_cast<std::int64_t>(fq.L) *
                    static_cast<std::int64_t>(fk.L) *
                    static_cast<std::int64_t>(fq.D + fv.D));

    return ForwardCore{std::move(out), std::move(weights_storage),
                       fq.B, fq.L, fk.L, fq.D, fv.D,
                       std::move(out_shape), std::move(weights_shape)};
}

}  // namespace

TensorImplPtr ScaledDotProductAttentionBackward::forward(
    const TensorImplPtr& q, const TensorImplPtr& k, const TensorImplPtr& v,
    const TensorImplPtr& attn_mask, double scale, bool is_causal) {

    auto core = run_forward(q, k, v, attn_mask, scale, is_causal);

    if (!GradMode::is_enabled() ||
        !(q->requires_grad_ || k->requires_grad_ || v->requires_grad_))
        return core.output;

    auto q_edge = detail::ensure_grad_fn(q);
    auto k_edge = detail::ensure_grad_fn(k);
    auto v_edge = detail::ensure_grad_fn(v);
    auto bwd = std::make_shared<ScaledDotProductAttentionBackward>();
    bwd->input_shapes_   = {q->shape_, k->shape_, v->shape_};
    bwd->out_shape_      = core.out_shape;
    bwd->dtype_          = q->dtype_;
    bwd->device_         = q->device_;
    bwd->input_tensors_  = {q, k, v};
    bwd->saved_inputs_   = {q->storage_, k->storage_, v->storage_};
    bwd->saved_weights_  = std::move(core.weights_storage);
    bwd->scale_          = scale;
    bwd->orig_q_shape_   = q->shape_;
    bwd->orig_k_shape_   = k->shape_;
    bwd->orig_v_shape_   = v->shape_;
    bwd->set_next_edges(std::vector<Edge>{
        Edge(q_edge, 0), Edge(k_edge, 0), Edge(v_edge, 0)});
    bwd->set_saved_versions(std::vector<std::int64_t>{
        static_cast<std::int64_t>(q->version_),
        static_cast<std::int64_t>(k->version_),
        static_cast<std::int64_t>(v->version_)});
    core.output->grad_fn_       = std::move(bwd);
    core.output->is_leaf_       = false;
    core.output->requires_grad_ = true;
    return core.output;
}

std::vector<Storage> ScaledDotProductAttentionBackward::apply(Storage grad_out) {
    // Reconstruct flat dims.
    const auto fq = flatten_qkv(orig_q_shape_, "Q");
    const auto fk = flatten_qkv(orig_k_shape_, "K");
    const auto fv = flatten_qkv(orig_v_shape_, "V");

    const std::size_t Lq = fq.L, Lk = fk.L, Dk = fq.D, Dv = fv.D, B = fq.B;

    if (device_ == Device::GPU) {
        const auto& gQ = std::get<GpuStorage>(saved_inputs_[0]);
        const auto& gK = std::get<GpuStorage>(saved_inputs_[1]);
        const auto& gV = std::get<GpuStorage>(saved_inputs_[2]);
        const auto& gW = std::get<GpuStorage>(saved_weights_);
        const auto& gG = std::get<GpuStorage>(grad_out);
        const auto mlx_dt = gpu::to_mlx_dtype(dtype_);
        auto scale_arr = ::mlx::core::astype(::mlx::core::array(static_cast<float>(scale_)),
                                                mlx_dt);

        // dV = weights^T @ dout
        auto W_t = ::mlx::core::swapaxes(*gW.arr, -2, -1);
        auto dV  = ::mlx::core::matmul(W_t, *gG.arr);
        // dweights = dout @ V^T
        auto V_t = ::mlx::core::swapaxes(*gV.arr, -2, -1);
        auto dW  = ::mlx::core::matmul(*gG.arr, V_t);
        // softmax-bwd: dscores = w * (dW - sum(w*dW, axis=-1, keepdim=True))
        auto wdw = ::mlx::core::multiply(*gW.arr, dW);
        auto sum_wdw = ::mlx::core::sum(wdw, std::vector<int>{-1}, /*keepdims=*/true);
        auto dscores = ::mlx::core::multiply(*gW.arr,
                          ::mlx::core::subtract(dW, sum_wdw));
        // dQ = scale * dscores @ K
        auto dQ = ::mlx::core::multiply(scale_arr, ::mlx::core::matmul(dscores, *gK.arr));
        // dK = scale * dscores^T @ Q
        auto dscores_t = ::mlx::core::swapaxes(dscores, -2, -1);
        auto dK = ::mlx::core::multiply(scale_arr, ::mlx::core::matmul(dscores_t, *gQ.arr));

        return {Storage{gpu::wrap_mlx_array(std::move(dQ), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dK), dtype_)},
                Storage{gpu::wrap_mlx_array(std::move(dV), dtype_)}};
    }

    auto dQ = allocate_size(B * Lq * Dk, dtype_);
    auto dK = allocate_size(B * Lk * Dk, dtype_);
    auto dV = allocate_size(B * Lk * Dv, dtype_);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const auto& qs = std::get<CpuStorage>(saved_inputs_[0]);
        const auto& ks = std::get<CpuStorage>(saved_inputs_[1]);
        const auto& vs = std::get<CpuStorage>(saved_inputs_[2]);
        const auto& ws = std::get<CpuStorage>(saved_weights_);
        const auto& gs = std::get<CpuStorage>(grad_out);

        const T* Qp  = reinterpret_cast<const T*>(qs.ptr.get());
        const T* Kp  = reinterpret_cast<const T*>(ks.ptr.get());
        const T* Vp  = reinterpret_cast<const T*>(vs.ptr.get());
        const T* Wp  = reinterpret_cast<const T*>(ws.ptr.get());
        const T* Gp  = reinterpret_cast<const T*>(gs.ptr.get());
        T* dQp = reinterpret_cast<T*>(dQ.ptr.get());
        T* dKp = reinterpret_cast<T*>(dK.ptr.get());
        T* dVp = reinterpret_cast<T*>(dV.ptr.get());

        // Per-batch scratch for dweights and dscores.
        std::vector<T> dweights(Lq * Lk);
        std::vector<T> dscores(Lq * Lk);
        const T scale_t = static_cast<T>(scale_);

        for (std::size_t b = 0; b < B; ++b) {
            const T* Qb  = Qp  + b * Lq * Dk;
            const T* Kb  = Kp  + b * Lk * Dk;
            const T* Vb  = Vp  + b * Lk * Dv;
            const T* Wb  = Wp  + b * Lq * Lk;
            const T* Gb  = Gp  + b * Lq * Dv;
            T* dQb       = dQp + b * Lq * Dk;
            T* dKb       = dKp + b * Lk * Dk;
            T* dVb       = dVp + b * Lk * Dv;

            // dV[b] = weights[b]^T @ dout[b].
            //   shape: (Lk, Lq) @ (Lq, Dv) -> (Lk, Dv)
            gemm_typed<T>(/*transA=*/true, /*transB=*/false,
                           static_cast<int>(Lk), static_cast<int>(Dv),
                           static_cast<int>(Lq),
                           T{1}, Wb, static_cast<int>(Lk),
                           Gb, static_cast<int>(Dv),
                           T{0}, dVb, static_cast<int>(Dv));

            // dweights[b] = dout[b] @ V[b]^T.
            //   shape: (Lq, Dv) @ (Dv, Lk) -> (Lq, Lk)
            gemm_typed<T>(false, true,
                           static_cast<int>(Lq), static_cast<int>(Lk),
                           static_cast<int>(Dv),
                           T{1}, Gb, static_cast<int>(Dv),
                           Vb, static_cast<int>(Dv),
                           T{0}, dweights.data(), static_cast<int>(Lk));

            // dscores[b] = softmax_backward(weights[b], dweights[b]).
            softmax_backward_rows<T>(Wb, dweights.data(), dscores.data(),
                                      Lq, Lk);

            // dQ[b] = scale * dscores[b] @ K[b].
            //   shape: (Lq, Lk) @ (Lk, Dk) -> (Lq, Dk)
            gemm_typed<T>(false, false,
                           static_cast<int>(Lq), static_cast<int>(Dk),
                           static_cast<int>(Lk),
                           scale_t, dscores.data(), static_cast<int>(Lk),
                           Kb, static_cast<int>(Dk),
                           T{0}, dQb, static_cast<int>(Dk));

            // dK[b] = scale * dscores[b]^T @ Q[b].
            //   shape: (Lk, Lq) @ (Lq, Dk) -> (Lk, Dk)
            gemm_typed<T>(true, false,
                           static_cast<int>(Lk), static_cast<int>(Dk),
                           static_cast<int>(Lq),
                           scale_t, dscores.data(), static_cast<int>(Lk),
                           Qb, static_cast<int>(Dk),
                           T{0}, dKb, static_cast<int>(Dk));
        }
    };
    if (dtype_ == Dtype::F32) run(float{});
    else if (dtype_ == Dtype::F64) run(double{});
    else throw NotImplementedError("attention backward: dtype not supported");

    return {Storage{std::move(dQ)}, Storage{std::move(dK)}, Storage{std::move(dV)}};
}

TensorImplPtr scaled_dot_product_attention_op(
    const TensorImplPtr& q, const TensorImplPtr& k, const TensorImplPtr& v,
    const TensorImplPtr& attn_mask_or_null,
    double scale, bool is_causal) {
    return ScaledDotProductAttentionBackward::forward(
        q, k, v, attn_mask_or_null, scale, is_causal);
}

AttentionWithWeightsResult scaled_dot_product_attention_with_weights_op(
    const TensorImplPtr& q, const TensorImplPtr& k, const TensorImplPtr& v,
    const TensorImplPtr& attn_mask_or_null,
    double scale, bool is_causal) {
    // For the "return weights too" overload, run the forward without grad
    // tracking on weights (weights are an internal probability tensor; their
    // own grad_fn would double-count gradients via the chain rule).
    auto core = run_forward(q, k, v, attn_mask_or_null, scale, is_causal);

    Shape weights_shape = core.weights_shape;
    auto weights = std::make_shared<TensorImpl>(std::move(core.weights_storage),
                                                 std::move(weights_shape),
                                                 q->dtype_, q->device_, false);

    if (GradMode::is_enabled() &&
        (q->requires_grad_ || k->requires_grad_ || v->requires_grad_)) {
        // Re-run forward to wire up the autograd path for `output` so the
        // primary gradient chain is preserved. The first run above only
        // produced detached weights; this second pass attaches grad_fn to
        // `output`.
        auto with_grad = ScaledDotProductAttentionBackward::forward(
            q, k, v, attn_mask_or_null, scale, is_causal);
        return AttentionWithWeightsResult{std::move(with_grad), std::move(weights)};
    }
    return AttentionWithWeightsResult{std::move(core.output), std::move(weights)};
}

LUCID_REGISTER_OP(ScaledDotProductAttentionBackward)

}  // namespace lucid
