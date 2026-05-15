// lucid/_C/nn/Attention.cpp
//
// Implementation of Scaled Dot-Product Attention forward and backward.
//
// The shared run_forward() helper validates inputs, dispatches to
// IBackend::sdpa_forward, and returns a ForwardCore bundle containing the
// output tensor, the attention weight storage, batch/sequence/dim sizes, and
// the output/weights shapes.  Both public entry points call run_forward and
// then either wire the backward node (forward()) or expose the weights tensor
// as a second return value (scaled_dot_product_attention_with_weights_op()).
//
// FLOP estimate: 2 * B * Lq * Lk * (Dk + Dv) — covers Q@K^T and W@V.

#include "Attention.h"

#include <vector>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema ScaledDotProductAttentionBackward::schema_v1{"scaled_dot_product_attention", 1,
                                                            AmpPolicy::ForceFP32, true};

namespace {

// Batch size B plus the last two dims (L, D) extracted from a Q/K/V shape.
struct Flat3 {
    std::size_t B;
    std::size_t L;
    std::size_t D;
};

// Flatten all leading dims of s (except last two) into a single batch B.
// Throws if s has fewer than 2 dimensions.
Flat3 flatten_qkv(const Shape& s, const char* name) {
    if (s.size() < 2) {
        ErrorBuilder("attention").fail(std::string(name) + " must be at least 2-D ([..., L, d])");
    }
    std::size_t b = 1;
    for (std::size_t i = 0; i + 2 < s.size(); ++i)
        b *= static_cast<std::size_t>(s[i]);
    return {b, static_cast<std::size_t>(s[s.size() - 2]), static_cast<std::size_t>(s.back())};
}

// Construct the SDPA output shape (..., Lq, Dv) from the Q and V shapes.
Shape build_output_shape(const Shape& q_shape, const Shape& v_shape) {
    Shape out;
    out.reserve(q_shape.size());
    for (std::size_t i = 0; i + 2 < q_shape.size(); ++i)
        out.push_back(q_shape[i]);
    out.push_back(q_shape[q_shape.size() - 2]);
    out.push_back(v_shape.back());
    return out;
}

// All data produced by the shared forward kernel, bundled for the two callers.
struct ForwardCore {
    TensorImplPtr output;
    Storage weights_storage;  // Attention weights W; may be {1} on GPU path.
    std::size_t B;
    std::size_t Lq;
    std::size_t Lk;
    std::size_t Dk;
    std::size_t Dv;
    Shape out_shape;
    Shape weights_shape;  // (..., Lq, Lk).
};

// Validate Q/K/V shapes and dtypes, dispatch to IBackend::sdpa_forward, and
// return a ForwardCore.  The backend returns [weights, output] in results[].
ForwardCore run_forward(const TensorImplPtr& q,
                        const TensorImplPtr& k,
                        const TensorImplPtr& v,
                        const TensorImplPtr& attn_mask,
                        double scale,
                        bool is_causal) {
    if (!q || !k || !v)
        ErrorBuilder("attention").fail("null input");
    if (q->device() != k->device() || q->device() != v->device())
        throw DeviceMismatch(std::string(device_name(q->device())),
                             std::string(device_name(k->device())),
                             "attention: Q/K/V device mismatch");
    if (q->dtype() != k->dtype() || q->dtype() != v->dtype())
        throw DtypeMismatch(std::string(dtype_name(q->dtype())),
                            std::string(dtype_name(k->dtype())), "attention: Q/K/V dtype mismatch");
    if (q->shape().size() < 2 || k->shape().size() < 2 || v->shape().size() < 2)
        ErrorBuilder("attention").fail("Q/K/V must be at least 2-D");

    const auto fq = flatten_qkv(q->shape(), "Q");
    const auto fk = flatten_qkv(k->shape(), "K");
    const auto fv = flatten_qkv(v->shape(), "V");
    if (fq.B != fk.B || fq.B != fv.B)
        throw ShapeMismatch(q->shape(), k->shape(),
                            "attention: leading dims of Q/K/V must be equal");
    if (fq.D != fk.D)
        throw ShapeMismatch(q->shape(), k->shape(), "attention: Q.last_dim must equal K.last_dim");
    if (fk.L != fv.L)
        throw ShapeMismatch(k->shape(), v->shape(), "attention: K.L_k must equal V.L_k");

    OpScopeFull scope{ScaledDotProductAttentionBackward::schema_v1.name, q->device(), q->dtype(),
                      build_output_shape(q->shape(), v->shape())};

    Shape out_shape = build_output_shape(q->shape(), v->shape());
    Shape weights_shape;
    weights_shape.reserve(q->shape().size());
    for (std::size_t i = 0; i + 2 < q->shape().size(); ++i)
        weights_shape.push_back(q->shape()[i]);
    weights_shape.push_back(q->shape()[q->shape().size() - 2]);
    weights_shape.push_back(k->shape()[k->shape().size() - 2]);

    const Storage* mask_storage = attn_mask ? &attn_mask->storage() : nullptr;
    // sdpa_forward returns {weights, output}.  On the GPU path weights may be a
    // dummy {1}-shape tensor; the backward detects this and recomputes W.
    auto results =
        backend::Dispatcher::for_device(q->device())
            .sdpa_forward(q->storage(), k->storage(), v->storage(), mask_storage, q->shape(),
                          k->shape(), v->shape(), attn_mask ? attn_mask->dtype() : Dtype::F32,
                          attn_mask ? static_cast<std::size_t>(attn_mask->numel()) : std::size_t{0},
                          scale, is_causal, q->dtype());

    // results[0] = weights storage; results[1] = output storage.
    auto out = std::make_shared<TensorImpl>(std::move(results[1]), out_shape, q->dtype(),
                                            q->device(), false);

    scope.set_flops(static_cast<std::int64_t>(2) * static_cast<std::int64_t>(fq.B) *
                    static_cast<std::int64_t>(fq.L) * static_cast<std::int64_t>(fk.L) *
                    static_cast<std::int64_t>(fq.D + fv.D));

    return ForwardCore{std::move(out),       std::move(results[0]),   fq.B, fq.L, fk.L, fq.D, fv.D,
                       std::move(out_shape), std::move(weights_shape)};
}

}  // namespace

TensorImplPtr ScaledDotProductAttentionBackward::forward(const TensorImplPtr& q,
                                                         const TensorImplPtr& k,
                                                         const TensorImplPtr& v,
                                                         const TensorImplPtr& attn_mask,
                                                         double scale,
                                                         bool is_causal) {
    auto core = run_forward(q, k, v, attn_mask, scale, is_causal);

    auto bwd = std::make_shared<ScaledDotProductAttentionBackward>();
    bwd->saved_weights_ = std::move(core.weights_storage);
    bwd->scale_ = scale;
    bwd->orig_q_shape_ = q->shape();
    bwd->orig_k_shape_ = k->shape();
    bwd->orig_v_shape_ = v->shape();
    kernel::NaryKernel<ScaledDotProductAttentionBackward, 3>::wire_autograd(std::move(bwd),
                                                                            {q, k, v}, core.output);
    return core.output;
}

std::vector<Storage> ScaledDotProductAttentionBackward::apply(Storage grad_out) {
    return backend::Dispatcher::for_device(device_).sdpa_backward(
        grad_out, saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], saved_weights_,
        orig_q_shape_, orig_k_shape_, orig_v_shape_, scale_, dtype_);
}

TensorImplPtr scaled_dot_product_attention_op(const TensorImplPtr& q,
                                              const TensorImplPtr& k,
                                              const TensorImplPtr& v,
                                              const TensorImplPtr& attn_mask_or_null,
                                              double scale,
                                              bool is_causal) {
    return ScaledDotProductAttentionBackward::forward(q, k, v, attn_mask_or_null, scale, is_causal);
}

std::vector<TensorImplPtr>
scaled_dot_product_attention_with_weights_op(const TensorImplPtr& q,
                                             const TensorImplPtr& k,
                                             const TensorImplPtr& v,
                                             const TensorImplPtr& attn_mask_or_null,
                                             double scale,
                                             bool is_causal) {
    auto core = run_forward(q, k, v, attn_mask_or_null, scale, is_causal);

    Shape weights_shape = core.weights_shape;
    auto weights = std::make_shared<TensorImpl>(
        std::move(core.weights_storage), std::move(weights_shape), q->dtype(), q->device(), false);

    if (GradMode::is_enabled() &&
        (q->requires_grad() || k->requires_grad() || v->requires_grad())) {
        auto with_grad = ScaledDotProductAttentionBackward::forward(q, k, v, attn_mask_or_null,
                                                                    scale, is_causal);
        return {std::move(with_grad), std::move(weights)};
    }
    return {std::move(core.output), std::move(weights)};
}

LUCID_REGISTER_OP(ScaledDotProductAttentionBackward)

}  // namespace lucid
