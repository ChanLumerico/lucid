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
#include "../compile/Tracer.h"
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
                        bool is_causal,
                        bool need_weights) {
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
    scope.set_attr("scale", scale);
    scope.set_attr("is_causal", is_causal);
    scope.set_attr("has_mask", attn_mask != nullptr);

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
                          scale, is_causal, need_weights, q->dtype());

    // results[0] = weights storage; results[1] = output storage.
    auto out = std::make_shared<TensorImpl>(std::move(results[1]), out_shape, q->dtype(),
                                            q->device(), false);

    scope.set_flops(static_cast<std::int64_t>(2) * static_cast<std::int64_t>(fq.B) *
                    static_cast<std::int64_t>(fq.L) * static_cast<std::int64_t>(fk.L) *
                    static_cast<std::int64_t>(fq.D + fv.D));

    // Caller wire_autograd records on_op_io internally — no explicit call.

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
    // Output-only op: use the memory-efficient fused path (no dense weights).
    auto core = run_forward(q, k, v, attn_mask, scale, is_causal, /*need_weights=*/false);

    auto bwd = std::make_shared<ScaledDotProductAttentionBackward>();
    bwd->saved_weights_ = std::move(core.weights_storage);
    bwd->scale_ = scale;
    bwd->is_causal_ = is_causal;
    bwd->orig_q_shape_ = q->shape();
    bwd->orig_k_shape_ = k->shape();
    bwd->orig_v_shape_ = v->shape();
    if (attn_mask) {
        // Persist the mask so the GPU VJP backward can replay the exact masked
        // attention.  Copying the Storage retains the underlying buffer
        // independently of the mask tensor.
        bwd->has_mask_ = true;
        bwd->saved_mask_ = attn_mask->storage();
        bwd->mask_dtype_ = attn_mask->dtype();
        bwd->orig_mask_shape_ = attn_mask->shape();
    }

    // An *additive* (float) mask is a differentiable input when the caller
    // actually wants its gradient: relative-position bias tables are learned
    // parameters that reach the loss only through this argument.  A Bool
    // keep-mask selects rather than adds, and a constant padding or causal
    // mask needs no gradient — both get a null edge, which keeps the node
    // (and the compile trace) exactly as it was for those callers.
    const bool mask_is_differentiable =
        attn_mask && attn_mask->dtype() != Dtype::Bool && attn_mask->requires_grad();
    const TensorImplPtr& mask_edge = mask_is_differentiable ? attn_mask : TensorImplPtr{};
    bwd->mask_differentiable_ = mask_is_differentiable;

    kernel::NaryKernel<ScaledDotProductAttentionBackward, 4>::wire_autograd(
        std::move(bwd), {q, k, v, mask_edge}, core.output);

    // ``wire_autograd`` records its raw input array — including the null 4th
    // slot when there is no mask, or a Bool one that carries no gradient.  The
    // trace must not see that null, and a Bool mask must still appear so the
    // sdpa emitter can fold it in rather than bailing to eager.  Re-record the
    // exact operand list (``Tracer::on_op_io`` is last-write-wins), mirroring
    // the embedding op's handling of its non-diff ``indices`` input.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        if (attn_mask) {
            trc->on_op_io({q, k, v, attn_mask}, core.output);
        } else {
            trc->on_op_io({q, k, v}, core.output);
        }
    }
    return core.output;
}

std::vector<Storage> ScaledDotProductAttentionBackward::apply(Storage grad_out) {
    const Storage* mask_ptr = has_mask_ ? &saved_mask_ : nullptr;
    auto grads = backend::Dispatcher::for_device(device_).sdpa_backward(
        grad_out, saved_inputs_[0], saved_inputs_[1], saved_inputs_[2], saved_weights_, mask_ptr,
        orig_q_shape_, orig_k_shape_, orig_v_shape_, mask_dtype_, scale_, is_causal_, dtype_);

    // The backends hand back dM at the *full* score shape (…, L_q, L_k); the
    // caller's mask may have broadcast into it, so sum the expanded axes away.
    // A Bool keep-mask has a null edge and a placeholder slot — leave it be.
    if (grads.size() >= 4 && mask_differentiable_ && !orig_mask_shape_.empty()) {
        Shape score_shape;
        score_shape.reserve(orig_q_shape_.size());
        for (std::size_t i = 0; i + 2 < orig_q_shape_.size(); ++i)
            score_shape.push_back(orig_q_shape_[i]);
        score_shape.push_back(orig_q_shape_[orig_q_shape_.size() - 2]);
        score_shape.push_back(orig_k_shape_[orig_k_shape_.size() - 2]);

        if (score_shape != orig_mask_shape_) {
            grads[3] =
                reduce_grad_to_shape(grads[3], score_shape, orig_mask_shape_, dtype_, device_);
        }
    }
    return grads;
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
    // Weights variant: materialize the dense softmax weight matrix for the caller.
    auto core = run_forward(q, k, v, attn_mask_or_null, scale, is_causal, /*need_weights=*/true);

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
