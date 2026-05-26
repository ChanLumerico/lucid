// lucid/_C/compile/OpEmitters/nn/Dropout.mm
//
// Dropout-family emitters.  Coverage strategy:
//
//   * Inference / p==0 passthrough — emit an identity bind (the
//     downstream id resolves to the same MPSGraph tensor as the
//     input).  This is bit-exact: eager's ``!training || p==0``
//     branch clones the input verbatim.
//
//   * Training-mode dropout (RNG-driven mask) — return nullptr so
//     the compile-trace path falls back to eager.  Real RNG-emit
//     coverage is layered in once the RNG family lands; this file's
//     scope is the structural skeleton + the inference fast path.
//
// ``drop_block`` and ``drop_path`` are always called from a
// training-context Python wrapper, so they always need RNG — stubbed
// for now.  They're registered (rather than left unregistered) so a
// model that accidentally enters this branch in eager-fallback mode
// gets a clean ``eager-only`` cache entry instead of a hard compile
// error.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

// Helper: forward the single input tensor as the output (identity bind).
// Binds outputs[0].id to the same MPSGraphTensor* as the input — zero copy.
inline bool emit_identity(BuilderContext& ctx, const OpNode& node) {
    if (node.inputs.empty() || node.outputs.empty()) return false;
    TensorId x_id = node.inputs[0];
    if (x_id < 0) return false;
    void* x_t = ctx.resolve(x_id);
    if (x_t == nullptr) return false;
    ctx.bind(node.outputs[0].id, x_t);
    return true;
}

// dropout / dropoutnd / alpha_dropout
// ────────────────────────────────────
// Coverage:
//   * training == false  OR  p == 0  → identity bind (zero-copy passthrough,
//     applies to BERT / GPT / ViT in eval mode where dropout is a no-op).
//   * training == true   AND p  > 0  → eager fallback (return false).
//
// **Why training-mode dropout still falls back to eager.**  MPSGraph's
// ``randomUniformTensorWithShape:`` is *dispatch-deterministic* — the
// graph compiles a fixed random tensor handle whose values stay
// identical across dispatches of the same executable.  Compiling
// dropout this way would silently break the regularising effect: the
// same mask applies every step, every batch.  The
// ``test_dropout_training_produces_random_outputs`` test (which
// compares two consecutive calls) caught this empirically when the
// RNG mask path was prototyped (2026-05-26).
//
// The clean path requires either:
//   (a) MPSGraph's stateful Philox RNG with a state tensor that's
//       updated per dispatch — needs the saved-state side-table
//       (now in :class:`BuilderContext`) plus a state-feed plumbing
//       layer in :file:`_fused_step.py` that allocates the state and
//       rotates it each call.
//   (b) A dropout op that re-seeds via an external feed per call.
// Both are tractable additions but each is its own ~150-line plan.
// Until then, training-mode dropout falls back to eager; the dropout
// VJP only runs on the identity / p==0 path where dx == grad.
template <bool ALPHA>
class DropoutPassthroughEmitterT final : public OpEmitter {
public:
    explicit DropoutPassthroughEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const bool training = bool_attr(node, "training", true);
        const double p = double_attr(node, "p", 0.5);
        if (!training || p == 0.0)
            return emit_identity(ctx, node);
        // Training-mode + p > 0 → eager fallback (see header comment
        // above for why MPSGraph's random op is unsuitable).
        return false;
    }

private:
    std::string name_;
};

// drop_block / drop_path — no ``training`` parameter at the op level
// (their Python wrappers gate training-mode entry), so the C++ side
// only sees them when they really do need to apply a mask.  Honor a
// ``p == 0`` passthrough but otherwise fall back to eager.
template <const char* OP_NAME>
class DropMaskEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return OP_NAME; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        const double p = double_attr(node, "p", 0.5);
        if (p == 0.0)
            return emit_identity(ctx, node);
        return false;
    }
};

constexpr char kDropBlock[] = "drop_block";
constexpr char kDropPath[] = "drop_path";

struct DropoutEmitterRegistrar {
    DropoutEmitterRegistrar() {
        register_emitter(std::make_unique<DropoutPassthroughEmitterT<false>>("dropout"));
        register_emitter(std::make_unique<DropoutPassthroughEmitterT<false>>("dropoutnd"));
        register_emitter(std::make_unique<DropoutPassthroughEmitterT<true>>("alpha_dropout"));
        register_emitter(std::make_unique<DropMaskEmitter<kDropBlock>>());
        register_emitter(std::make_unique<DropMaskEmitter<kDropPath>>());
    }
};

[[maybe_unused]] static const DropoutEmitterRegistrar g_dropout_registrar;

}  // namespace

}  // namespace lucid::compile
