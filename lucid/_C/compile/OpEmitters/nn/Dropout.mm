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
//   * training == true   AND p  > 0  → emit identity (= eager fallback would
//     pollute the trace, see :class:`DropoutPassthroughEmitterT::emit`).
//
// **Why training-mode dropout falls back via this emitter rather than
// running stateful Philox here.**  The standard ``"dropout"`` trace node
// records a single input (``x``).  MPSGraph's stateful Philox API
// requires a second tensor (the Philox state) as input + emits a second
// output (the new state).  That schema is incompatible with the
// single-in/single-out ``"dropout"`` op — so the Python wrapper in
// :file:`lucid/nn/functional/dropout.py` routes training-mode dropout
// through a **sibling op name ``"dropout_stateful"``** that carries the
// 2-input/2-output schema and is handled by the
// :class:`DropoutStatefulEmitter` below.  The plain ``"dropout"`` op
// only sees inference / p==0 calls in the compile path; training-mode
// dispatches go through ``dropout_stateful``.
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
        // Training-mode + p > 0 on the *plain* dropout op only reaches
        // here when the Python wrapper missed the tracer-routing check
        // (e.g. eager-only callers).  Fall back to eager — the dropout
        // sibling ``"dropout_stateful"`` op handles the compile path.
        return false;
    }

private:
    std::string name_;
};

// dropout_stateful — training-mode dropout with explicit Philox state I/O.
// ─────────────────────────────────────────────────────────────────────────
// Sibling of the plain ``"dropout"`` op designed exclusively for the
// compile path's training-mode dispatch.  Two inputs (``x``,
// ``state_in``) and two outputs (``y``, ``state_out``).  Uses MPSGraph's
// 2-output stateful Philox API
// ``randomTensorWithShape:descriptor:stateTensor:`` which returns
// ``(uniform, new_state)`` — by feeding the new state back into the
// state buffer between dispatches (either as an in/out feed pair or as
// an MPSGraph variable via
// :func:`compile_generic_fused_step_with_vars`) the executable
// produces genuinely-per-dispatch varying masks instead of the
// dispatch-deterministic ones the stateless seed-only path would give.
//
// Emit order:
//   1. mask = (uniform > p) * (1 / (1 - p))   ← inverted-dropout scale
//   2. y = x * mask                            ← bound to outputs[0]
//   3. state_out bound to outputs[1]
//   4. mask stashed via ``ctx.stash_saved(outputs[0].id, "mask", mask)``
//      so the VJP can multiply through without recomputing random
//      values (the same MPSGraph node is referenced by both
//      forward and backward subgraphs).
//
// Inference / p==0 fast path: if the Python wrapper still routed
// through this op for some reason in eval mode, fall back to the
// identity bind (state_out becomes a clone of state_in via identity
// pass-through — no randomness needed).
class DropoutStatefulEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "dropout_stateful"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.outputs.size() < 2)
            return false;
        const bool training = bool_attr(node, "training", true);
        const double p = double_attr(node, "p", 0.5);

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        if (graph == nil) return false;

        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
        MPSGraphTensor* state_in = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[1]);
        if (x == nil || state_in == nil) return false;

        // Inference / p==0 → identity bind (y = x; state_out = state_in).
        // Avoids spurious RNG cost when this op gets dispatched eval-mode.
        if (!training || p == 0.0) {
            ctx.bind(node.outputs[0].id, (__bridge void*)x);
            ctx.bind(node.outputs[1].id, (__bridge void*)state_in);
            return true;
        }

        // Training: stateful Philox via 2-output random API.
        const MPSDataType dtype = x.dataType;
        MPSGraphRandomOpDescriptor* desc =
            [MPSGraphRandomOpDescriptor descriptorWithDistribution:
                                            MPSGraphRandomDistributionUniform
                                                          dataType:dtype];
        desc.min = 0.0f;
        desc.max = 1.0f;

        // ``randomTensorWithShape:descriptor:stateTensor:`` returns an
        // NSArray<MPSGraphTensor*> of length 2: [uniform_values, new_state].
        // The shape argument here is the *output* shape — same as ``x``.
        NSArray<MPSGraphTensor*>* rng_pair =
            [graph randomTensorWithShape:x.shape
                              descriptor:desc
                             stateTensor:state_in
                                    name:@"dropout_stateful_rng"];
        if (rng_pair == nil || rng_pair.count < 2) return false;

        MPSGraphTensor* uniform = rng_pair[0];
        MPSGraphTensor* state_out = rng_pair[1];

        // mask = (uniform > p) * (1 / (1 - p)) — inverted-dropout scale.
        // The boolean comparison gets cast back to the input dtype
        // before the multiplicative scaling so MPSGraph's downstream
        // ``multiplicationWithPrimaryTensor:`` sees matching dtypes.
        MPSGraphTensor* p_const =
            [graph constantWithScalar:p dataType:dtype];
        MPSGraphTensor* keep_inv =
            [graph constantWithScalar:(1.0 / (1.0 - p)) dataType:dtype];
        MPSGraphTensor* keep_bool =
            [graph greaterThanWithPrimaryTensor:uniform
                                secondaryTensor:p_const
                                           name:@"dropout_keep_bool"];
        MPSGraphTensor* keep_f =
            [graph castTensor:keep_bool toType:dtype name:@"dropout_keep_cast"];
        MPSGraphTensor* mask =
            [graph multiplicationWithPrimaryTensor:keep_f
                                   secondaryTensor:keep_inv
                                              name:@"dropout_mask"];

        // y = x * mask.
        MPSGraphTensor* y =
            [graph multiplicationWithPrimaryTensor:x
                                   secondaryTensor:mask
                                              name:@"dropout_y"];

        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        ctx.bind(node.outputs[1].id, (__bridge void*)state_out);

        // Stash mask so the matching VJP (registered under
        // ``"dropout_stateful"``) can multiply through without
        // recomputing random values — MPSGraph evaluates each node
        // exactly once per dispatch.
        ctx.stash_saved(node.outputs[0].id, "mask", (__bridge void*)mask);

        return true;
    }
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
        register_emitter(std::make_unique<DropoutStatefulEmitter>());
    }
};

[[maybe_unused]] static const DropoutEmitterRegistrar g_dropout_registrar;

}  // namespace

}  // namespace lucid::compile
