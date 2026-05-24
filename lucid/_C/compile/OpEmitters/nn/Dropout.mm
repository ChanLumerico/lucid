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
inline void* emit_identity(BuilderContext& ctx, const OpNode& node) {
    if (node.inputs.empty()) return nullptr;
    TensorId x_id = node.inputs[0];
    if (x_id < 0) return nullptr;
    return ctx.resolve(x_id);  // same MPSGraphTensor* — zero copy
}

// dropout / dropoutnd / alpha_dropout — passthrough when ``training``
// is false or ``p`` is zero; otherwise nullptr (eager fallback) until
// RNG-emitter coverage lands.
template <bool ALPHA>
class DropoutPassthroughEmitterT final : public OpEmitter {
public:
    explicit DropoutPassthroughEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        const bool training = bool_attr(node, "training", true);
        const double p = double_attr(node, "p", 0.5);
        if (!training || p == 0.0)
            return emit_identity(ctx, node);
        // RNG path not yet emit-able — caller falls back to eager.
        return nullptr;
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
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        const double p = double_attr(node, "p", 0.5);
        if (p == 0.0)
            return emit_identity(ctx, node);
        return nullptr;
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
