// lucid/_C/compile/VjpEmitters/nn/Dropout.mm
//
// Dropout VJP.  Forward emitter (see :file:`OpEmitters/nn/Dropout.mm`)
// stashes the RNG-derived scaled mask via
// ``ctx.stash_saved(output_id, "mask", mask)`` during training mode
// emission.  This VJP reads that same MPSGraph tensor handle so both
// forward (``y = x * mask``) and backward (``dx = grad * mask``) see
// identical mask values within a dispatch.
//
// Eval / p==0 case: forward emits identity (no mask stashed).  The
// VJP detects that and falls through to grad pass-through.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string>
#include <string_view>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

template <const char* OP_NAME>
class DropoutVjpT final : public VjpEmitter {
public:
    std::string_view op_name() const override { return OP_NAME; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || node.outputs.empty() ||
            grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        if (g == nil || grad == nil) return false;

        // Look up the saved mask from forward emit.  Absent → eval-mode
        // identity dropout; backward is identity pass-through.
        void* mask_void =
            bctx.forward_ctx().resolve_saved(node.outputs[0].id, "mask");
        if (mask_void == nullptr) {
            // Identity pass-through (eval-mode or p==0).
            bctx.accumulate_grad(x_id, from_tensor(grad));
            return true;
        }
        MPSGraphTensor* mask = as_tensor(mask_void);
        // dx = grad * mask (mask already carries the 1/(1-p) scale).
        MPSGraphTensor* dx =
            [g multiplicationWithPrimaryTensor:grad
                                secondaryTensor:mask
                                           name:@"dropout_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

constexpr char kDropout[] = "dropout";
constexpr char kDropoutNd[] = "dropoutnd";
constexpr char kAlphaDropout[] = "alpha_dropout";

struct DropoutVjpRegistrar {
    DropoutVjpRegistrar() {
        register_vjp_emitter(std::make_unique<DropoutVjpT<kDropout>>());
        register_vjp_emitter(std::make_unique<DropoutVjpT<kDropoutNd>>());
        register_vjp_emitter(std::make_unique<DropoutVjpT<kAlphaDropout>>());
    }
};

[[maybe_unused]] static const DropoutVjpRegistrar g_dropout_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
