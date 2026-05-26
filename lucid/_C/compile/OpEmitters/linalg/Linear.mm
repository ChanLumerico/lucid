// lucid/_C/compile/OpEmitters/Linear.mm
//
// MPSGraph emitter for ``nn.Linear`` forward:
//
//     y = matmul(x, W^T) + b
//
// Inputs (in :func:`NaryKernel<LinearBackward,3>::wire_autograd` order):
//
//   - ``inputs[0]`` — x   (..., in_features)
//   - ``inputs[1]`` — W   (out_features, in_features)
//   - ``inputs[2]`` — b   (out_features,)  — optional (``-1`` when bias=False)
//
// Output:
//
//   - (..., out_features)
//
// The emitter is registered at process startup via a file-scope
// initialiser; :class:`MpsBuilder` finds it by op name ``"linear"``.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

class LinearEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "linear"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() < 2 || node.inputs.size() > 3 || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        TensorId W_id = node.inputs[1];
        TensorId b_id = node.inputs.size() == 3
            ? node.inputs[2]
            : TraceId::external_feed();
        if (x_id < 0 || W_id < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        MPSGraphTensor* W_t = (__bridge MPSGraphTensor*)ctx.resolve(W_id);
        if (x_t == nil || W_t == nil)
            return false;
        MPSGraphTensor* b_t = nil;
        if (b_id >= 0)
            b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);

        // W has shape (out, in); transpose to (in, out) for matmul.
        MPSGraphTensor* W_T = [graph transposeTensor:W_t
                                         permutation:@[@1, @0]
                                                name:@"linear_W_T"];

        // matmul(x, W^T) — MPSGraph supports batched 2-D and N-D.
        MPSGraphTensor* xW = [graph matrixMultiplicationWithPrimaryTensor:x_t
                                                          secondaryTensor:W_T
                                                                     name:@"linear_xWt"];

        // Optional bias add (broadcasts over leading batch dims).
        MPSGraphTensor* y = (b_t != nil)
            ? [graph additionWithPrimaryTensor:xW
                              secondaryTensor:b_t
                                         name:@"linear_out"]
            : xW;

        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

// File-scope registrar — runs once at .so load before any user code,
// so the emitter is available the first time :class:`MpsBuilder` looks
// up an op named "linear".
struct LinearEmitterRegistrar {
    LinearEmitterRegistrar() {
        register_emitter(std::make_unique<LinearEmitter>());
    }
};

[[maybe_unused]] static const LinearEmitterRegistrar g_linear_registrar;

}  // namespace

}  // namespace lucid::compile
