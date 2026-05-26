// lucid/_C/compile/OpEmitters/linalg/Matmul.mm
//
// Two-input matrix-multiplication emitter.  Pulled out of the
// elementwise/Arith.mm bucket so it lives next to the rest of the
// linear-algebra primitives (Linear / Inner / Outer / Tensordot /
// MatrixOps).
//
// Engine schema: ``matmul`` (lucid/_C/ops/bfunc/Matmul.cpp).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

class MatmulEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "matmul"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        TensorId a_id = node.inputs[0];
        TensorId b_id = node.inputs[1];
        if (a_id < 0 || b_id < 0)
            return false;
        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* a_t = (__bridge MPSGraphTensor*)ctx.resolve(a_id);
        MPSGraphTensor* b_t = (__bridge MPSGraphTensor*)ctx.resolve(b_id);
        if (graph == nil || a_t == nil || b_t == nil)
            return false;
        MPSGraphTensor* y =
            [graph matrixMultiplicationWithPrimaryTensor:a_t
                                          secondaryTensor:b_t
                                                     name:@"matmul"];
        ctx.bind(node.outputs[0].id, (__bridge void*)y);
        return true;
    }
};

struct MatmulEmitterRegistrar {
    MatmulEmitterRegistrar() {
        register_emitter(std::make_unique<MatmulEmitter>());
    }
};

[[maybe_unused]] static const MatmulEmitterRegistrar g_matmul_registrar;

}  // namespace

}  // namespace lucid::compile
