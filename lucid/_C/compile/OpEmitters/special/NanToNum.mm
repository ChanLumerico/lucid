// lucid/_C/compile/OpEmitters/misc/NanToNum.mm
//
// ``nan_to_num`` emitter — replace each special value with its
// configured substitute via a chain of ``selectWithPredicate``.
//
// Predicates:
//   - NaN  : x != x
//   - +Inf : x > FLT_MAX
//   - -Inf : x < -FLT_MAX
//
// The substitution values come from attrs ``nan``, ``posinf``,
// ``neginf`` (all double, default 0.0).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

class NanToNumEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "nan_to_num"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        double nan_v = double_attr(node, "nan", 0.0);
        double pos_v = double_attr(node, "posinf", 0.0);
        double neg_v = double_attr(node, "neginf", 0.0);
        MPSGraphTensor* nan_c = [g constantWithScalar:nan_v dataType:x.dataType];
        MPSGraphTensor* pos_c = [g constantWithScalar:pos_v dataType:x.dataType];
        MPSGraphTensor* neg_c = [g constantWithScalar:neg_v dataType:x.dataType];
        MPSGraphTensor* is_nan =
            [g notEqualWithPrimaryTensor:x secondaryTensor:x name:nil];
        MPSGraphTensor* max_c =
            [g constantWithScalar:3.4028234663852886e38 dataType:x.dataType];
        MPSGraphTensor* neg_max_c =
            [g constantWithScalar:-3.4028234663852886e38 dataType:x.dataType];
        MPSGraphTensor* is_pinf =
            [g greaterThanWithPrimaryTensor:x secondaryTensor:max_c name:nil];
        MPSGraphTensor* is_ninf =
            [g lessThanWithPrimaryTensor:x secondaryTensor:neg_max_c name:nil];
        MPSGraphTensor* tmp1 =
            [g selectWithPredicateTensor:is_nan
                     truePredicateTensor:nan_c
                    falsePredicateTensor:x
                                    name:nil];
        MPSGraphTensor* tmp2 =
            [g selectWithPredicateTensor:is_pinf
                     truePredicateTensor:pos_c
                    falsePredicateTensor:tmp1
                                    name:nil];
        return (__bridge void*)[g selectWithPredicateTensor:is_ninf
                                        truePredicateTensor:neg_c
                                       falsePredicateTensor:tmp2
                                                       name:@"nan_to_num"];
    }
};

struct NanToNumRegistrar {
    NanToNumRegistrar() {
        register_emitter(std::make_unique<NanToNumEmitter>());
    }
};

[[maybe_unused]] static const NanToNumRegistrar g_nan_to_num_registrar;

}  // namespace

}  // namespace lucid::compile
