// lucid/_C/compile/OpEmitters/elementwise/DtypeCast.mm
//
// Element-wise dtype conversion: ``astype``.  Each output element
// comes from the same-position input element with the target dtype.
//
// Engine schema: ``astype`` (lucid/_C/ops/ufunc/Astype.cpp) — attr
// ``dst_dtype`` (int).  Falls back to the output meta's dtype when
// the attr is missing (always set by the tracer).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>

#include "../../../core/Dtype.h"
#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

inline MPSDataType lucid_dtype_to_mps(Dtype dt) {
    switch (dt) {
        case Dtype::F32:
            return MPSDataTypeFloat32;
        case Dtype::F16:
            return MPSDataTypeFloat16;
        case Dtype::I32:
            return MPSDataTypeInt32;
        case Dtype::I64:
            return MPSDataTypeInt64;
        case Dtype::Bool:
            return MPSDataTypeBool;
        default:
            return MPSDataTypeFloat32;
    }
}

class AstypeEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "astype"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1 || node.outputs.empty())
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return false;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return false;

        // Prefer the recorded ``dst_dtype`` attr; fall back to the
        // output meta's dtype (always set by the tracer).
        MPSDataType dst = lucid_dtype_to_mps(node.outputs[0].dtype);
        auto it = node.attrs.find("dst_dtype");
        if (it != node.attrs.end()) {
            if (const auto* p = std::get_if<std::int64_t>(&it->second)) {
                Dtype dt = static_cast<Dtype>(static_cast<int>(*p));
                dst = lucid_dtype_to_mps(dt);
            }
        }
        ctx.bind(node.outputs[0].id, (__bridge void*)([graph castTensor:x_t toType:dst name:@"astype"]));
        return true;
    }
};

struct DtypeCastEmitterRegistrar {
    DtypeCastEmitterRegistrar() {
        register_emitter(std::make_unique<AstypeEmitter>());
    }
};

[[maybe_unused]] static const DtypeCastEmitterRegistrar g_dtype_cast_registrar;

}  // namespace

}  // namespace lucid::compile
