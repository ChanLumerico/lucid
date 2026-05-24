// lucid/_C/compile/OpEmitters/misc/Complex.mm
//
// Complex split / combine emitters.  Lucid's current ``Dtype::C64``
// path is eager-only (no MPSGraph complex storage), so these
// emitters treat their input as a real tensor and return:
//
//   - ``conj`` — identity (real input is its own conjugate)
//   - ``real`` — identity (real input is its own real part)
//   - ``imag`` — zeros with the same shape
//
// The ``reshapeTensor:withShape:`` call gives each output its own
// graph identity so the trace's ``outputs[0].id`` doesn't alias the
// input.  The full complex op (``complex(re, im)`` constructor) is a
// stub in :file:`Stubs.mm` because real-input pipeline can't
// express the two-storage backing path.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

class ConjEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "conj"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        return (__bridge void*)[g reshapeTensor:x withShape:x.shape name:@"conj"];
    }
};

class RealEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "real"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        return (__bridge void*)[g reshapeTensor:x withShape:x.shape name:@"real"];
    }
};

class ImagEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "imag"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        MPSGraphTensor* z = [g constantWithScalar:0.0 dataType:x.dataType];
        return (__bridge void*)[g broadcastTensor:z toShape:x.shape name:@"imag"];
    }
};

struct ComplexRegistrar {
    ComplexRegistrar() {
        register_emitter(std::make_unique<ConjEmitter>());
        register_emitter(std::make_unique<RealEmitter>());
        register_emitter(std::make_unique<ImagEmitter>());
    }
};

[[maybe_unused]] static const ComplexRegistrar g_complex_registrar;

}  // namespace

}  // namespace lucid::compile
