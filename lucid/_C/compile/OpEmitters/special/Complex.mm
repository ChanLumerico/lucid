// lucid/_C/compile/OpEmitters/special/Complex.mm
//
// The four operations that move between the real and complex lanes:
// ``complex`` builds a complex tensor out of two real ones, ``real`` and
// ``imag`` take a lane back out, and ``conj`` negates the imaginary one.
//
// These used to assume their input was real — ``real`` and ``conj``
// emitted the identity and ``imag`` emitted zeros, which is right for a
// real input and wrong for every other one.  That was safe only because
// no complex tensor could reach them: ``MpsBuilder``'s converter threw
// on C64, so a graph carrying one never got built.  Once the converter
// learned the type, the same three emitters would have started answering
// confidently with the wrong lane, so they are written properly here.
//
// Lucid's complex is not the "2-storage backing path" the old note
// described.  ``Dtype.h`` defines C64 as an interleaved pair of float32
// lanes in a single storage, eight bytes per element — the same bytes
// MPSGraph reads as ``MPSDataTypeComplexFloat32``.  Nothing is repacked;
// only the dtype case was missing.
//
// C128 has no counterpart: MPSGraph's complex types carry float32 and
// float16 lanes and nothing wider.  The converters name it and say so,
// and the graph fails to build before it reaches these emitters.
//
// Engine schemas (lucid/_C/ops/complex/): ``complex`` (two real inputs,
// complex output), ``real`` / ``imag`` (complex in, real lane out — the
// engine refuses a real input to either), ``conj`` (dtype-preserving,
// and wired for a real input too, where it is the identity).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

// Read off the tensor rather than the trace, so a node that declares
// the wrong dtype is declined here instead of building a graph that
// quietly means something else.
inline bool is_complex_tensor(MPSGraphTensor* t) {
    return t != nil &&
           (t.dataType == MPSDataTypeComplexFloat32 || t.dataType == MPSDataTypeComplexFloat16);
}

inline bool resolve_unary_complex_op(BuilderContext& ctx,
                                     const OpNode& node,
                                     MPSGraph* __strong& g,
                                     MPSGraphTensor* __strong& x) {
    if (node.inputs.empty() || node.outputs.empty() || node.inputs[0] < 0)
        return false;
    g = (__bridge MPSGraph*)ctx.graph();
    x = (__bridge MPSGraphTensor*)ctx.resolve(node.inputs[0]);
    return g != nil && x != nil;
}

inline bool bind_complex_output(BuilderContext& ctx, const OpNode& node, MPSGraphTensor* out) {
    if (out == nil)
        return false;
    ctx.bind(node.outputs[0].id, (__bridge void*)(out));
    return true;
}

// ── complex — two real tensors into one complex tensor.
class ComplexEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "complex"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 2 || node.outputs.empty())
            return false;
        const TensorId re_id = node.inputs[0];
        const TensorId im_id = node.inputs[1];
        if (re_id < 0 || im_id < 0)
            return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* re = (__bridge MPSGraphTensor*)ctx.resolve(re_id);
        MPSGraphTensor* im = (__bridge MPSGraphTensor*)ctx.resolve(im_id);
        if (g == nil || re == nil || im == nil)
            return false;
        // The engine widens half inputs and rejects a dtype mismatch
        // before tracing, so both lanes arrive real and equal.
        if (is_complex_tensor(re) || is_complex_tensor(im) || re.dataType != im.dataType)
            return false;
        return bind_complex_output(
            ctx, node, [g complexTensorWithRealTensor:re imaginaryTensor:im name:@"complex"]);
    }
};

// ── real / imag — one lane back out of a complex tensor.
template <bool IS_REAL>
class ComplexLaneEmitterT final : public OpEmitter {
public:
    explicit ComplexLaneEmitterT(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        MPSGraph* g = nil;
        MPSGraphTensor* x = nil;
        // ``real_op`` / ``imag_op`` require a complex input, so anything
        // else means the node was mis-declared upstream.  Declining is
        // the only honest answer: there is no lane to take out.
        if (!resolve_unary_complex_op(ctx, node, g, x) || !is_complex_tensor(x))
            return false;
        MPSGraphTensor* out = IS_REAL ? [g realPartOfTensor:x name:@"real"]
                                      : [g imaginaryPartOfTensor:x name:@"imag"];
        return bind_complex_output(ctx, node, out);
    }

private:
    std::string name_;
};

// ── conj — negate the imaginary lane, or pass a real tensor through.
class ConjEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "conj"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        MPSGraph* g = nil;
        MPSGraphTensor* x = nil;
        if (!resolve_unary_complex_op(ctx, node, g, x))
            return false;
        // The engine keeps a node for a real input so the gradient still
        // flows; there the conjugate is the identity.  The reshape gives
        // the output its own graph identity rather than aliasing the
        // input's binding.
        MPSGraphTensor* out = is_complex_tensor(x)
                                  ? [g conjugateWithTensor:x name:@"conj"]
                                  : [g reshapeTensor:x withShape:x.shape name:@"conj"];
        return bind_complex_output(ctx, node, out);
    }
};

class ViewAsRealEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "view_as_real"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        MPSGraph* g = nil;
        MPSGraphTensor* x = nil;
        if (!resolve_unary_complex_op(ctx, node, g, x) || !is_complex_tensor(x))
            return false;
        MPSGraphTensor* re = [g realPartOfTensor:x name:@"view_as_real_re"];
        MPSGraphTensor* im = [g imaginaryPartOfTensor:x name:@"view_as_real_im"];
        if (re == nil || im == nil)
            return false;
        return bind_complex_output(ctx, node,
                                   [g stackTensors:@[ re, im ] axis:-1 name:@"view_as_real"]);
    }
};

class ViewAsComplexEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "view_as_complex"; }

    bool emit(BuilderContext& ctx, const OpNode& node) override {
        MPSGraph* g = nil;
        MPSGraphTensor* x = nil;
        if (!resolve_unary_complex_op(ctx, node, g, x) || is_complex_tensor(x) || x.shape.count < 1)
            return false;
        MPSGraphTensor* re_slice = [g sliceTensor:x
                                        dimension:-1
                                            start:0
                                           length:1
                                             name:@"view_as_complex_re_slice"];
        MPSGraphTensor* im_slice = [g sliceTensor:x
                                        dimension:-1
                                            start:1
                                           length:1
                                             name:@"view_as_complex_im_slice"];
        if (re_slice == nil || im_slice == nil)
            return false;
        MPSGraphTensor* re = [g squeezeTensor:re_slice axis:-1 name:@"view_as_complex_re"];
        MPSGraphTensor* im = [g squeezeTensor:im_slice axis:-1 name:@"view_as_complex_im"];
        if (re == nil || im == nil)
            return false;
        return bind_complex_output(ctx, node,
                                   [g complexTensorWithRealTensor:re
                                                  imaginaryTensor:im
                                                             name:@"view_as_complex"]);
    }
};

struct ComplexRegistrar {
    ComplexRegistrar() {
        register_emitter(std::make_unique<ComplexEmitter>());
        register_emitter(std::make_unique<ComplexLaneEmitterT<true>>("real"));
        register_emitter(std::make_unique<ComplexLaneEmitterT<false>>("imag"));
        register_emitter(std::make_unique<ConjEmitter>());
        register_emitter(std::make_unique<ViewAsRealEmitter>());
        register_emitter(std::make_unique<ViewAsComplexEmitter>());
    }
};

[[maybe_unused]] static const ComplexRegistrar g_complex_registrar;

}  // namespace

}  // namespace lucid::compile
