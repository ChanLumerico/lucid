// lucid/_C/compile/VjpEmitters/elementwise/Arith.mm
//
// VJPs for the binary-elementwise family.  Mirrors the eager
// implementations in :file:`lucid/_C/ops/bfunc/{Add,Sub,Mul,Div,Pow}.cpp`,
// translated to MPSGraph subgraph emission.
//
// Standard recipe:
//   1. Pull ``grad_out = grad_outs[0]`` (single-output ops).
//   2. Look up forward activations a / b via :func:`BackwardContext::forward`.
//   3. Emit the gradient subgraph.
//   4. Call :func:`BackwardContext::unreduce` to broadcast-back to each
//      input's shape, then :func:`accumulate_grad`.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// Pair of full-shape gradients (before unreduce) for a binary VJP.
struct BinaryGradPair {
    MPSGraphTensor* da = nil;
    MPSGraphTensor* db = nil;
};

// All binary-arith VJPs share the same accumulation pattern: compute
// the full-shape gradient w.r.t. each input, then unreduce back to the
// pre-broadcast shape and accumulate.  Factored as a helper so each
// per-op body stays the math-only minimum.  ``body(c) -> BinaryGradPair``.
//
// Mixed-dtype reconciliation (autocast): the forward emit casts both
// operands to the recorded output dtype, but ``bctx.forward(*)`` returns
// the original (uncast) producer binding, so ``c.a`` / ``c.b`` may have
// different dtypes than each other and ``c.go``.  All arithmetic in the
// gradient body must run in ``c.go.dataType`` (the chain's dtype), so
// we cast both forward activations to that dtype before invoking the
// per-op body.
template <class Body>
inline bool accumulate_binary(BackwardContext& bctx, const OpNode& node,
                               const std::vector<void*>& grad_outs, Body body) {
    BinaryVjpCtx c = unpack_binary_vjp(bctx, node, grad_outs);
    if (!c.ok) return false;
    const MPSDataType chain_dt = c.go.dataType;
    c.a = cast_if_needed(c.g, c.a, chain_dt);
    c.b = cast_if_needed(c.g, c.b, chain_dt);
    BinaryGradPair g = body(c);
    if (g.da == nil || g.db == nil) return false;
    bctx.accumulate_grad(c.a_id,
                          bctx.unreduce(from_tensor(g.da), c.a_shape, c.out_shape));
    bctx.accumulate_grad(c.b_id,
                          bctx.unreduce(from_tensor(g.db), c.b_shape, c.out_shape));
    return true;
}

// ────────────────────────────────────────────────────────────────────
// add: dA = dB = grad_out (unreduce to each input shape).
// ────────────────────────────────────────────────────────────────────
class AddVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "add"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return accumulate_binary(bctx, node, grad_outs,
            [](const BinaryVjpCtx& c) -> BinaryGradPair {
                return { c.go, c.go };
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// sub: dA = grad_out, dB = -grad_out (unreduce).
// ────────────────────────────────────────────────────────────────────
class SubVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "sub"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return accumulate_binary(bctx, node, grad_outs,
            [](const BinaryVjpCtx& c) -> BinaryGradPair {
                return { c.go, [c.g negativeWithTensor:c.go name:nil] };
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// mul: dA = grad * b, dB = grad * a (unreduce).  Product rule.
// ────────────────────────────────────────────────────────────────────
class MulVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "mul"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return accumulate_binary(bctx, node, grad_outs,
            [](const BinaryVjpCtx& c) -> BinaryGradPair {
                return {
                    [c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:c.b name:nil],
                    [c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:c.a name:nil]
                };
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// div: dA = grad / b, dB = -(grad * a) / b² (unreduce).  Quotient rule.
// ────────────────────────────────────────────────────────────────────
class DivVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "div"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return accumulate_binary(bctx, node, grad_outs,
            [](const BinaryVjpCtx& c) -> BinaryGradPair {
                MPSGraphTensor* da =
                    [c.g divisionWithPrimaryTensor:c.go secondaryTensor:c.b name:nil];
                MPSGraphTensor* g_times_a =
                    [c.g multiplicationWithPrimaryTensor:c.go secondaryTensor:c.a name:nil];
                MPSGraphTensor* b_sq =
                    [c.g multiplicationWithPrimaryTensor:c.b secondaryTensor:c.b name:nil];
                MPSGraphTensor* div_b_sq =
                    [c.g divisionWithPrimaryTensor:g_times_a secondaryTensor:b_sq name:nil];
                MPSGraphTensor* db = [c.g negativeWithTensor:div_b_sq name:nil];
                return { da, db };
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// pow: dA = b * a^(b-1) * grad, dB = log(a) * a^b * grad (unreduce).
// ────────────────────────────────────────────────────────────────────
class PowVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "pow"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        return accumulate_binary(bctx, node, grad_outs,
            [](const BinaryVjpCtx& c) -> BinaryGradPair {
                // c.a / c.b already cast to chain dtype by accumulate_binary.
                MPSDataType dt = c.go.dataType;
                MPSGraphTensor* one = [c.g constantWithScalar:1.0 dataType:dt];
                MPSGraphTensor* bm1 =
                    [c.g subtractionWithPrimaryTensor:c.b secondaryTensor:one name:nil];
                MPSGraphTensor* a_pow_bm1 =
                    [c.g powerWithPrimaryTensor:c.a secondaryTensor:bm1 name:nil];
                MPSGraphTensor* b_times_apow =
                    [c.g multiplicationWithPrimaryTensor:c.b secondaryTensor:a_pow_bm1 name:nil];
                MPSGraphTensor* da =
                    [c.g multiplicationWithPrimaryTensor:b_times_apow secondaryTensor:c.go name:nil];

                MPSGraphTensor* log_a = [c.g logarithmWithTensor:c.a name:nil];
                MPSGraphTensor* a_pow_b =
                    [c.g powerWithPrimaryTensor:c.a secondaryTensor:c.b name:nil];
                MPSGraphTensor* prod =
                    [c.g multiplicationWithPrimaryTensor:log_a secondaryTensor:a_pow_b name:nil];
                MPSGraphTensor* db =
                    [c.g multiplicationWithPrimaryTensor:prod secondaryTensor:c.go name:nil];
                return { da, db };
            });
    }
};

// ────────────────────────────────────────────────────────────────────
// neg (unary): dA = -grad.
// ────────────────────────────────────────────────────────────────────
class NegVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "neg"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.size() != 1 || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* go = as_tensor(grad_outs[0]);
        MPSGraphTensor* neg = [g negativeWithTensor:go name:nil];
        bctx.accumulate_grad(x_id, from_tensor(neg));
        return true;
    }
};

struct ArithVjpRegistrar {
    ArithVjpRegistrar() {
        register_vjp_emitter(std::make_unique<AddVjp>());
        register_vjp_emitter(std::make_unique<SubVjp>());
        register_vjp_emitter(std::make_unique<MulVjp>());
        register_vjp_emitter(std::make_unique<DivVjp>());
        register_vjp_emitter(std::make_unique<PowVjp>());
        register_vjp_emitter(std::make_unique<NegVjp>());
    }
};

[[maybe_unused]] static const ArithVjpRegistrar g_arith_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
