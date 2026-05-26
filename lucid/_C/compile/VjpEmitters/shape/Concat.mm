// lucid/_C/compile/VjpEmitters/shape/Concat.mm
//
// VJPs for ``concatenate`` and ``split`` / ``split_at`` (the inverse
// pair).  Concat's backward is split; split's backward is concat.
//
// Concat forward: y = concatenate([x_0, x_1, ..., x_{N-1}], dim).
// Backward: split grad along dim at each input's size, accumulate onto
// the corresponding input slot.
//
// Split / split_at forward: outputs[i] = x[..., start_i:start_i+len_i, ...]
// Backward: concat all grad_outs along axis (missing-grad slots = zeros).
//
// These ops are central to attention QKV-split / multi-head concat;
// without them MPSGraph autograd asserts on the forward ``sliceTensor:``
// chain.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

// ────────────────────────────────────────────────────────────────────
// concatenate VJP — slice grad at each input's offset+size.
// ────────────────────────────────────────────────────────────────────
class ConcatVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return "concatenate"; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || grad_outs.empty() || grad_outs[0] == nullptr)
            return false;
        auto dim_it = node.attrs.find("dim");
        if (dim_it == node.attrs.end()) return false;
        const auto* dimp = std::get_if<std::int64_t>(&dim_it->second);
        if (dimp == nullptr) return false;
        const std::int64_t dim = *dimp;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        MPSGraphTensor* grad = as_tensor(grad_outs[0]);
        if (g == nil || grad == nil) return false;

        std::int64_t offset = 0;
        for (std::size_t k = 0; k < node.inputs.size(); ++k) {
            TensorId iid = node.inputs[k];
            if (iid < 0) return false;
            MPSGraphTensor* xk = as_tensor(bctx.forward(iid));
            if (xk == nil) return false;
            // Get the size along dim for this input.
            std::vector<std::int64_t> xk_shape = shape_of_mps(xk);
            if (dim < 0 || (std::size_t)dim >= xk_shape.size()) return false;
            const std::int64_t length = xk_shape[(std::size_t)dim];
            if (length <= 0) return false;

            MPSGraphTensor* piece =
                [g sliceTensor:grad
                     dimension:(NSInteger)dim
                         start:(NSInteger)offset
                        length:(NSInteger)length
                          name:[NSString stringWithFormat:@"cat_vjp_p%zu", k]];
            bctx.accumulate_grad(iid, from_tensor(piece));
            offset += length;
        }
        return true;
    }
};

// ────────────────────────────────────────────────────────────────────
// Split / split_at VJP — concat all grad_outs along axis.
// Missing-grad slots (output unused downstream) are filled with zeros
// shaped like the corresponding forward output piece.
// ────────────────────────────────────────────────────────────────────
template <const char* OP_NAME>
class SplitFamilyVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override { return OP_NAME; }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        auto ax_it = node.attrs.find("axis");
        if (ax_it == node.attrs.end()) return false;
        const auto* axp = std::get_if<std::int64_t>(&ax_it->second);
        if (axp == nullptr) return false;
        const std::int64_t axis = *axp;

        MPSGraph* g = (__bridge MPSGraph*)bctx.graph();
        if (g == nil) return false;

        // Build the list of per-piece grad tensors.  For pieces with no
        // downstream demand (grad_outs[i] == nullptr) we synthesise
        // zeros at the piece's recorded shape so concat shape matches
        // the forward x.
        if (grad_outs.size() != node.outputs.size()) return false;
        NSMutableArray<MPSGraphTensor*>* pieces =
            [NSMutableArray arrayWithCapacity:node.outputs.size()];
        // Use the first non-null grad's dtype for zero pieces — this is
        // the chain dtype (under autocast, the forward input may be F32
        // master while grads are F16; the synthesised zero pieces must
        // match the grad pieces' dtype so concat succeeds).  Fall back
        // to the forward input dtype only when no grads exist.
        MPSDataType dt = MPSDataTypeFloat32;
        bool dt_from_grad = false;
        for (void* gp : grad_outs) {
            if (gp != nullptr) {
                dt = as_tensor(gp).dataType;
                dt_from_grad = true;
                break;
            }
        }
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        MPSGraphTensor* x = as_tensor(bctx.forward(x_id));
        if (!dt_from_grad && x != nil)
            dt = x.dataType;

        for (std::size_t k = 0; k < node.outputs.size(); ++k) {
            MPSGraphTensor* p = nil;
            if (grad_outs[k] != nullptr) {
                p = as_tensor(grad_outs[k]);
            } else {
                // Piece had no demand — synthesise zeros at the trace's
                // recorded shape for outputs[k].
                const auto& sh = node.outputs[k].shape;
                if (sh.empty()) return false;
                NSArray<NSNumber*>* sh_ns = shape_to_ns(sh);
                p = [g constantWithScalar:0.0 shape:sh_ns dataType:dt];
            }
            [pieces addObject:p];
        }
        MPSGraphTensor* dx =
            [g concatTensors:pieces dimension:(NSInteger)axis name:@"split_vjp"];
        bctx.accumulate_grad(x_id, from_tensor(dx));
        return true;
    }
};

constexpr char kSplit[] = "split";
constexpr char kSplitAt[] = "split_at";

struct ConcatVjpRegistrar {
    ConcatVjpRegistrar() {
        register_vjp_emitter(std::make_unique<ConcatVjp>());
        register_vjp_emitter(std::make_unique<SplitFamilyVjp<kSplit>>());
        register_vjp_emitter(std::make_unique<SplitFamilyVjp<kSplitAt>>());
    }
};

[[maybe_unused]] static const ConcatVjpRegistrar g_concat_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
