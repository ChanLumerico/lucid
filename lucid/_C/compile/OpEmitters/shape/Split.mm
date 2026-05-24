// lucid/_C/compile/OpEmitters/shape/Split.mm
//
// Single-input → N-output emitters for ``split`` (equal-size pieces)
// and ``split_at`` (cut-point indices).  Each piece is built as an
// MPSGraph ``sliceTensor:dimension:start:length:`` op; piece 0 is
// returned by the emit() method (auto-bound to outputs[0].id by the
// builder), and pieces 1..N-1 are manually bound into the
// :class:`BuilderContext` via ``ctx.bind(node.outputs[i].id, …)``.
//
// Background: the OpEmitter API has a single-output signature, but the
// IR's ``OpNode::outputs`` already supports multiple entries.  We use
// the manual-bind path here so the slice emitters can stay inside the
// existing framework without an IR refactor.
//
// Engine schema names: ``"split"``, ``"split_at"`` (set by
// ``split_op`` / ``split_at_op`` in lucid/_C/ops/utils/Concat.cpp).

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

// Helper: emit one MPSGraph slice along ``axis`` for an arbitrary
// starting offset + length.
inline MPSGraphTensor* emit_slice(MPSGraph* g,
                                  MPSGraphTensor* x,
                                  std::int64_t axis,
                                  std::int64_t start,
                                  std::int64_t length,
                                  NSString* name) {
    return [g sliceTensor:x
                dimension:(NSInteger)axis
                    start:(NSInteger)start
                   length:(NSInteger)length
                     name:name];
}

// ── split — equal-size pieces along ``axis``.
//
// Note on inputs: ``split_op`` calls ``attach_split_grad`` once per
// piece, each invocation appending its single input to the current
// :class:`OpScopeFull`.  The resulting :class:`OpNode` therefore has
// ``inputs.size() == num_splits`` entries that all reference the same
// source TensorId.  We dedupe by reading inputs[0] only.
class SplitEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "split"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty())
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        const std::int64_t axis = int_attr(node, "axis", -1);
        const std::int64_t piece = int_attr(node, "piece", 0);
        const std::int64_t num_splits = int_attr(node, "num_splits", 0);
        if (axis < 0 || piece <= 0 || num_splits <= 0)
            return nullptr;
        if ((std::int64_t)node.outputs.size() != num_splits)
            return nullptr;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil)
            return nullptr;

        // Multi-output handling: builder skips auto-bind for ops with
        // outputs.size() > 1 — emit each consumed piece + explicitly
        // bind it.  Return any tensor we emitted as a non-null success
        // signal; if everything is dead, return piece 0 to avoid a
        // false failure (unusual case — the trace shouldn't have a
        // fully-dead split op, but we tolerate it).
        MPSGraphTensor* anchor = nil;
        for (std::int64_t i = 0; i < num_splits; ++i) {
            const std::int64_t start = i * piece;
            const TensorId piece_id = node.outputs[(std::size_t)i].id;
            const bool consumed = ctx.is_consumed(piece_id);
            if (!consumed && anchor != nil) {
                // Skip dead piece — but keep building piece 0 below
                // for the success-signal tensor if we haven't yet.
                continue;
            }
            MPSGraphTensor* p = emit_slice(
                g, x, axis, start, piece,
                [NSString stringWithFormat:@"split_p%lld", i]);
            if (consumed)
                ctx.bind(piece_id, (__bridge void*)p);
            if (anchor == nil)
                anchor = p;
        }
        return (__bridge void*)anchor;
    }
};

// ── split_at — explicit cut-point indices along ``axis``.
// Produces (indices.size() + 1) pieces; piece i spans
// [prev_cut, indices[i]) with 0 / shape[axis] as implicit boundaries.
class SplitAtEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "split_at"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        // See SplitEmitter for why inputs may be duplicated.  Read
        // inputs[0] and ignore the rest — all entries reference the
        // same source.
        if (node.inputs.empty())
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        const std::int64_t axis = int_attr(node, "axis", -1);
        if (axis < 0)
            return nullptr;
        const std::vector<std::int64_t>* indices_p = int_vec_attr(node, "indices");
        if (indices_p == nullptr || indices_p->empty())
            return nullptr;
        const std::vector<std::int64_t>& indices = *indices_p;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil)
            return nullptr;
        if ((NSInteger)axis >= (NSInteger)x.shape.count)
            return nullptr;
        const std::int64_t axis_size = x.shape[(NSUInteger)axis].longLongValue;

        const std::size_t n_pieces = indices.size() + 1;
        if (node.outputs.size() != n_pieces)
            return nullptr;

        // Multi-output: build only consumed pieces (or piece 0 as a
        // fallback success signal — see SplitEmitter for rationale).
        MPSGraphTensor* anchor = nil;
        std::int64_t prev = 0;
        for (std::size_t i = 0; i < n_pieces; ++i) {
            const std::int64_t next =
                (i < indices.size()) ? indices[i] : axis_size;
            const std::int64_t length = next - prev;
            if (length <= 0)
                return nullptr;
            const TensorId piece_id = node.outputs[i].id;
            const bool consumed = ctx.is_consumed(piece_id);
            if (!consumed && anchor != nil) {
                prev = next;
                continue;
            }
            MPSGraphTensor* p = emit_slice(
                g, x, axis, prev, length,
                [NSString stringWithFormat:@"split_at_p%zu", i]);
            if (consumed)
                ctx.bind(piece_id, (__bridge void*)p);
            if (anchor == nil)
                anchor = p;
            prev = next;
        }
        return (__bridge void*)anchor;
    }
};

// ── unbind — decompose along ``axis`` into individual size-1 slices
// with the axis dimension squeezed out.  N-output where N = shape[axis].
// Each piece = sliceTensor(start=k, length=1) + reshape(squeeze).
class UnbindEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "unbind"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty())
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        const std::int64_t axis = int_attr(node, "axis", -1);
        if (axis < 0)
            return nullptr;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil)
            return nullptr;
        if ((NSInteger)axis >= (NSInteger)x.shape.count)
            return nullptr;
        const std::int64_t n_pieces = x.shape[(NSUInteger)axis].longLongValue;
        if ((std::int64_t)node.outputs.size() != n_pieces)
            return nullptr;

        // Build the squeezed shape (drop the axis dim) once — every piece
        // ends up with this shape.
        NSMutableArray<NSNumber*>* squeezed_shape =
            [NSMutableArray arrayWithCapacity:x.shape.count - 1];
        for (NSUInteger d = 0; d < x.shape.count; ++d) {
            if ((std::int64_t)d == axis) continue;
            [squeezed_shape addObject:x.shape[d]];
        }

        MPSGraphTensor* anchor = nil;
        for (std::int64_t k = 0; k < n_pieces; ++k) {
            const TensorId piece_id = node.outputs[(std::size_t)k].id;
            const bool consumed = ctx.is_consumed(piece_id);
            if (!consumed && anchor != nil)
                continue;
            MPSGraphTensor* slab = emit_slice(
                g, x, axis, k, 1,
                [NSString stringWithFormat:@"unbind_slab%lld", k]);
            MPSGraphTensor* squeezed =
                [g reshapeTensor:slab
                       withShape:squeezed_shape
                            name:[NSString stringWithFormat:@"unbind_p%lld", k]];
            if (consumed)
                ctx.bind(piece_id, (__bridge void*)squeezed);
            if (anchor == nil)
                anchor = squeezed;
        }
        return (__bridge void*)anchor;
    }
};

// ── topk — multi-output (values + indices).  Uses MPSGraph's
// ``topKWithSourceTensor:axis:k:name:`` (macOS 14+).
class TopKEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "topk"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty())
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        const std::int64_t axis = int_attr(node, "axis", -1);
        const std::int64_t k = int_attr(node, "k", 0);
        if (axis < 0 || k <= 0)
            return nullptr;
        // topk_op returns {values, indices} — exactly 2 outputs.
        if (node.outputs.size() != 2)
            return nullptr;

        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil)
            return nullptr;

        NSArray<MPSGraphTensor*>* res =
            [g topKWithSourceTensor:x
                               axis:(NSInteger)axis
                                  k:(NSUInteger)k
                               name:@"topk"];
        if (res == nil || res.count != 2)
            return nullptr;

        MPSGraphTensor* values = res[0];
        MPSGraphTensor* indices = res[1];

        // Lucid's topk indices are I32 but MPSGraph returns I32 by
        // default — no cast needed for I32.  If a future macOS bumps
        // this to I64, add a cast here.

        const TensorId values_id = node.outputs[0].id;
        const TensorId indices_id = node.outputs[1].id;
        const bool values_consumed = ctx.is_consumed(values_id);
        const bool indices_consumed = ctx.is_consumed(indices_id);

        // Bind whichever pieces are consumed.  Multi-output ops don't
        // get auto-bind — see MpsBuilder loop comment.
        if (values_consumed)
            ctx.bind(values_id, (__bridge void*)values);
        if (indices_consumed)
            ctx.bind(indices_id, (__bridge void*)indices);

        // Return a non-null sentinel (values is always a valid MPSGraph
        // tensor we built).  Builder's auto-bind is skipped because
        // outputs.size() > 1.
        return (__bridge void*)values;
    }
};

struct SplitEmitterRegistrar {
    SplitEmitterRegistrar() {
        register_emitter(std::make_unique<SplitEmitter>());
        register_emitter(std::make_unique<SplitAtEmitter>());
        register_emitter(std::make_unique<TopKEmitter>());
        register_emitter(std::make_unique<UnbindEmitter>());
    }
};

[[maybe_unused]] static const SplitEmitterRegistrar g_split_registrar;

}  // namespace

}  // namespace lucid::compile
