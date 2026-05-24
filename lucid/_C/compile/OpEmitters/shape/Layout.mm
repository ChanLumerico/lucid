// lucid/_C/compile/OpEmitters/shape/Layout.mm
//
// Non-view layout rearrangement emitters:
//
//   - ``flip``     — reverse along listed axes
//   - ``roll``     — wrap-around shift via slice + concat
//   - ``tril`` / ``triu`` — lower / upper triangular mask via bandPart (k=0)
//   - ``diagonal`` — main diagonal extraction (offset=0, 2-D only)
//
// These all produce a same-shape (or strictly smaller-rank for
// ``diagonal``) output by rearranging or masking the input — no
// data-dependent reads, no extra inputs.  They sit in ``shape/``
// rather than ``index/`` because they don't take an index tensor.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

class FlipEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "flip"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        auto it = node.attrs.find("dims");
        if (it == node.attrs.end())
            return nullptr;
        const auto* dims = std::get_if<std::vector<std::int64_t>>(&it->second);
        if (dims == nullptr)
            return nullptr;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return nullptr;

        NSMutableArray<NSNumber*>* axes =
            [NSMutableArray arrayWithCapacity:dims->size()];
        for (std::int64_t d : *dims)
            [axes addObject:[NSNumber numberWithLongLong:d]];
        return (__bridge void*)[graph reverseTensor:x_t axes:axes name:@"flip"];
    }
};

// ``roll`` — MPSGraph's stable ``rollWithTensor:...`` selector differs
// across SDK levels.  Implement via per-axis slice + concatenate
// (wraparound), which works on every MPSGraph version Lucid targets.
class RollEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "roll"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty())
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;
        auto sh_it = node.attrs.find("shifts");
        auto ax_it = node.attrs.find("axes");
        if (sh_it == node.attrs.end() || ax_it == node.attrs.end())
            return nullptr;
        const auto* shifts =
            std::get_if<std::vector<std::int64_t>>(&sh_it->second);
        const auto* axes =
            std::get_if<std::vector<std::int64_t>>(&ax_it->second);
        if (shifts == nullptr || axes == nullptr || shifts->size() != axes->size())
            return nullptr;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (graph == nil || x_t == nil)
            return nullptr;

        MPSGraphTensor* cur = x_t;
        NSUInteger ndim = cur.shape.count;
        for (std::size_t i = 0; i < shifts->size(); ++i) {
            std::int64_t ax = (*axes)[i];
            if (ax < 0) ax += (std::int64_t)ndim;
            if (ax < 0 || ax >= (std::int64_t)ndim)
                return nullptr;
            std::int64_t dim = cur.shape[(NSUInteger)ax].longLongValue;
            if (dim <= 0) continue;
            std::int64_t s = (*shifts)[i] % dim;
            if (s < 0) s += dim;
            if (s == 0) continue;

            NSMutableArray<NSNumber*>* starts_tail =
                [NSMutableArray arrayWithCapacity:ndim];
            NSMutableArray<NSNumber*>* ends_tail =
                [NSMutableArray arrayWithCapacity:ndim];
            NSMutableArray<NSNumber*>* strides =
                [NSMutableArray arrayWithCapacity:ndim];
            NSMutableArray<NSNumber*>* starts_head =
                [NSMutableArray arrayWithCapacity:ndim];
            NSMutableArray<NSNumber*>* ends_head =
                [NSMutableArray arrayWithCapacity:ndim];
            for (NSUInteger d = 0; d < ndim; ++d) {
                std::int64_t dim_d = cur.shape[d].longLongValue;
                [strides addObject:[NSNumber numberWithLongLong:1]];
                if ((std::int64_t)d == ax) {
                    [starts_tail addObject:[NSNumber numberWithLongLong:dim - s]];
                    [ends_tail addObject:[NSNumber numberWithLongLong:dim]];
                    [starts_head addObject:[NSNumber numberWithLongLong:0]];
                    [ends_head addObject:[NSNumber numberWithLongLong:dim - s]];
                } else {
                    [starts_tail addObject:[NSNumber numberWithLongLong:0]];
                    [ends_tail addObject:[NSNumber numberWithLongLong:dim_d]];
                    [starts_head addObject:[NSNumber numberWithLongLong:0]];
                    [ends_head addObject:[NSNumber numberWithLongLong:dim_d]];
                }
            }
            MPSGraphTensor* tail = [graph sliceTensor:cur
                                              starts:starts_tail
                                                ends:ends_tail
                                             strides:strides
                                                name:nil];
            MPSGraphTensor* head = [graph sliceTensor:cur
                                              starts:starts_head
                                                ends:ends_head
                                             strides:strides
                                                name:nil];
            cur = [graph concatTensors:@[tail, head]
                            dimension:(NSInteger)ax
                                 name:nil];
        }
        return (__bridge void*)cur;
    }
};

// ── tril / triu — k=0 only via bandPart(numLower, numUpper).
// MPSGraph's ``bandPart`` semantics differ subtly across SDKs for
// non-zero ``k`` offsets; we restrict to ``k=0`` (the main-diagonal
// split) where both SDKs agree.
class TriEmitter final : public OpEmitter {
public:
    explicit TriEmitter(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        std::int64_t k = int_attr(node, "k", 0);
        bool upper = bool_attr(node, "upper", name_ == "triu");
        if (k != 0) return nullptr;  // k != 0 → eager
        NSInteger nl = upper ? 0 : -1;
        NSInteger nu = upper ? -1 : 0;
        return (__bridge void*)[g bandPartWithTensor:x
                                            numLower:nl
                                            numUpper:nu
                                                name:@"tri"];
    }

private:
    std::string name_;
};

// ── diagonal — main diagonal (offset=0) of a 2-D tensor.
class DiagonalEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "diagonal"; }
    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return nullptr;
        std::int64_t offset = int_attr(node, "offset", 0);
        if (offset != 0) return nullptr;  // only the main diagonal
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return nullptr;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return nullptr;
        NSArray<NSNumber*>* sh = x.shape;
        if (sh.count != 2) return nullptr;
        std::int64_t N = sh[0].longLongValue;
        std::int64_t M = sh[1].longLongValue;
        std::int64_t K = N < M ? N : M;
        MPSGraphTensor* band =
            [g bandPartWithTensor:x numLower:0 numUpper:0 name:nil];
        NSArray<NSNumber*>* reduce_axes = (N >= M) ? @[@0] : @[@1];
        MPSGraphTensor* r =
            [g reductionSumWithTensor:band axes:reduce_axes name:nil];
        return (__bridge void*)[g reshapeTensor:r
                                       withShape:@[[NSNumber numberWithLongLong:K]]
                                            name:@"diagonal"];
    }
};

struct LayoutEmitterRegistrar {
    LayoutEmitterRegistrar() {
        register_emitter(std::make_unique<FlipEmitter>());
        register_emitter(std::make_unique<RollEmitter>());
        register_emitter(std::make_unique<TriEmitter>("tril"));
        register_emitter(std::make_unique<TriEmitter>("triu"));
        register_emitter(std::make_unique<DiagonalEmitter>());
    }
};

[[maybe_unused]] static const LayoutEmitterRegistrar g_layout_registrar;

}  // namespace

}  // namespace lucid::compile
