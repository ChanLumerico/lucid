// lucid/_C/compile/OpEmitters/nn/Embedding.mm
//
// Embedding-family emitters:
//
//   - ``one_hot``               — MPSGraph ``oneHotWithIndicesTensor:``
//   - ``rotary_pos_embedding``  — slice + sin/cos table mul + concat
//
// ``embedding`` itself (the bog-standard lookup table) lives in
// :file:`../shape/IndexCast.mm` since it boils down to a gather.
// ``embedding_bag`` stays in :file:`../misc/Stubs.mm`: its dynamic
// per-bag offsets need a segment-reduce primitive MPSGraph doesn't
// expose.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <memory>
#include <string_view>
#include <vector>

#include "../_AttrHelpers.h"

namespace lucid::compile {

namespace {

// ── one_hot — depth = num_classes, axis = last; F32/F16/I32/I64/Bool dtype.
class OneHotEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "one_hot"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        std::int64_t depth = int_attr(node, "num_classes", 0);
        if (depth <= 0) return false;
        MPSDataType out_mps;
        switch (node.outputs[0].dtype) {
            case Dtype::F32: out_mps = MPSDataTypeFloat32; break;
            case Dtype::F16: out_mps = MPSDataTypeFloat16; break;
            case Dtype::I32: out_mps = MPSDataTypeInt32; break;
            case Dtype::I64: out_mps = MPSDataTypeInt64; break;
            case Dtype::Bool: out_mps = MPSDataTypeBool; break;
            default: return false;
        }
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSUInteger axis = (NSUInteger)x.shape.count;  // append as last
        ctx.bind(node.outputs[0].id, (__bridge void*)([g oneHotWithIndicesTensor:x
                                                    depth:(NSUInteger)depth
                                                     axis:axis
                                                 dataType:out_mps
                                                  onValue:1.0
                                                 offValue:0.0
                                                     name:@"one_hot"]));
        return true;
    }
};

// ── rotary_pos_embedding — half-rotation pair multiply.
// Input shape: (..., L, D) with even D.  ``interleaved=False`` form:
//
//     x1 = input[..., :D/2],  x2 = input[..., D/2:]
//     y1 = x1*cos - x2*sin
//     y2 = x1*sin + x2*cos
//     out = concat([y1, y2], axis=-1)
//
// Tables (cos, sin) at positions 0..L-1 are pre-computed at emit
// time and materialised as MPSGraph constants — fast and exact.
// Dynamic position_ids (``has_pos_ids=True``) and the interleaved
// variant return nullptr → eager fallback.
class RotaryPosEmbeddingEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "rotary_pos_embedding"; }
    bool emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.empty() || node.outputs.empty()) return false;
        TensorId x_id = node.inputs[0];
        if (x_id < 0) return false;
        if (bool_attr(node, "has_pos_ids", false)) return false;
        if (bool_attr(node, "interleaved", false)) return false;
        MPSGraph* g = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (g == nil || x == nil) return false;
        NSArray<NSNumber*>* sh = x.shape;
        NSUInteger nd = sh.count;
        if (nd < 2) return false;
        std::int64_t L = sh[nd - 2].longLongValue;
        std::int64_t D = sh[nd - 1].longLongValue;
        if (D % 2 != 0) return false;
        std::int64_t half = D / 2;
        std::vector<float> cos_data(L * half), sin_data(L * half);
        for (std::int64_t i = 0; i < L; ++i) {
            for (std::int64_t k = 0; k < half; ++k) {
                double freq = std::pow(10000.0, -2.0 * (double)k / (double)D);
                double angle = (double)i * freq;
                cos_data[i * half + k] = (float)std::cos(angle);
                sin_data[i * half + k] = (float)std::sin(angle);
            }
        }
        NSData* cos_nsd = [NSData dataWithBytes:cos_data.data()
                                          length:cos_data.size() * sizeof(float)];
        NSData* sin_nsd = [NSData dataWithBytes:sin_data.data()
                                          length:sin_data.size() * sizeof(float)];
        NSArray<NSNumber*>* tbl_sh = @[[NSNumber numberWithLongLong:L],
                                        [NSNumber numberWithLongLong:half]];
        MPSGraphTensor* cos_t =
            [g constantWithData:cos_nsd shape:tbl_sh dataType:MPSDataTypeFloat32];
        MPSGraphTensor* sin_t =
            [g constantWithData:sin_nsd shape:tbl_sh dataType:MPSDataTypeFloat32];
        if (cos_t.dataType != x.dataType) {
            cos_t = [g castTensor:cos_t toType:x.dataType name:nil];
            sin_t = [g castTensor:sin_t toType:x.dataType name:nil];
        }
        NSMutableArray<NSNumber*>* tbl_bcast = [NSMutableArray array];
        for (NSUInteger d = 0; d + 2 < nd; ++d) [tbl_bcast addObject:@1];
        [tbl_bcast addObject:[NSNumber numberWithLongLong:L]];
        [tbl_bcast addObject:[NSNumber numberWithLongLong:half]];
        cos_t = [g reshapeTensor:cos_t withShape:tbl_bcast name:nil];
        sin_t = [g reshapeTensor:sin_t withShape:tbl_bcast name:nil];
        MPSGraphTensor* x1 =
            [g sliceTensor:x dimension:(NSInteger)(nd - 1) start:0 length:half name:nil];
        MPSGraphTensor* x2 = [g sliceTensor:x
                                  dimension:(NSInteger)(nd - 1)
                                       start:half
                                      length:half
                                        name:nil];
        MPSGraphTensor* t1a =
            [g multiplicationWithPrimaryTensor:x1 secondaryTensor:cos_t name:nil];
        MPSGraphTensor* t1b =
            [g multiplicationWithPrimaryTensor:x2 secondaryTensor:sin_t name:nil];
        MPSGraphTensor* y1 =
            [g subtractionWithPrimaryTensor:t1a secondaryTensor:t1b name:nil];
        MPSGraphTensor* t2a =
            [g multiplicationWithPrimaryTensor:x1 secondaryTensor:sin_t name:nil];
        MPSGraphTensor* t2b =
            [g multiplicationWithPrimaryTensor:x2 secondaryTensor:cos_t name:nil];
        MPSGraphTensor* y2 =
            [g additionWithPrimaryTensor:t2a secondaryTensor:t2b name:nil];
        ctx.bind(node.outputs[0].id, (__bridge void*)([g concatTensors:@[y1, y2]
                                       dimension:(NSInteger)(nd - 1)
                                            name:@"rope"]));
        return true;
    }
};

struct EmbeddingRegistrar {
    EmbeddingRegistrar() {
        register_emitter(std::make_unique<OneHotEmitter>());
        register_emitter(std::make_unique<RotaryPosEmbeddingEmitter>());
    }
};

[[maybe_unused]] static const EmbeddingRegistrar g_embedding_registrar;

}  // namespace

}  // namespace lucid::compile
