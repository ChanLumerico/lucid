// lucid/_C/compile/VjpEmitters/nn/Spatial.mm
//
// VJPs for ``interpolate_bilinear`` and ``interpolate_nearest_2d``.
//
// Each is MPSGraph's resize-gradient kernel called with exactly the
// parameters the forward emitter (``OpEmitters/nn/Spatial.mm``) resizes
// with — bilinear with ``centerResult`` on and ``align_corners`` passed
// through, nearest with floor rounding and neither — so the gradient
// scatters through the same index map the forward gathered by.  Without
// them, upsampling put every FCN / MaskFormer / YOLO training step on
// MPSGraph's autodiff, which the train-mode batch norms beside it rule
// out: the steps ran eager.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "../VjpEmitter.h"
#include "../_VjpHelpers.h"

namespace lucid::compile {

namespace {

bool align_corners_of(const OpNode& node) {
    auto it = node.attrs.find("align_corners");
    if (it == node.attrs.end())
        return false;
    if (const auto* b = std::get_if<bool>(&it->second))
        return *b;
    if (const auto* i = std::get_if<std::int64_t>(&it->second))
        return *i != 0;
    return false;
}

template <bool IS_BILINEAR>
class Interpolate2dVjp final : public VjpEmitter {
public:
    std::string_view op_name() const override {
        return IS_BILINEAR ? "interpolate_bilinear" : "interpolate_nearest_2d";
    }
    bool emit(BackwardContext& bctx, const OpNode& node,
              const std::vector<void*>& grad_outs) override {
        const bool align = align_corners_of(node);
        return emit_unary_vjp(bctx, node, grad_outs,
            [&](MPSGraph* g, MPSGraphTensor* x, MPSGraphTensor* go) -> MPSGraphTensor* {
                if (x.shape.count != 4 || node.outputs.empty() ||
                    node.outputs[0].shape.size() != 4)
                    return nil;
                for (NSNumber* d in x.shape)
                    if (d.longLongValue <= 0)
                        return nil;
                // Static shapes in and out: the gradient kernel inherits an
                // unresolved shape otherwise (see the nearest forward).
                NSMutableArray<NSNumber*>* out_shape = [NSMutableArray array];
                for (const auto d : node.outputs[0].shape)
                    [out_shape addObject:@(d)];
                go = [g reshapeTensor:go withShape:out_shape name:nil];
                MPSGraphTensor* dx = nil;
                if (IS_BILINEAR)
                    dx = [g resizeWithGradientTensor:go
                                                 input:x
                                                  mode:MPSGraphResizeBilinear
                                          centerResult:YES
                                          alignCorners:align ? YES : NO
                                                layout:MPSGraphTensorNamedDataLayoutNCHW
                                                  name:@"interp2d_bilinear_vjp"];
                else
                    dx = [g resizeNearestWithGradientTensor:go
                                                    input:x
                                      nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
                                             centerResult:NO
                                             alignCorners:NO
                                                   layout:MPSGraphTensorNamedDataLayoutNCHW
                                                     name:@"interp2d_nearest_vjp"];
                return [g reshapeTensor:dx withShape:x.shape name:nil];
            });
    }
};

struct SpatialVjpRegistrar {
    SpatialVjpRegistrar() {
        register_vjp_emitter(std::make_unique<Interpolate2dVjp<true>>());
        register_vjp_emitter(std::make_unique<Interpolate2dVjp<false>>());
    }
};

[[maybe_unused]] static const SpatialVjpRegistrar g_spatial_vjp_registrar;

}  // namespace

}  // namespace lucid::compile
