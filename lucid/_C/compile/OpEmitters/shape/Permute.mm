// lucid/_C/compile/OpEmitters/Permute.mm
//
// Permute emitter — needs the ``permutation`` attribute that
// :file:`ops/ufunc/Transpose.cpp` reports via :func:`OpScopeFull::set_attr`.
//
// Op name "permute" matches the engine schema in Transpose.cpp.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <memory>
#include <string_view>
#include <variant>
#include <vector>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

class PermuteEmitter final : public OpEmitter {
public:
    std::string_view op_name() const override { return "permute"; }

    void* emit(BuilderContext& ctx, const OpNode& node) override {
        if (node.inputs.size() != 1)
            return nullptr;
        TensorId x_id = node.inputs[0];
        if (x_id < 0)
            return nullptr;

        // Pull the permutation payload from attrs; abort on miss or
        // wrong variant alternative (signature → eager-only).
        auto it = node.attrs.find("permutation");
        if (it == node.attrs.end())
            return nullptr;
        const auto* perm =
            std::get_if<std::vector<std::int64_t>>(&it->second);
        if (perm == nullptr)
            return nullptr;

        MPSGraph* graph = (__bridge MPSGraph*)ctx.graph();
        MPSGraphTensor* x_t = (__bridge MPSGraphTensor*)ctx.resolve(x_id);
        if (x_t == nil || graph == nil)
            return nullptr;

        NSMutableArray<NSNumber*>* ns_perm =
            [NSMutableArray arrayWithCapacity:perm->size()];
        for (std::int64_t p : *perm)
            [ns_perm addObject:[NSNumber numberWithLongLong:p]];

        MPSGraphTensor* y = [graph transposeTensor:x_t
                                       permutation:ns_perm
                                              name:@"permute"];
        return (__bridge void*)y;
    }
};

struct PermuteEmitterRegistrar {
    PermuteEmitterRegistrar() {
        register_emitter(std::make_unique<PermuteEmitter>());
    }
};

[[maybe_unused]] static const PermuteEmitterRegistrar g_permute_registrar;

}  // namespace

}  // namespace lucid::compile
