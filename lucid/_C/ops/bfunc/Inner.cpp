// lucid/_C/ops/bfunc/Inner.cpp
//
// Implements inner_op.  For gradient-tracked tensors, the inner product is
// expressed as an equivalent einsum string and forwarded to einsum_op so that
// the full einsum autograd machinery handles the backward pass.  For non-
// gradient paths the backend inner primitive is called directly.

#include "Inner.h"

#include <string>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../einops/Einops.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Build an einsum contraction string equivalent to numpy.inner for tensors
// with rank na and nb.
//
// The last axis of each tensor is labelled 'z' (the contracted axis).  Free
// axes of A receive letters starting at 'a'; free axes of B receive letters
// starting at 'p'.  The two alphabets are kept disjoint so that no free axis
// of A aliases a free axis of B.
//
// Example — inner_einsum_pattern(3, 2):
//   A has axes [a, b, z], B has axes [p, z]  → "abz,pz->abp"
std::string inner_einsum_pattern(std::size_t na, std::size_t nb) {
    std::string a_lhs, b_lhs, rhs;
    for (std::size_t i = 0; i + 1 < na; ++i) {
        char c = static_cast<char>('a' + i);
        a_lhs.push_back(c);
        rhs.push_back(c);
    }
    a_lhs.push_back('z');
    for (std::size_t i = 0; i + 1 < nb; ++i) {
        char c = static_cast<char>('p' + i);
        b_lhs.push_back(c);
        rhs.push_back(c);
    }
    b_lhs.push_back('z');
    return a_lhs + "," + b_lhs + "->" + rhs;
}

}  // namespace

TensorImplPtr inner_op(const TensorImplPtr& a, const TensorImplPtr& b) {
    validate_pair(a, b, "inner");

    // When gradient tracking is enabled, delegate to einsum_op so that the
    // backward pass is handled automatically by the einsum backward node.
    if (GradMode::is_enabled() && (a->requires_grad() || b->requires_grad())) {
        return einsum_op(inner_einsum_pattern(a->shape().size(), b->shape().size()), {a, b});
    }

    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"inner", device, dt, Shape{}};

    const auto& sa = a->shape();
    const auto& sb = b->shape();
    // Both tensors must be non-scalar and have the same last-axis size.
    if (sa.empty() || sb.empty() || sa.back() != sb.back())
        throw ShapeMismatch(sa, sb, "inner");
    // Output shape: all-but-last dims of A followed by all-but-last dims of B.
    Shape out_shape(sa.begin(), sa.end() - 1);
    out_shape.insert(out_shape.end(), sb.begin(), sb.end() - 1);

    if (device == Device::GPU) {
        auto out_storage = backend::Dispatcher::for_device(device).inner(a->storage(), b->storage(),
                                                                         sa, sb, out_shape, dt);
        // MLX may adjust the output shape slightly (e.g. scalar squeeze), so
        // read the actual shape back from the returned GpuStorage.
        const auto& gs = storage_gpu(out_storage);
        Shape actual_shape;
        for (auto d : gs.arr->shape())
            actual_shape.push_back(static_cast<std::int64_t>(d));
        return fresh(std::move(out_storage), std::move(actual_shape), dt, device);
    }

    auto out_storage = backend::Dispatcher::for_device(device).inner(a->storage(), b->storage(), sa,
                                                                     sb, out_shape, dt);
    return fresh(std::move(out_storage), out_shape, dt, device);
}

}  // namespace lucid
