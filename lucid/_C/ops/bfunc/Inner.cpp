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

// Build an einsum pattern equivalent to inner: contract last axis of a and b.
// `a`'s leading axes use 'a','b',... and `b`'s use 'p','q',... ; contraction
// uses 'z'. Each side may have up to 16 leading axes.
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
    // Route autograd-tracked computations through einsum so the backward
    // chain is well-formed via the primitive ops einsum is built on.
    // Pure-inference calls keep the native fast path below.
    if (GradMode::is_enabled() && (a->requires_grad() || b->requires_grad())) {
        return einsum_op(inner_einsum_pattern(a->shape().size(), b->shape().size()), {a, b});
    }

    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"inner", device, dt, Shape{}};

    const auto& sa = a->shape();
    const auto& sb = b->shape();
    if (sa.empty() || sb.empty() || sa.back() != sb.back())
        throw ShapeMismatch(sa, sb, "inner");
    Shape out_shape(sa.begin(), sa.end() - 1);
    out_shape.insert(out_shape.end(), sb.begin(), sb.end() - 1);

    if (device == Device::GPU) {
        auto out_storage = backend::Dispatcher::for_device(device).inner(a->storage(), b->storage(),
                                                                         sa, sb, out_shape, dt);
        // For GPU, inner() returns storage with shape embedded in MLX array.
        const auto& gs = std::get<GpuStorage>(out_storage);
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
