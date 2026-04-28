#include "Norm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <variant>
#include <vector>

#include <mlx/linalg.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

namespace {

// Reduce shape after collapsing `axes` (with keepdims).
Shape reduced_shape(const Shape& sh, const std::vector<int>& axes, bool keepdims) {
    if (axes.empty()) {
        // Empty axes → full reduction: scalar (or all-ones if keepdims).
        if (keepdims)
            return Shape(sh.size(), 1);
        return Shape{};
    }
    std::vector<bool> mask(sh.size(), false);
    for (int a : axes) {
        int p = a < 0 ? a + static_cast<int>(sh.size()) : a;
        mask[p] = true;
    }
    Shape out;
    for (std::size_t i = 0; i < sh.size(); ++i) {
        if (mask[i]) {
            if (keepdims)
                out.push_back(1);
        } else {
            out.push_back(sh[i]);
        }
    }
    return out;
}

// Iterate every element of `sh`, calling `fn(in_flat, out_flat)`.
// Reduction axes collapse to coord 0 in the output index. Layout: row-major.
template <typename T, typename Fn>
void elementwise_loop(const T* in,
                      std::size_t /*in_numel*/,
                      const Shape& sh,
                      const std::vector<bool>& reduce_mask,
                      const Shape& out_shape,
                      Fn fn) {
    const std::size_t nd = sh.size();
    Stride in_stride(nd), out_stride(out_shape.size());
    if (nd > 0) {
        in_stride[nd - 1] = 1;
        for (std::ptrdiff_t i = (std::ptrdiff_t)nd - 2; i >= 0; --i)
            in_stride[i] = in_stride[i + 1] * sh[i + 1];
    }
    if (!out_shape.empty()) {
        const std::size_t ond = out_shape.size();
        out_stride[ond - 1] = 1;
        for (std::ptrdiff_t i = (std::ptrdiff_t)ond - 2; i >= 0; --i)
            out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }

    // Walk all input coords, project to output coords.
    std::vector<std::int64_t> coord(nd, 0);
    const std::size_t in_numel = shape_numel(sh);
    for (std::size_t f = 0; f < in_numel; ++f) {
        std::size_t in_flat = 0;
        std::size_t out_flat = 0;
        std::size_t out_axis = 0;
        for (std::size_t d = 0; d < nd; ++d) {
            in_flat += static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
            if (!reduce_mask[d]) {
                out_flat += static_cast<std::size_t>(coord[d]) *
                            static_cast<std::size_t>(out_stride[out_axis]);
                ++out_axis;
            }
        }
        fn(in[in_flat], out_flat);
        for (std::ptrdiff_t d = (std::ptrdiff_t)nd - 1; d >= 0; --d) {
            if (++coord[d] < sh[d])
                break;
            coord[d] = 0;
        }
    }
}

// Compute ord-norm along the requested axes for a single dtype.
template <typename T>
void norm_typed(const T* in,
                T* out,
                const Shape& sh,
                const std::vector<int>& axes,
                bool keepdims,
                double ord,
                const Shape& out_shape) {
    std::vector<bool> reduce_mask(sh.size(), false);
    if (axes.empty()) {
        for (std::size_t i = 0; i < sh.size(); ++i)
            reduce_mask[i] = true;
    } else {
        for (int a : axes) {
            int p = a < 0 ? a + static_cast<int>(sh.size()) : a;
            reduce_mask[p] = true;
        }
    }
    const std::size_t out_numel = std::max<std::size_t>(1, shape_numel(out_shape));

    // For p == inf / -inf, accumulate max / min of |x|.
    // For p == 0, count nonzeros.
    // For p == 1, accumulate |x|.
    // For p == 2, accumulate x*x then sqrt.
    // For other p, accumulate |x|^p then ^(1/p).
    if (std::isinf(ord)) {
        const bool pos = ord > 0;
        std::vector<T> acc(out_numel, pos ? T{0} : std::numeric_limits<T>::infinity());
        elementwise_loop<T>(in, 0, sh, reduce_mask, out_shape, [&](T v, std::size_t o) {
            T av = std::abs(v);
            acc[o] = pos ? std::max(acc[o], av) : std::min(acc[o], av);
        });
        std::memcpy(out, acc.data(), out_numel * sizeof(T));
        return;
    }
    if (ord == 0.0) {
        std::vector<T> acc(out_numel, T{0});
        elementwise_loop<T>(in, 0, sh, reduce_mask, out_shape, [&](T v, std::size_t o) {
            if (v != T{0})
                acc[o] += T{1};
        });
        std::memcpy(out, acc.data(), out_numel * sizeof(T));
        return;
    }
    std::vector<T> acc(out_numel, T{0});
    elementwise_loop<T>(in, 0, sh, reduce_mask, out_shape, [&](T v, std::size_t o) {
        T av = std::abs(v);
        if (ord == 2.0)
            acc[o] += v * v;
        else if (ord == 1.0)
            acc[o] += av;
        else
            acc[o] += std::pow(av, static_cast<T>(ord));
    });
    if (ord == 2.0) {
        for (std::size_t i = 0; i < out_numel; ++i)
            acc[i] = std::sqrt(acc[i]);
    } else if (ord != 1.0) {
        const T inv_ord = static_cast<T>(1.0 / ord);
        for (std::size_t i = 0; i < out_numel; ++i)
            acc[i] = std::pow(acc[i], inv_ord);
    }
    std::memcpy(out, acc.data(), out_numel * sizeof(T));
}

}  // namespace

TensorImplPtr norm_op(const TensorImplPtr& a, double ord, std::vector<int> axis, bool keepdims) {
    using namespace linalg_detail;
    Validator::input(a, "norm.a").non_null();
    require_float(a->dtype(), "norm");
    OpScopeFull scope{"norm", a->device(), a->dtype(), a->shape()};

    if (a->device() == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        std::optional<std::vector<int>> axis_opt;
        if (!axis.empty())
            axis_opt = std::move(axis);
        auto out = ::mlx::core::linalg::norm(in, ord, axis_opt, keepdims, kMlxLinalgStream);
        Shape sh = mlx_shape_to_lucid(out.shape());
        return fresh(wrap_gpu_result(std::move(out), a->dtype()), std::move(sh), a->dtype(),
                     a->device());
    }

    // CPU path: hand-rolled p-norm reductions in pure Accelerate-friendly
    // C++ (no LAPACK needed for p-norms). Matrix-induced norms (nuc, p=inf
    // matrix-style) would route through SVD; not supported here.
    Shape out_shape = reduced_shape(a->shape(), axis, keepdims);
    auto out_cpu = allocate_cpu(out_shape, a->dtype());
    const auto& in_cpu = std::get<CpuStorage>(a->storage());

    if (a->dtype() == Dtype::F32) {
        norm_typed<float>(reinterpret_cast<const float*>(in_cpu.ptr.get()),
                          reinterpret_cast<float*>(out_cpu.ptr.get()), a->shape(), axis, keepdims,
                          ord, out_shape);
    } else {
        norm_typed<double>(reinterpret_cast<const double*>(in_cpu.ptr.get()),
                           reinterpret_cast<double*>(out_cpu.ptr.get()), a->shape(), axis, keepdims,
                           ord, out_shape);
    }
    return fresh(Storage{std::move(out_cpu)}, out_shape, a->dtype(), a->device());
}

}  // namespace lucid
