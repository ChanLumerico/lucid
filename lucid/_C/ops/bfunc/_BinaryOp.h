#pragma once

// =====================================================================
// Lucid C++ engine — BinaryOp CRTP base.
// =====================================================================
//
// All element-wise (and reduction-via-broadcast) binary ops inherit from
// `BinaryOp<Derived>`. Derived implements:
//
//   1. `static const OpSchema schema_v1;`
//   2. `static CpuStorage cpu_kernel(const CpuStorage& a, const CpuStorage& b,
//                                    const Shape& out_shape, Dtype dt);`
//   3. `std::pair<Storage, Storage> grad_formula(const Storage& grad_out);`
//   4. (optional) `static constexpr bool kSavesInputs = false;` — set to
//      false when grad is independent of input values (e.g. add, sub).
//
// The base handles:
//   - validation (dtype, device match; Phase 3.0 also requires equal shapes,
//     broadcast forward arrives in Phase 3.1)
//   - output allocation
//   - profiler scope
//   - autograd graph wiring (gather edges, save metadata, set grad_fn_)
//   - broadcast-undo on backward (reduce_grad_to_shape)
//
// Layer: autograd/ops/binary/. Depends on core/, backend/cpu/.

#include <memory>
#include <utility>
#include <vector>

#include <mlx/ops.h>

#include "../../api.h"
#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/AmpPolicy.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Result.h"
#include "../../core/Scope.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../utils/Contiguous.h"  // contiguous_op for auto-materialization

namespace lucid {

namespace detail {

template <class T>
concept HasGpuKernel = requires(GpuStorage a, GpuStorage b, Shape s, Dtype d) {
    { T::gpu_kernel(a, b, s, d) } -> std::same_as<GpuStorage>;
};

// NumPy-style broadcast shape resolution. Right-aligned: missing leading
// axes are treated as size 1, and any axis pair (1, n) or (n, 1) is
// allowed.
//
// Two flavors:
//   - `try_broadcast_shapes(a, b)` returns `Result<Shape>`. Internal
//     callers in fallible fast-paths (e.g. dispatch fall-through, AMP
//     promotion try-cast) use this so the failure path stays branch-y
//     instead of paying exception unwind cost on every miss.
//   - `broadcast_shapes(a, b)` is the public/forward-time variant that
//     throws `ShapeMismatch` on incompatible shapes — same behavior the
//     bfunc forwards have always had.
//
// `broadcast_shapes` is implemented in terms of `try_broadcast_shapes`
// so the inference logic lives in one place.
inline Result<Shape> try_broadcast_shapes(const Shape& a, const Shape& b) {
    const std::size_t ra = a.size();
    const std::size_t rb = b.size();
    const std::size_t r = ra > rb ? ra : rb;
    Shape out(r, 1);
    for (std::size_t i = 0; i < r; ++i) {
        const std::size_t ai = (ra >= r - i) ? ra - (r - i) : SIZE_MAX;
        const std::size_t bi = (rb >= r - i) ? rb - (r - i) : SIZE_MAX;
        const std::int64_t da = (ai != SIZE_MAX) ? a[ai] : 1;
        const std::int64_t db = (bi != SIZE_MAX) ? b[bi] : 1;
        if (da == db || da == 1 || db == 1) {
            out[i] = da == 1 ? db : da;
        } else {
            return Err(ErrorCode::ShapeMismatch, "broadcast: incompatible shapes");
        }
    }
    return Ok(std::move(out));
}

inline Shape broadcast_shapes(const Shape& a, const Shape& b) {
    auto r = try_broadcast_shapes(a, b);
    if (r.is_ok())
        return std::move(r).value();
    // Throw the typed exception with full shape payload — `try_*` only
    // knows the message; the throwing form preserves the structured
    // ShapeMismatch fields for the pybind11 translator.
    throw ShapeMismatch(a, b, "broadcast: incompatible shapes");
}

// CPU broadcast-materialize: read input contiguously per output element.
inline CpuStorage broadcast_cpu(const CpuStorage& src,
                                const Shape& src_shape,
                                const Shape& out_shape,
                                Dtype dt) {
    const std::size_t ndim_out = out_shape.size();
    const std::size_t ndim_in = src_shape.size();
    Shape padded(ndim_out, 1);
    for (std::size_t i = 0; i < ndim_in; ++i) {
        padded[ndim_out - ndim_in + i] = src_shape[i];
    }
    // Strides over the *padded* input shape, with 0 where the axis was 1.
    std::vector<std::size_t> in_str(ndim_out, 0);
    std::size_t s = 1;
    for (std::ptrdiff_t d = (std::ptrdiff_t)ndim_out - 1; d >= 0; --d) {
        in_str[d] = (padded[d] == 1) ? 0 : s;
        s *= static_cast<std::size_t>(padded[d]);
    }
    const std::size_t out_numel = shape_numel(out_shape);
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = out_numel * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* sp = reinterpret_cast<const T*>(src.ptr.get());
        T* dp = reinterpret_cast<T*>(out.ptr.get());
        std::vector<std::size_t> coord(ndim_out, 0);
        for (std::size_t f = 0; f < out_numel; ++f) {
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim_out; ++d)
                in_flat += coord[d] * in_str[d];
            dp[f] = sp[in_flat];
            for (std::ptrdiff_t d = (std::ptrdiff_t)ndim_out - 1; d >= 0; --d) {
                if (++coord[d] < static_cast<std::size_t>(out_shape[d]))
                    break;
                coord[d] = 0;
            }
        }
    };
    switch (dt) {
        case Dtype::F32:
            run(float{});
            break;
        case Dtype::F64:
            run(double{});
            break;
        case Dtype::I32:
            run(std::int32_t{});
            break;
        case Dtype::I64:
            run(std::int64_t{});
            break;
        case Dtype::I16:
            run(std::int16_t{});
            break;
        case Dtype::I8:
        case Dtype::Bool:
            run(std::uint8_t{});
            break;
        default:
            ErrorBuilder("broadcast").not_implemented("dtype not supported");
    }
    return out;
}

/// Walk a tensor; if it's a leaf needing grad, attach (or reuse) an
/// AccumulateGrad as its grad_fn. Else return its existing grad_fn.
/// Returns null when the tensor is non-grad — the caller skips that edge.
inline std::shared_ptr<Node> ensure_grad_fn(const std::shared_ptr<TensorImpl>& t) {
    if (!t || !t->requires_grad())
        return nullptr;
    if (t->grad_fn())
        return t->grad_fn();
    if (t->is_leaf()) {
        t->set_grad_fn(std::make_shared<AccumulateGrad>(t));
        return t->grad_fn();
    }
    return nullptr;
}

}  // namespace detail

template <class Derived>
class BinaryOp : public FuncOp<Derived, 2> {
public:
    /// Forward pass: validate, compute, wire grad graph if needed.
    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a,
                                               const std::shared_ptr<TensorImpl>& b);

    /// Backward — calls Derived::grad_formula then applies broadcast-undo.
    std::vector<Storage> apply(Storage grad_out) override;

    static constexpr bool kSavesInputs = true;

protected:
    /// Materialize a saved input at `out_shape_` (broadcasting if needed).
    /// Used by grad_formula in derivatives that consume the inputs (e.g.
    /// Mul/Div/Pow), which need them aligned with grad_out's broadcast shape.
    Storage saved_input_broadcasted(std::size_t k) const {
        const Shape& src = this->input_shapes_[k];
        if (src == this->out_shape_)
            return this->saved_inputs_[k];
        if (this->device_ == Device::GPU) {
            const auto& g = std::get<GpuStorage>(this->saved_inputs_[k]);
            auto bcast = ::mlx::core::contiguous(
                ::mlx::core::broadcast_to(*g.arr, gpu::to_mlx_shape(this->out_shape_)));
            return Storage{gpu::wrap_mlx_array(std::move(bcast), this->dtype_)};
        }
        const auto& c = std::get<CpuStorage>(this->saved_inputs_[k]);
        return Storage{detail::broadcast_cpu(c, src, this->out_shape_, this->dtype_)};
    }
};

// ---------------- implementation ----------------

template <class Derived>
std::shared_ptr<TensorImpl> BinaryOp<Derived>::forward(const std::shared_ptr<TensorImpl>& a,
                                                       const std::shared_ptr<TensorImpl>& b) {
    if (!a || !b) {
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");
    }
    if (a->dtype() != b->dtype()) {
        throw DtypeMismatch(std::string(dtype_name(a->dtype())),
                            std::string(dtype_name(b->dtype())),
                            std::string(Derived::schema_v1.name));
    }
    if (a->device() != b->device()) {
        throw DeviceMismatch(std::string(device_name(a->device())),
                             std::string(device_name(b->device())),
                             std::string(Derived::schema_v1.name));
    }
    // Auto-materialize non-contiguous CPU inputs — view ops produce tensors
    // with non-contiguous strides; kernels require contiguous buffers.
    // GPU contiguity is handled internally by MLX, so GPU inputs pass as-is.
    const TensorImplPtr a_ptr =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;
    const TensorImplPtr b_ptr =
        (b->device() == Device::CPU && !b->is_contiguous()) ? contiguous_op(b) : b;

    // Resolve broadcast output shape (NumPy semantics). Backward already
    // accumulates back to each operand's original shape via
    // reduce_grad_to_shape, so we only have to materialize the broadcast
    // forward inputs here.
    Shape out_shape = (a_ptr->shape() == b_ptr->shape())
                          ? a_ptr->shape()
                          : detail::broadcast_shapes(a_ptr->shape(), b_ptr->shape());

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), a_ptr->dtype(), out_shape};

    Storage out_storage;
    if (a_ptr->device() == Device::GPU) {
        if constexpr (detail::HasGpuKernel<Derived>) {
            // Materialize broadcast on GPU via mlx::broadcast_to (lazy view
            // → contiguous() to give the kernel a real buffer).
            const auto& ga = std::get<GpuStorage>(a_ptr->storage());
            const auto& gb = std::get<GpuStorage>(b_ptr->storage());
            ::mlx::core::array a_arr = (a_ptr->shape() == out_shape)
                                           ? *ga.arr
                                           : ::mlx::core::contiguous(::mlx::core::broadcast_to(
                                                 *ga.arr, gpu::to_mlx_shape(out_shape)));
            ::mlx::core::array b_arr = (b_ptr->shape() == out_shape)
                                           ? *gb.arr
                                           : ::mlx::core::contiguous(::mlx::core::broadcast_to(
                                                 *gb.arr, gpu::to_mlx_shape(out_shape)));
            GpuStorage a_g, b_g;
            a_g.arr = std::make_shared<::mlx::core::array>(std::move(a_arr));
            b_g.arr = std::make_shared<::mlx::core::array>(std::move(b_arr));
            out_storage = Storage{Derived::gpu_kernel(a_g, b_g, out_shape, a_ptr->dtype())};
        } else {
            ErrorBuilder(Derived::schema_v1.name)
                .not_implemented("GPU kernel not yet implemented (Phase 3.7.x in progress)");
        }
    } else {
        // Materialize broadcast on CPU when shapes differ.
        const CpuStorage& a_raw = std::get<CpuStorage>(a_ptr->storage());
        const CpuStorage& b_raw = std::get<CpuStorage>(b_ptr->storage());
        CpuStorage a_buf;
        CpuStorage b_buf;
        const CpuStorage* a_use = &a_raw;
        const CpuStorage* b_use = &b_raw;
        if (a_ptr->shape() != out_shape) {
            a_buf = detail::broadcast_cpu(a_raw, a_ptr->shape(), out_shape, a_ptr->dtype());
            a_use = &a_buf;
        }
        if (b_ptr->shape() != out_shape) {
            b_buf = detail::broadcast_cpu(b_raw, b_ptr->shape(), out_shape, a_ptr->dtype());
            b_use = &b_buf;
        }
        out_storage = Storage{Derived::cpu_kernel(*a_use, *b_use, out_shape, a_ptr->dtype())};
    }

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(), a->device(),
                                     /*requires_grad=*/false);

    // Approximate flop count for the profiler.
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    const bool needs_grad = GradMode::is_enabled() && (a->requires_grad() || b->requires_grad());
    if (!needs_grad)
        return out;

    auto a_edge = detail::ensure_grad_fn(a);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<Derived>();
    bwd->input_shapes_ = {a->shape(), b->shape()};
    bwd->out_shape_ = out->shape();  // broadcasted output shape, not a->shape()
    bwd->dtype_ = a->dtype();
    bwd->device_ = a->device();
    bwd->input_tensors_ = {a, b};  // Item #9 — for version check at backward
    if constexpr (Derived::kSavesInputs) {
        bwd->saved_inputs_ = {a->storage(), b->storage()};
    }

    std::vector<Edge> edges;
    edges.emplace_back(a_edge, /*input_nr=*/0);
    edges.emplace_back(b_edge, /*input_nr=*/0);
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions({a->version(), b->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

template <class Derived>
std::vector<Storage> BinaryOp<Derived>::apply(Storage grad_out) {
    auto [da, db] = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {
        reduce_grad_to_shape(da, this->out_shape_, this->input_shapes_[0], this->dtype_,
                             this->device_),
        reduce_grad_to_shape(db, this->out_shape_, this->input_shapes_[1], this->dtype_,
                             this->device_),
    };
}

}  // namespace lucid
