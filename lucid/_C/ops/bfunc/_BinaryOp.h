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
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"

namespace lucid {

namespace detail {

template <class T>
concept HasGpuKernel = requires(GpuStorage a, GpuStorage b, Shape s, Dtype d) {
    { T::gpu_kernel(a, b, s, d) } -> std::same_as<GpuStorage>;
};

// NumPy-style broadcast shape resolution.  Returns the broadcast result
// shape; throws ShapeMismatch when shapes are incompatible.  Right-aligned:
// missing leading axes are treated as size 1, and any axis pair (1, n) or
// (n, 1) is allowed.
inline Shape broadcast_shapes(const Shape& a, const Shape& b) {
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
            throw ShapeMismatch(a, b, "broadcast: incompatible shapes");
        }
    }
    return out;
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
            throw NotImplementedError("broadcast: dtype not supported");
    }
    return out;
}

/// Walk a tensor; if it's a leaf needing grad, attach (or reuse) an
/// AccumulateGrad as its grad_fn. Else return its existing grad_fn.
/// Returns null when the tensor is non-grad — the caller skips that edge.
inline std::shared_ptr<Node> ensure_grad_fn(const std::shared_ptr<TensorImpl>& t) {
    if (!t || !t->requires_grad_)
        return nullptr;
    if (t->grad_fn_)
        return t->grad_fn_;
    if (t->is_leaf_) {
        t->grad_fn_ = std::make_shared<AccumulateGrad>(t);
        return t->grad_fn_;
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
        throw LucidError(std::string(Derived::schema_v1.name) + ": null input tensor");
    }
    if (a->dtype_ != b->dtype_) {
        throw DtypeMismatch(std::string(dtype_name(a->dtype_)), std::string(dtype_name(b->dtype_)),
                            std::string(Derived::schema_v1.name));
    }
    if (a->device_ != b->device_) {
        throw DeviceMismatch(std::string(device_name(a->device_)),
                             std::string(device_name(b->device_)),
                             std::string(Derived::schema_v1.name));
    }
    // Item #8 — non-contiguous input guard. Phase 3.4 view ops produce
    // non-contiguous tensors; user must call .contiguous() before compute.
    // CPU only — GPU contiguity is internal to MLX, not exposed via stride_.
    if (a->device_ == Device::CPU && (!a->is_contiguous() || !b->is_contiguous())) {
        throw NotImplementedError(
            std::string(Derived::schema_v1.name) +
            ": non-contiguous input not supported (call .contiguous() first)");
    }

    // Resolve broadcast output shape (NumPy semantics). Backward already
    // accumulates back to each operand's original shape via
    // reduce_grad_to_shape, so we only have to materialize the broadcast
    // forward inputs here.
    Shape out_shape =
        (a->shape_ == b->shape_) ? a->shape_ : detail::broadcast_shapes(a->shape_, b->shape_);

    OpScope scope{Derived::schema_v1.name, a->device_, a->dtype_, out_shape};

    Storage out_storage;
    if (a->device_ == Device::GPU) {
        if constexpr (detail::HasGpuKernel<Derived>) {
            // Materialize broadcast on GPU via mlx::broadcast_to (lazy view
            // → contiguous() to give the kernel a real buffer).
            const auto& ga = std::get<GpuStorage>(a->storage_);
            const auto& gb = std::get<GpuStorage>(b->storage_);
            ::mlx::core::array a_arr = (a->shape_ == out_shape)
                                           ? *ga.arr
                                           : ::mlx::core::contiguous(::mlx::core::broadcast_to(
                                                 *ga.arr, gpu::to_mlx_shape(out_shape)));
            ::mlx::core::array b_arr = (b->shape_ == out_shape)
                                           ? *gb.arr
                                           : ::mlx::core::contiguous(::mlx::core::broadcast_to(
                                                 *gb.arr, gpu::to_mlx_shape(out_shape)));
            GpuStorage a_g, b_g;
            a_g.arr = std::make_shared<::mlx::core::array>(std::move(a_arr));
            b_g.arr = std::make_shared<::mlx::core::array>(std::move(b_arr));
            out_storage = Storage{Derived::gpu_kernel(a_g, b_g, out_shape, a->dtype_)};
        } else {
            throw NotImplementedError(std::string(Derived::schema_v1.name) +
                                      ": GPU kernel not yet implemented (Phase 3.7.x in progress)");
        }
    } else {
        // Materialize broadcast on CPU when shapes differ.
        const CpuStorage& a_raw = std::get<CpuStorage>(a->storage_);
        const CpuStorage& b_raw = std::get<CpuStorage>(b->storage_);
        CpuStorage a_buf;
        CpuStorage b_buf;
        const CpuStorage* a_use = &a_raw;
        const CpuStorage* b_use = &b_raw;
        if (a->shape_ != out_shape) {
            a_buf = detail::broadcast_cpu(a_raw, a->shape_, out_shape, a->dtype_);
            a_use = &a_buf;
        }
        if (b->shape_ != out_shape) {
            b_buf = detail::broadcast_cpu(b_raw, b->shape_, out_shape, a->dtype_);
            b_use = &b_buf;
        }
        out_storage = Storage{Derived::cpu_kernel(*a_use, *b_use, out_shape, a->dtype_)};
    }

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype_, a->device_,
                                     /*requires_grad=*/false);

    // Approximate flop count for the profiler.
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    const bool needs_grad = GradMode::is_enabled() && (a->requires_grad_ || b->requires_grad_);
    if (!needs_grad)
        return out;

    auto a_edge = detail::ensure_grad_fn(a);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<Derived>();
    bwd->input_shapes_ = {a->shape_, b->shape_};
    bwd->out_shape_ = out->shape_;  // broadcasted output shape, not a->shape_
    bwd->dtype_ = a->dtype_;
    bwd->device_ = a->device_;
    bwd->input_tensors_ = {a, b};  // Item #9 — for version check at backward
    if constexpr (Derived::kSavesInputs) {
        bwd->saved_inputs_ = {a->storage_, b->storage_};
    }

    std::vector<Edge> edges;
    edges.emplace_back(a_edge, /*input_nr=*/0);
    edges.emplace_back(b_edge, /*input_nr=*/0);
    bwd->set_next_edges(std::move(edges));
    bwd->set_saved_versions({a->version_, b->version_});

    out->grad_fn_ = std::move(bwd);
    out->is_leaf_ = false;
    out->requires_grad_ = true;
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
