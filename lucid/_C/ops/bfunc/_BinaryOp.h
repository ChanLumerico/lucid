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

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"

namespace lucid {

namespace detail {

template <class T>
concept HasGpuKernel = requires(GpuStorage a, GpuStorage b, Shape s, Dtype d) {
    { T::gpu_kernel(a, b, s, d) } -> std::same_as<GpuStorage>;
};

/// Walk a tensor; if it's a leaf needing grad, attach (or reuse) an
/// AccumulateGrad as its grad_fn. Else return its existing grad_fn.
/// Returns null when the tensor is non-grad — the caller skips that edge.
inline std::shared_ptr<Node> ensure_grad_fn(const std::shared_ptr<TensorImpl>& t) {
    if (!t || !t->requires_grad_) return nullptr;
    if (t->grad_fn_) return t->grad_fn_;
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
    static std::shared_ptr<TensorImpl> forward(
        const std::shared_ptr<TensorImpl>& a,
        const std::shared_ptr<TensorImpl>& b);

    /// Backward — calls Derived::grad_formula then applies broadcast-undo.
    std::vector<Storage> apply(Storage grad_out) override;

    static constexpr bool kSavesInputs = true;
};

// ---------------- implementation ----------------

template <class Derived>
std::shared_ptr<TensorImpl> BinaryOp<Derived>::forward(
    const std::shared_ptr<TensorImpl>& a,
    const std::shared_ptr<TensorImpl>& b) {
    if (!a || !b) {
        throw LucidError(std::string(Derived::schema_v1.name) +
                         ": null input tensor");
    }
    if (a->dtype_ != b->dtype_) {
        throw DtypeMismatch(std::string(dtype_name(a->dtype_)),
                            std::string(dtype_name(b->dtype_)),
                            std::string(Derived::schema_v1.name));
    }
    if (a->device_ != b->device_) {
        throw DeviceMismatch(std::string(device_name(a->device_)),
                             std::string(device_name(b->device_)),
                             std::string(Derived::schema_v1.name));
    }
    if (a->shape_ != b->shape_) {
        // Phase 3.0: equal shapes only. Broadcast forward arrives in Phase 3.1
        // (will be a separate op-level concern; backward already supports it
        // via reduce_grad_to_shape).
        throw ShapeMismatch(a->shape_, b->shape_,
                            std::string(Derived::schema_v1.name) +
                                " (broadcast forward not yet implemented)");
    }
    // Item #8 — non-contiguous input guard. Phase 3.4 view ops produce
    // non-contiguous tensors; user must call .contiguous() before compute.
    // CPU only — GPU contiguity is internal to MLX, not exposed via stride_.
    if (a->device_ == Device::CPU &&
        (!a->is_contiguous() || !b->is_contiguous())) {
        throw NotImplementedError(
            std::string(Derived::schema_v1.name) +
            ": non-contiguous input not supported (call .contiguous() first)");
    }

    OpScope scope{Derived::schema_v1.name, a->device_, a->dtype_, a->shape_};

    Storage out_storage;
    if (a->device_ == Device::GPU) {
        if constexpr (detail::HasGpuKernel<Derived>) {
            out_storage = Storage{Derived::gpu_kernel(
                std::get<GpuStorage>(a->storage_),
                std::get<GpuStorage>(b->storage_), a->shape_, a->dtype_)};
        } else {
            throw NotImplementedError(
                std::string(Derived::schema_v1.name) +
                ": GPU kernel not yet implemented (Phase 3.7.x in progress)");
        }
    } else {
        // CPU forward — Derived returns the output Storage. Allocation is the
        // kernel's responsibility (lets it choose layout for fused ops).
        out_storage =
            Storage{Derived::cpu_kernel(std::get<CpuStorage>(a->storage_),
                                        std::get<CpuStorage>(b->storage_),
                                        a->shape_, a->dtype_)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a->shape_,
                                            a->dtype_, a->device_,
                                            /*requires_grad=*/false);

    // Approximate flop count for the profiler.
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    const bool needs_grad = GradMode::is_enabled() &&
                            (a->requires_grad_ || b->requires_grad_);
    if (!needs_grad) return out;

    auto a_edge = detail::ensure_grad_fn(a);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<Derived>();
    bwd->input_shapes_ = {a->shape_, b->shape_};
    bwd->out_shape_ = a->shape_;
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
        reduce_grad_to_shape(da, this->out_shape_, this->input_shapes_[0],
                             this->dtype_, this->device_),
        reduce_grad_to_shape(db, this->out_shape_, this->input_shapes_[1],
                             this->dtype_, this->device_),
    };
}

}  // namespace lucid
