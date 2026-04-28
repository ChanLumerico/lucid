#pragma once

// =====================================================================
// Lucid C++ engine — UnaryOp CRTP base.
// =====================================================================
//
// All single-input element-wise ops inherit from `UnaryOp<Derived>`. Derived
// implements:
//
//   1. `static const OpSchema schema_v1;`
//   2. `static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& out_shape, Dtype dt);`
//   3. `Storage grad_formula(const Storage& grad_out);`  — single output
//   4. (optional) `static constexpr bool kSavesInput  = false;` — input not needed in backward
//   5. (optional) `static constexpr bool kSavesOutput = true;`  — backward uses output value
//   6. (optional) `static constexpr bool kHasGradient = false;` — sign/floor/ceil etc.
//
// The base handles validation, allocation, profiler scope, autograd graph
// wiring, and broadcast-undo (a no-op for unary; same shape in/out).
//
// Layer: autograd/ops/unary/. Depends on core/, autograd/, backend/cpu/.

#include <memory>
#include <utility>
#include <vector>

#include "../../api.h"
#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Exceptions.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn

namespace lucid {

namespace detail {

template <class T>
concept HasUnaryGpuKernel = requires(GpuStorage a, Shape s, Dtype d) {
    { T::gpu_kernel(a, s, d) } -> std::same_as<GpuStorage>;
};

}  // namespace detail

template <class Derived>
class UnaryOp : public FuncOp<Derived, 1> {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = false;
    static constexpr bool kHasGradient = true;

    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a);

    std::vector<Storage> apply(Storage grad_out) override;
};

// ---------------- implementation ----------------

template <class Derived>
std::shared_ptr<TensorImpl> UnaryOp<Derived>::forward(const std::shared_ptr<TensorImpl>& a) {
    if (!a) {
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");
    }
    // Item #8 — non-contiguous input guard. CPU only; GPU stride is internal.
    if (a->device_ == Device::CPU && !a->is_contiguous()) {
        ErrorBuilder(Derived::schema_v1.name)
            .not_implemented("non-contiguous input not supported (call .contiguous() first)");
    }

    OpScopeFull scope{Derived::schema_v1.name, a->device_, a->dtype_, a->shape_};

    Storage out_storage;
    if (a->device_ == Device::GPU) {
        if constexpr (detail::HasUnaryGpuKernel<Derived>) {
            out_storage = Storage{
                Derived::gpu_kernel(std::get<GpuStorage>(a->storage_), a->shape_, a->dtype_)};
        } else {
            ErrorBuilder(Derived::schema_v1.name)
                .not_implemented("GPU kernel not yet implemented (Phase 3.7.x in progress)");
        }
    } else {
        out_storage =
            Storage{Derived::cpu_kernel(std::get<CpuStorage>(a->storage_), a->shape_, a->dtype_)};
    }

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), a->shape_, a->dtype_, a->device_,
                                     /*requires_grad=*/false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    if constexpr (!Derived::kHasGradient) {
        return out;
    } else {
        const bool needs_grad = GradMode::is_enabled() && a->requires_grad_;
        if (!needs_grad)
            return out;

        auto a_edge = detail::ensure_grad_fn(a);

        auto bwd = std::make_shared<Derived>();
        bwd->input_shapes_ = {a->shape_};
        bwd->out_shape_ = a->shape_;
        bwd->dtype_ = a->dtype_;
        bwd->device_ = a->device_;
        bwd->input_tensors_ = {a};  // Item #9 — for version check at backward
        if constexpr (Derived::kSavesInput) {
            bwd->saved_inputs_ = {a->storage_};
        }
        if constexpr (Derived::kSavesOutput) {
            bwd->saved_output_ = out->storage_;
        }

        std::vector<Edge> edges;
        edges.emplace_back(a_edge, /*input_nr=*/0);
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version_});

        out->grad_fn_ = std::move(bwd);
        out->is_leaf_ = false;
        out->requires_grad_ = true;
        return out;
    }
}

template <class Derived>
std::vector<Storage> UnaryOp<Derived>::apply(Storage grad_out) {
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    // Unary forward keeps shape; reduce-to-shape becomes a no-op clone, but we
    // route through it so the engine always owns its grad buffer.
    return {reduce_grad_to_shape(dx, this->out_shape_, this->input_shapes_[0], this->dtype_,
                                 this->device_)};
}

}  // namespace lucid
