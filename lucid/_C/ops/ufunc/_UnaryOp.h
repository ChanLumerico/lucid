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
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/_BinaryOp.h"   // detail::ensure_grad_fn
#include "../utils/Contiguous.h"  // contiguous_op for auto-materialization

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
    // Auto-materialize non-contiguous CPU inputs.
    const TensorImplPtr a_ptr =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), a_ptr->dtype(), a_ptr->shape()};

    Storage out_storage;
    if (a_ptr->device() == Device::GPU) {
        if constexpr (detail::HasUnaryGpuKernel<Derived>) {
            out_storage = Storage{Derived::gpu_kernel(std::get<GpuStorage>(a_ptr->storage()),
                                                      a_ptr->shape(), a_ptr->dtype())};
        } else {
            ErrorBuilder(Derived::schema_v1.name)
                .not_implemented("GPU kernel not yet implemented (Phase 3.7.x in progress)");
        }
    } else {
        out_storage = Storage{Derived::cpu_kernel(std::get<CpuStorage>(a_ptr->storage()),
                                                  a_ptr->shape(), a_ptr->dtype())};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a_ptr->shape(), a_ptr->dtype(),
                                            a_ptr->device(), /*requires_grad=*/false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    if constexpr (!Derived::kHasGradient) {
        return out;
    } else {
        const bool needs_grad = GradMode::is_enabled() && a->requires_grad();
        if (!needs_grad)
            return out;

        auto a_edge = detail::ensure_grad_fn(a);

        auto bwd = std::make_shared<Derived>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = a->shape();
        bwd->dtype_ = a->dtype();
        bwd->device_ = a->device();
        bwd->input_tensors_ = {a};  // Item #9 — for version check at backward
        if constexpr (Derived::kSavesInput) {
            bwd->saved_inputs_ = {a->storage()};
        }
        if constexpr (Derived::kSavesOutput) {
            bwd->saved_output_ = out->storage();
        }

        std::vector<Edge> edges;
        edges.emplace_back(a_edge, /*input_nr=*/0);
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version()});

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
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
