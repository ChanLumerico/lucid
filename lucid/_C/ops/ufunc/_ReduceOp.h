#pragma once

// =====================================================================
// Lucid C++ engine — ReduceOp CRTP base.
// =====================================================================
//
// Reduction ops collapse one or more axes. Each Derived implements:
//
//   1. `static const OpSchema schema_v1;`
//   2. `static CpuStorage cpu_kernel(const CpuStorage& a,
//                                    const Shape& input_shape,
//                                    const std::vector<int>& axes,
//                                    bool keepdims, Dtype dt);`
//   3. `Storage grad_formula(const Storage& grad_out);`
//      — backward returns input-shaped storage, typically via
//        `broadcast_back_for_reduce` followed by op-specific scaling.
//
// Forward signature is uniform: `(input, axes, keepdims)`. The base stores
// the metadata (axes, keepdims, input_shape) and invokes the kernel; backward
// uses the stored metadata to broadcast the gradient back.
//
// Layer: autograd/ops/reduce/.

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
concept HasReduceGpuKernel =
    requires(GpuStorage a, Shape s, std::vector<int> ax, bool kd, Dtype d) {
        { T::gpu_kernel(a, s, ax, kd, d) } -> std::same_as<GpuStorage>;
    };

}  // namespace detail

template <class Derived>
class ReduceOp : public FuncOp<Derived, 1> {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = false;
    static constexpr bool kHasGradient = true;

    // Reduce-specific saved state — populated by `forward`, consumed by
    // `apply` -> `Derived::grad_formula`.
    std::vector<int> reduce_axes_;
    bool keepdims_ = false;
    Shape full_input_shape_;

    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a,
                                               const std::vector<int>& axes_user,
                                               bool keepdims);

    std::vector<Storage> apply(Storage grad_out) override;
};

// ---------------- implementation ----------------

template <class Derived>
std::shared_ptr<TensorImpl> ReduceOp<Derived>::forward(const std::shared_ptr<TensorImpl>& a,
                                                       const std::vector<int>& axes_user,
                                                       bool keepdims) {
    if (!a) {
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");
    }
    // Auto-materialize non-contiguous CPU inputs.
    const TensorImplPtr a_ptr =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;

    const auto axes = normalize_axes(axes_user, static_cast<int>(a_ptr->shape().size()));
    Shape out_shape = reduce_output_shape(a_ptr->shape(), axes, keepdims);

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), a_ptr->dtype(), out_shape};

    Storage out_storage;
    if (a_ptr->device() == Device::GPU) {
        if constexpr (detail::HasReduceGpuKernel<Derived>) {
            out_storage =
                Storage{Derived::gpu_kernel(std::get<GpuStorage>(a_ptr->storage()), a_ptr->shape(),
                                            axes, keepdims, a_ptr->dtype())};
        } else {
            ErrorBuilder(Derived::schema_v1.name)
                .not_implemented("GPU kernel not yet implemented (Phase 3.7.x in progress)");
        }
    } else {
        out_storage = Storage{Derived::cpu_kernel(std::get<CpuStorage>(a_ptr->storage()),
                                                  a_ptr->shape(), axes, keepdims, a_ptr->dtype())};
    }

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(), a->device(),
                                     /*requires_grad=*/false);
    scope.set_flops(static_cast<std::int64_t>(a->numel()));

    if constexpr (!Derived::kHasGradient) {
        return out;
    } else {
        const bool needs_grad = GradMode::is_enabled() && a->requires_grad();
        if (!needs_grad)
            return out;

        auto a_edge = detail::ensure_grad_fn(a);

        auto bwd = std::make_shared<Derived>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = out_shape;
        bwd->dtype_ = a->dtype();
        bwd->device_ = a->device();
        bwd->input_tensors_ = {a};  // Item #9 — for version check
        if constexpr (Derived::kSavesInput)
            bwd->saved_inputs_ = {a->storage()};
        if constexpr (Derived::kSavesOutput)
            bwd->saved_output_ = out->storage();

        bwd->reduce_axes_ = axes;
        bwd->keepdims_ = keepdims;
        bwd->full_input_shape_ = a->shape();

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
std::vector<Storage> ReduceOp<Derived>::apply(Storage grad_out) {
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {std::move(dx)};
}

}  // namespace lucid
