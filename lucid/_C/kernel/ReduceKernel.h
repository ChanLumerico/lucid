// lucid/_C/kernel/ReduceKernel.h
//
// CRTP base for single-input reduction ops (sum, mean, max, min, etc.).
// Unlike UnaryKernel, the output shape differs from the input shape so
// the backward pass must broadcast the gradient back to the full input
// extent via restore_shape() / broadcast before calling grad_formula.
//
// A concrete reduction op is declared as:
//
//   struct SumOp : ReduceKernel<SumOp> {
//       static constexpr OpSchema schema_v1 = {"sum", ...};
//       static CpuStorage cpu_kernel(const CpuStorage&, const Shape&,
//                                    const std::vector<int>& axes,
//                                    bool keepdims, Dtype);
//       static GpuStorage gpu_kernel(const GpuStorage&, const Shape&,
//                                    const std::vector<int>& axes,
//                                    bool keepdims, Dtype);
//       Storage grad_formula(Storage grad_out);
//   };
//
// The extra state members (reduce_axes_, keepdims_, full_input_shape_)
// are stored on the backward node so grad_formula can reconstruct the
// shape transformation done during forward.

#pragma once

#include <memory>
#include <vector>

#include "../api.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/AutogradNode.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpSchema.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "BinaryKernel.h"
#include "Contig.h"
#include "IKernel.h"

namespace lucid {

namespace detail {

// Satisfied when Derived provides a gpu_kernel static method matching the
// reduction GPU kernel signature (axes and keepdims are additional params).
template <class T>
concept HasReduceGpuKernel =
    requires(GpuStorage a, Shape s, std::vector<int> ax, bool kd, Dtype d) {
        { T::gpu_kernel(a, s, ax, kd, d) } -> std::same_as<GpuStorage>;
    };

// Satisfied when Derived routes through the IBackend dispatch interface
// for the reduction operation.
template <class T>
concept HasReduceDispatch =
    requires(backend::IBackend& be, Storage a, Shape s, std::vector<int> ax, bool kd, Dtype d) {
        { T::dispatch(be, a, s, ax, kd, d) } -> std::same_as<Storage>;
    };

}  // namespace detail

// CRTP reduction-op base. Inherits AutogradNode<Derived, 1>.
// Extra backward state is stored in the instance fields below and
// populated by forward() before the node is attached to the graph.
template <class Derived>
class ReduceKernel : public AutogradNode<Derived, 1>, public kernel::IKernel {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = false;
    static constexpr bool kHasGradient = true;

    std::string_view name() const noexcept override { return Derived::schema_v1.name; }
    std::string node_name() const override { return std::string(Derived::schema_v1.name); }

    // Axes that were reduced during forward(), stored for use in grad_formula.
    std::vector<int> reduce_axes_;
    // Whether keepdims was requested; grad_formula uses this to know whether
    // to insert size-1 dimensions before broadcasting back to full_input_shape_.
    bool keepdims_ = false;
    // The shape of the input tensor before reduction; needed to broadcast
    // the gradient back to the original extent during backward.
    Shape full_input_shape_;

    // Default scale: pass gradient through unchanged (used by SumBackward).
    // Override in Derived for scaling ops like MeanBackward.
    TensorImplPtr scale_graph_grad(const TensorImplPtr& g) { return g; }

    // Forward pass: normalize axes, compute output shape, dispatch kernel,
    // and wire the backward node with the extra reduction state.
    static std::shared_ptr<TensorImpl>
    forward(const std::shared_ptr<TensorImpl>& a, const std::vector<int>& axes_user, bool keepdims);

    // Backward pass: call Derived::grad_formula(grad_out) and return the
    // single input gradient (no shape reduction needed here; grad_formula
    // is responsible for restoring the reduced dimensions).
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward: broadcast grad_out back to full_input_shape_ via
    // sum_op and reshape_op, keeping the computation tracked for 2nd-order grad.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

// Normalize axes to canonical positive form, compute the output shape,
// enforce contiguity on CPU, dispatch the reduce kernel, and record the
// reduction axes + keepdims on the backward node for grad_formula.
template <class Derived>
std::shared_ptr<TensorImpl> ReduceKernel<Derived>::forward(const std::shared_ptr<TensorImpl>& a,
                                                           const std::vector<int>& axes_user,
                                                           bool keepdims) {
    if (!a)
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");

    SchemaGuard sg{Derived::schema_v1, a->dtype(), a->device()};
    const Dtype eff_dt = sg.effective_dtype();

    const TensorImplPtr a_contig =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;
    const TensorImplPtr a_ptr = detail::maybe_cast_for_kernel(a_contig, eff_dt);

    // normalize_axes converts negative indices and deduplicates; the
    // result is a sorted, non-negative axis list suitable for the kernels.
    const auto axes = normalize_axes(axes_user, static_cast<int>(a_ptr->shape().size()));
    Shape out_shape = reduce_output_shape(a_ptr->shape(), axes, keepdims);

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), eff_dt, out_shape};

    Storage out_storage;
    if constexpr (detail::HasReduceDispatch<Derived>) {
        out_storage = Derived::dispatch(backend::Dispatcher::for_device(a_ptr->device()),
                                        a_ptr->storage(), a_ptr->shape(), axes, keepdims, eff_dt);
    } else if (a_ptr->device() == Device::GPU) {
        if constexpr (detail::HasReduceGpuKernel<Derived>) {
            out_storage = Storage{Derived::gpu_kernel(std::get<GpuStorage>(a_ptr->storage()),
                                                      a_ptr->shape(), axes, keepdims, eff_dt)};
        } else {
            ErrorBuilder(Derived::schema_v1.name).not_implemented("GPU kernel not yet implemented");
        }
    } else {
        out_storage = Storage{Derived::cpu_kernel(std::get<CpuStorage>(a_ptr->storage()),
                                                  a_ptr->shape(), axes, keepdims, eff_dt)};
    }

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), out_shape, eff_dt, a->device(), false);
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
        bwd->dtype_ = eff_dt;
        bwd->device_ = a->device();
        bwd->input_tensors_ = {a};
        if constexpr (Derived::kSavesInput)
            bwd->saved_inputs_ = {a_ptr->storage()};
        if constexpr (Derived::kSavesOutput)
            bwd->saved_output_ = out->storage();
        bwd->saved_impl_inputs_ = {a};

        // Store reduction metadata so grad_formula can invert the shape change.
        bwd->reduce_axes_ = axes;
        bwd->keepdims_ = keepdims;
        bwd->full_input_shape_ = a->shape();

        std::vector<Edge> edges;
        edges.emplace_back(a_edge, a->grad_output_nr());
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version()});

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return out;
    }
}

// Delegate entirely to Derived::grad_formula. Unlike UnaryKernel::apply(),
// no reduce_grad_to_shape call is made here because the reduction kernel's
// grad_formula is responsible for broadcasting grad_out back to
// full_input_shape_ using the saved reduce_axes_ and keepdims_ fields.
template <class Derived>
std::vector<Storage> ReduceKernel<Derived>::apply(Storage grad_out) {
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {std::move(dx)};
}

// Graph-mode backward for reductions: broadcast the incoming gradient back to
// full_input_shape_ using unsqueeze + broadcast_to_op so the expansion itself
// is tracked in the autograd graph.
template <class Derived>
std::vector<TensorImplPtr> ReduceKernel<Derived>::apply_for_graph(const TensorImplPtr& grad_out) {
    extern TensorImplPtr broadcast_to_op(const TensorImplPtr&, const Shape&);
    extern TensorImplPtr unsqueeze_op(const TensorImplPtr&, int);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    TensorImplPtr g = grad_out;

    // If keepdims was false, the reduced axes are missing from grad_out.
    // Re-insert size-1 axes at the correct positions before broadcasting.
    if (!this->keepdims_) {
        std::vector<int> sorted_axes = this->reduce_axes_;
        std::sort(sorted_axes.begin(), sorted_axes.end());
        for (int ax : sorted_axes) {
            g = unsqueeze_op(g, ax);
        }
    }

    // Broadcast to the original input shape — this replicates the gradient
    // to all elements that were reduced (sum backward).
    auto dx = broadcast_to_op(g, this->full_input_shape_);

    // Derived may scale the gradient (e.g. MeanBackward divides by n_reduced).
    return {static_cast<Derived*>(this)->scale_graph_grad(dx)};
}

}  // namespace lucid
