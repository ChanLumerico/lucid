// lucid/_C/kernel/UnaryKernel.h
//
// CRTP base for single-input, single-output op kernels. A concrete op
// declares itself as:
//
//   struct ReluOp : UnaryKernel<ReluOp> {
//       static constexpr OpSchema schema_v1 = {"relu", ...};
//       static CpuStorage cpu_kernel(const CpuStorage&, const Shape&, Dtype);
//       static GpuStorage gpu_kernel(const GpuStorage&, const Shape&, Dtype);
//       Storage grad_formula(Storage grad_out);
//   };
//
// UnaryKernel::forward() then handles dtype negotiation (SchemaGuard),
// contiguous enforcement, dispatch to cpu_kernel / gpu_kernel / dispatch,
// and autograd graph wiring. Derived ops only implement the math.
//
// Dispatch priority inside forward():
//   1. Derived::dispatch(IBackend&, ...)  — if HasUnaryDispatch<Derived>
//   2. Derived::gpu_kernel(GpuStorage, ...)  — GPU path
//   3. Derived::cpu_kernel(CpuStorage, ...)  — CPU path
//
// The apply() override calls Derived::grad_formula(grad_out) and
// broadcasts the result back to the original input shape.

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
// signature required for unary GPU dispatch.
template <class T>
concept HasUnaryGpuKernel = requires(GpuStorage a, Shape s, Dtype d) {
    { T::gpu_kernel(a, s, d) } -> std::same_as<GpuStorage>;
};

// Satisfied when Derived provides a dispatch static method that routes
// through the IBackend abstraction (used by ops with Accelerate BNNS
// or other backend-specific implementations).
template <class T>
concept HasUnaryDispatch = requires(backend::IBackend& be, Storage a, Shape s, Dtype d) {
    { T::dispatch(be, a, s, d) } -> std::same_as<Storage>;
};

}  // namespace detail

// CRTP unary-op base. Inherits AutogradNode<Derived, 1> (one saved input)
// and IKernel so it can be used via the polymorphic IKernel* interface.
//
// kSavesInput, kSavesOutput, and kHasGradient may be overridden by
// Derived as static constexpr bool constants to suppress unnecessary
// allocations when e.g. no backward pass is required.
template <class Derived>
class UnaryKernel : public AutogradNode<Derived, 1>, public kernel::IKernel {
public:
    // Whether to snapshot a->storage() for use in grad_formula(). When
    // the gradient formula does not need the forward input (e.g. for
    // ReLU using the saved output) set to false to avoid the copy.
    static constexpr bool kSavesInput = true;
    // Whether to snapshot the output storage for use in grad_formula().
    static constexpr bool kSavesOutput = false;
    // Set to false for in-place or non-differentiable ops to skip graph wiring.
    static constexpr bool kHasGradient = true;

    // Default graph-mode gradient formula: throws NotImplementedError.
    // Override in concrete Derived classes to support create_graph=True.
    TensorImplPtr grad_formula_impl(const TensorImplPtr& /*g*/, const TensorImplPtr& /*a*/,
                                    const TensorImplPtr& /*out*/) {
        throw std::runtime_error(
            "create_graph=True is not supported for op '" +
            std::string(Derived::schema_v1.name) + "'. "
            "Implement grad_formula_impl() to add support.");
    }

    std::string_view name() const noexcept override { return Derived::schema_v1.name; }
    std::string node_name() const override { return std::string(Derived::schema_v1.name); }

    // Forward pass: validate, cast dtype, enforce contiguity, dispatch,
    // then wire the autograd graph if any input requires gradients.
    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a);

    // Backward pass: delegate to Derived::grad_formula, then reduce the
    // gradient back to the original input shape (for broadcast ops).
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward: Derived::grad_formula_impl(grad_out, a_impl, out_impl)
    // returns a TensorImplPtr gradient so the backward itself is differentiable.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;
};

template <class Derived>
std::shared_ptr<TensorImpl> UnaryKernel<Derived>::forward(const std::shared_ptr<TensorImpl>& a) {
    if (!a)
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");

    // SchemaGuard resolves the effective dtype (e.g., promoting F16 → F32
    // when the schema requires full precision) and validates device support.
    SchemaGuard sg{Derived::schema_v1, a->dtype(), a->device()};
    const Dtype eff_dt = sg.effective_dtype();

    // Non-contiguous CPU tensors must be materialized before the typed
    // cpu_kernel loop can index them with a simple pointer + offset.
    const TensorImplPtr a_contig =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;
    const TensorImplPtr a_ptr = detail::maybe_cast_for_kernel(a_contig, eff_dt);

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), eff_dt, a_ptr->shape()};

    Storage out_storage;
    if constexpr (detail::HasUnaryDispatch<Derived>) {
        // Dispatch path: uses the backend abstraction (Accelerate / MLX).
        out_storage = Derived::dispatch(backend::Dispatcher::for_device(a_ptr->device()),
                                        a_ptr->storage(), a_ptr->shape(), eff_dt);
    } else if (a_ptr->device() == Device::GPU) {
        if constexpr (detail::HasUnaryGpuKernel<Derived>) {
            out_storage = Storage{Derived::gpu_kernel(std::get<GpuStorage>(a_ptr->storage()),
                                                      a_ptr->shape(), eff_dt)};
        } else {
            ErrorBuilder(Derived::schema_v1.name).not_implemented("GPU kernel not yet implemented");
        }
    } else {
        out_storage = Storage{
            Derived::cpu_kernel(std::get<CpuStorage>(a_ptr->storage()), a_ptr->shape(), eff_dt)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), a_ptr->shape(), eff_dt,
                                            a_ptr->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    if constexpr (!Derived::kHasGradient) {
        return out;
    } else {
        const bool needs_grad = GradMode::is_enabled() && a->requires_grad();
        if (!needs_grad)
            return out;

        // ensure_grad_fn wraps a leaf parameter in an AccumulateGrad node
        // so the autograd engine knows where to accumulate its gradient.
        auto a_edge = detail::ensure_grad_fn(a);

        auto bwd = std::make_shared<Derived>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = a->shape();
        bwd->dtype_ = eff_dt;
        bwd->device_ = a->device();
        bwd->input_tensors_ = {a};
        // Conditionally snapshot inputs/output for use in grad_formula.
        if constexpr (Derived::kSavesInput)
            bwd->saved_inputs_ = {a_ptr->storage()};
        if constexpr (Derived::kSavesOutput)
            bwd->saved_output_ = out->storage();
        // Always save the original TensorImpl for graph-mode backward.
        bwd->saved_impl_inputs_ = {a};
        if constexpr (Derived::kSavesOutput)
            bwd->saved_impl_output_ = out;

        std::vector<Edge> edges;
        edges.emplace_back(a_edge, a->grad_output_nr());
        bwd->set_next_edges(std::move(edges));
        // Version is captured so in-place modifications after this forward
        // call are detected as version mismatches during backward.
        bwd->set_saved_versions({a->version()});

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return out;
    }
}

// Delegate to the concrete grad_formula, then broadcast-reduce the result
// to match the original input shape if the op changed the shape.
template <class Derived>
std::vector<Storage> UnaryKernel<Derived>::apply(Storage grad_out) {
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {reduce_grad_to_shape(dx, this->out_shape_, this->input_shapes_[0], this->dtype_,
                                 this->device_)};
}

// Graph-mode backward: call Derived::grad_formula_impl(grad_out, a_impl, out_impl).
// The returned TensorImplPtr carries grad_fn for higher-order differentiation.
template <class Derived>
std::vector<TensorImplPtr> UnaryKernel<Derived>::apply_for_graph(const TensorImplPtr& grad_out) {
    extern TensorImplPtr sum_op(const TensorImplPtr&, const std::vector<int>&, bool);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    auto& a = this->saved_impl_inputs_[0];
    if (!a) {
        throw std::runtime_error(
            "apply_for_graph: saved_impl_inputs_[0] not set for op '" +
            std::string(Derived::schema_v1.name) + "'.");
    }

    auto dx = static_cast<Derived*>(this)->grad_formula_impl(
        grad_out, a, this->saved_impl_output_);

    // Reduce back to input shape if needed (same as apply()).
    if (dx->shape() == this->input_shapes_[0]) return {dx};
    std::vector<int> axes;
    const int ng = static_cast<int>(dx->shape().size());
    const int nt = static_cast<int>(this->input_shapes_[0].size());
    for (int i = 0; i < ng - nt; ++i) axes.push_back(i);
    for (int i = 0; i < nt; ++i) {
        if (this->input_shapes_[0][static_cast<std::size_t>(i)] == 1 &&
            dx->shape()[static_cast<std::size_t>(i + ng - nt)] != 1)
            axes.push_back(i + ng - nt);
    }
    if (!axes.empty()) dx = sum_op(dx, axes, false);
    if (dx->shape() != this->input_shapes_[0]) dx = reshape_op(dx, this->input_shapes_[0]);
    return {dx};
}

}  // namespace lucid
