#pragma once

// =====================================================================
// Lucid C++ engine — ReduceKernel<Derived>: reduction op base.
// =====================================================================
//
// Replaces `ops/ufunc/_ReduceOp.h::ReduceOp<Derived>`.
//
// Derived implements:
//   1. `static const OpSchema schema_v1;`
//   2. `static CpuStorage cpu_kernel(const CpuStorage&, const Shape& input_shape,
//                                    const std::vector<int>& axes, bool keepdims, Dtype);`
//   3. (optional) `static GpuStorage gpu_kernel(...);`
//   4. `Storage grad_formula(const Storage& grad_out);`
//
// `ops/ufunc/_ReduceOp.h` re-exports as:
//   template<class D> using ReduceOp = kernel::ReduceKernel<D>;
//
// Layer: kernel/. Depends on kernel/AutogradNode.h, autograd/, backend/, core/.

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
#include "BinaryKernel.h"  // detail::ensure_grad_fn
#include "Contig.h"        // contiguous_op forward-decl

namespace lucid {

namespace detail {

template <class T>
concept HasReduceGpuKernel =
    requires(GpuStorage a, Shape s, std::vector<int> ax, bool kd, Dtype d) {
        { T::gpu_kernel(a, s, ax, kd, d) } -> std::same_as<GpuStorage>;
    };

template <class T>
concept HasReduceDispatch =
    requires(backend::IBackend& be, Storage a, Shape s, std::vector<int> ax, bool kd, Dtype d) {
        { T::dispatch(be, a, s, ax, kd, d) } -> std::same_as<Storage>;
    };

}  // namespace detail

template <class Derived>
class ReduceKernel : public AutogradNode<Derived, 1> {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = false;
    static constexpr bool kHasGradient = true;

    /// Reduce-specific saved state.
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
std::shared_ptr<TensorImpl> ReduceKernel<Derived>::forward(const std::shared_ptr<TensorImpl>& a,
                                                           const std::vector<int>& axes_user,
                                                           bool keepdims) {
    if (!a)
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");

    // Phase 5: determinism gate + AMP dtype resolution.
    SchemaGuard sg{Derived::schema_v1, a->dtype(), a->device()};
    const Dtype eff_dt = sg.effective_dtype();

    const TensorImplPtr a_contig =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;
    const TensorImplPtr a_ptr = sg.maybe_cast(a_contig);

    const auto axes = normalize_axes(axes_user, static_cast<int>(a_ptr->shape().size()));
    Shape out_shape = reduce_output_shape(a_ptr->shape(), axes, keepdims);

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), eff_dt, out_shape};

    Storage out_storage;
    if constexpr (detail::HasReduceDispatch<Derived>) {
        out_storage = Derived::dispatch(backend::Dispatcher::for_device(a_ptr->device()),
                                        a_ptr->storage(), a_ptr->shape(), axes, keepdims,
                                        eff_dt);
    } else if (a_ptr->device() == Device::GPU) {
        if constexpr (detail::HasReduceGpuKernel<Derived>) {
            out_storage =
                Storage{Derived::gpu_kernel(std::get<GpuStorage>(a_ptr->storage()), a_ptr->shape(),
                                            axes, keepdims, eff_dt)};
        } else {
            ErrorBuilder(Derived::schema_v1.name).not_implemented("GPU kernel not yet implemented");
        }
    } else {
        out_storage = Storage{Derived::cpu_kernel(std::get<CpuStorage>(a_ptr->storage()),
                                                  a_ptr->shape(), axes, keepdims, eff_dt)};
    }

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, eff_dt,
                                            a->device(), /*requires_grad=*/false);
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
            bwd->saved_inputs_ = {a_ptr->storage()};  // cast storage
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
std::vector<Storage> ReduceKernel<Derived>::apply(Storage grad_out) {
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {std::move(dx)};
}

}  // namespace lucid
