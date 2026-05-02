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

template <class T>
concept HasUnaryGpuKernel = requires(GpuStorage a, Shape s, Dtype d) {
    { T::gpu_kernel(a, s, d) } -> std::same_as<GpuStorage>;
};

template <class T>
concept HasUnaryDispatch = requires(backend::IBackend& be, Storage a, Shape s, Dtype d) {
    { T::dispatch(be, a, s, d) } -> std::same_as<Storage>;
};

}  // namespace detail

template <class Derived>
class UnaryKernel : public AutogradNode<Derived, 1>, public kernel::IKernel {
public:
    static constexpr bool kSavesInput = true;
    static constexpr bool kSavesOutput = false;
    static constexpr bool kHasGradient = true;

    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a);

    std::vector<Storage> apply(Storage grad_out) override;
};

template <class Derived>
std::shared_ptr<TensorImpl> UnaryKernel<Derived>::forward(const std::shared_ptr<TensorImpl>& a) {
    if (!a)
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");

    SchemaGuard sg{Derived::schema_v1, a->dtype(), a->device()};
    const Dtype eff_dt = sg.effective_dtype();

    const TensorImplPtr a_contig =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;
    const TensorImplPtr a_ptr = detail::maybe_cast_for_kernel(a_contig, eff_dt);

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), eff_dt, a_ptr->shape()};

    Storage out_storage;
    if constexpr (detail::HasUnaryDispatch<Derived>) {
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

        auto a_edge = detail::ensure_grad_fn(a);

        auto bwd = std::make_shared<Derived>();
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = a->shape();
        bwd->dtype_ = eff_dt;
        bwd->device_ = a->device();
        bwd->input_tensors_ = {a};
        if constexpr (Derived::kSavesInput)
            bwd->saved_inputs_ = {a_ptr->storage()};
        if constexpr (Derived::kSavesOutput)
            bwd->saved_output_ = out->storage();

        std::vector<Edge> edges;
        edges.emplace_back(a_edge, 0);
        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions({a->version()});

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return out;
    }
}

template <class Derived>
std::vector<Storage> UnaryKernel<Derived>::apply(Storage grad_out) {
    Storage dx = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {reduce_grad_to_shape(dx, this->out_shape_, this->input_shapes_[0], this->dtype_,
                                 this->device_)};
}

}  // namespace lucid
