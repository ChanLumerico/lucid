// lucid/_C/kernel/BinaryKernel.h
//
// CRTP base for two-input, single-output op kernels, as well as the
// broadcast and autograd utility helpers shared by the entire kernel layer.
//
// A concrete binary op is declared as:
//
//   struct AddOp : BinaryKernel<AddOp> {
//       static constexpr OpSchema schema_v1 = {"add", ...};
//       static CpuStorage cpu_kernel(const CpuStorage& a,
//                                    const CpuStorage& b,
//                                    const Shape&, Dtype);
//       static GpuStorage gpu_kernel(const GpuStorage& a,
//                                    const GpuStorage& b,
//                                    const Shape&, Dtype);
//       std::tuple<Storage, Storage> grad_formula(Storage grad_out);
//   };
//
// BinaryKernel::forward() handles dtype+device validation, broadcast
// shape inference, contiguous enforcement, dispatch to the right backend,
// and full autograd graph wiring. apply() calls grad_formula and
// broadcast-reduces both gradients back to their original input shapes.
//
// The detail:: namespace below also provides the broadcast helpers
// (try_broadcast_shapes, broadcast_cpu) and autograd utilities
// (ensure_grad_fn, maybe_cast_for_kernel) used by all kernel bases.

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mlx/ops.h>

#include "../api.h"
#include "../autograd/AccumulateGrad.h"
#include "../autograd/AutogradNode.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/AmpPolicy.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpSchema.h"
#include "../core/Profiler.h"
#include "../core/Result.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "Contig.h"
#include "IKernel.h"

namespace lucid {

namespace detail {

// Satisfied when Derived provides a gpu_kernel static method matching the
// binary GPU kernel signature.
template <class T>
concept HasGpuKernel = requires(GpuStorage a, GpuStorage b, Shape s, Dtype d) {
    { T::gpu_kernel(a, b, s, d) } -> std::same_as<GpuStorage>;
};

// Satisfied when Derived routes through the IBackend dispatch interface
// rather than calling a typed cpu_kernel/gpu_kernel directly.
template <class T>
concept HasBinaryDispatch =
    requires(backend::IBackend& be, Storage a, Storage b, Shape s, Dtype d) {
        { T::dispatch(be, a, b, s, d) } -> std::same_as<Storage>;
    };

// Compute the NumPy-style broadcast shape of a and b, returning an error
// Result if the shapes are incompatible (no dimension pair is broadcastable).
// Used by both forward() and saved_input_broadcasted().
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

// Throwing wrapper around try_broadcast_shapes. Raises ShapeMismatch on failure.
inline Shape broadcast_shapes(const Shape& a, const Shape& b) {
    auto r = try_broadcast_shapes(a, b);
    if (r.is_ok())
        return std::move(r).value();
    throw ShapeMismatch(a, b, "broadcast: incompatible shapes");
}

// Materialize a CPU broadcast of src from src_shape to out_shape.
// Uses a stride-based mapping: dimensions of size 1 get stride 0 so all
// output positions in that dimension read from the same source element.
// This is the pure-CPU analogue of mlx::core::broadcast_to.
inline CpuStorage
broadcast_cpu(const CpuStorage& src, const Shape& src_shape, const Shape& out_shape, Dtype dt) {
    const std::size_t ndim_out = out_shape.size();
    const std::size_t ndim_in = src_shape.size();
    Shape padded(ndim_out, 1);
    for (std::size_t i = 0; i < ndim_in; ++i)
        padded[ndim_out - ndim_in + i] = src_shape[i];
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

// Retrieve or create the autograd function for a tensor. Leaf parameters
// that haven't been involved in a computation yet get an AccumulateGrad
// node installed lazily here so the engine has a sink to write into.
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

// Return t unchanged if its dtype already matches dt, or allocate a new
// TensorImpl with a type-cast copy of the storage. Used by forward() to
// normalize both inputs to the effective dtype before compute.
inline TensorImplPtr maybe_cast_for_kernel(const TensorImplPtr& t, Dtype dt) {
    if (!t || t->dtype() == dt)
        return t;
    auto& be = backend::Dispatcher::for_device(t->device());
    Storage cast_storage = be.cast(t->storage(), t->shape(), t->dtype(), dt);
    return std::make_shared<TensorImpl>(std::move(cast_storage), t->shape(), dt, t->device(),
                                        false);
}

}  // namespace detail

// CRTP binary-op base. Inherits AutogradNode<Derived, 2> (two saved inputs)
// and IKernel. forward() owns all validation, broadcast, dispatch, and graph
// wiring. apply() calls grad_formula and reduces each gradient to its input
// shape.
//
// kSavesInputs may be overridden to false by ops whose backward pass does
// not need the original inputs (e.g. Add, where the backward is identity).
template <class Derived>
class BinaryKernel : public AutogradNode<Derived, 2>, public kernel::IKernel {
public:
    // Forward pass: validate dtype/device consistency, infer broadcast output
    // shape, cast inputs to eff_dt, dispatch, wire autograd graph.
    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a,
                                               const std::shared_ptr<TensorImpl>& b);

    std::string_view name() const noexcept override { return Derived::schema_v1.name; }
    std::string node_name() const override { return std::string(Derived::schema_v1.name); }

    // Backward pass: call Derived::grad_formula to get (da, db), then
    // broadcast-reduce each to its original input shape.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward: call Derived::grad_formula_impl to get (da, db) as
    // TensorImplPtrs (so operations are tracked for higher-order grad), then
    // reduce each back to its input shape via sum_op.
    // Concrete ops opt in by implementing grad_formula_impl(grad_out, a, b).
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

    // Controls whether forward() saves a_ptr->storage() and b_ptr->storage()
    // into saved_inputs_ for use in grad_formula.
    static constexpr bool kSavesInputs = true;

    // Default graph-mode gradient formula: throws NotImplementedError.
    // Override in concrete Derived classes to support create_graph=True.
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(
        const TensorImplPtr& /*grad_out*/, const TensorImplPtr& /*a*/, const TensorImplPtr& /*b*/) {
        throw std::runtime_error(
            "create_graph=True is not supported for op '" +
            std::string(Derived::schema_v1.name) + "'. "
            "Implement grad_formula_impl() to add support.");
    }

protected:
    static backend::IBackend& backend_for(Device d) { return backend::Dispatcher::for_device(d); }

    // Reduce a gradient TensorImpl from grad_shape back to target_shape by
    // summing over broadcast axes.  Used in apply_for_graph implementations.
    static TensorImplPtr reduce_impl_to_shape(const TensorImplPtr& grad,
                                              const Shape& grad_shape,
                                              const Shape& target_shape);

    // Return the saved input k broadcast to out_shape_, materializing the
    // broadcast copy lazily. Used by grad_formula implementations that need
    // the full-rank operand rather than the original possibly-smaller tensor.
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

// Validate, broadcast, dispatch, and wire autograd for a two-input op.
// The broadcast shape is inferred from a and b; if the shapes are equal
// no copy is performed. On CPU, non-contiguous inputs are materialized
// via contiguous_op before entering the typed compute loop.
template <class Derived>
std::shared_ptr<TensorImpl> BinaryKernel<Derived>::forward(const std::shared_ptr<TensorImpl>& a,
                                                           const std::shared_ptr<TensorImpl>& b) {
    if (!a || !b)
        ErrorBuilder(Derived::schema_v1.name).fail("null input tensor");
    // Mixed-dtype binary ops are not implicitly promoted; callers must cast.
    if (a->dtype() != b->dtype())
        throw DtypeMismatch(std::string(dtype_name(a->dtype())),
                            std::string(dtype_name(b->dtype())),
                            std::string(Derived::schema_v1.name));
    // Cross-device binary ops are never silently moved; callers must migrate.
    if (a->device() != b->device())
        throw DeviceMismatch(std::string(device_name(a->device())),
                             std::string(device_name(b->device())),
                             std::string(Derived::schema_v1.name));

    SchemaGuard sg{Derived::schema_v1, a->dtype(), a->device()};
    const Dtype eff_dt = sg.effective_dtype();

    const TensorImplPtr a_contig =
        (a->device() == Device::CPU && !a->is_contiguous()) ? contiguous_op(a) : a;
    const TensorImplPtr b_contig =
        (b->device() == Device::CPU && !b->is_contiguous()) ? contiguous_op(b) : b;
    const TensorImplPtr a_ptr = detail::maybe_cast_for_kernel(a_contig, eff_dt);
    const TensorImplPtr b_ptr = detail::maybe_cast_for_kernel(b_contig, eff_dt);

    Shape out_shape = (a_ptr->shape() == b_ptr->shape())
                          ? a_ptr->shape()
                          : detail::broadcast_shapes(a_ptr->shape(), b_ptr->shape());

    OpScopeFull scope{Derived::schema_v1.name, a_ptr->device(), eff_dt, out_shape};

    Storage out_storage;

    if constexpr (detail::HasBinaryDispatch<Derived>) {
        if (a_ptr->device() == Device::CPU) {
            const CpuStorage& a_raw = std::get<CpuStorage>(a_ptr->storage());
            const CpuStorage& b_raw = std::get<CpuStorage>(b_ptr->storage());
            CpuStorage a_buf, b_buf;
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
            out_storage = Derived::dispatch(backend::Dispatcher::for_device(Device::CPU),
                                            Storage{*a_use}, Storage{*b_use}, out_shape, eff_dt);
        } else {
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
            out_storage = Derived::dispatch(backend::Dispatcher::for_device(Device::GPU),
                                            Storage{a_g}, Storage{b_g}, out_shape, eff_dt);
        }
    } else if (a_ptr->device() == Device::GPU) {
        if constexpr (detail::HasGpuKernel<Derived>) {
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
            out_storage = Storage{Derived::gpu_kernel(a_g, b_g, out_shape, eff_dt)};
        } else {
            ErrorBuilder(Derived::schema_v1.name).not_implemented("GPU kernel not yet implemented");
        }
    } else {
        const CpuStorage& a_raw = std::get<CpuStorage>(a_ptr->storage());
        const CpuStorage& b_raw = std::get<CpuStorage>(b_ptr->storage());
        CpuStorage a_buf, b_buf;
        const CpuStorage* a_use = &a_raw;
        const CpuStorage* b_use = &b_raw;
        if (a_ptr->shape() != out_shape) {
            a_buf = detail::broadcast_cpu(a_raw, a_ptr->shape(), out_shape, eff_dt);
            a_use = &a_buf;
        }
        if (b_ptr->shape() != out_shape) {
            b_buf = detail::broadcast_cpu(b_raw, b_ptr->shape(), out_shape, eff_dt);
            b_use = &b_buf;
        }
        out_storage = Storage{Derived::cpu_kernel(*a_use, *b_use, out_shape, eff_dt)};
    }

    auto out =
        std::make_shared<TensorImpl>(std::move(out_storage), out_shape, eff_dt, a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(out->numel()));

    const bool needs_grad = GradMode::is_enabled() && (a->requires_grad() || b->requires_grad());
    if (!needs_grad)
        return out;

    // Install AccumulateGrad sinks for leaf parameters before recording edges.
    auto a_edge = detail::ensure_grad_fn(a);
    auto b_edge = detail::ensure_grad_fn(b);

    auto bwd = std::make_shared<Derived>();
    bwd->input_shapes_ = {a->shape(), b->shape()};
    bwd->out_shape_ = out->shape();
    bwd->dtype_ = eff_dt;
    bwd->device_ = a->device();
    bwd->input_tensors_ = {a, b};
    // saved_inputs_ holds the (possibly cast) pre-broadcast input storages
    // so that grad_formula can inspect the original per-element values.
    if constexpr (Derived::kSavesInputs)
        bwd->saved_inputs_ = {a_ptr->storage(), b_ptr->storage()};
    // Always save the original TensorImpl pointers (before any dtype cast) so
    // that apply_for_graph() can invoke forward ops on them and trace the
    // gradient computation back through the original computation graph.
    bwd->saved_impl_inputs_ = {a, b};

    std::vector<Edge> edges;
    edges.emplace_back(a_edge, 0);
    edges.emplace_back(b_edge, 0);
    bwd->set_next_edges(std::move(edges));
    // Both input versions are captured to detect in-place mutations.
    bwd->set_saved_versions({a->version(), b->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

// Call grad_formula to obtain (da, db), then broadcast-reduce each
// gradient back to the original input shapes in case the forward pass
// broadcast either operand to a larger output shape.
template <class Derived>
std::vector<Storage> BinaryKernel<Derived>::apply(Storage grad_out) {
    auto [da, db] = static_cast<Derived*>(this)->grad_formula(grad_out);
    return {
        reduce_grad_to_shape(da, this->out_shape_, this->input_shapes_[0], this->dtype_,
                             this->device_),
        reduce_grad_to_shape(db, this->out_shape_, this->input_shapes_[1], this->dtype_,
                             this->device_),
    };
}

// Reduce a gradient TensorImpl from grad_shape back to target_shape by summing
// over the axes that were broadcast in the forward pass.
// Mirrors reduce_grad_to_shape but operates on TensorImplPtr via sum_op.
template <class Derived>
TensorImplPtr BinaryKernel<Derived>::reduce_impl_to_shape(const TensorImplPtr& grad,
                                                           const Shape& grad_shape,
                                                           const Shape& target_shape) {
    if (grad_shape == target_shape) return grad;

    // Forward declaration — defined in ops/ufunc/Reductions.cpp.
    extern TensorImplPtr sum_op(const TensorImplPtr&, const std::vector<int>&, bool);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    // Compute axes that were broadcast: leading axes added by ndim expansion,
    // plus any axis where target_shape has size 1 but grad_shape does not.
    std::vector<int> axes;
    const int ndim_g = static_cast<int>(grad_shape.size());
    const int ndim_t = static_cast<int>(target_shape.size());
    const int leading = ndim_g - ndim_t;
    for (int i = 0; i < leading; ++i) axes.push_back(i);
    for (int i = 0; i < ndim_t; ++i) {
        if (target_shape[static_cast<std::size_t>(i)] == 1 &&
            grad_shape[static_cast<std::size_t>(i + leading)] != 1)
            axes.push_back(i + leading);
    }

    if (axes.empty()) return grad;

    auto reduced = sum_op(grad, axes, /*keepdims=*/false);

    // After summing, we may have fewer dimensions than target_shape if we
    // reduced leading axes.  Reshape to match exactly.
    if (reduced->shape() != target_shape) {
        reduced = reshape_op(reduced, target_shape);
    }
    return reduced;
}

// Graph-mode backward: Derived::grad_formula_impl(grad_out, a_impl, b_impl)
// returns (da, db) as TensorImplPtrs that retain grad_fn for higher-order diff.
// If Derived does not implement grad_formula_impl the base Node::apply_for_graph
// throws NotImplementedError, giving the user a clear message.
template <class Derived>
std::vector<TensorImplPtr> BinaryKernel<Derived>::apply_for_graph(const TensorImplPtr& grad_out) {
    // Forward declaration — defined in ops/utils/Layout.cpp.
    extern TensorImplPtr broadcast_to_op(const TensorImplPtr&, const Shape&);

    auto& a = this->saved_impl_inputs_[0];
    auto& b = this->saved_impl_inputs_[1];
    if (!a || !b) {
        throw std::runtime_error(
            "apply_for_graph called but saved_impl_inputs_ were not set for op '" +
            std::string(Derived::schema_v1.name) + "'. "
            "Ensure create_graph=True was set before the forward pass.");
    }

    // Broadcast saved inputs to out_shape_ if needed so grad_formula_impl can
    // do simple element-wise ops without worrying about shape mismatches.
    auto a_b = (a->shape() == this->out_shape_) ? a : broadcast_to_op(a, this->out_shape_);
    auto b_b = (b->shape() == this->out_shape_) ? b : broadcast_to_op(b, this->out_shape_);

    auto [da, db] = static_cast<Derived*>(this)->grad_formula_impl(grad_out, a_b, b_b);

    return {
        reduce_impl_to_shape(da, this->out_shape_, this->input_shapes_[0]),
        reduce_impl_to_shape(db, this->out_shape_, this->input_shapes_[1]),
    };
}

}  // namespace lucid
