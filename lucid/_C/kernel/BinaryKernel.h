// lucid/_C/kernel/BinaryKernel.h
//
// CRTP base for two-input, single-output op kernels, plus the broadcast
// and autograd utility helpers (``try_broadcast_shapes``, ``broadcast_cpu``,
// ``ensure_grad_fn``, ``maybe_cast_for_kernel``) shared by the entire
// kernel layer.
//
// Concrete binary ops (:class:`AddBackward`, :class:`SubBackward`,
// :class:`MulBackward`, :class:`DivBackward`, :class:`PowBackward`,
// :class:`MatmulBackward`, …) inherit as
// ``class FooBackward : public BinaryKernel<FooBackward>`` and supply
// op-specific schema, compute hooks, and gradient formulas; the base
// owns dtype/device validation, NumPy-style broadcasting, contiguity
// enforcement, backend dispatch, and full autograd graph wiring.
//
// :meth:`forward` infers the broadcast shape (no-op when shapes are
// identical), prefers ``Derived::dispatch`` when available, otherwise
// falls back to ``Derived::cpu_kernel`` / ``Derived::gpu_kernel``.
// :meth:`apply` calls ``Derived::grad_formula`` and broadcast-reduces
// both gradients back to their original input shapes; the create_graph
// path goes through :meth:`apply_for_graph` and
// ``Derived::grad_formula_impl``.

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
#include "../compile/Tracer.h"  // 3.5 Phase 1.2 step 2: trace I/O wiring at forward
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

// Concept satisfied when ``Derived`` exposes a typed binary GPU kernel.
//
// Template Parameters
// -------------------
// T : class
//     The candidate Derived kernel class.
//
// Notes
// -----
// Matches ``T::gpu_kernel(GpuStorage, GpuStorage, Shape, Dtype) -> GpuStorage``.
// Used by :meth:`BinaryKernel::forward` to gate the typed GPU path; ops
// lacking a GPU kernel surface a clear ``not_implemented`` error.
template <class T>
concept HasGpuKernel = requires(GpuStorage a, GpuStorage b, Shape s, Dtype d) {
    { T::gpu_kernel(a, b, s, d) } -> std::same_as<GpuStorage>;
};

// Concept satisfied when ``Derived`` opts into backend-aware dispatch.
//
// Template Parameters
// -------------------
// T : class
//     The candidate Derived kernel class.
//
// Notes
// -----
// Matches ``T::dispatch(IBackend&, Storage, Storage, Shape, Dtype) -> Storage``.
// When present this path is preferred over the typed CPU/GPU kernels —
// it lets the op funnel through a backend-aware code path that may
// pick e.g. an Accelerate BLAS routine on CPU and an MLX op on GPU.
template <class T>
concept HasBinaryDispatch =
    requires(backend::IBackend& be, Storage a, Storage b, Shape s, Dtype d) {
        { T::dispatch(be, a, b, s, d) } -> std::same_as<Storage>;
    };

// Compute the NumPy-style broadcast shape of two operands.
//
// Parameters
// ----------
// a : const Shape&
//     Shape of the first operand.
// b : const Shape&
//     Shape of the second operand.
//
// Returns
// -------
// Result<Shape>
//     ``Ok(out)`` containing the broadcast shape on success, otherwise
//     ``Err(ShapeMismatch, ...)`` when at least one aligned dimension
//     pair is neither equal nor unit-extendable.
//
// Notes
// -----
// Right-aligned NumPy broadcasting: dimensions are matched from the
// trailing end; missing leading dimensions are treated as ``1``.
//
// See Also
// --------
// :func:`broadcast_shapes` — throwing wrapper used in the hot path.
// :meth:`BinaryKernel::saved_input_broadcasted`.
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

// Throwing wrapper around :func:`try_broadcast_shapes`.
//
// Parameters
// ----------
// a : const Shape&
//     Shape of the first operand.
// b : const Shape&
//     Shape of the second operand.
//
// Returns
// -------
// Shape
//     The NumPy-style broadcast shape.
//
// Raises
// ------
// ShapeMismatch
//     If the two shapes are not broadcast-compatible.
inline Shape broadcast_shapes(const Shape& a, const Shape& b) {
    auto r = try_broadcast_shapes(a, b);
    if (r.is_ok())
        return std::move(r).value();
    throw ShapeMismatch(a, b, "broadcast: incompatible shapes");
}

// Materialise a CPU broadcast of ``src`` from ``src_shape`` to ``out_shape``.
//
// Parameters
// ----------
// src : const CpuStorage&
//     Source storage holding the original (smaller) tensor data.
// src_shape : const Shape&
//     Shape of ``src``.
// out_shape : const Shape&
//     Target broadcast shape; must be NumPy-broadcast-compatible with
//     ``src_shape``.
// dt : Dtype
//     Element dtype.  Must match ``src.dtype``.
//
// Returns
// -------
// CpuStorage
//     A freshly allocated, contiguous storage of shape ``out_shape``
//     filled by replicating ``src`` along its size-1 axes.
//
// Notes
// -----
// Uses a stride-based mapping where size-1 source dimensions get
// stride 0, so every position along that axis reads the same source
// element.  This is the pure-CPU analogue of :func:`mlx::core::broadcast_to`.
//
// Raises
// ------
// LucidNotImplementedError
//     If the dtype is not one of ``F32``, ``F64``, ``I8/I16/I32/I64``,
//     ``Bool``.
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

// Resolve (or lazily create) the autograd grad_fn for a tensor.
//
// Parameters
// ----------
// t : const std::shared_ptr<TensorImpl>&
//     The tensor whose backward sink is needed.  May be ``nullptr`` or
//     not require a gradient — handled gracefully.
//
// Returns
// -------
// std::shared_ptr<Node>
//     The tensor's existing ``grad_fn``; an :class:`AccumulateGrad`
//     sink freshly installed on a leaf parameter; or ``nullptr`` when
//     ``t`` is null, does not require a gradient, or is a non-leaf
//     without a ``grad_fn`` (a programming error caught elsewhere).
//
// Notes
// -----
// Leaf parameters that have not yet participated in a computation
// don't carry a ``grad_fn`` — the first forward op that touches them
// installs an :class:`AccumulateGrad` here so the engine has a sink
// to write into during backward.
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

// Cast a tensor to a target dtype, returning the original when no cast is needed.
//
// Parameters
// ----------
// t : const TensorImplPtr&
//     The input tensor.  ``nullptr`` is propagated through unchanged.
// dt : Dtype
//     The desired effective dtype.
//
// Returns
// -------
// TensorImplPtr
//     ``t`` itself when ``t->dtype() == dt`` (zero copy), otherwise a
//     fresh :class:`TensorImpl` wrapping the backend-cast storage.
//
// Notes
// -----
// Used by kernel ``forward()`` trampolines to normalise inputs to the
// effective dtype returned by :class:`SchemaGuard` before compute.
// The cast routes through ``IBackend::cast`` so CPU and GPU paths
// share a single typed cast implementation.
inline TensorImplPtr maybe_cast_for_kernel(const TensorImplPtr& t, Dtype dt) {
    if (!t || t->dtype() == dt)
        return t;
    auto& be = backend::Dispatcher::for_device(t->device());
    Storage cast_storage = be.cast(t->storage(), t->shape(), t->dtype(), dt);
    return std::make_shared<TensorImpl>(std::move(cast_storage), t->shape(), dt, t->device(),
                                        false);
}

}  // namespace detail

// CRTP base for two-input, single-output op kernels — pairs forward
// output computation with the two-input saved-tensor bookkeeping every
// binary op needs.
//
// Inherits :class:`AutogradNode\<Derived, 2\>` (two saved input slots)
// and :class:`kernel::IKernel`.  :meth:`forward` owns all validation,
// broadcast inference, dispatch, and graph wiring; :meth:`apply` calls
// ``Derived::grad_formula`` and broadcast-reduces each gradient back to
// its original input shape.
//
// Concrete ops declare themselves as
// ``class FooBackward : public BinaryKernel<FooBackward>`` and provide:
//
//   - ``static constexpr OpSchema schema_v1`` — op name + AMP policy.
//   - ``static cpu_kernel(a, b, out_shape, dtype) -> CpuStorage`` and/or
//     ``static gpu_kernel(a, b, out_shape, dtype) -> GpuStorage``, **or**
//     ``static dispatch(IBackend&, a, b, out_shape, dtype) -> Storage``.
//   - ``grad_formula(grad_out) -> std::tuple<Storage, Storage>`` for the
//     local Jacobian product.
//   - Optional
//     ``grad_formula_impl(grad_out, a_impl, b_impl) -> std::pair<TensorImplPtr, TensorImplPtr>``
//     for create_graph higher-order differentiation.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete CRTP self-type.
//
// Attributes
// ----------
// kSavesInputs : static constexpr bool
//     Whether ``forward()`` snapshots both input storages into
//     ``saved_inputs_``.  Defaults to ``true``; set ``false`` for ops
//     whose backward does not consult the original inputs (e.g. Add
//     whose backward is identity).
//
// Notes
// -----
// Slot count is 2 (one per input).  Mixed-dtype binary ops are **not**
// implicitly promoted — callers must cast.  Cross-device operands are
// never silently moved — callers must migrate.  Both checks throw
// :class:`DtypeMismatch` / :class:`DeviceMismatch` from :meth:`forward`.
//
// See Also
// --------
// :class:`UnaryKernel`, :class:`NaryKernel`, :class:`VariadicKernel`.
// :class:`IKernel` — the abstract base above the CRTP layer.
template <class Derived>
class BinaryKernel : public AutogradNode<Derived, 2>, public kernel::IKernel {
public:
    // Typed forward trampoline for a two-input op.
    //
    // Parameters
    // ----------
    // a, b : const std::shared_ptr<TensorImpl>&
    //     The two input tensors.  Must share dtype and device.
    //
    // Returns
    // -------
    // std::shared_ptr<TensorImpl>
    //     The output tensor.  If grad mode is on and either input
    //     requires a gradient the result carries a fresh ``Derived``
    //     ``grad_fn`` with two outbound edges.
    //
    // Raises
    // ------
    // DtypeMismatch
    //     If ``a->dtype() != b->dtype()``.
    // DeviceMismatch
    //     If ``a->device() != b->device()``.
    // ShapeMismatch
    //     If the shapes are not NumPy-broadcast-compatible.
    static std::shared_ptr<TensorImpl> forward(const std::shared_ptr<TensorImpl>& a,
                                               const std::shared_ptr<TensorImpl>& b);

    // Return the canonical schema name of the concrete op.
    std::string_view name() const noexcept override { return Derived::schema_v1.name; }

    // Return the autograd node label (same string as :meth:`name`).
    std::string node_name() const override { return std::string(Derived::schema_v1.name); }

    // Backward implementation invoked by the autograd engine.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     The upstream gradient with respect to this op's output.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     ``{grad_a, grad_b}`` each reduced back to its original input
    //     shape via :func:`reduce_grad_to_shape`.
    //
    // Notes
    // -----
    // Reduction is needed because ``forward()`` may have broadcast
    // either operand to a larger output shape.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward — supports create_graph=True.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     The upstream gradient retained as a :class:`TensorImpl` with
    //     ``grad_fn``.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     ``{grad_a, grad_b}`` as fully-traced :class:`TensorImpl`
    //     pointers, each reduced to its original input shape via
    //     :func:`sum_op` / :func:`reshape_op`.
    //
    // Raises
    // ------
    // std::runtime_error
    //     If the saved input pointers were not captured, or if
    //     ``Derived`` did not override :meth:`grad_formula_impl`.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

    // Whether ``forward()`` snapshots the inputs.  See class docstring.
    static constexpr bool kSavesInputs = true;

    // Default graph-mode gradient formula — concrete ops override.
    //
    // Parameters
    // ----------
    // grad_out : const TensorImplPtr&
    //     The upstream gradient.
    // a, b : const TensorImplPtr&
    //     The saved forward inputs already broadcast to ``out_shape_``.
    //
    // Returns
    // -------
    // std::pair<TensorImplPtr, TensorImplPtr>
    //     ``(grad_a, grad_b)`` as fully-traced tensors.
    //
    // Raises
    // ------
    // std::runtime_error
    //     The default implementation always throws.  Concrete ops
    //     must override to support ``create_graph=True``.
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& /*grad_out*/,
                                                              const TensorImplPtr& /*a*/,
                                                              const TensorImplPtr& /*b*/) {
        throw std::runtime_error("create_graph=True is not supported for op '" +
                                 std::string(Derived::schema_v1.name) +
                                 "'. "
                                 "Implement grad_formula_impl() to add support.");
    }

protected:
    // Convenience accessor for the backend matching a device.
    static backend::IBackend& backend_for(Device d) { return backend::Dispatcher::for_device(d); }

    // Reduce a traced gradient tensor back to a target input shape.
    //
    // Parameters
    // ----------
    // grad : const TensorImplPtr&
    //     The traced gradient with shape ``grad_shape``.
    // grad_shape : const Shape&
    //     The current shape of ``grad``.
    // target_shape : const Shape&
    //     The desired output shape (typically an original input shape).
    //
    // Returns
    // -------
    // TensorImplPtr
    //     A new tensor whose shape equals ``target_shape``.  Sums over
    //     leading broadcast axes and over any axis where
    //     ``target_shape`` is unit but ``grad_shape`` is not, then
    //     reshapes to ``target_shape`` exactly.
    //
    // Notes
    // -----
    // Used inside :meth:`apply_for_graph` implementations and mirrors
    // the Storage-level helper :func:`reduce_grad_to_shape`.
    static TensorImplPtr reduce_impl_to_shape(const TensorImplPtr& grad,
                                              const Shape& grad_shape,
                                              const Shape& target_shape);

    // Return the saved input ``k`` broadcast to ``out_shape_``.
    //
    // Parameters
    // ----------
    // k : std::size_t
    //     Index into ``saved_inputs_`` (0 or 1).
    //
    // Returns
    // -------
    // Storage
    //     The saved storage when ``input_shapes_[k] == out_shape_``,
    //     otherwise a freshly materialised broadcast copy.  On GPU
    //     this calls :func:`mlx::core::broadcast_to`; on CPU it calls
    //     :func:`detail::broadcast_cpu`.
    //
    // Notes
    // -----
    // Lazy: the broadcast copy is only materialised when needed.
    // Used by ``grad_formula`` implementations that need the
    // full-rank operand rather than the possibly-smaller original.
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

// Out-of-class definition of :meth:`BinaryKernel::forward`.
//
// See the in-class declaration for parameter and return semantics.
// The broadcast shape is inferred from ``a`` and ``b``; equal shapes
// short-circuit the broadcast copy.  On CPU, non-contiguous inputs are
// materialised via :func:`contiguous_op` before entering the typed
// compute loop so :class:`Derived` may rely on flat pointer arithmetic.
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

    // 3.5 Phase 1.2 step 2: trace I/O wiring at the forward boundary.
    // BinaryKernel doesn't go through NaryKernel::wire_autograd, so the
    // hook here keeps elementwise binary ops in the trace IR.  Outside
    // any _tracing() scope this is one TLS load + null check.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a, b}, out);
    }

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
    edges.emplace_back(a_edge, a->grad_output_nr());
    edges.emplace_back(b_edge, b->grad_output_nr());
    bwd->set_next_edges(std::move(edges));
    // Both input versions are captured to detect in-place mutations.
    bwd->set_saved_versions({a->version(), b->version()});

    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

// Out-of-class definition of :meth:`BinaryKernel::apply`.
//
// Calls ``Derived::grad_formula(grad_out)`` to obtain ``(da, db)`` then
// reduces each back to its original input shape via
// :func:`reduce_grad_to_shape`, accounting for any broadcasting the
// forward pass performed.
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

// Out-of-class definition of :meth:`BinaryKernel::reduce_impl_to_shape`.
//
// Computes the broadcast-axis set (leading axes + axes where
// ``target_shape`` is unit but ``grad_shape`` is not), sums over them,
// then reshapes to ``target_shape``.  Mirrors :func:`reduce_grad_to_shape`
// but operates on traced :class:`TensorImpl` pointers via :func:`sum_op`.
template <class Derived>
TensorImplPtr BinaryKernel<Derived>::reduce_impl_to_shape(const TensorImplPtr& grad,
                                                          const Shape& grad_shape,
                                                          const Shape& target_shape) {
    if (grad_shape == target_shape)
        return grad;

    // Forward declaration — defined in ops/ufunc/Reductions.cpp.
    extern TensorImplPtr sum_op(const TensorImplPtr&, const std::vector<int>&, bool);
    extern TensorImplPtr reshape_op(const TensorImplPtr&, const Shape&);

    // Compute axes that were broadcast: leading axes added by ndim expansion,
    // plus any axis where target_shape has size 1 but grad_shape does not.
    std::vector<int> axes;
    const int ndim_g = static_cast<int>(grad_shape.size());
    const int ndim_t = static_cast<int>(target_shape.size());
    const int leading = ndim_g - ndim_t;
    for (int i = 0; i < leading; ++i)
        axes.push_back(i);
    for (int i = 0; i < ndim_t; ++i) {
        if (target_shape[static_cast<std::size_t>(i)] == 1 &&
            grad_shape[static_cast<std::size_t>(i + leading)] != 1)
            axes.push_back(i + leading);
    }

    if (axes.empty())
        return grad;

    auto reduced = sum_op(grad, axes, /*keepdims=*/false);

    // After summing, we may have fewer dimensions than target_shape if we
    // reduced leading axes.  Reshape to match exactly.
    if (reduced->shape() != target_shape) {
        reduced = reshape_op(reduced, target_shape);
    }
    return reduced;
}

// Out-of-class definition of :meth:`BinaryKernel::apply_for_graph`.
//
// Broadcasts each saved input to ``out_shape_`` (when needed) so
// ``grad_formula_impl`` can write simple element-wise ops, calls
// ``Derived::grad_formula_impl(grad_out, a, b)``, then reduces each
// returned gradient back to its original input shape via
// :meth:`reduce_impl_to_shape`.  If ``Derived`` did not override
// :meth:`grad_formula_impl` the base throws with a clear error.
template <class Derived>
std::vector<TensorImplPtr> BinaryKernel<Derived>::apply_for_graph(const TensorImplPtr& grad_out) {
    // Forward declaration — defined in ops/utils/Layout.cpp.
    extern TensorImplPtr broadcast_to_op(const TensorImplPtr&, const Shape&);

    auto& a = this->saved_impl_inputs_[0];
    auto& b = this->saved_impl_inputs_[1];
    if (!a || !b) {
        throw std::runtime_error(
            "apply_for_graph called but saved_impl_inputs_ were not set for op '" +
            std::string(Derived::schema_v1.name) +
            "'. "
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
