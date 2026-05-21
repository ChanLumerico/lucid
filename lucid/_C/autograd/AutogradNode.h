// lucid/_C/autograd/AutogradNode.h
//
// CRTP base class for every concrete backward node in the engine's ``ops/``
// layer.
//
// :class:`AutogradNode` extends :class:`Node` with the boilerplate that
// almost every op shares: fixed-size arrays of saved input / output
// :class:`Storage`, weak pointers to the source :class:`TensorImpl` objects
// for version checking, and the shape / dtype / device metadata needed to
// rebuild gradient tensors during backward.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "Helpers.h"
#include "Node.h"

namespace lucid {

class TensorImpl;

// CRTP base supplying boilerplate fields and methods to concrete backward
// nodes with a fixed number of forward inputs.
//
// Inherit as ``class FooBackward : public AutogradNode<FooBackward, N>`` —
// or, more commonly, through the :class:`FuncOp` alias — to obtain
// ``N``-slot saved-input storage, automatic version validation, and an
// engine-friendly ``release_saved`` implementation.  The derived class
// must expose a ``static constexpr OpSchema schema_v1`` whose ``.name``
// field identifies the op in error messages and traces, and must
// implement :meth:`apply` returning one :class:`Storage` per forward
// input.
//
// Template Parameters
// -------------------
// Derived : class
//     The concrete backward node class (CRTP self-type).  Must expose a
//     ``static constexpr`` member ``schema_v1`` with at least a ``.name``
//     :class:`std::string_view` field.
// N_IN : std::size_t
//     Number of input tensors the corresponding forward op consumes.
//     Determines the compile-time size of :attr:`saved_inputs_`,
//     :attr:`input_shapes_`, :attr:`input_tensors_`, and
//     :attr:`saved_impl_inputs_`.
//
// Attributes
// ----------
// kNumInputs : std::size_t
//     Compile-time constant mirror of ``N_IN`` for callers that need it
//     as a value (e.g. when constructing the ``next_edges`` vector).
// input_tensors_ : std::array<std::weak_ptr<TensorImpl>, N_IN>
//     Weak references to the original input :class:`TensorImpl` objects,
//     used by :meth:`validate_versions` and :meth:`retainable_inputs`.
//     Weak so the backward graph never extends an input's lifetime.
// input_shapes_ : std::array<Shape, N_IN>
//     Shapes of the forward inputs, captured so that broadcasts and
//     reductions can be inverted in :meth:`apply` without needing the
//     live :class:`TensorImpl`.
// out_shape_ : Shape
//     Shape of the forward output; used to allocate or validate the
//     incoming gradient in :meth:`apply`.
// dtype_ : Dtype
//     Dtype of the forward computation.  Gradient tensors are created
//     with this dtype.  Defaults to :data:`Dtype::F32`.
// device_ : Device
//     Device of the forward computation.  Gradient tensors live on the
//     same device.  Defaults to :data:`Device::CPU`.
// saved_inputs_ : std::array<Storage, N_IN>
//     Copies of forward input :class:`Storage` values that :meth:`apply`
//     reads — for example, the input activations of an element-wise
//     multiply.  Populated by the forward op builder.
// saved_output_ : Storage
//     Copy of the forward output :class:`Storage`, populated only when
//     the backward formula needs the output activation (sigmoid,
//     softmax, tanh, etc.).  Left as an empty :class:`CpuStorage` when
//     unused.
// saved_impl_inputs_ : std::array<std::shared_ptr<TensorImpl>, N_IN>
//     Strong references to the source :class:`TensorImpl` objects.
//     Set in lockstep with :attr:`saved_inputs_` so that
//     :meth:`Node::apply_for_graph` can call into ``grad_fn``-bearing
//     ops when ``create_graph=True``.
// saved_impl_output_ : std::shared_ptr<TensorImpl>
//     Strong reference to the forward output :class:`TensorImpl`, set
//     only for ops whose graph-mode backward needs the output value
//     (e.g. :class:`SigmoidBackward`).
//
// Notes
// -----
// **Lifecycle.** The forward op builder constructs an instance, fills
// :attr:`input_tensors_`, :attr:`input_shapes_`, :attr:`saved_inputs_`,
// :attr:`saved_output_`, :attr:`dtype_`, and :attr:`device_`, then calls
// :meth:`Node::set_next_edges` and :meth:`Node::set_saved_versions`
// before attaching the node to the output :class:`TensorImpl` via
// ``set_grad_fn``.  During backward the engine calls
// :meth:`validate_versions` then :meth:`apply`.  After :meth:`apply`
// returns, the engine calls :meth:`release_saved`, which resets every
// :class:`Storage` slot to a default-constructed (zero-byte)
// :class:`CpuStorage` so referenced memory can be freed immediately.
//
// **Thread safety.** None.  All access happens on the single backward
// thread.
//
// **Slot ordering.** :attr:`saved_inputs_` and the gradients returned
// from :meth:`apply` are ordered to match the forward op's positional
// argument list — changing that order is a wire-format break.
//
// See Also
// --------
// :class:`Node` — the abstract base whose virtuals this class overrides.
// :class:`FuncOp` — alias used by most concrete backward classes.
template <class Derived, std::size_t N_IN>
class AutogradNode : public Node {
public:
    // Compile-time count of forward inputs.  Equivalent to ``N_IN``;
    // exposed as a value so non-template callers can read it.
    static constexpr std::size_t kNumInputs = N_IN;

    // Human-readable name of the op, sourced from ``Derived::schema_v1``.
    //
    // Returns
    // -------
    // std::string_view
    //     The schema name (e.g. ``"linear"``, ``"matmul"``).  Used in
    //     error messages, profiler traces, and ``create_graph`` failure
    //     reports.
    std::string_view name() const noexcept { return Derived::schema_v1.name; }

    // Return weak pointers to the forward-input :class:`TensorImpl`
    // objects so that the engine can honour ``retain_grad`` on non-leaf
    // tensors.
    //
    // Returns
    // -------
    // std::vector<std::weak_ptr<TensorImpl>>
    //     One weak pointer per forward input, in input order.
    std::vector<std::weak_ptr<TensorImpl>> retainable_inputs() const override {
        std::vector<std::weak_ptr<TensorImpl>> result;
        result.reserve(N_IN);
        for (std::size_t i = 0; i < N_IN; ++i)
            result.push_back(input_tensors_[i]);
        return result;
    }

    // Verify that no saved input has been mutated in-place since forward.
    //
    // Walks every input slot and calls :func:`check_version_match` with
    // the live :class:`TensorImpl` weak reference and the version
    // captured at forward time.  Expired weak references are silently
    // skipped — if the input has been destroyed, no consumer can observe
    // a corrupted gradient.
    //
    // Raises
    // ------
    // VersionMismatch
    //     When any saved input's live version counter differs from the
    //     value captured at forward.  The error names this op (via
    //     ``Derived::schema_v1.name``) and the offending input index.
    void validate_versions() override {
        for (std::size_t i = 0; i < N_IN; ++i) {
            ::lucid::check_version_match(input_tensors_[i],
                                         saved_versions_.size() > i ? saved_versions_[i] : 0,
                                         Derived::schema_v1.name, i);
        }
    }

    // Drop every reference held for backward so that memory can be
    // reclaimed immediately after the engine finishes with this node.
    //
    // Resets all entries of :attr:`saved_inputs_`, clears
    // :attr:`saved_output_`, empties the :attr:`input_tensors_` weak
    // array, and releases the strong :class:`TensorImpl` references
    // in :attr:`saved_impl_inputs_` and :attr:`saved_impl_output_`.
    //
    // Notes
    // -----
    // Called by :class:`Engine` once :meth:`apply` returns, except when
    // ``retain_graph=True``.  After this call the node retains only its
    // sequence number and (now-empty) metadata; calling :meth:`apply`
    // again is undefined.
    void release_saved() override {
        for (auto& s : saved_inputs_)
            s = Storage{CpuStorage{}};
        saved_output_ = Storage{CpuStorage{}};
        input_tensors_ = {};
        for (auto& p : saved_impl_inputs_)
            p.reset();
        saved_impl_output_.reset();
    }

    std::array<std::weak_ptr<TensorImpl>, N_IN> input_tensors_;
    std::array<Shape, N_IN> input_shapes_;
    Shape out_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;
    std::array<Storage, N_IN> saved_inputs_;
    Storage saved_output_;
    std::array<std::shared_ptr<TensorImpl>, N_IN> saved_impl_inputs_;
    std::shared_ptr<TensorImpl> saved_impl_output_;
};

}  // namespace lucid
