// lucid/_C/core/fwd.h
//
// Compile-firewall header that forward-declares the core engine types.
//
// Including this header in place of the full type headers (``Device.h``,
// ``Dtype.h``, ``TensorImpl.h``, ``Storage.h``, ``Error.h``, ``Node.h``,
// ``Engine.h`` …) keeps header dependency graphs small and breaks the
// circular include cycles that arise between the storage, tensor, and
// autograd subsystems.  Translation units that need the full
// definitions — to call member functions, take ``sizeof``, or
// instantiate by value — must still include the corresponding ``.h``
// directly.
//
// Notes
// -----
// The enumerations declared here (:class:`Device`, :class:`Dtype`)
// pin their underlying integer type so that pointers and references
// can be formed across translation units without needing the full
// enumerator list visible.  Adding a new enumerator therefore does
// **not** require touching this file — only updates to the concrete
// ``.h`` are needed.
//
// When to include
// ---------------
//
//   * Function and method **declarations** that take :class:`TensorImpl`,
//     :class:`Node`, or related types only by pointer / reference.
//   * Type aliases (e.g. :type:`TensorImplPtr`) reused across many
//     headers.
//   * Headers that need to friend-declare an engine class without
//     pulling in its full definition.
//
// Switch to the full header when you need to:
//
//   * Call a member function or member access (``->``, ``.``).
//   * Take ``sizeof``, construct by value, derive from, or use as a
//     template parameter requiring complete types.
//   * Catch a specific exception by value or reference (``Error.h``).
//
// See Also
// --------
// :class:`TensorImpl` — full definition in ``TensorImpl.h``.
// :class:`Node`       — full definition in ``Node.h``.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace lucid {

// Memory-domain tag.
//
// Underlying type is fixed at ``std::uint8_t`` so pointers and
// references to :class:`Device` can be formed without the full
// enumerator list.  Full definition: ``Device.h``.
enum class Device : std::uint8_t;

// Element-type tag.
//
// Underlying type is fixed at ``std::uint8_t`` so pointers and
// references to :class:`Dtype` can be formed without the full
// enumerator list.  Full definition: ``Dtype.h``.
enum class Dtype : std::uint8_t;

// Owning tensor implementation — shape, storage, dtype, device, and
// optional autograd state.  Full definition: ``TensorImpl.h``.
class TensorImpl;

// Per-stream pseudo-random number generator.  Full definition:
// ``Generator.h``.
class Generator;

// Root of the Lucid exception hierarchy.  Forward-declared so
// error-handling headers can refer to it without dragging in the
// full ``Error.h`` chain.  Full definition: ``Error.h``.
class LucidError;

// Concrete byte buffer wrappers used by :class:`TensorImpl`'s
// :class:`Storage`.  Full definitions: ``Storage.h``.
struct CpuStorage;

// GPU companion to :class:`CpuStorage`; holds an MLX-managed buffer.
// Full definition: ``Storage.h``.
struct GpuStorage;

// Per-device allocation telemetry snapshot.  Full definition:
// ``MemoryStats.h``.
struct MemoryStats;

// Autograd graph node — one differentiable operation, with input
// edges and a ``backward`` implementation.  Full definition:
// ``Node.h``.
class Node;

// Connection from one :class:`Node`'s output to another's input.
// Full definition: ``Node.h``.
struct Edge;

// Leaf-tensor gradient accumulator :class:`Node` subclass.  Full
// definition: ``AccumulateGrad.h``.
class AccumulateGrad;

// Top-level autograd executor — runs the backward pass over an
// assembled graph of :class:`Node` s.  Full definition: ``Engine.h``.
class Engine;

// Shared owning handle to a :class:`TensorImpl`.
//
// Pervasive across the autograd and op layers — most engine APIs
// traffic in :type:`TensorImplPtr` rather than raw pointers so
// lifetime is managed by reference counting.
using TensorImplPtr = std::shared_ptr<TensorImpl>;

// Shared owning handle to a :class:`Node`.
//
// Autograd graphs are built out of :type:`NodePtr` references; the
// graph is implicitly destroyed when the last :type:`NodePtr` to its
// root drops.
using NodePtr = std::shared_ptr<Node>;

}  // namespace lucid
