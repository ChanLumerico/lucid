// lucid/_C/compile/Tracer.h
//
// Op-level tracer used by :mod:`lucid.compile`.
//
// :class:`Tracer` is a thread-local sink that records every op
// dispatched while it is installed via :func:`set_current_tracer`.
// The single hook site is the body of :class:`OpScopeFull`'s
// constructor — when an op enters its scope, ``OpScopeFull`` consults
// :func:`current_tracer` and, if non-null, forwards the op name and
// output meta into :func:`Tracer::on_op_enter`.  Ops outside any
// ``_tracing()`` scope take a single null check and incur no further
// cost (matches the :class:`OpScope` / :class:`Profiler` fast-path
// invariant in :file:`Profiler.h`).
//
// Phase 1.1 only records :class:`OpNode` headers (name + output
// metadata).  Input/output id wiring is added in Phase 1.2 inside the
// shared :func:`wire_autograd` boundary — the single site every op
// passes through after backend dispatch.  Until that wiring lands,
// ``inputs`` is left empty and the MpsBuilder treats such a node as
// a starting placeholder.
//
// Notes
// -----
// Thread safety:
//
//   - :func:`current_tracer` / :func:`set_current_tracer` operate on
//     a ``thread_local`` slot — each thread chooses its own tracer
//     independently.  Cross-thread tracing is not supported in 3.5.0
//     (training step is single-threaded by convention).
//   - :class:`Tracer` itself is **not** internally synchronised.  The
//     installing thread owns the object and must not share it with
//     other threads while it is the active tracer.  This matches the
//     trace lifecycle: install → run a single forward pass → detach.
//
// See Also
// --------
// :class:`lucid::compile::TraceGraph` — the recorded IR.
// :class:`lucid::OpScopeFull` — the single hook site.
// :func:`lucid::current_profiler` (:file:`Profiler.h:234`) — the TLS
//     pattern this header mirrors.

#pragma once

#include <string_view>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/fwd.h"  // TensorImpl forward declaration
#include "TraceIR.h"

namespace lucid::compile {

// Thread-local op recorder consumed by :class:`OpScopeFull`.
//
// Constructing a Tracer is cheap — the underlying :class:`TraceGraph`
// starts empty.  Hooks fire only between :func:`set_current_tracer`
// (this) and :func:`set_current_tracer` (nullptr); outside that
// window the tracer accumulates nothing.
//
// Attributes
// ----------
// graph_ : TraceGraph
//     The recorded op DAG.  Appended to by :func:`on_op_enter` and
//     exposed read-only via :func:`graph`.
//
// Notes
// -----
// Non-copyable, non-movable — the active-tracer slot is a raw pointer
// owned by the caller; copying or moving the Tracer would either
// invalidate that pointer or duplicate the recording sink.
//
// Examples
// --------
// Native usage (Python binding mirrors this)::
//
//     Tracer t;
//     set_current_tracer(&t);
//     // ... run a forward pass — every OpScopeFull ctor records ...
//     set_current_tracer(nullptr);
//     const auto& g = t.graph();  // inspect recorded nodes
class LUCID_API Tracer {
public:
    // Construct an empty ``Tracer`` ready to be installed via
    // :func:`set_current_tracer`.
    //
    // Defaulted; ``graph_`` starts empty and ``impl_to_id_`` starts with
    // no entries.  No op hooks fire until the tracer is installed as the
    // active TLS sink — construction itself has zero side effects.
    Tracer() = default;

    Tracer(const Tracer&) = delete;
    Tracer& operator=(const Tracer&) = delete;
    Tracer(Tracer&&) = delete;
    Tracer& operator=(Tracer&&) = delete;

    // Records the entry of a new op into the active scope.
    //
    // Parameters
    // ----------
    // name : std::string_view
    //     Op name as passed to :class:`OpScopeFull`.  Copied into the
    //     resulting :class:`OpNode::name`.
    // device : Device
    //     Device on which the op is dispatching.  Stored in the
    //     single :class:`TensorMeta` appended to ``outputs``.
    // dtype : Dtype
    //     Output element dtype at op entry.
    // shape : Shape
    //     Output shape at op entry.  Moved into the appended
    //     :class:`TensorMeta`.
    //
    // Notes
    // -----
    // Phase 1.1 appends one :class:`OpNode` per call with an empty
    // ``inputs`` vector and a single-element ``outputs`` vector.
    // Phase 1.2 backfills ``inputs`` from the autograd wiring site
    // and may append additional outputs for multi-output ops.
    void on_op_enter(std::string_view name, Device device, Dtype dtype, Shape shape);

    // Wires the input/output TensorImpl identities of the most recently
    // recorded :class:`OpNode` into the trace IR.
    //
    // Parameters
    // ----------
    // inputs : const std::vector<TensorImplPtr>&
    //     Owning shared pointers to the input tensors in op-defined
    //     order.  ``nullptr`` slots (e.g. an optional bias) are passed
    //     through as ``-1`` in the resulting :class:`OpNode::inputs``.
    //     The Tracer retains a strong reference to every external
    //     feed (input not produced by an earlier traced op) so the
    //     :class:`MpsBuilder` can read its shape/dtype + the run path
    //     can bind its data buffer.
    // output : const TensorImplPtr&
    //     Output tensor.  The id minted earlier by :func:`on_op_enter`
    //     (stored in ``node.outputs[0].id``) is associated with this
    //     pointer in the internal map so subsequent ops that consume
    //     this tensor can resolve it back to the same id.  Only the
    //     raw pointer is kept (no strong ref) — the output's lifetime
    //     is the caller's responsibility.
    //
    // Notes
    // -----
    // Phase 1.1 wires only kernel families that go through
    // :class:`kernel::NaryKernel` or :class:`kernel::VariadicKernel`'s
    // ``wire_autograd`` boundary (Conv / Linear / BatchNorm / Norm /
    // Loss / …).  Ops dispatched through :class:`UnaryKernel` or
    // :class:`BinaryKernel`'s in-line ``forward()`` reach
    // :func:`on_op_enter` but skip this step — their ``inputs`` slot
    // stays empty, marking the node "pending" for the Phase 1.2
    // builder, which then routes that signature to the eager path.
    //
    // Calling this with an empty :class:`TraceGraph` (no preceding
    // :func:`on_op_enter`) is a no-op — defensive against ordering
    // mistakes at hook sites.
    void on_op_io(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output);

    // Attach a single attribute to the most recently recorded
    // :class:`OpNode`.  Used by op forwards via
    // :func:`OpScopeFull::set_attr` to thread payloads (permutation,
    // axis, stride, padding, …) that the emitter would otherwise be
    // unable to recover from inputs + output shapes alone.
    //
    // Parameters
    // ----------
    // key : std::string_view
    //     Attribute name; chosen by the op forward and read by the
    //     matching emitter.  Convention: snake_case, matches the
    //     reference-framework keyword argument when one exists
    //     (``"permutation"``, ``"stride"``, ``"padding"``, ``"axis"``,
    //     ``"keepdim"``, ``"eps"``, …).
    // value : AttributeValue
    //     The payload, moved into the attribute map.
    //
    // Notes
    // -----
    // Calling this with an empty :class:`TraceGraph` (no preceding
    // :func:`on_op_enter`) is a no-op — defensive against ordering
    // mistakes at hook sites.  Overwrites any prior value under
    // ``key`` on the same node, so duplicate calls report the last
    // value (matches ``unordered_map`` semantics).
    void on_op_attr(std::string_view key, AttributeValue value);

    // Returns a read-only reference to the recorded graph.
    //
    // Returns
    // -------
    // const TraceGraph&
    //     The op DAG accumulated so far.  Caller must not access this
    //     reference after the Tracer is destroyed or a subsequent
    //     :func:`on_op_enter` invalidates iterators into ``ops``.
    const TraceGraph& graph() const { return graph_; }

    // Returns the id → external :type:`TensorImplPtr` map populated by
    // :func:`on_op_io` when an input wasn't produced by any earlier
    // traced op.  The builder uses this to materialise an MPSGraph
    // placeholder per external feed and to bind the corresponding
    // input data at execution time.
    //
    // The Tracer holds an owning :type:`TensorImplPtr` for every
    // external feed so the underlying buffers cannot be released
    // mid-trace.
    const std::unordered_map<TensorId, TensorImplPtr>& external_feeds() const {
        return external_feeds_;
    }

    // Reverse lookup: find the :type:`TensorId` for ``impl`` if it was
    // observed during the trace (either as an external feed or as the
    // output of a traced op).  Returns ``-1`` when ``impl`` is not in
    // the trace — used by the Python compile harness to map the user's
    // return-value tensors back to graph ids so they can be passed to
    // :func:`compile_or_cached` as explicit graph outputs (which lets
    // the builder drop "unconsumed but Python-discarded" intermediates
    // like RNN's ``h_n`` tensor when the user only kept ``out``).
    TensorId lookup_id(const TensorImpl* impl) const {
        if (impl == nullptr) return -1;
        const auto it = impl_to_id_.find(const_cast<TensorImpl*>(impl));
        return (it != impl_to_id_.end()) ? it->second : -1;
    }

private:
    TraceGraph graph_;
    // Maps the raw :class:`TensorImpl` pointer of every traced tensor
    // (op output *or* external feed) back to the :type:`TensorId` it
    // carries inside the IR.  Used by :func:`on_op_io` to resolve
    // input ids — if the pointer isn't in this map yet, the op's
    // input is an external feed and a fresh id is minted on the spot.
    std::unordered_map<TensorImpl*, TensorId> impl_to_id_;
    // Owning subset of ``impl_to_id_``: every external feed (inputs
    // not produced by any earlier op) carries a strong reference so
    // its TensorImpl can't be released mid-trace.  The builder
    // consults this to build placeholders + record feed-order for
    // runtime data binding.
    std::unordered_map<TensorId, TensorImplPtr> external_feeds_;
    // Owning references to every TensorImpl observed during the trace
    // (both inputs and outputs).  Without this, short-lived tensors
    // (e.g. the zero-bias tensors materialised by
    // ``conv_bias_or_zero`` when ``Conv2d(bias=False)`` is used) can
    // be destroyed and their raw pointer recycled by the allocator
    // for the *next* op's output — at which point the recycled
    // pointer is already in ``impl_to_id_`` and gets resolved to the
    // wrong id.  Keeping every TensorImplPtr alive for the trace's
    // lifetime makes raw-pointer keys unambiguous.
    std::vector<TensorImplPtr> live_refs_;
};

// Returns the :class:`Tracer` currently installed on the calling
// thread.
//
// Returns
// -------
// Tracer*
//     Non-owning pointer to the active tracer, or ``nullptr`` if no
//     tracer is installed.
//
// Notes
// -----
// The pointer lives in ``thread_local`` storage; each thread reads its
// own value.  Ownership of the pointed-to object remains with the
// caller of :func:`set_current_tracer`.
LUCID_API Tracer* current_tracer();

// Installs ``t`` as the active :class:`Tracer` for the calling thread.
//
// Parameters
// ----------
// t : Tracer*
//     Tracer to install, or ``nullptr`` to detach the active tracer
//     without destroying the object.
//
// Notes
// -----
// Ownership is not transferred — the caller keeps the Tracer alive
// for as long as any :class:`OpScopeFull` could observe the pointer.
// The Python binding holds the underlying ``shared_ptr`` for exactly
// this reason (matches :func:`lucid::set_current_profiler`).
//
// See Also
// --------
// :func:`current_tracer` — reader for the same TLS slot.
LUCID_API void set_current_tracer(Tracer* t);

}  // namespace lucid::compile
