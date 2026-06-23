// lucid/_C/compile/OpEmitters/OpEmitter.h
//
// Abstract base for MPSGraph emitters + the process-global registry.
//
// One :class:`OpEmitter` instance handles one op family (e.g. all
// linear-layer dispatches, all batch-norm forwards).  The
// :class:`MpsBuilder` walks a :class:`TraceGraph` op-by-op and looks
// each op name up in the registry; a missing entry signals an
// unsupported op and the builder aborts (signature → eager-only).
//
// Header is pure C++.  Objective-C types (``MPSGraph*``,
// ``MPSGraphTensor*``) are erased to ``void*`` at this boundary —
// emitter implementations (``.mm`` files in this directory) re-cast
// via ``__bridge`` on the way through.  Matches the same pattern used
// by :file:`backend/gpu/mps/MpsBridge.h`.

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "../../api.h"
#include "../../core/Device.h"
#include "../TraceIR.h"

namespace lucid::compile {

// Builder-side state threaded through each emitter call.
//
// Holds the active :class:`MPSGraph` plus the id → tensor map that
// resolves an input :type:`TensorId` to the :class:`MPSGraphTensor`
// produced by an earlier op (or by a placeholder for external feeds).
//
// Attributes
// ----------
// graph_ : void* (MPSGraph*)
//     The graph being assembled.  Cast back to ``MPSGraph*`` in .mm
//     files via ``__bridge MPSGraph*``.
// device_ : Device
//     Device for the trace.  Phase 1.2 supports only ``Device::GPU``
//     since MPSGraph is Metal-only; CPU traces escape to eager.
// tensors_ : std::unordered_map<TensorId, void*>
//     Resolves a TraceIR id to the :class:`MPSGraphTensor` that
//     represents it inside the graph.
class LUCID_API BuilderContext {
public:
    // Construct with the active :class:`MPSGraph` (passed as
    // ``__bridge void*``) and the target device.
    BuilderContext(void* graph_void, Device device);

    // Resolve a :type:`TensorId` to its :class:`MPSGraphTensor*`.
    // Returns ``nullptr`` for unregistered ids — used by the builder
    // to detect "input not produced by any traced op" (i.e. external
    // feed candidates).
    void* resolve(TensorId id) const;

    // Register a :class:`MPSGraphTensor` produced by an emitter (or a
    // placeholder for an external feed) under a :type:`TensorId`.
    void bind(TensorId id, void* tensor);

    // Direct access to the underlying graph + device.  Used inside
    // emitter ``.mm`` files for placeholder creation and op assembly.
    void* graph() const { return graph_; }

    // Device the executable is being compiled for (e.g. ``Device::GPU``).
    // Emitters route MPSGraph vs Metal-shader paths off this value.
    Device device() const { return device_; }

    // Multi-output ops (``split`` / ``split_at`` / ``unbind`` / ``chunk``
    // / ``topk`` / ``lstm`` …) can use this to skip building / binding
    // pieces that no downstream op consumes.  ``consumed_inputs`` is
    // populated by :class:`MpsBuilder` before the emit loop as the union
    // of every op's input ids across the whole trace.  An output id NOT
    // in this set is either a final graph output OR a dead piece —
    // emitters that wish to suppress dead pieces from becoming spurious
    // graph outputs check this before calling :func:`bind`.
    bool is_consumed(TensorId id) const {
        return consumed_inputs_.find(id) != consumed_inputs_.end();
    }
    void set_consumed_inputs(const std::unordered_set<TensorId>& s) { consumed_inputs_ = s; }

    // Trace ids that are the output of a ``softmax`` op.  A matmul that reads
    // one is the value-projection of an attention block; the MatmulEmitter
    // emits it transposed to dodge a buggy MPSGraph fused-attention pass (it
    // miscompiles ``softmax(...) @ V`` for some shapes — N in [17,24], B>=3 on
    // macOS 26).  Populated by :class:`MpsBuilder` before the emit loop.
    bool is_softmax_output(TensorId id) const {
        return softmax_outputs_.find(id) != softmax_outputs_.end();
    }
    void set_softmax_outputs(const std::unordered_set<TensorId>& s) { softmax_outputs_ = s; }

    // Saved-state side-table.
    //
    // Used by ops whose backward needs an intermediate tensor computed
    // during forward emission that ISN'T derivable from the inputs +
    // attrs alone — most notably the RNG mask of training-mode
    // dropout (the same random tensor must be referenced by both
    // forward and backward subgraphs; recomputing would draw fresh
    // random values).  The forward emitter stashes the tensor via
    // ``stash_saved(out_id, name, t)``; the VJP emitter retrieves it
    // via ``resolve_saved(out_id, name)`` and ties its backward
    // subgraph to the same MPSGraph node — MPSGraph evaluates each
    // node exactly once per dispatch, so the value is shared.
    //
    // Keyed on the forward op's primary output id + a short
    // op-defined slot name ("mask", "y", "rstd", …).  ``nullptr``
    // return signals "no slot found" — VJP must handle that fallback
    // (typically by recomputing or returning false).
    void stash_saved(TensorId out_id, const std::string& name, void* tensor) {
        saved_[out_id][name] = tensor;
    }
    void* resolve_saved(TensorId out_id, const std::string& name) const {
        auto it = saved_.find(out_id);
        if (it == saved_.end())
            return nullptr;
        auto jt = it->second.find(name);
        return jt == it->second.end() ? nullptr : jt->second;
    }

private:
    void* graph_;
    Device device_;
    std::unordered_map<TensorId, void*> tensors_;
    std::unordered_set<TensorId> consumed_inputs_;
    std::unordered_set<TensorId> softmax_outputs_;
    // Saved tensors keyed by (forward output id, slot name).  Owned
    // for the duration of this BuilderContext (== one compile call).
    std::unordered_map<TensorId, std::unordered_map<std::string, void*>> saved_;
};

// Abstract emitter — one per op family.
//
// Concrete subclasses live in this directory (``Linear.mm``,
// ``Conv.mm``, …).  Each subclass registers itself at process startup
// via :func:`register_emitter`.
//
// Emitter contract
// ----------------
// ``OpEmitter::emit`` returns ``bool`` and is responsible for binding
// each of its outputs explicitly via :func:`BuilderContext::bind`.
// Single-output ops bind ``node.outputs[0].id``; multi-output ops
// (``split``, ``topk``, ``lstm``, ...) bind every consumed slot
// ``node.outputs[k].id``.  The builder does **not** auto-bind —
// emitters own their output namespace.  ``true`` signals success;
// ``false`` signals "unsupported under this configuration" and the
// builder aborts the compile, marking the trace signature eager-only.
//
// This symmetric design matches :class:`VjpEmitter::emit` (which
// writes input gradients via :func:`BackwardContext::accumulate_grad`
// and also returns ``bool``).  A single mental model for both emitter
// families: *resolve inputs from context, build subgraph, bind
// outputs into context, return success*.
class LUCID_API OpEmitter {
public:
    // Virtual destructor — ensures derived per-op emitter destructors
    // run when held by the registry's base-class pointer.  Defaulted.
    virtual ~OpEmitter() = default;

    // Emit the MPSGraph subgraph for ``node``.
    //
    // Parameters
    // ----------
    // ctx : BuilderContext&
    //     Carries the in-progress graph and the id → tensor map.
    //     Implementations call :func:`BuilderContext::resolve` to find
    //     input tensors and :func:`BuilderContext::bind` to register
    //     their output(s).
    // node : const OpNode&
    //     The TraceIR node to emit.  Implementations may inspect
    //     ``node.inputs`` (TraceIds; a value of ``-1`` indicates an
    //     external feed that the builder must materialise as a
    //     placeholder before calling this emit) and ``node.outputs``
    //     (shape / dtype metadata for the produced tensor).
    //
    // Returns
    // -------
    // bool
    //     ``true`` on success — the emitter has bound each of its
    //     outputs via :func:`BuilderContext::bind`.  ``false`` signals
    //     "unsupported under this configuration"; the builder aborts
    //     and marks the trace signature as eager-only.
    virtual bool emit(BuilderContext& ctx, const OpNode& node) = 0;

    // Op name handled by this emitter — matched against ``OpNode::name``.
    virtual std::string_view op_name() const = 0;
};

// Look up the emitter for ``op_name``; returns ``nullptr`` if no
// emitter is registered.  Used by :class:`MpsBuilder` to decide
// between emit and eager-fallback.
LUCID_API OpEmitter* find_emitter(std::string_view op_name);

// Register ``emitter`` under its ``op_name()``.  Replaces any
// previously-registered emitter for the same name.  Process-global
// state; typically called from a static initialiser in the emitter's
// own .mm file.
LUCID_API void register_emitter(std::unique_ptr<OpEmitter> emitter);

}  // namespace lucid::compile
