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
    void set_consumed_inputs(const std::unordered_set<TensorId>& s) {
        consumed_inputs_ = s;
    }

private:
    void* graph_;
    Device device_;
    std::unordered_map<TensorId, void*> tensors_;
    std::unordered_set<TensorId> consumed_inputs_;
};

// Abstract emitter — one per op family.
//
// Concrete subclasses live in this directory (``Linear.mm``,
// ``Conv.mm``, …).  Each subclass registers itself at process startup
// via :func:`register_emitter`.
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
    // void* (MPSGraphTensor*)
    //     The op's output tensor inside the graph, or ``nullptr`` to
    //     signal "unsupported under this configuration" — the builder
    //     aborts and marks the trace signature as eager-only.  The
    //     returned tensor is also bound into ``ctx`` automatically by
    //     :class:`MpsBuilder` so the next op can consume it.
    virtual void* emit(BuilderContext& ctx, const OpNode& node) = 0;

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
