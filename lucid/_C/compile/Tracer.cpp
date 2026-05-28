// lucid/_C/compile/Tracer.cpp
//
// Implementation of the thread-local op tracer for :mod:`lucid.compile`.
//
// The active Tracer for each thread lives in a single thread_local pointer
// (g_current).  :class:`OpScopeFull` reads this pointer at construction time
// and, if non-null, forwards the entering op into :func:`Tracer::on_op_enter`.
// Outside any ``_tracing()`` scope the pointer is nullptr and the hook
// short-circuits to a single TLS load — matching the un-profiled fast path
// described in :file:`Profiler.cpp`.
//
// Phase 1.1 records OpNode headers only — name, output dtype/shape/device,
// and a freshly-minted output TensorId.  ``inputs`` stays empty; Phase 1.2
// backfills it from the shared :func:`wire_autograd` site.

#include "Tracer.h"

#include "../core/TensorImpl.h"

namespace lucid::compile {

namespace {
// Per-thread active Tracer.  nullptr means "no tracing" for this thread —
// the OpScopeFull hook does nothing.
thread_local Tracer* g_current = nullptr;
}  // namespace

Tracer* current_tracer() {
    return g_current;
}

void set_current_tracer(Tracer* t) {
    g_current = t;
}

void Tracer::on_op_enter(std::string_view name, Device device, Dtype dtype, Shape shape) {
    // Mint a fresh TensorId for the (single, Phase-1.1) output.  Phase 1.2 will
    // append additional TensorMeta entries here for multi-output ops; the
    // append-after-mint pattern keeps ids monotone within one TraceGraph.
    const TensorId out_id = graph_.next_id++;

    OpNode node;
    node.name = std::string(name);
    // inputs filled by on_op_io if the op goes through wire_autograd.
    node.outputs.push_back(TensorMeta{out_id, std::move(shape), dtype, device});
    graph_.ops.push_back(std::move(node));
}

void Tracer::on_op_io(const std::vector<TensorImplPtr>& inputs, const TensorImplPtr& output) {
    // Defensive: if on_op_enter never ran for this op (e.g. an unexpected
    // hook ordering), there is no node to attach I/O to.  Drop silently
    // rather than corrupting the IR.
    if (graph_.ops.empty())
        return;

    // Hold owning references to every TensorImpl we see for the rest
    // of the trace's lifetime.  Short-lived tensors (e.g. the
    // zero-bias materialised by ``conv_bias_or_zero``) would otherwise
    // be destroyed before the trace ends, freeing their raw pointer
    // back to the allocator pool — the next allocation in the same
    // trace can then reuse that pointer, and the stale
    // ``impl_to_id_`` entry would resolve to the wrong id.
    for (const auto& inp : inputs) {
        if (inp)
            live_refs_.push_back(inp);
    }
    if (output)
        live_refs_.push_back(output);

    OpNode& node = graph_.ops.back();

    // Resolve each input id.
    //   - nullptr slot (e.g. an optional bias) → -1 sentinel.
    //   - Pointer already known (output of an earlier traced op) →
    //     look up its id in ``impl_to_id_``.
    //   - Pointer not yet known → external feed.  Mint a fresh id,
    //     remember it both in ``impl_to_id_`` (so a later op that
    //     consumes the same buffer resolves to the same id) and in
    //     ``external_feeds_`` (with an owning :type:`TensorImplPtr`
    //     so the builder can materialise an MPSGraph placeholder and
    //     bind the input data at run time without races against
    //     buffer release).
    // Last-write-wins semantics: callers (e.g. ops with non-diff
    // inputs like ``embedding``'s indices) may invoke ``on_op_io``
    // a second time after the autograd kernel's internal call to
    // fill in the full graph-level input list.  Clear here so the
    // most recent call is authoritative; append would otherwise
    // duplicate ids and confuse the emitter's resolve step.
    node.inputs.clear();
    node.inputs.reserve(inputs.size());
    for (const auto& inp : inputs) {
        TensorImpl* raw = inp.get();
        if (raw == nullptr) {
            node.inputs.push_back(-1);
            continue;
        }
        const auto it = impl_to_id_.find(raw);
        if (it != impl_to_id_.end()) {
            node.inputs.push_back(it->second);
        } else {
            const TensorId fresh = graph_.next_id++;
            impl_to_id_[raw] = fresh;
            external_feeds_[fresh] = inp;  // owning ref
            node.inputs.push_back(fresh);
        }
    }

    // Register the output's identity so the next op that consumes it can
    // resolve its input id.  In-place ops (output pointer aliases an
    // input) intentionally overwrite the prior mapping — the new id now
    // represents the post-mutation state of the buffer, which is what
    // any later consumer should see.
    //
    // Multi-output handling: a single op (split / split_at / unbind /
    // chunk / lstm / topk …) may call ``on_op_io`` multiple times within
    // one ``OpScopeFull``, once per piece.  ``on_op_enter`` always seeds
    // ``node.outputs`` with a single placeholder TensorMeta.  We detect
    // "is this the first ``on_op_io`` call for this op?" by checking
    // whether the placeholder's id has been claimed by any impl in
    // ``impl_to_id_``.  First call → update ``outputs[0]`` (existing
    // single-output semantics).  Subsequent calls → append a fresh
    // ``TensorMeta`` so each piece gets its own id; the emit-side
    // emitter binds piece-N's MPSGraphTensor to ``outputs[N].id`` via
    // :func:`BuilderContext::bind`.
    if (output && !node.outputs.empty()) {
        // In-place / alias detection: if this output impl is already
        // mapped, just overwrite the mapping to the latest "post-
        // mutation" id (preserves the original single-output in-place
        // semantics).
        const auto existing = impl_to_id_.find(output.get());
        if (existing != impl_to_id_.end()) {
            existing->second = node.outputs[0].id;
            // Refresh the metadata too — same rationale as below.
            node.outputs[0].shape = output->shape();
            node.outputs[0].dtype = output->dtype();
            node.outputs[0].device = output->device();
        } else {
            // Has any prior impl claimed outputs[0].id?  Walk
            // ``impl_to_id_`` (small map — order of dozens) once.
            const TensorId slot0_id = node.outputs[0].id;
            bool first_call = true;
            for (const auto& kv : impl_to_id_) {
                if (kv.second == slot0_id) {
                    first_call = false;
                    break;
                }
            }
            if (first_call) {
                // First on_op_io call for this op — update outputs[0].
                // OpScopeFull ctor records the shape that was known at
                // scope entry — for some ops (tile / repeat / pad …)
                // that's stale by the time the actual output is
                // materialised.  Refresh it from the real output so
                // downstream emitters get the right shape / dtype /
                // device meta.
                impl_to_id_[output.get()] = slot0_id;
                node.outputs[0].shape = output->shape();
                node.outputs[0].dtype = output->dtype();
                node.outputs[0].device = output->device();
            } else {
                // Subsequent call — multi-output op.  Mint a fresh id
                // and append a new TensorMeta so the emitter can bind
                // this piece independently.
                const TensorId new_id = graph_.next_id++;
                node.outputs.push_back(
                    TensorMeta{new_id, output->shape(), output->dtype(), output->device()});
                impl_to_id_[output.get()] = new_id;
            }
        }
    }
}

void Tracer::on_op_attr(std::string_view key, AttributeValue value) {
    if (graph_.ops.empty())
        return;
    graph_.ops.back().attrs.insert_or_assign(std::string(key), std::move(value));
}

}  // namespace lucid::compile
