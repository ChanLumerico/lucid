// lucid/_C/compile/MpsBuilder.mm
//
// Walks a :class:`TraceGraph` and produces a compiled MPSGraph
// executable.  See :file:`MpsBuilder.h` for the contract.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "../backend/gpu/mps/MpsBridge.h"
#include "../core/Determinism.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "MpsBuilder.h"
#include "OpEmitters/OpEmitter.h"

namespace lucid::compile {

// Defined in CompiledExecutable.mm — friend-private state of the
// :class:`CompiledExecutable` aggregate we build below.
class CompiledExecutable;

namespace {

inline MPSDataType to_mps_dtype(Dtype dt) {
    switch (dt) {
        case Dtype::F32:
            return MPSDataTypeFloat32;
        case Dtype::F16:
            return MPSDataTypeFloat16;
        case Dtype::I32:
            return MPSDataTypeInt32;
        case Dtype::I64:
            return MPSDataTypeInt64;
        case Dtype::Bool:
            return MPSDataTypeBool;
        default:
            throw std::runtime_error(
                "lucid::compile: dtype not supported on the MPSGraph compile path");
    }
}

inline NSArray<NSNumber*>* shape_to_nsarray(const Shape& shape) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:shape.size()];
    for (std::int64_t d : shape)
        [out addObject:[NSNumber numberWithLongLong:d]];
    return out;
}

// Builds a compile descriptor.
//
//   * Determinism ON (uncommon path): ``optimizationLevel = Level0``
//     to guarantee bit-stable kernel selection across runs.
//   * Determinism OFF (default): ``optimizationLevel = Level1`` — pay
//     extra compile-time analysis for runtime fusion wins.  On macOS
//     26 SDK this opens up kernel-merge passes MPSGraph keeps gated
//     behind Level1; the cost is a one-time ~5-20ms compile-time
//     increase per signature, amortised over every subsequent run.
inline MPSGraphCompilationDescriptor* make_compile_descriptor() {
    MPSGraphCompilationDescriptor* desc =
        [[MPSGraphCompilationDescriptor alloc] init];
    if (::lucid::Determinism::is_enabled())
        desc.optimizationLevel = MPSGraphOptimizationLevel0;
    else
        desc.optimizationLevel = MPSGraphOptimizationLevel1;
    return desc;
}

}  // namespace

}  // namespace lucid::compile

// Bring the .mm-only class definition into scope.  CompiledExecutable
// is fully defined inside the other translation unit; here we redeclare
// it as a complete class to access its fields (one TU pattern).
namespace lucid::compile {

// Full definition that matches CompiledExecutable.mm.  The two .mm
// files must keep these in sync — they are compiled into separate
// object files but expose the same layout, so the cache + builder can
// read/write the same fields without breaking the ABI.
class CompiledExecutable {
public:
    MPSGraphExecutable* executable = nil;  // ARC strong (matches .mm)
    // Phase 1.9: retained source MPSGraph for variable-bearing
    // executables — keeps the variable storage live for the executable's
    // lifetime.  See the matching field in CompiledExecutable.mm for
    // rationale.  nullptr for non-variable compiles (the common case).
    void* source_graph = nullptr;  // __bridge_retained MPSGraph*
    std::vector<TensorId> input_ids;
    std::vector<TensorId> output_ids;
    std::vector<Shape> input_shapes;
    std::vector<Dtype> input_dtypes;
    std::vector<Shape> output_shapes;
    std::vector<Dtype> output_dtypes;
    Device device = Device::GPU;
    std::vector<TensorId> grad_output_ids;  // Phase 1.3
    bool dynamic_batch = false;             // Phase 1.6
    std::unordered_set<std::size_t> static_feed_slots;  // Phase 1.6

    ~CompiledExecutable() {
        // Mirror the destructor in CompiledExecutable.mm — release the
        // retained source MPSGraph (if any).  Defined inline here so the
        // C++ destructor calls the right release; ARC handles
        // ``executable``.
        if (source_graph != nullptr) {
            @autoreleasepool {
                MPSGraph* g = (__bridge_transfer MPSGraph*)source_graph;
                (void)g;
            }
            source_graph = nullptr;
        }
    }
};

}  // namespace lucid::compile

namespace lucid::compile {

CompiledExecutable* compile_trace(
        const TraceGraph& graph,
        const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
        std::string* error_msg,
        bool dynamic_batch,
        const std::vector<TensorId>& param_ids,
        const std::vector<TensorId>& explicit_outputs) {
    // ``dynamic_batch`` (Phase 1.6): leading dim of every non-parameter
    // feed becomes a symbolic placeholder (-1) so a single executable
    // handles variable batch size.  ``param_ids`` supplies which feed
    // ids are model parameters whose first dim is structural (not a
    // batch axis) — those stay fixed.  The forward-only path can do
    // this safely because it does not call ``gradientForPrimaryTensor:``,
    // which currently refuses to build gradients for dynamic-shape
    // primaries.  ``make_step`` / the backward-aware path therefore
    // continues to compile a separate executable per batch size.
    const std::unordered_set<TensorId> param_id_set(
        param_ids.begin(), param_ids.end());
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_trace: empty TraceGraph");

    // Trace must be single-device — Phase 1.2 supports GPU only.
    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (op.outputs.empty())
            continue;
        device = op.outputs[0].device;
        break;
    }
    if (device != Device::GPU)
        return fail("compile_trace: only Device::GPU traces are supported in Phase 1.2");

    // Every op must have all of its inputs resolved (no -1 sentinels)
    // and must have an emitter registered.  These are the two reasons
    // we fall back to eager at this stage.
    //
    // Exception: a host-precomputed factory (arange / eye / linspace /
    // logspace / meshgrid …) records an :class:`OpScopeFull` but
    // doesn't wire its output through the tracer.  Downstream consumers
    // therefore see the factory's :type:`TensorImpl` as a *fresh
    // external feed*, not an intermediate, and the op header in the
    // trace is dead code.  We skip the emitter check for such
    // ``inputs.empty()`` ops; the emit loop below mirrors the skip.
    for (const auto& op : graph.ops) {
        if (op.outputs.empty() || op.outputs[0].device != device)
            return fail("compile_trace: mixed-device trace (only single-device Phase 1.2)");
        for (TensorId iid : op.inputs) {
            if (iid < 0)
                return fail("compile_trace: op '" + op.name +
                            "' has an unresolved input slot (pending/eager-only)");
        }
        if (find_emitter(op.name) == nullptr) {
            if (op.inputs.empty())
                continue;  // dead-code factory header — skip below too
            return fail("compile_trace: no emitter registered for op '" + op.name + "'");
        }
    }

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Build placeholders for every external feed.  The id order is
        // the order MPSGraph will expect inputs in at run time — we
        // sort by id for determinism (so two compiled traces with the
        // same structure always have the same feed order regardless
        // of map iteration order).
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_trace: external feed for id " +
                            std::to_string(tid) + " is null");
            if (impl->device() != device)
                return fail("compile_trace: external feed device mismatch");

            const Shape feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            const bool is_user_input_ct =
                (param_id_set.find(tid) == param_id_set.end());
            NSArray<NSNumber*>* ns_shape;
            if (dynamic_batch && is_user_input_ct && !feed_shape.empty()) {
                NSMutableArray<NSNumber*>* dyn =
                    [NSMutableArray arrayWithCapacity:feed_shape.size()];
                [dyn addObject:[NSNumber numberWithLongLong:-1]];
                for (std::size_t k = 1; k < feed_shape.size(); ++k)
                    [dyn addObject:[NSNumber numberWithLongLong:feed_shape[k]]];
                ns_shape = dyn;
            } else {
                ns_shape = shape_to_nsarray(feed_shape);
            }
            MPSDataType ns_dt;
            try {
                ns_dt = to_mps_dtype(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_trace: ") + e.what());
            }

            MPSGraphTensor* ph =
                [graph_obj placeholderWithShape:ns_shape
                                       dataType:ns_dt
                                           name:[NSString stringWithFormat:@"feed_%lld",
                                                                            (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Verbose: when LUCID_COMPILE_VERBOSE=1 print per-op input/output
        // MPSGraph tensor shapes so reshape-compat failures can be
        // localised without external profilers.
        const bool verbose = []() {
            const char* s = std::getenv("LUCID_COMPILE_VERBOSE");
            return s && *s == '1';
        }();

        // Pre-compute the trace-wide "consumed by some op" set and hand
        // it to BuilderContext so multi-output emitters (split / split_at
        // / unbind / chunk …) can skip binding pieces that no downstream
        // op uses.  Without this filter, dead pieces would silently
        // appear as additional graph outputs (a single ``a, _ = x.split``
        // becomes a 2-output executable returning ``_`` to the user).
        std::unordered_set<TensorId> trace_consumed;
        for (const auto& n : graph.ops)
            for (TensorId iid : n.inputs)
                if (iid >= 0)
                    trace_consumed.insert(iid);
        ctx.set_consumed_inputs(trace_consumed);

        // Emit ops in dispatch order.
        std::size_t op_idx = 0;
        for (const auto& node : graph.ops) {
            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty()) {
                    // Dead-code host-factory header (arange / eye / …)
                    // — its output already shows up as an external feed
                    // through whatever consumer used the tensor next.
                    ++op_idx;
                    continue;
                }
                // Defensive against a racy registry mutation between
                // the precheck above and this loop.
                return fail("compile_trace: emitter vanished for op '" + node.name + "'");
            }

            if (verbose) {
                fprintf(stderr, "[compile] op[%zu] %s  inputs=[", op_idx, node.name.c_str());
                for (std::size_t i = 0; i < node.inputs.size(); ++i) {
                    if (i) fputc(',', stderr);
                    TensorId iid = node.inputs[i];
                    if (iid < 0) { fputs("none", stderr); continue; }
                    MPSGraphTensor* t = (__bridge MPSGraphTensor*)ctx.resolve(iid);
                    if (t == nil) { fprintf(stderr, "id%lld[unresolved]", (long long)iid); continue; }
                    fputc('(', stderr);
                    NSArray<NSNumber*>* sh = t.shape;
                    for (NSUInteger k = 0; k < sh.count; ++k) {
                        if (k) fputc(',', stderr);
                        fprintf(stderr, "%lld", [sh[k] longLongValue]);
                    }
                    fputc(')', stderr);
                }
                fputs("] expected_out=(", stderr);
                if (!node.outputs.empty()) {
                    const auto& sh = node.outputs[0].shape;
                    for (std::size_t k = 0; k < sh.size(); ++k) {
                        if (k) fputc(',', stderr);
                        fprintf(stderr, "%lld", (long long)sh[k]);
                    }
                }
                fputs(")\n", stderr);
            }

            void* result = emitter->emit(ctx, node);
            if (result == nullptr)
                return fail("compile_trace: emitter for op '" + node.name +
                            "' returned nullptr (unsupported variant)");
            if (verbose) {
                MPSGraphTensor* rt = (__bridge MPSGraphTensor*)result;
                fprintf(stderr, "[compile]   → emitted (");
                NSArray<NSNumber*>* sh = rt.shape;
                for (NSUInteger k = 0; k < sh.count; ++k) {
                    if (k) fputc(',', stderr);
                    fprintf(stderr, "%lld", [sh[k] longLongValue]);
                }
                fputs(")\n", stderr);
                fflush(stderr);
            }
            // Auto-bind outputs[0] to the emit() return value for
            // single-output ops.  Multi-output emitters (split /
            // split_at / unbind / chunk / …) handle their own binding
            // via :func:`BuilderContext::bind` because outputs[0] may
            // be a dead piece the user didn't pick — auto-binding
            // it would leak the dead tensor into the graph's output
            // list.  They still return a non-null sentinel tensor to
            // signal "emit succeeded".
            if (!node.outputs.empty() && node.outputs.size() == 1)
                ctx.bind(node.outputs[0].id, result);
            ++op_idx;
        }

        // Build the target list: any output id that is not consumed by
        // a later op is a graph output.  Phase 1.2 step 1 typically
        // has a single Linear → single target.
        std::unordered_set<TensorId> consumed;
        for (const auto& node : graph.ops)
            for (TensorId iid : node.inputs)
                if (iid >= 0)
                    consumed.insert(iid);

        std::vector<TensorId> target_ids;
        std::vector<MPSGraphTensor*> target_tensors;
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;

        auto add_target = [&](TensorId id, const Shape& shape, Dtype dtype) -> bool {
            void* t_void = ctx.resolve(id);
            if (t_void == nullptr) return false;
            target_ids.push_back(id);
            target_tensors.push_back((__bridge MPSGraphTensor*)t_void);
            output_shapes.push_back(shape);
            output_dtypes.push_back(dtype);
            return true;
        };

        if (!explicit_outputs.empty()) {
            // Caller specified exactly which ids are graph outputs (e.g.
            // the Python compile harness extracted them from the user's
            // return value).  Look up each id in the trace metadata and
            // bind it as a target.  Honoring this list drops "unconsumed
            // but Python-discarded" intermediates (RNN's ``h_n`` when
            // user does ``out, _ = rnn(x)``).
            std::unordered_map<TensorId, std::pair<Shape, Dtype>> id_to_meta;
            for (const auto& node : graph.ops)
                for (const auto& meta : node.outputs)
                    id_to_meta[meta.id] = {meta.shape, meta.dtype};
            // External feeds can also be returned (`return x` directly).
            // Look those up from the feed map; the placeholder is already
            // bound in ctx.
            for (TensorId tid : explicit_outputs) {
                auto it = id_to_meta.find(tid);
                if (it != id_to_meta.end()) {
                    if (!add_target(tid, it->second.first, it->second.second))
                        return fail("compile_trace: explicit_output id " +
                                    std::to_string(tid) + " has no MPSGraph binding");
                } else {
                    // Try external feed.
                    auto fit = external_feeds.find(tid);
                    if (fit == external_feeds.end() || !fit->second)
                        return fail("compile_trace: explicit_output id " +
                                    std::to_string(tid) + " not in trace");
                    if (!add_target(tid, fit->second->shape(), fit->second->dtype()))
                        return fail("compile_trace: explicit_output id " +
                                    std::to_string(tid) + " external feed not bound");
                }
            }
        } else {
            // Auto-detect: walk every output slot of each op, skip ones
            // consumed by a later op, skip un-bound (dead-piece / host
            // factory) ids.  The emitter for multi-output ops handles
            // ctx.bind selectively so dead pieces resolve to nullptr.
            for (const auto& node : graph.ops) {
                for (const auto& meta : node.outputs) {
                    if (consumed.count(meta.id))
                        continue;
                    if (!add_target(meta.id, meta.shape, meta.dtype)) {
                        // Either a host-side factory header (arange /
                        // eye / linspace …) or an un-bound dead piece
                        // of a multi-output op.  Drop silently.
                        continue;
                    }
                }
            }
        }
        if (target_tensors.empty())
            return fail("compile_trace: no graph outputs found (every op output consumed)");

        // Build the feed dictionary for compileWithDevice:feeds: — maps
        // each placeholder to a shape descriptor.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = to_mps_dtype(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            MPSGraphShapedType* st =
                [[MPSGraphShapedType alloc] initWithShape:ns_shape dataType:ns_dt];
            feed_dict[feed_tensors[i]] = st;
        }
        NSMutableArray<MPSGraphTensor*>* target_arr =
            [NSMutableArray arrayWithCapacity:target_tensors.size()];
        for (MPSGraphTensor* t : target_tensors)
            [target_arr addObject:t];

        id<MTLDevice> mtl_device =
            (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled = [graph_obj compileWithDevice:mps_device
                                                              feeds:feed_dict
                                                      targetTensors:target_arr
                                                   targetOperations:nil
                                              compilationDescriptor:
                                                  make_compile_descriptor()];
        if (compiled == nil)
            return fail("compile_trace: MPSGraph compilation returned nil");

        auto* exe = new CompiledExecutable();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = target_ids;
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        // Phase 1.6: tell the runner which feed slots are parameters
        // (whose first dim is structural) so a BS != trace-BS call
        // only rewrites the leading axis for non-parameter slots.
        if (dynamic_batch) {
            exe->dynamic_batch = true;
            const std::unordered_set<TensorId> param_id_set_for_exe(
                param_ids.begin(), param_ids.end());
            for (std::size_t i = 0; i < exe->input_ids.size(); ++i) {
                if (param_id_set_for_exe.find(exe->input_ids[i]) !=
                    param_id_set_for_exe.end()) {
                    exe->static_feed_slots.insert(i);
                }
            }
        }
        return exe;
    }
}

// ── Phase 1.3 ────────────────────────────────────────────────────────────────

CompiledExecutable* compile_trace_with_backward(
        const TraceGraph& graph,
        const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
        TensorId loss_id,
        const std::vector<TensorId>& param_ids,
        std::string* error_msg,
        bool dynamic_batch) {
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_trace_with_backward: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_trace_with_backward: param_ids must be non-empty");

    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (op.outputs.empty())
            continue;
        device = op.outputs[0].device;
        break;
    }
    if (device != Device::GPU)
        return fail("compile_trace_with_backward: only Device::GPU traces are supported");

    for (const auto& op : graph.ops) {
        if (op.outputs.empty() || op.outputs[0].device != device)
            return fail("compile_trace_with_backward: mixed-device trace");
        for (TensorId iid : op.inputs) {
            if (iid < 0)
                return fail("compile_trace_with_backward: op '" + op.name +
                            "' has an unresolved input slot");
        }
        if (find_emitter(op.name) == nullptr) {
            if (op.inputs.empty())
                continue;  // dead-code host-precomputed factory header
            return fail("compile_trace_with_backward: no emitter for op '" + op.name + "'");
        }
    }

    // Phase 1.5 step 5.3 precheck: MPSGraph's ``gradientForPrimaryTensor:``
    // abort-on-LLVM-level when fwd ops have heterogeneous floating
    // dtypes (e.g. an AMP-autocast block leaves some ops in F16 and
    // others in F32, and the auto-derived backward inserts an
    // incompatible broadcast).  We can't catch that abort, so guard
    // by refusing to compile a trace whose floating dtypes are mixed.
    // Integral / Bool dtypes are exempt — they live in CE's gather /
    // mask path and never feed into the backward graph.
    {
        Dtype floating_dtype = Dtype::F32;
        bool seen_floating = false;
        for (const auto& op : graph.ops) {
            for (const auto& meta : op.outputs) {
                if (meta.dtype != Dtype::F32 && meta.dtype != Dtype::F16)
                    continue;
                if (!seen_floating) {
                    floating_dtype = meta.dtype;
                    seen_floating = true;
                } else if (meta.dtype != floating_dtype) {
                    return fail(
                        "compile_trace_with_backward: mixed floating dtypes in "
                        "trace (AMP autocast not yet supported by the compile path)");
                }
            }
        }
    }

    // Every param id must be an external feed (gradient is meaningful only
    // for trace inputs; intermediate-tensor gradients are not user state).
    for (TensorId pid : param_ids) {
        if (external_feeds.find(pid) == external_feeds.end())
            return fail("compile_trace_with_backward: param id " + std::to_string(pid) +
                        " is not an external feed");
    }

    // Phase 1.6: ``dynamic_batch`` opts the leading dim of every
    // non-parameter feed into a symbolic placeholder (-1).  Parameter
    // shapes (weights, biases, running stats) stay fixed because
    // their first dim is structural (out_channels, etc.) and not a
    // batch axis.
    std::unordered_set<TensorId> param_id_set(param_ids.begin(), param_ids.end());

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_trace_with_backward: feed id " + std::to_string(tid) +
                            " is null");
            if (impl->device() != device)
                return fail("compile_trace_with_backward: feed device mismatch");

            const Shape feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            const bool is_user_input =
                (param_id_set.find(tid) == param_id_set.end());
            NSArray<NSNumber*>* ns_shape;
            if (dynamic_batch && is_user_input && !feed_shape.empty()) {
                NSMutableArray<NSNumber*>* dyn =
                    [NSMutableArray arrayWithCapacity:feed_shape.size()];
                [dyn addObject:[NSNumber numberWithLongLong:-1]];
                for (std::size_t k = 1; k < feed_shape.size(); ++k)
                    [dyn addObject:[NSNumber numberWithLongLong:feed_shape[k]]];
                ns_shape = dyn;
            } else {
                ns_shape = shape_to_nsarray(feed_shape);
            }
            MPSDataType ns_dt;
            try {
                ns_dt = to_mps_dtype(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_trace_with_backward: ") + e.what());
            }
            MPSGraphTensor* ph =
                [graph_obj placeholderWithShape:ns_shape
                                       dataType:ns_dt
                                           name:[NSString stringWithFormat:@"feed_%lld",
                                                                            (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Emit forward ops.
        for (const auto& node : graph.ops) {
            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty())
                    continue;  // dead-code host-factory header
                return fail("compile_trace_with_backward: emitter vanished for op '" +
                            node.name + "'");
            }
            void* result = emitter->emit(ctx, node);
            if (result == nullptr)
                return fail("compile_trace_with_backward: emitter for op '" + node.name +
                            "' returned nullptr");
            if (!node.outputs.empty())
                ctx.bind(node.outputs[0].id, result);
        }

        // Resolve the loss tensor.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_trace_with_backward: loss id " + std::to_string(loss_id) +
                        " has no MPSGraph tensor (not produced by trace?)");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

        // Note: MPSGraph's gradientForPrimaryTensor mishandles 0-D
        // scalar primaries.  The Reduction emitter (OpEmitters/Reduction.mm)
        // works around this by leaving one size-1 axis when reducing
        // every dim, so any loss that originated as a full-reduce sum/mean
        // is at least 1-D from MPSGraph's point of view (output_shapes
        // still records the user-visible 0-D shape so the buffer wrap
        // matches eager exactly).

        // Resolve param placeholders.
        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_trace_with_backward: param id " + std::to_string(pid) +
                            " has no placeholder");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
        }

        // Auto-derive gradients via MPSGraph.
        NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
            [graph_obj gradientForPrimaryTensor:loss_t
                                    withTensors:param_arr
                                           name:@"lucid_grads"];
        if (grad_map == nil)
            return fail("compile_trace_with_backward: gradientForPrimaryTensor returned nil");

        // Look up the grad tensor per param (the map keys are the param
        // placeholders we just passed in).
        std::vector<MPSGraphTensor*> grad_tensors;
        std::vector<Shape> grad_shapes;
        std::vector<Dtype> grad_dtypes;
        grad_tensors.reserve(param_ids.size());
        grad_shapes.reserve(param_ids.size());
        grad_dtypes.reserve(param_ids.size());
        for (std::size_t i = 0; i < param_ids.size(); ++i) {
            MPSGraphTensor* g_t = grad_map[param_arr[i]];
            if (g_t == nil)
                return fail("compile_trace_with_backward: no gradient produced for param id " +
                            std::to_string(param_ids[i]));
            grad_tensors.push_back(g_t);
            // The gradient shape always matches the param shape, dtype too.
            const auto& p_impl = external_feeds.at(param_ids[i]);
            grad_shapes.push_back(p_impl->shape());
            grad_dtypes.push_back(p_impl->dtype());
        }

        // Targets = [loss] + grad tensors.
        NSMutableArray<MPSGraphTensor*>* target_arr =
            [NSMutableArray arrayWithCapacity:1 + grad_tensors.size()];
        [target_arr addObject:loss_t];
        for (MPSGraphTensor* g_t : grad_tensors)
            [target_arr addObject:g_t];

        // Build feed dict.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = to_mps_dtype(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] =
                [[MPSGraphShapedType alloc] initWithShape:ns_shape dataType:ns_dt];
        }

        id<MTLDevice> mtl_device =
            (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled = [graph_obj compileWithDevice:mps_device
                                                              feeds:feed_dict
                                                      targetTensors:target_arr
                                                   targetOperations:nil
                                              compilationDescriptor:
                                                  make_compile_descriptor()];
        if (compiled == nil)
            return fail("compile_trace_with_backward: MPSGraph compilation returned nil");

        // Mint fresh ids for the grad outputs, well past the trace's
        // id space.  Convention: grad_id_for_param[i] = graph.next_id + 1 + i
        // (offset by 1 to leave room for any future loss-id alias).
        std::vector<TensorId> grad_output_ids;
        grad_output_ids.reserve(param_ids.size());
        TensorId next_grad_id = graph.next_id + 1;
        for (std::size_t i = 0; i < param_ids.size(); ++i)
            grad_output_ids.push_back(next_grad_id + static_cast<TensorId>(i));

        // Assemble the per-output meta vectors in target order: loss first,
        // then grads in param_ids order.
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;
        output_shapes.reserve(1 + grad_shapes.size());
        output_dtypes.reserve(1 + grad_dtypes.size());

        // Find the loss meta in the trace.
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops) {
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        }
        if (!loss_found)
            return fail("compile_trace_with_backward: loss id meta missing from trace");
        output_shapes.push_back(std::move(loss_shape));
        output_dtypes.push_back(loss_dtype);
        for (std::size_t i = 0; i < grad_shapes.size(); ++i) {
            output_shapes.push_back(std::move(grad_shapes[i]));
            output_dtypes.push_back(grad_dtypes[i]);
        }

        auto* exe = new CompiledExecutable();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(grad_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe;
    }
}

// ── Fused training step (Phase 1.7) ──────────────────────────────────

namespace {

// Build a 0-D constant tensor on ``g`` with the given scalar/dtype.
inline MPSGraphTensor* scalar_const(MPSGraph* g, double v, MPSDataType dt) {
    return [g constantWithScalar:v dataType:dt];
}

// Per-optimizer MPSGraph update emitters.  Each takes the current
// param + grad + state buffers (already resolved as MPSGraphTensors)
// and per-step scalar feeds, and returns (new_param, new_state[]).
//
// SGD with optional momentum / nesterov / weight_decay.  ``mom`` is
// the current momentum tensor (or nullptr if momentum == 0).
struct SgdOutputs {
    MPSGraphTensor* new_param;
    MPSGraphTensor* new_mom;  // nil if no momentum
};

SgdOutputs emit_sgd_update(MPSGraph* g,
                           MPSGraphTensor* param,
                           MPSGraphTensor* grad,
                           MPSGraphTensor* mom,
                           const OptimizerSpec& s,
                           MPSDataType dt) {
    MPSGraphTensor* g_eff = grad;
    if (s.weight_decay != 0.0) {
        MPSGraphTensor* wd = scalar_const(g, s.weight_decay, dt);
        g_eff = [g additionWithPrimaryTensor:g_eff
                              secondaryTensor:[g multiplicationWithPrimaryTensor:wd
                                                                   secondaryTensor:param
                                                                              name:nil]
                                         name:nil];
    }
    MPSGraphTensor* lr_c = scalar_const(g, s.lr, dt);
    if (s.momentum != 0.0) {
        MPSGraphTensor* mu = scalar_const(g, s.momentum, dt);
        MPSGraphTensor* one_minus_damp =
            scalar_const(g, 1.0 - s.dampening, dt);
        MPSGraphTensor* new_m = [g
            additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:mu
                                                          secondaryTensor:mom
                                                                     name:nil]
                      secondaryTensor:[g multiplicationWithPrimaryTensor:one_minus_damp
                                                          secondaryTensor:g_eff
                                                                     name:nil]
                                 name:nil];
        MPSGraphTensor* eff_g;
        if (s.nesterov) {
            eff_g = [g additionWithPrimaryTensor:g_eff
                                  secondaryTensor:[g multiplicationWithPrimaryTensor:mu
                                                                       secondaryTensor:new_m
                                                                                  name:nil]
                                             name:nil];
        } else {
            eff_g = new_m;
        }
        MPSGraphTensor* new_p =
            [g subtractionWithPrimaryTensor:param
                            secondaryTensor:[g multiplicationWithPrimaryTensor:lr_c
                                                                 secondaryTensor:eff_g
                                                                            name:nil]
                                       name:nil];
        return {new_p, new_m};
    }
    MPSGraphTensor* new_p =
        [g subtractionWithPrimaryTensor:param
                        secondaryTensor:[g multiplicationWithPrimaryTensor:lr_c
                                                             secondaryTensor:g_eff
                                                                        name:nil]
                                   name:nil];
    return {new_p, nil};
}

struct AdamOutputs {
    MPSGraphTensor* new_param;
    MPSGraphTensor* new_m;
    MPSGraphTensor* new_v;
};

// Adam / AdamW share the same moment update; the only difference is
// where weight decay is applied: Adam folds it into ``grad`` before
// the moments, AdamW applies it directly to the param outside.
AdamOutputs emit_adam_update(MPSGraph* g,
                             MPSGraphTensor* param,
                             MPSGraphTensor* grad,
                             MPSGraphTensor* m_buf,
                             MPSGraphTensor* v_buf,
                             MPSGraphTensor* bias1,
                             MPSGraphTensor* bias2,
                             const OptimizerSpec& s,
                             MPSDataType dt,
                             bool decoupled_wd) {
    MPSGraphTensor* g_eff = grad;
    if (!decoupled_wd && s.weight_decay != 0.0) {
        MPSGraphTensor* wd = scalar_const(g, s.weight_decay, dt);
        g_eff = [g additionWithPrimaryTensor:g_eff
                              secondaryTensor:[g multiplicationWithPrimaryTensor:wd
                                                                   secondaryTensor:param
                                                                              name:nil]
                                         name:nil];
    }
    MPSGraphTensor* beta1_c = scalar_const(g, s.beta1, dt);
    MPSGraphTensor* beta2_c = scalar_const(g, s.beta2, dt);
    MPSGraphTensor* one_minus_b1 = scalar_const(g, 1.0 - s.beta1, dt);
    MPSGraphTensor* one_minus_b2 = scalar_const(g, 1.0 - s.beta2, dt);
    MPSGraphTensor* eps_c = scalar_const(g, s.eps, dt);
    MPSGraphTensor* lr_c = scalar_const(g, s.lr, dt);

    MPSGraphTensor* g_sq =
        [g multiplicationWithPrimaryTensor:g_eff secondaryTensor:g_eff name:nil];

    MPSGraphTensor* new_m = [g
        additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:beta1_c
                                                      secondaryTensor:m_buf
                                                                 name:nil]
                  secondaryTensor:[g multiplicationWithPrimaryTensor:one_minus_b1
                                                      secondaryTensor:g_eff
                                                                 name:nil]
                             name:nil];
    MPSGraphTensor* new_v = [g
        additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:beta2_c
                                                      secondaryTensor:v_buf
                                                                 name:nil]
                  secondaryTensor:[g multiplicationWithPrimaryTensor:one_minus_b2
                                                      secondaryTensor:g_sq
                                                                 name:nil]
                             name:nil];
    MPSGraphTensor* m_hat = [g divisionWithPrimaryTensor:new_m
                                          secondaryTensor:bias1
                                                     name:nil];
    MPSGraphTensor* v_hat = [g divisionWithPrimaryTensor:new_v
                                          secondaryTensor:bias2
                                                     name:nil];
    MPSGraphTensor* v_sqrt = [g squareRootWithTensor:v_hat name:nil];
    MPSGraphTensor* denom =
        [g additionWithPrimaryTensor:v_sqrt secondaryTensor:eps_c name:nil];
    MPSGraphTensor* step =
        [g divisionWithPrimaryTensor:m_hat secondaryTensor:denom name:nil];
    if (decoupled_wd && s.weight_decay != 0.0) {
        MPSGraphTensor* wd = scalar_const(g, s.weight_decay, dt);
        step = [g additionWithPrimaryTensor:step
                            secondaryTensor:[g multiplicationWithPrimaryTensor:wd
                                                                 secondaryTensor:param
                                                                            name:nil]
                                       name:nil];
    }
    MPSGraphTensor* new_p =
        [g subtractionWithPrimaryTensor:param
                        secondaryTensor:[g multiplicationWithPrimaryTensor:lr_c
                                                             secondaryTensor:step
                                                                        name:nil]
                                   name:nil];
    return {new_p, new_m, new_v};
}

}  // namespace

CompiledExecutable* compile_fused_training_step(
        const TraceGraph& graph,
        const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
        TensorId loss_id,
        const std::vector<TensorId>& param_ids,
        const OptimizerSpec& opt_spec,
        const std::vector<std::vector<TensorId>>& state_buf_ids_per_param,
        const std::vector<TensorId>& scalar_input_ids,
        std::string* error_msg) {
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg) *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_fused_training_step: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_fused_training_step: param_ids must be non-empty");
    if (state_buf_ids_per_param.size() != param_ids.size())
        return fail("compile_fused_training_step: state_buf_ids size mismatch");

    // Device + emitter precheck — mirrors compile_trace_with_backward.
    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (!op.outputs.empty()) {
            device = op.outputs[0].device;
            break;
        }
    }
    if (device != Device::GPU)
        return fail("compile_fused_training_step: only Device::GPU traces "
                    "are supported");
    for (const auto& op : graph.ops) {
        if (op.outputs.empty() || op.outputs[0].device != device)
            return fail("compile_fused_training_step: mixed-device trace");
        for (TensorId iid : op.inputs)
            if (iid < 0)
                return fail("compile_fused_training_step: op '" + op.name +
                            "' has an unresolved input slot");
        if (find_emitter(op.name) == nullptr && !op.inputs.empty())
            return fail("compile_fused_training_step: no emitter for op '" +
                        op.name + "'");
    }

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Ordered feed ids — same scheme as the other compile paths.
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_fused_training_step: feed " +
                            std::to_string(tid) + " is null");
            const Shape& feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(feed_shape);
            MPSDataType ns_dt;
            try {
                ns_dt = to_mps_dtype(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string(
                    "compile_fused_training_step: ") + e.what());
            }
            MPSGraphTensor* ph =
                [graph_obj placeholderWithShape:ns_shape
                                       dataType:ns_dt
                                           name:[NSString stringWithFormat:
                                                                @"feed_%lld",
                                                                (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Emit the forward + loss.
        for (const auto& node : graph.ops) {
            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty()) continue;  // dead-code header
                return fail("compile_fused_training_step: emitter vanished "
                            "for op '" + node.name + "'");
            }
            void* result = emitter->emit(ctx, node);
            if (result == nullptr)
                return fail("compile_fused_training_step: emitter '" +
                            node.name + "' returned nullptr");
            if (!node.outputs.empty() && node.outputs.size() == 1)
                ctx.bind(node.outputs[0].id, result);
        }

        // Resolve the loss tensor and param placeholders.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_fused_training_step: loss id has no "
                        "MPSGraph binding");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        std::vector<Shape> param_shapes;
        std::vector<Dtype> param_dtypes;
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_fused_training_step: param id " +
                            std::to_string(pid) + " has no placeholder");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
            const auto& p_impl = external_feeds.at(pid);
            param_shapes.push_back(p_impl->shape());
            param_dtypes.push_back(p_impl->dtype());
        }

        // Auto-derive gradients.
        NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
            [graph_obj gradientForPrimaryTensor:loss_t
                                    withTensors:param_arr
                                           name:@"lucid_fused_grads"];
        if (grad_map == nil)
            return fail("compile_fused_training_step: gradientForPrimaryTensor "
                        "returned nil");

        // Resolve scalar inputs (bias1, bias2 for Adam).
        std::vector<MPSGraphTensor*> scalar_tensors;
        scalar_tensors.reserve(scalar_input_ids.size());
        for (TensorId sid : scalar_input_ids) {
            void* s_void = ctx.resolve(sid);
            if (s_void == nullptr)
                return fail("compile_fused_training_step: scalar id " +
                            std::to_string(sid) + " has no placeholder");
            scalar_tensors.push_back((__bridge MPSGraphTensor*)s_void);
        }

        // Emit the optimizer update per param.  Collect (new_param,
        // new_state[]) tensors in flat output order.
        NSMutableArray<MPSGraphTensor*>* new_param_tensors =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        std::vector<std::vector<MPSGraphTensor*>> new_state_tensors;
        new_state_tensors.resize(param_ids.size());

        for (std::size_t i = 0; i < param_ids.size(); ++i) {
            MPSGraphTensor* p = param_arr[i];
            MPSGraphTensor* grad = grad_map[p];
            if (grad == nil)
                return fail("compile_fused_training_step: no gradient for "
                            "param " + std::to_string(param_ids[i]));
            MPSDataType dt = p.dataType;

            if (opt_spec.kind == OptimizerSpec::Kind::SGD) {
                MPSGraphTensor* mom = nil;
                if (opt_spec.momentum != 0.0) {
                    if (state_buf_ids_per_param[i].empty())
                        return fail("compile_fused_training_step: SGD with "
                                    "momentum requires a momentum state buffer");
                    void* m_void = ctx.resolve(state_buf_ids_per_param[i][0]);
                    if (m_void == nullptr)
                        return fail("compile_fused_training_step: momentum "
                                    "buffer placeholder missing");
                    mom = (__bridge MPSGraphTensor*)m_void;
                }
                SgdOutputs out =
                    emit_sgd_update(graph_obj, p, grad, mom, opt_spec, dt);
                [new_param_tensors addObject:out.new_param];
                if (out.new_mom != nil)
                    new_state_tensors[i].push_back(out.new_mom);
            } else {
                // Adam / AdamW.
                if (state_buf_ids_per_param[i].size() != 2)
                    return fail("compile_fused_training_step: Adam family "
                                "requires m,v state buffers (2 per param)");
                if (scalar_tensors.size() != 2)
                    return fail("compile_fused_training_step: Adam family "
                                "requires bias1,bias2 scalar feeds");
                void* m_void = ctx.resolve(state_buf_ids_per_param[i][0]);
                void* v_void = ctx.resolve(state_buf_ids_per_param[i][1]);
                if (m_void == nullptr || v_void == nullptr)
                    return fail("compile_fused_training_step: Adam state "
                                "buffer placeholders missing");
                MPSGraphTensor* m_buf = (__bridge MPSGraphTensor*)m_void;
                MPSGraphTensor* v_buf = (__bridge MPSGraphTensor*)v_void;
                bool decoupled =
                    (opt_spec.kind == OptimizerSpec::Kind::ADAMW);
                AdamOutputs out = emit_adam_update(
                    graph_obj, p, grad, m_buf, v_buf,
                    scalar_tensors[0], scalar_tensors[1],
                    opt_spec, dt, decoupled);
                [new_param_tensors addObject:out.new_param];
                new_state_tensors[i].push_back(out.new_m);
                new_state_tensors[i].push_back(out.new_v);
            }
        }

        // Assemble target list in declared output order:
        // [loss, new_p_0, ..., new_p_N-1, new_state[0][0], new_state[0][1], ...]
        NSMutableArray<MPSGraphTensor*>* target_arr =
            [NSMutableArray array];
        [target_arr addObject:loss_t];
        for (NSUInteger i = 0; i < [new_param_tensors count]; ++i)
            [target_arr addObject:new_param_tensors[i]];
        for (std::size_t i = 0; i < new_state_tensors.size(); ++i)
            for (MPSGraphTensor* t : new_state_tensors[i])
                [target_arr addObject:t];

        // Build feed dict.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = to_mps_dtype(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] =
                [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                  dataType:ns_dt];
        }

        id<MTLDevice> mtl_device =
            (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device = [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled = [graph_obj
            compileWithDevice:mps_device
                        feeds:feed_dict
                targetTensors:target_arr
             targetOperations:nil
        compilationDescriptor:make_compile_descriptor()];
        if (compiled == nil)
            return fail("compile_fused_training_step: MPSGraph compile "
                        "returned nil");

        // Mint trace ids for the new_param / new_state outputs (so the
        // CompiledExecutable's id list is unambiguous).  Start past
        // the max id we've seen — the Python wrapper may have
        // injected state-buffer / scalar feeds with ids past
        // ``graph.next_id`` (which is itself a read-only property on
        // the Python side, so the wrapper can't update it), so we
        // can't trust ``graph.next_id + 1`` alone.
        TensorId max_id = graph.next_id;
        for (const auto& [tid, _] : external_feeds)
            if (tid > max_id) max_id = tid;
        std::vector<TensorId> aux_output_ids;
        TensorId next_id = max_id + 1;
        for (std::size_t i = 0; i < param_ids.size(); ++i)
            aux_output_ids.push_back(next_id++);
        for (std::size_t i = 0; i < new_state_tensors.size(); ++i)
            for (std::size_t k = 0; k < new_state_tensors[i].size(); ++k)
                aux_output_ids.push_back(next_id++);

        // Find loss meta in the trace.
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops) {
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        }
        if (!loss_found)
            return fail("compile_fused_training_step: loss meta missing");

        // Build the per-output shape/dtype list parallel to target_arr.
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;
        output_shapes.push_back(loss_shape);
        output_dtypes.push_back(loss_dtype);
        // new_param_i: shape == param_shapes[i].
        for (std::size_t i = 0; i < param_ids.size(); ++i) {
            output_shapes.push_back(param_shapes[i]);
            output_dtypes.push_back(param_dtypes[i]);
        }
        // new_state[i][k]: shape == state buffer's trace impl shape.
        for (std::size_t i = 0; i < param_ids.size(); ++i) {
            for (std::size_t k = 0; k < state_buf_ids_per_param[i].size(); ++k) {
                TensorId sb = state_buf_ids_per_param[i][k];
                const auto& sb_impl = external_feeds.at(sb);
                output_shapes.push_back(sb_impl->shape());
                output_dtypes.push_back(sb_impl->dtype());
            }
        }

        auto* exe = new CompiledExecutable();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(aux_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe;
    }
}

// ── Generic fused step (Phase 1.8) ──────────────────────────────────

CompiledExecutable* compile_generic_fused_step(
        const TraceGraph& graph,
        const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
        TensorId loss_id,
        const std::vector<TensorId>& param_ids,
        const std::vector<TensorId>& ghost_grad_ids,
        const std::vector<TensorId>& output_target_ids,
        std::string* error_msg) {
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg) *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_generic_fused_step: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_generic_fused_step: param_ids must be non-empty");
    if (ghost_grad_ids.size() != param_ids.size())
        return fail("compile_generic_fused_step: ghost_grad_ids size != param_ids size");

    const std::unordered_set<TensorId> ghost_set(
        ghost_grad_ids.begin(), ghost_grad_ids.end());

    // Device + emitter precheck.
    Device device = Device::CPU;
    for (const auto& op : graph.ops) {
        if (!op.outputs.empty()) {
            device = op.outputs[0].device;
            break;
        }
    }
    if (device != Device::GPU)
        return fail("compile_generic_fused_step: only Device::GPU traces supported");
    for (const auto& op : graph.ops) {
        if (op.outputs.empty() || op.outputs[0].device != device)
            return fail("compile_generic_fused_step: mixed-device trace");
        for (TensorId iid : op.inputs)
            if (iid < 0)
                return fail("compile_generic_fused_step: op '" + op.name +
                            "' has unresolved input slot");
        if (find_emitter(op.name) == nullptr && !op.inputs.empty())
            return fail("compile_generic_fused_step: no emitter for op '" +
                        op.name + "'");
    }

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Build placeholders for every external_feed EXCEPT the ghost
        // grads (those are bound to derived gradients after the forward).
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            if (ghost_set.find(tid) == ghost_set.end())
                ordered_feed_ids.push_back(tid);
        std::sort(ordered_feed_ids.begin(), ordered_feed_ids.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;
        feed_tensors.reserve(ordered_feed_ids.size());
        input_shapes.reserve(ordered_feed_ids.size());
        input_dtypes.reserve(ordered_feed_ids.size());

        for (TensorId tid : ordered_feed_ids) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_generic_fused_step: null feed " +
                            std::to_string(tid));
            const Shape& feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(feed_shape);
            MPSDataType ns_dt;
            try {
                ns_dt = to_mps_dtype(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_generic_fused_step: ") + e.what());
            }
            MPSGraphTensor* ph =
                [graph_obj placeholderWithShape:ns_shape
                                       dataType:ns_dt
                                           name:[NSString stringWithFormat:
                                                                @"feed_%lld",
                                                                (long long)tid]];
            ctx.bind(tid, (__bridge void*)ph);
            feed_tensors.push_back(ph);
            input_shapes.push_back(feed_shape);
            input_dtypes.push_back(feed_dtype);
        }

        // Build the param-tensor array (we'll need it for the grad call).
        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_generic_fused_step: param id " +
                            std::to_string(pid) +
                            " has no placeholder (not in external_feeds?)");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
        }

        // Emit ops in trace order.  Before reaching the first op that
        // consumes a ghost grad, derive grads via
        // ``gradientForPrimaryTensor:withTensors:`` and bind each
        // ghost_grad_id to the derived gradient tensor for the
        // corresponding param.
        bool grads_derived = false;

        auto derive_grads_now = [&]() -> bool {
            if (grads_derived) return true;
            void* loss_void = ctx.resolve(loss_id);
            if (loss_void == nullptr) {
                if (error_msg)
                    *error_msg =
                        "compile_generic_fused_step: loss id has no "
                        "MPSGraph binding when reaching opt phase";
                return false;
            }
            MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
                [graph_obj gradientForPrimaryTensor:loss_t
                                        withTensors:param_arr
                                               name:@"lucid_generic_grads"];
            if (grad_map == nil) {
                if (error_msg)
                    *error_msg =
                        "compile_generic_fused_step: gradientForPrimaryTensor "
                        "returned nil";
                return false;
            }
            for (std::size_t i = 0; i < param_ids.size(); ++i) {
                MPSGraphTensor* g_t = grad_map[param_arr[i]];
                if (g_t == nil) {
                    if (error_msg)
                        *error_msg =
                            "compile_generic_fused_step: no gradient for "
                            "param " + std::to_string(param_ids[i]);
                    return false;
                }
                ctx.bind(ghost_grad_ids[i], (__bridge void*)g_t);
            }
            grads_derived = true;
            return true;
        };

        for (const auto& node : graph.ops) {
            // Detect entry into the opt-update phase: any input is a
            // ghost grad id.
            bool uses_ghost = false;
            for (TensorId iid : node.inputs) {
                if (ghost_set.find(iid) != ghost_set.end()) {
                    uses_ghost = true;
                    break;
                }
            }
            if (uses_ghost && !grads_derived) {
                if (!derive_grads_now())
                    return nullptr;
            }

            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty()) continue;  // dead-code factory
                return fail("compile_generic_fused_step: emitter vanished "
                            "for op '" + node.name + "'");
            }
            void* result = emitter->emit(ctx, node);
            if (result == nullptr)
                return fail("compile_generic_fused_step: emitter '" +
                            node.name + "' returned nullptr");
            // Auto-bind outputs[0] for single-output ops; multi-output
            // emitters (split/topk/...) handle their own binding.
            if (!node.outputs.empty() && node.outputs.size() == 1)
                ctx.bind(node.outputs[0].id, result);
        }

        // If the trace doesn't actually have any op that consumes a
        // ghost grad (degenerate case — opt is a no-op?), derive grads
        // anyway so the output_target_ids can resolve gradient-only
        // outputs.  Cheap when grad map isn't consumed.
        if (!grads_derived && !ghost_grad_ids.empty()) {
            if (!derive_grads_now()) return nullptr;
        }

        // Resolve targets: loss + output_target_ids.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_generic_fused_step: loss id has no "
                        "MPSGraph binding");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;

        NSMutableArray<MPSGraphTensor*>* target_arr = [NSMutableArray array];
        [target_arr addObject:loss_t];
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;

        // Loss meta.
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops) {
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        }
        if (!loss_found)
            return fail("compile_generic_fused_step: loss meta missing");
        output_shapes.push_back(loss_shape);
        output_dtypes.push_back(loss_dtype);

        // Per-output_target_id meta.  Search graph.ops for matching
        // TensorMeta.
        std::unordered_map<TensorId, std::pair<Shape, Dtype>> id_to_meta;
        for (const auto& node : graph.ops)
            for (const auto& meta : node.outputs)
                id_to_meta[meta.id] = {meta.shape, meta.dtype};

        for (TensorId tid : output_target_ids) {
            void* t_void = ctx.resolve(tid);
            if (t_void == nullptr)
                return fail("compile_generic_fused_step: output target id " +
                            std::to_string(tid) + " has no MPSGraph binding");
            [target_arr addObject:(__bridge MPSGraphTensor*)t_void];
            auto it = id_to_meta.find(tid);
            if (it == id_to_meta.end()) {
                // Maybe it's an external feed (shouldn't happen for opt
                // outputs but handle defensively).
                auto fit = external_feeds.find(tid);
                if (fit == external_feeds.end() || !fit->second)
                    return fail("compile_generic_fused_step: output target "
                                "id " + std::to_string(tid) +
                                " not in trace");
                output_shapes.push_back(fit->second->shape());
                output_dtypes.push_back(fit->second->dtype());
            } else {
                output_shapes.push_back(it->second.first);
                output_dtypes.push_back(it->second.second);
            }
        }

        // Feed dict.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = to_mps_dtype(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] =
                [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                  dataType:ns_dt];
        }

        id<MTLDevice> mtl_device =
            (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device =
            [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled = [graph_obj
            compileWithDevice:mps_device
                        feeds:feed_dict
                targetTensors:target_arr
             targetOperations:nil
        compilationDescriptor:make_compile_descriptor()];
        if (compiled == nil)
            return fail("compile_generic_fused_step: MPSGraph compile returned nil");

        // Mint aux ids past the max observed id (output_target_ids are
        // already in the trace, so the executable's grad_output_ids
        // can carry them directly).
        std::vector<TensorId> aux_output_ids(output_target_ids);

        auto* exe = new CompiledExecutable();
        exe->executable = compiled;
        exe->input_ids = ordered_feed_ids;
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(aux_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Stateful-variables variant of compile_generic_fused_step.
//
// See the header doc for the full design rationale.  Implementation
// mirrors ``compile_generic_fused_step`` line-by-line but with three
// surgical insertions:
//
//   1. For every feed in ``variable_pairs``, the feed becomes
//      ``variableWithData:`` (initialized from the Lucid Tensor's
//      current MTLBuffer contents) rather than ``placeholderWithShape:``.
//      The feed slot is removed from the executable's ``input_ids``.
//   2. After the trace emits the new value for each variable, the
//      builder issues ``assignVariable:`` and adds the operation to
//      the ``targetOperations`` array so it executes on every call.
//      A ``readVariable:`` op is also created and added to
//      ``targetTensors`` in place of the original write_id — so
//      ``run_executable_inplace`` can bind the Lucid Tensor's
//      existing MTLBuffer and receive the post-assign value with
//      no buffer alloc.
//   3. The original write_id entry in ``output_target_ids`` is
//      replaced by the readVariable's trace id in
//      ``grad_output_ids`` (same slot index — caller is unaffected).
// ─────────────────────────────────────────────────────────────────────
CompiledExecutable* compile_generic_fused_step_with_vars(
        const TraceGraph& graph,
        const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
        TensorId loss_id,
        const std::vector<TensorId>& param_ids,
        const std::vector<TensorId>& ghost_grad_ids,
        const std::vector<TensorId>& output_target_ids,
        const std::vector<std::pair<TensorId, TensorId>>& variable_pairs,
        std::string* error_msg) {
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg) *error_msg = std::move(msg);
        return nullptr;
    };

    if (graph.ops.empty())
        return fail("compile_generic_fused_step_with_vars: empty TraceGraph");
    if (param_ids.empty())
        return fail("compile_generic_fused_step_with_vars: param_ids must be non-empty");
    if (ghost_grad_ids.size() != param_ids.size())
        return fail(
            "compile_generic_fused_step_with_vars: ghost_grad_ids size != param_ids size");

    // If no variables requested, defer to the original path.
    if (variable_pairs.empty()) {
        return compile_generic_fused_step(
            graph, external_feeds, loss_id, param_ids, ghost_grad_ids,
            output_target_ids, error_msg);
    }

    const std::unordered_set<TensorId> ghost_set(
        ghost_grad_ids.begin(), ghost_grad_ids.end());

    // Set of feed ids that are promoted to variables, and a map from
    // each var-feed-id to its corresponding write_id (the new value
    // produced by the trace that gets assigned back into the variable).
    std::unordered_set<TensorId> var_feed_set;
    std::unordered_map<TensorId, TensorId> feed_to_write;
    std::unordered_set<TensorId> var_write_set;
    for (const auto& [f, w] : variable_pairs) {
        if (external_feeds.find(f) == external_feeds.end())
            return fail(
                "compile_generic_fused_step_with_vars: variable feed id " +
                std::to_string(f) + " not in external_feeds");
        var_feed_set.insert(f);
        feed_to_write[f] = w;
        var_write_set.insert(w);
    }

    Device device = Device::CPU;
    for (const auto& op : graph.ops)
        if (!op.outputs.empty()) {
            device = op.outputs[0].device;
            break;
        }
    if (device != Device::GPU)
        return fail("compile_generic_fused_step_with_vars: only GPU supported");
    for (const auto& op : graph.ops) {
        if (op.outputs.empty() || op.outputs[0].device != device)
            return fail("compile_generic_fused_step_with_vars: mixed-device trace");
        for (TensorId iid : op.inputs)
            if (iid < 0)
                return fail("compile_generic_fused_step_with_vars: op '" +
                            op.name + "' has unresolved input slot");
        if (find_emitter(op.name) == nullptr && !op.inputs.empty())
            return fail("compile_generic_fused_step_with_vars: no emitter for op '" +
                        op.name + "'");
    }

    @autoreleasepool {
        MPSGraph* graph_obj = [[MPSGraph alloc] init];
        BuilderContext ctx{(__bridge void*)graph_obj, device};

        // Variables init reads raw MTLBuffer contents at compile time
        // — flush pending MLX async work first so the snapshot reflects
        // the current parameter values rather than half-evaluated lazy
        // state.  Without this the variableWithData: source can be
        // mid-stream and the resulting variable holds garbage (or the
        // contents read blocks indefinitely).
        lucid::gpu::mps::wait_all();

        // Step 1: build placeholders for non-var, non-ghost feeds AND
        // variables for var_feed_set.  ``ordered_feed_ids`` records only
        // the placeholder feeds so ``input_ids`` excludes variables.
        std::vector<TensorId> ordered_feed_ids;
        ordered_feed_ids.reserve(external_feeds.size());
        std::vector<TensorId> sorted_all_feeds;
        sorted_all_feeds.reserve(external_feeds.size());
        for (const auto& [tid, _] : external_feeds)
            if (ghost_set.find(tid) == ghost_set.end())
                sorted_all_feeds.push_back(tid);
        std::sort(sorted_all_feeds.begin(), sorted_all_feeds.end());

        std::vector<MPSGraphTensor*> feed_tensors;
        std::vector<Shape> input_shapes;
        std::vector<Dtype> input_dtypes;

        for (TensorId tid : sorted_all_feeds) {
            const auto& impl = external_feeds.at(tid);
            if (!impl)
                return fail("compile_generic_fused_step_with_vars: null feed " +
                            std::to_string(tid));
            const Shape& feed_shape = impl->shape();
            const Dtype feed_dtype = impl->dtype();
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(feed_shape);
            MPSDataType ns_dt;
            try {
                ns_dt = to_mps_dtype(feed_dtype);
            } catch (const std::exception& e) {
                return fail(std::string("compile_generic_fused_step_with_vars: ") +
                            e.what());
            }

            if (var_feed_set.find(tid) != var_feed_set.end()) {
                // Variable: snapshot the current MTLBuffer contents
                // into NSData for ``variableWithData:``.  The data is
                // copied into MPSGraph's own internal storage; we can
                // safely free the NSData after.
                if (impl->device() != Device::GPU)
                    return fail(
                        "compile_generic_fused_step_with_vars: variable feed not on GPU");
                const auto& gs = std::get<GpuStorage>(impl->storage());
                if (!gs.arr)
                    return fail(
                        "compile_generic_fused_step_with_vars: variable feed has no MLX array");
                lucid::gpu::mps::BufferView v =
                    lucid::gpu::mps::array_to_buffer(*gs.arr);
                id<MTLBuffer> src_buf = (__bridge id<MTLBuffer>)v.mtl_buffer;
                // Inline shape_nbytes — same formula as
                // CompiledExecutable.mm's detail::shape_nbytes (anon
                // namespace there is unreachable from this TU).
                std::size_t nelem = 1;
                for (std::int64_t d : feed_shape)
                    nelem *= static_cast<std::size_t>(d);
                const std::size_t itemsize = (feed_dtype == Dtype::F16) ? 2 : 4;
                const std::size_t nbytes = nelem * itemsize;
                NSData* init_data =
                    [NSData dataWithBytes:[src_buf contents] length:nbytes];
                MPSGraphTensor* var =
                    [graph_obj variableWithData:init_data
                                          shape:ns_shape
                                       dataType:ns_dt
                                           name:[NSString stringWithFormat:
                                                              @"var_%lld",
                                                              (long long)tid]];
                ctx.bind(tid, (__bridge void*)var);
                // NOTE: var feeds NOT added to feed_tensors / input_shapes.
            } else {
                MPSGraphTensor* ph =
                    [graph_obj placeholderWithShape:ns_shape
                                           dataType:ns_dt
                                               name:[NSString stringWithFormat:
                                                                    @"feed_%lld",
                                                                    (long long)tid]];
                ctx.bind(tid, (__bridge void*)ph);
                feed_tensors.push_back(ph);
                input_shapes.push_back(feed_shape);
                input_dtypes.push_back(feed_dtype);
                ordered_feed_ids.push_back(tid);
            }
        }

        // Step 2: param array for gradient derivation.  Param feeds
        // are now variables — ctx.resolve returns the variable tensor.
        NSMutableArray<MPSGraphTensor*>* param_arr =
            [NSMutableArray arrayWithCapacity:param_ids.size()];
        for (TensorId pid : param_ids) {
            void* p_void = ctx.resolve(pid);
            if (p_void == nullptr)
                return fail("compile_generic_fused_step_with_vars: param id " +
                            std::to_string(pid) + " has no binding");
            [param_arr addObject:(__bridge MPSGraphTensor*)p_void];
        }

        // Step 3: emit trace ops with ghost-grad derivation, same as
        // the non-variables path.
        bool grads_derived = false;
        auto derive_grads_now = [&]() -> bool {
            if (grads_derived) return true;
            void* loss_void = ctx.resolve(loss_id);
            if (loss_void == nullptr) {
                if (error_msg)
                    *error_msg =
                        "compile_generic_fused_step_with_vars: loss id has no binding";
                return false;
            }
            MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;
            NSDictionary<MPSGraphTensor*, MPSGraphTensor*>* grad_map =
                [graph_obj gradientForPrimaryTensor:loss_t
                                        withTensors:param_arr
                                               name:@"lucid_var_grads"];
            if (grad_map == nil) {
                if (error_msg)
                    *error_msg =
                        "compile_generic_fused_step_with_vars: gradientForPrimaryTensor nil";
                return false;
            }
            for (std::size_t i = 0; i < param_ids.size(); ++i) {
                MPSGraphTensor* g_t = grad_map[param_arr[i]];
                if (g_t == nil) {
                    if (error_msg)
                        *error_msg =
                            "compile_generic_fused_step_with_vars: no gradient for param " +
                            std::to_string(param_ids[i]);
                    return false;
                }
                ctx.bind(ghost_grad_ids[i], (__bridge void*)g_t);
            }
            grads_derived = true;
            return true;
        };

        for (const auto& node : graph.ops) {
            bool uses_ghost = false;
            for (TensorId iid : node.inputs)
                if (ghost_set.find(iid) != ghost_set.end()) {
                    uses_ghost = true;
                    break;
                }
            if (uses_ghost && !grads_derived)
                if (!derive_grads_now()) return nullptr;

            OpEmitter* emitter = find_emitter(node.name);
            if (emitter == nullptr) {
                if (node.inputs.empty()) continue;
                return fail("compile_generic_fused_step_with_vars: emitter vanished for op '" +
                            node.name + "'");
            }
            void* result = emitter->emit(ctx, node);
            if (result == nullptr)
                return fail("compile_generic_fused_step_with_vars: emitter '" +
                            node.name + "' returned nullptr");
            if (!node.outputs.empty() && node.outputs.size() == 1)
                ctx.bind(node.outputs[0].id, result);
        }
        if (!grads_derived && !ghost_grad_ids.empty()) {
            if (!derive_grads_now()) return nullptr;
        }

        // Step 4: emit assignVariable for each (feed, write) pair +
        // readVariable into a target tensor we'll expose as an output
        // (so per-call the new value flushes into the Lucid Tensor's
        // existing MTLBuffer with no fresh allocation).
        NSMutableArray<MPSGraphOperation*>* target_ops_arr = [NSMutableArray array];

        // Loss target.
        void* loss_void = ctx.resolve(loss_id);
        if (loss_void == nullptr)
            return fail("compile_generic_fused_step_with_vars: loss has no binding");
        MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;
        NSMutableArray<MPSGraphTensor*>* target_arr = [NSMutableArray array];
        [target_arr addObject:loss_t];

        // Loss meta.
        std::vector<Shape> output_shapes;
        std::vector<Dtype> output_dtypes;
        Shape loss_shape;
        Dtype loss_dtype = Dtype::F32;
        bool loss_found = false;
        for (const auto& node : graph.ops)
            if (!node.outputs.empty() && node.outputs[0].id == loss_id) {
                loss_shape = node.outputs[0].shape;
                loss_dtype = node.outputs[0].dtype;
                loss_found = true;
                break;
            }
        if (!loss_found)
            return fail("compile_generic_fused_step_with_vars: loss meta missing");
        output_shapes.push_back(loss_shape);
        output_dtypes.push_back(loss_dtype);

        // Build id_to_meta map once for output_target_ids lookups.
        std::unordered_map<TensorId, std::pair<Shape, Dtype>> id_to_meta;
        for (const auto& node : graph.ops)
            for (const auto& meta : node.outputs)
                id_to_meta[meta.id] = {meta.shape, meta.dtype};

        // Process each output_target_id.  If it's a var write, emit
        // assignVariable + readVariable; otherwise add directly as a
        // target tensor.
        std::vector<TensorId> aux_output_ids;
        aux_output_ids.reserve(output_target_ids.size());
        for (TensorId tid : output_target_ids) {
            void* new_value_void = ctx.resolve(tid);
            if (new_value_void == nullptr)
                return fail(
                    "compile_generic_fused_step_with_vars: output target id " +
                    std::to_string(tid) + " has no binding");
            MPSGraphTensor* new_value_t = (__bridge MPSGraphTensor*)new_value_void;

            // Is this id a "write" for any variable?
            TensorId var_feed_for_write = -1;
            for (const auto& [f, w] : feed_to_write)
                if (w == tid) {
                    var_feed_for_write = f;
                    break;
                }

            if (var_feed_for_write >= 0) {
                // Variable write: emit assignVariable as a SIDE
                // EFFECT (added to targetOperations) and use the
                // computed ``new_value_t`` directly as the output
                // tensor.  This mirrors the working standalone
                // pattern in /tmp/var_grad.mm — using
                // ``readVariable:`` after ``assignVariable:`` inside
                // the same compile unit creates a read-after-write
                // dependency on the same variable, which interacts
                // poorly with MPSGraph's autograd scheduler (the
                // backward pass already reads the variable, so an
                // additional read introduces extra scheduling
                // constraints that we empirically observed cause a
                // hang on macOS 26 SDK).  ``new_value_t`` and
                // ``readVariable`` post-assign are bit-identical —
                // assignVariable copies new_value into the variable
                // verbatim — so we lose nothing semantically.
                void* var_void = ctx.resolve(var_feed_for_write);
                MPSGraphTensor* var_t = (__bridge MPSGraphTensor*)var_void;
                MPSGraphOperation* assign_op =
                    [graph_obj assignVariable:var_t
                            withValueOfTensor:new_value_t
                                         name:[NSString stringWithFormat:
                                                            @"assign_%lld",
                                                            (long long)tid]];
                [target_ops_arr addObject:assign_op];
                [target_arr addObject:new_value_t];
            } else {
                // Plain output target: add to targetTensors directly.
                [target_arr addObject:new_value_t];
            }

            auto it = id_to_meta.find(tid);
            if (it == id_to_meta.end()) {
                auto fit = external_feeds.find(tid);
                if (fit == external_feeds.end() || !fit->second)
                    return fail("compile_generic_fused_step_with_vars: output target id " +
                                std::to_string(tid) + " not in trace");
                output_shapes.push_back(fit->second->shape());
                output_dtypes.push_back(fit->second->dtype());
            } else {
                output_shapes.push_back(it->second.first);
                output_dtypes.push_back(it->second.second);
            }
            aux_output_ids.push_back(tid);
        }

        // Feed dict only includes non-variable feeds.
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_dict =
            [NSMutableDictionary dictionaryWithCapacity:feed_tensors.size()];
        for (std::size_t i = 0; i < feed_tensors.size(); ++i) {
            MPSDataType ns_dt = to_mps_dtype(input_dtypes[i]);
            NSArray<NSNumber*>* ns_shape = shape_to_nsarray(input_shapes[i]);
            feed_dict[feed_tensors[i]] =
                [[MPSGraphShapedType alloc] initWithShape:ns_shape
                                                  dataType:ns_dt];
        }

        id<MTLDevice> mtl_device =
            (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
        MPSGraphDevice* mps_device =
            [MPSGraphDevice deviceWithMTLDevice:mtl_device];
        MPSGraphExecutable* compiled = [graph_obj
            compileWithDevice:mps_device
                        feeds:feed_dict
                targetTensors:target_arr
             targetOperations:[target_ops_arr count] > 0 ? target_ops_arr : nil
        compilationDescriptor:make_compile_descriptor()];
        if (compiled == nil)
            return fail(
                "compile_generic_fused_step_with_vars: MPSGraph compile returned nil");

        auto* exe = new CompiledExecutable();
        exe->executable = compiled;
        // Retain the source MPSGraph so its variable storage outlives
        // the compile autoreleasepool.  Without this, MPSGraph releases
        // the variable's backing MTLBuffer along with the graph object;
        // subsequent ``runWithMTLCommandQueue:`` calls segfault inside
        // ``GPU::VarHandleOpHandler::encodeOp`` when the executor tries
        // to dereference the dangling variable handle.  The retain is
        // released in ``~CompiledExecutable``.
        exe->source_graph = (__bridge_retained void*)graph_obj;
        exe->input_ids = std::move(ordered_feed_ids);
        exe->output_ids = std::vector<TensorId>{loss_id};
        exe->grad_output_ids = std::move(aux_output_ids);
        exe->input_shapes = std::move(input_shapes);
        exe->input_dtypes = std::move(input_dtypes);
        exe->output_shapes = std::move(output_shapes);
        exe->output_dtypes = std::move(output_dtypes);
        exe->device = device;
        return exe;
    }
}

}  // namespace lucid::compile
