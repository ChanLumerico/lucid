// lucid/_C/compile/CompiledExecutable.mm
//
// Objective-C++ side of the opaque :class:`CompiledExecutable` handle.
// Defines the actual class (with an ARC-managed ``MPSGraphExecutable*``
// member) and the free-function helpers declared in
// :file:`CompiledExecutable.h`.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <mlx/array.h>

#include "../backend/gpu/MlxBridge.h"
#include "../backend/gpu/mps/MpsBridge.h"
#include "../core/Determinism.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "CompiledExecutable.h"

namespace lucid::compile {

// Class definition (visible only inside .mm callers).  Holds the
// ARC-strong reference to the MPSGraphExecutable plus the I/O ordering
// plan minted by :class:`MpsBuilder`.
class CompiledExecutable {
public:
    MPSGraphExecutable* executable = nil;  // ARC strong

    // When the executable was compiled from a graph that contains
    // ``MPSGraphVariable``-backed tensors (``variableWithData:`` /
    // ``assignVariable:``), the variable's internal storage belongs to
    // the source MPSGraph object, not to the executable.  Releasing the
    // graph (e.g. at the end of the compile autoreleasepool) frees the
    // variable's buffer; subsequent ``runWithMTLCommandQueue:`` calls
    // then crash inside ``GPU::VarHandleOpHandler::encodeOp`` when the
    // executor tries to dereference the dangling variable handle.  We
    // retain the source graph here for variable-bearing compiles so the
    // variable storage stays live for the executable's lifetime.  For
    // pure-forward / placeholder-only compiles this field is nil.
    void* source_graph = nullptr;  // __bridge_retained MPSGraph*

    // Trace ids for inputs (feed order) and outputs (target order).
    std::vector<TensorId> input_ids;
    std::vector<TensorId> output_ids;

    // Shape + dtype per input/output, parallel to the id vectors.
    // Used to build MPSGraphTensorData at run time.
    std::vector<Shape> input_shapes;
    std::vector<Dtype> input_dtypes;
    std::vector<Shape> output_shapes;
    std::vector<Dtype> output_dtypes;

    Device device = Device::GPU;

    // Phase 1.3: optional backward outputs.  When non-empty, the
    // executable also produces ``len(grad_output_ids)`` gradient
    // tensors after the forward outputs, one per parameter passed to
    // :func:`compile_trace_with_backward`.  Empty for pure-forward
    // executables.  ``grad_output_ids[i]`` is the TraceId minted for
    // the gradient of ``param_ids[i]``; the corresponding shape and
    // dtype are appended to ``output_shapes`` / ``output_dtypes``
    // immediately after the loss tensor's entry.
    std::vector<TensorId> grad_output_ids;

    // Phase 1.6: when ``true`` the executable was compiled with a
    // symbolic leading dim on every non-parameter feed.  At run time
    // we replace the recorded shape's first axis with the actual
    // batch size pulled off the call-site input, both for the input
    // MPSGraphTensorData binding and for the output buffer allocation.
    bool dynamic_batch = false;
    // Set of feed-slot indices (into ``input_ids``) that were
    // declared parameters and therefore keep their static shape even
    // in dynamic-batch mode.
    std::unordered_set<std::size_t> static_feed_slots;

    ~CompiledExecutable() {
        // Release the variable-storage-owning MPSGraph if we retained
        // one during compile (``compile_generic_fused_step_with_vars``).
        // ARC handles ``executable``.
        if (source_graph != nullptr) {
            @autoreleasepool {
                MPSGraph* g = (__bridge_transfer MPSGraph*)source_graph;
                (void)g;
            }
            source_graph = nullptr;
        }
    }
};

std::size_t executable_num_inputs(const CompiledExecutable* exe) {
    return exe ? exe->input_ids.size() : 0;
}

std::size_t executable_num_outputs(const CompiledExecutable* exe) {
    return exe ? exe->output_ids.size() : 0;
}

std::vector<TensorId> executable_input_ids(const CompiledExecutable* exe) {
    return exe ? exe->input_ids : std::vector<TensorId>{};
}

std::vector<TensorId> executable_output_ids(const CompiledExecutable* exe) {
    return exe ? exe->output_ids : std::vector<TensorId>{};
}

std::vector<TensorId> executable_grad_output_ids(const CompiledExecutable* exe) {
    return exe ? exe->grad_output_ids : std::vector<TensorId>{};
}

void destroy_executable(CompiledExecutable* exe) {
    // ``delete`` invokes the C++ destructor which releases the
    // ARC-strong reference to ``executable``.  Safe with ``nullptr``.
    delete exe;
}

// Helpers shared with Linear emitter / MpsBuilder (defined here so the
// .mm can use them; not exported to other TUs).
namespace detail {

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

inline std::size_t shape_nbytes(const Shape& shape, Dtype dt) {
    std::size_t n = 1;
    for (std::int64_t d : shape)
        n *= static_cast<std::size_t>(d);
    std::size_t itemsize = (dt == Dtype::F16) ? 2 : 4;
    return n * itemsize;
}

}  // namespace detail

// Run a compiled executable.  Feed-order inputs and target-order
// outputs are paired against the executable's recorded ordering.
// Output ``TensorImpl`` objects are freshly allocated GPU buffers
// produced by the executable; their MLX-array wrappers reuse the
// output ``MTLBuffer`` (no copy).
//
// Defined here rather than in MpsBuilder.mm because all the
// MPSGraph-side machinery is in this translation unit already.
LUCID_API std::vector<TensorImplPtr> run_executable(
        CompiledExecutable* exe,
        const std::vector<TensorImplPtr>& input_feeds) {
    if (exe == nullptr)
        throw std::invalid_argument("run_executable: null executable");
    if (input_feeds.size() != exe->input_ids.size())
        throw std::invalid_argument(
            "run_executable: input count mismatch (expected feed-order list of "
            "length matching executable_num_inputs)");

    // Total output count = forward outputs + grad outputs.  The output_*
    // vectors (shapes/dtypes) are sized to the combined total; output_ids
    // only enumerates the forward portion, grad_output_ids the backward.
    const std::size_t total_outs =
        exe->output_ids.size() + exe->grad_output_ids.size();
    if (exe->output_shapes.size() != total_outs ||
        exe->output_dtypes.size() != total_outs) {
        throw std::runtime_error(
            "run_executable: output_shapes/dtypes vector mismatch "
            "(internal bug — builder did not populate consistently)");
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)lucid::gpu::mps::shared_mtl_queue();

    std::vector<TensorImplPtr> outputs;
    outputs.reserve(exe->output_ids.size());

    // Phase 1.6: in dynamic-batch mode the actual batch size for this
    // call is pulled off the first non-parameter feed.  We then use
    // that BS to rewrite the leading axis of (a) input MPSGraphTensorData
    // shapes and (b) output buffer allocations.
    std::int64_t dyn_batch_size = -1;
    if (exe->dynamic_batch) {
        for (std::size_t i = 0; i < input_feeds.size(); ++i) {
            if (exe->static_feed_slots.find(i) != exe->static_feed_slots.end())
                continue;
            const auto& impl = input_feeds[i];
            if (!impl || impl->shape().empty())
                continue;
            dyn_batch_size = impl->shape()[0];
            break;
        }
    }

    @autoreleasepool {
        NSMutableArray<MPSGraphTensorData*>* feeds =
            [NSMutableArray arrayWithCapacity:input_feeds.size()];
        for (std::size_t i = 0; i < input_feeds.size(); ++i) {
            const auto& impl = input_feeds[i];
            if (!impl)
                throw std::invalid_argument("run_executable: null input feed");
            if (impl->device() != Device::GPU)
                throw std::invalid_argument(
                    "run_executable: every input feed must be on Device::GPU");

            const auto& gs = std::get<GpuStorage>(impl->storage());
            if (!gs.arr)
                throw std::runtime_error(
                    "run_executable: input feed GpuStorage has no MLX array");

            lucid::gpu::mps::BufferView view = lucid::gpu::mps::array_to_buffer(*gs.arr);
            id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
            // Choose the shape: dynamic mode uses the actual input
            // tensor's shape (so the leading BS axis matches the
            // call-site data); static mode uses the trace-time shape.
            Shape feed_shape;
            const bool is_static_slot =
                exe->dynamic_batch &&
                exe->static_feed_slots.find(i) != exe->static_feed_slots.end();
            if (exe->dynamic_batch && !is_static_slot) {
                feed_shape = impl->shape();
            } else {
                feed_shape = exe->input_shapes[i];
            }
            NSArray<NSNumber*>* ns_shape = detail::shape_to_nsarray(feed_shape);
            MPSDataType ns_dt = detail::to_mps_dtype(exe->input_dtypes[i]);
            MPSGraphTensorData* td =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                        shape:ns_shape
                                                     dataType:ns_dt];
            [feeds addObject:td];
        }

        NSMutableArray<MPSGraphTensorData*>* results =
            [NSMutableArray arrayWithCapacity:total_outs];
        std::vector<id<MTLBuffer>> out_bufs;
        out_bufs.reserve(total_outs);

        // Per-output shape decision: in dynamic-batch mode we replace
        // the leading axis with the call-time BS for any output whose
        // first dim matches the trace-time BS (i.e. it inherits the
        // batch axis).  Scalar reductions over the batch axis stay
        // 0-D; intermediate outputs unrelated to BS stay static.
        std::vector<Shape> realized_output_shapes;
        realized_output_shapes.reserve(total_outs);
        for (std::size_t j = 0; j < total_outs; ++j) {
            Shape s = exe->output_shapes[j];
            if (exe->dynamic_batch && dyn_batch_size >= 0 && !s.empty()) {
                // Heuristic: any output whose first dim matches the
                // *traced* batch size is itself batch-shaped.  Look
                // up the first non-parameter input's trace BS as the
                // reference value.
                std::int64_t trace_bs = -1;
                for (std::size_t i = 0; i < exe->input_shapes.size(); ++i) {
                    if (exe->static_feed_slots.find(i) !=
                        exe->static_feed_slots.end())
                        continue;
                    if (!exe->input_shapes[i].empty()) {
                        trace_bs = exe->input_shapes[i][0];
                        break;
                    }
                }
                if (trace_bs >= 0 && s[0] == trace_bs) {
                    s[0] = dyn_batch_size;
                }
            }
            realized_output_shapes.push_back(std::move(s));
        }

        for (std::size_t j = 0; j < total_outs; ++j) {
            const std::size_t nbytes =
                detail::shape_nbytes(realized_output_shapes[j], exe->output_dtypes[j]);
            id<MTLBuffer> out_buf =
                [device newBufferWithLength:nbytes
                                    options:MTLResourceStorageModeShared];
            if (!out_buf)
                throw std::runtime_error(
                    "run_executable: MTLBuffer allocation for output failed");
            out_bufs.push_back(out_buf);

            NSArray<NSNumber*>* ns_shape =
                detail::shape_to_nsarray(realized_output_shapes[j]);
            MPSDataType ns_dt = detail::to_mps_dtype(exe->output_dtypes[j]);
            MPSGraphTensorData* td =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                        shape:ns_shape
                                                     dataType:ns_dt];
            [results addObject:td];
        }

        // ``encodeToCommandBuffer:`` + async ``commit`` (no wait) —
        // ~100-200μs cheaper per call on M-series than the
        // synchronous ``runWithMTLCommandQueue:`` path.  Output
        // dependency is enforced by MLX's tracker on the wrapped
        // output arrays (line ~290 below) — every user-side read
        // (``.item()`` / ``.numpy()`` / ``metal.synchronize()``)
        // blocks via MLX's eval path until the GPU finishes.  Matches
        // the async model MLX eager uses, so compile-mode now
        // pipelines the same way.
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        // EXPERIMENTAL ``LUCID_COMPILE_SYNC=1``: force the synchronous
        // ``runWithMTLCommandQueue:`` path.  Used to diagnose hangs in
        // the async commit path (notably for variable-bearing
        // executables where assignVariable/readVariable scheduling
        // behaves differently than placeholder outputs).
        static const bool force_sync = []() {
            const char* s = std::getenv("LUCID_COMPILE_SYNC");
            return s && std::string(s) == "1";
        }();
        if (force_sync) {
            desc.waitUntilCompleted = YES;
            (void)[exe->executable runWithMTLCommandQueue:queue
                                              inputsArray:feeds
                                             resultsArray:results
                                      executionDescriptor:desc];
        } else {
            desc.waitUntilCompleted = NO;
            MPSCommandBuffer* mps_cb =
                [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            (void)[exe->executable encodeToCommandBuffer:mps_cb
                                             inputsArray:feeds
                                            resultsArray:results
                                     executionDescriptor:desc];
            [mps_cb commit];
        }

        // Wrap each output MTLBuffer back into a GpuStorage-backed
        // TensorImpl.  ``__bridge_retained`` transfers the strong ref
        // into the MLX deleter.
        for (std::size_t j = 0; j < total_outs; ++j) {
            void* raw = (__bridge_retained void*)out_bufs[j];
            std::vector<int> mlx_shape(exe->output_shapes[j].begin(),
                                       exe->output_shapes[j].end());
            ::mlx::core::array arr = lucid::gpu::mps::buffer_to_array(
                raw, std::move(mlx_shape), exe->output_dtypes[j]);
            GpuStorage gs;
            gs.arr = std::make_shared<::mlx::core::array>(std::move(arr));
            outputs.push_back(std::make_shared<TensorImpl>(
                Storage{std::move(gs)}, exe->output_shapes[j], exe->output_dtypes[j],
                Device::GPU, false));
        }
    }

    return outputs;
}

LUCID_API void run_executable_inplace(
        CompiledExecutable* exe,
        const std::vector<TensorImplPtr>& input_feeds,
        const std::vector<TensorImplPtr>& output_targets) {
    // Swap-buffer variant: instead of allocating fresh MTLBuffers per
    // output and returning new TensorImpls, allocate one fresh
    // MTLBuffer per output target and then *replace* the target's
    // ``GpuStorage::arr`` pointer with a new MLX array wrapping the
    // fresh buffer.  No ``mlx::copy`` runs — the target's old MLX
    // array (and the buffer it owned) is released, and the new buffer
    // takes over.  Conceptually equivalent to ``target.copy_(new_t)``
    // but free of the per-output MLX evaluation sync.
    //
    // Why not bind the target's existing buffer as the output target?
    // MPSGraph's executable doesn't guarantee read-before-write
    // ordering when the same MTLBuffer appears in both the inputs and
    // results arrays.  Optimizer-step graphs frequently have the
    // momentum / m / v state on both sides of the dataflow (read old,
    // write new) — aliasing the buffer corrupts the read.  The
    // swap-buffer path sidesteps the hazard by always producing
    // outputs in a fresh allocation, then replacing the target's
    // pointer atomically once the executable has finished.
    //
    // Contract:
    //   * Each target must be on Device::GPU with a GpuStorage.
    //   * Shape + dtype must match ``exe->output_shapes[j]`` /
    //     ``output_dtypes[j]`` exactly; we do NOT broadcast.
    //   * The target's ``GpuStorage::arr`` is replaced by a fresh
    //     array; any external references to the previous ``arr``
    //     remain valid (MLX's own refcounting keeps the old buffer
    //     alive until they drop).
    //   * ``bump_version()`` is called on each target so autograd's
    //     mutation tracker sees the write.
    if (exe == nullptr)
        throw std::invalid_argument("run_executable_inplace: null executable");
    if (input_feeds.size() != exe->input_ids.size())
        throw std::invalid_argument(
            "run_executable_inplace: input count mismatch");
    const std::size_t total_outs =
        exe->output_ids.size() + exe->grad_output_ids.size();
    if (output_targets.size() != total_outs)
        throw std::invalid_argument(
            "run_executable_inplace: output_targets count mismatch (expected " +
            std::to_string(total_outs) + ", got " +
            std::to_string(output_targets.size()) + ")");
    if (exe->dynamic_batch)
        throw std::runtime_error(
            "run_executable_inplace: dynamic-batch executables are not "
            "supported (output shapes resolved at run time, no stable "
            "target buffers).");

    id<MTLDevice> device =
        (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)lucid::gpu::mps::shared_mtl_queue();

    // Pre-validate every target's shape/dtype — fails the whole call
    // early instead of mid-MPS-dispatch.
    for (std::size_t j = 0; j < total_outs; ++j) {
        const auto& target = output_targets[j];
        if (!target)
            throw std::invalid_argument(
                "run_executable_inplace: null output target at slot " +
                std::to_string(j));
        if (target->device() != Device::GPU)
            throw std::invalid_argument(
                "run_executable_inplace: output target not on GPU at slot " +
                std::to_string(j));
        if (target->shape() != exe->output_shapes[j])
            throw std::invalid_argument(
                "run_executable_inplace: output target shape mismatch at "
                "slot " + std::to_string(j));
        if (target->dtype() != exe->output_dtypes[j])
            throw std::invalid_argument(
                "run_executable_inplace: output target dtype mismatch at "
                "slot " + std::to_string(j));
    }

    std::vector<id<MTLBuffer>> fresh_out_bufs;
    fresh_out_bufs.reserve(total_outs);

    @autoreleasepool {
        NSMutableArray<MPSGraphTensorData*>* feeds =
            [NSMutableArray arrayWithCapacity:input_feeds.size()];
        for (std::size_t i = 0; i < input_feeds.size(); ++i) {
            const auto& impl = input_feeds[i];
            if (!impl)
                throw std::invalid_argument(
                    "run_executable_inplace: null input feed");
            if (impl->device() != Device::GPU)
                throw std::invalid_argument(
                    "run_executable_inplace: input feed not on GPU");
            const auto& gs = std::get<GpuStorage>(impl->storage());
            if (!gs.arr)
                throw std::runtime_error(
                    "run_executable_inplace: input GpuStorage has no MLX array");
            lucid::gpu::mps::BufferView view =
                lucid::gpu::mps::array_to_buffer(*gs.arr);
            id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
            NSArray<NSNumber*>* ns_shape =
                detail::shape_to_nsarray(exe->input_shapes[i]);
            MPSDataType ns_dt = detail::to_mps_dtype(exe->input_dtypes[i]);
            MPSGraphTensorData* td =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                        shape:ns_shape
                                                     dataType:ns_dt];
            [feeds addObject:td];
        }

        // ``LUCID_COMPILE_ALIAS_INPLACE=1``: diagnostic — bind the
        // target's existing MTLBuffer directly as the output result.
        // Re-validated 2026-05-25 on macOS 26 SDK: **read-write hazard
        // confirmed** — Adam-step trace produces zero loss / divergence
        // when input and output share a buffer.  Path is retained as
        // an opt-in diagnostic for future SDK versions but MUST stay
        // off by default.  The production fix for the per-step
        // ``newBufferWithLength`` cost is MPSGraph stateful variables
        // (``assignVariable``), not aliasing — variables move state
        // inside the executable and remove the param/state I/O
        // entirely.  See ``obsidian/engine/engine-compile-stateful-variables.md``.
        static const bool alias_mode = []() {
            const char* s = std::getenv("LUCID_COMPILE_ALIAS_INPLACE");
            return s && std::string(s) == "1";
        }();

        NSMutableArray<MPSGraphTensorData*>* results =
            [NSMutableArray arrayWithCapacity:total_outs];
        for (std::size_t j = 0; j < total_outs; ++j) {
            NSArray<NSNumber*>* ns_shape =
                detail::shape_to_nsarray(exe->output_shapes[j]);
            MPSDataType ns_dt = detail::to_mps_dtype(exe->output_dtypes[j]);

            id<MTLBuffer> out_buf;
            if (alias_mode) {
                // Reuse the target's existing MTLBuffer as the result
                // slot.  No allocation, no swap.
                const auto& target = output_targets[j];
                const auto& gs = std::get<GpuStorage>(target->storage());
                lucid::gpu::mps::BufferView v =
                    lucid::gpu::mps::array_to_buffer(*gs.arr);
                out_buf = (__bridge id<MTLBuffer>)v.mtl_buffer;
                fresh_out_bufs.push_back(nil);  // sentinel: no swap needed
            } else {
                const std::size_t nbytes = detail::shape_nbytes(
                    exe->output_shapes[j], exe->output_dtypes[j]);
                out_buf =
                    [device newBufferWithLength:nbytes
                                        options:MTLResourceStorageModeShared];
                if (!out_buf)
                    throw std::runtime_error(
                        "run_executable_inplace: MTLBuffer allocation failed");
                fresh_out_bufs.push_back(out_buf);
            }

            MPSGraphTensorData* td =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                        shape:ns_shape
                                                     dataType:ns_dt];
            [results addObject:td];
        }

        // Synchronous ``runWithMTLCommandQueue:`` — the freshly-
        // allocated output MTLBuffer (esp. the 0-D loss scalar) is
        // wrapped as an MLX leaf array right after this call, and
        // ``loss.item()`` in user code reads its contents directly.
        // The async ``encodeToCommandBuffer:`` path (used in
        // ``run_executable`` for inference) breaks training because
        // MLX has no way to track the pending MPSGraph write on a
        // brand-new buffer; reads come back as zeros.  Forcing
        // ``waitUntilCompleted = YES`` here keeps the swap-buffer
        // dance correct.  Cost: ~30-100μs/step for the host sync;
        // still cheap relative to the executable's GPU work for
        // anything bigger than a toy MLP.
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe->executable runWithMTLCommandQueue:queue
                                          inputsArray:feeds
                                         resultsArray:results
                                  executionDescriptor:desc];

        // Swap (or skip in alias mode): wrap each fresh MTLBuffer as
        // a leaf MLX array and overwrite the target's ``GpuStorage::arr``
        // pointer.  In alias mode the buffer is already the target's
        // own — bump_version is still needed so downstream readers
        // observe the update.
        for (std::size_t j = 0; j < total_outs; ++j) {
            auto& target = const_cast<TensorImplPtr&>(output_targets[j]);
            id<MTLBuffer> buf = fresh_out_bufs[j];
            if (buf != nil) {
                auto& gs = std::get<GpuStorage>(target->mutable_storage());
                void* raw = (__bridge_retained void*)buf;
                std::vector<int> mlx_shape(target->shape().begin(),
                                           target->shape().end());
                ::mlx::core::array fresh = lucid::gpu::mps::buffer_to_array(
                    raw, std::move(mlx_shape), target->dtype());
                gs.arr = std::make_shared<::mlx::core::array>(std::move(fresh));
            }
            target->bump_version();
        }
    }
}

// ── Disk cache implementation ───────────────────────────────────────
//
// Binary metadata format (.meta sidecar), all little-endian:
//
//   byte  0:        magic = 'L', 'M', 'P', 'S' (4 bytes)
//   byte  4:        format version = 1 (1 byte)
//   byte  5:        device (1 byte; 0=CPU, 1=GPU)
//   byte  6:        dynamic_batch (1 byte; 0=false, 1=true)
//   byte  7:        padding (1 byte, zero)
//   then, in this order:
//     vec<int64> input_ids        (u32 count + count × i64)
//     vec<int64> output_ids
//     vec<int64> grad_output_ids
//     vec<vec<int64>> input_shapes   (u32 count + count × (u32 + count × i64))
//     vec<vec<int64>> output_shapes
//     vec<u8> input_dtypes           (u32 count + count × u8)
//     vec<u8> output_dtypes
//     vec<u64> static_feed_slots     (u32 count + count × u64)
//
// Hand-rolled to avoid pulling in protobuf / json deps.  Bumping
// the version byte lets future Lucid releases reject incompatible
// caches without crashing.

namespace {
constexpr uint8_t kMetaFormatVersion = 1;
constexpr char kMetaMagic[4] = {'L', 'M', 'P', 'S'};

template <typename T>
inline void write_pod(std::vector<uint8_t>& out, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
inline void write_vec(std::vector<uint8_t>& out, const std::vector<T>& v) {
    write_pod<uint32_t>(out, static_cast<uint32_t>(v.size()));
    for (const T& x : v) write_pod<T>(out, x);
}

inline void write_vec_vec_i64(std::vector<uint8_t>& out,
                              const std::vector<Shape>& vv) {
    write_pod<uint32_t>(out, static_cast<uint32_t>(vv.size()));
    for (const auto& v : vv) {
        write_pod<uint32_t>(out, static_cast<uint32_t>(v.size()));
        for (std::int64_t x : v) write_pod<std::int64_t>(out, x);
    }
}

template <typename T>
inline bool read_pod(const uint8_t*& p, const uint8_t* end, T& out) {
    if (end - p < (ptrdiff_t)sizeof(T)) return false;
    std::memcpy(&out, p, sizeof(T));
    p += sizeof(T);
    return true;
}

template <typename T>
inline bool read_vec(const uint8_t*& p, const uint8_t* end,
                     std::vector<T>& v) {
    uint32_t n;
    if (!read_pod(p, end, n)) return false;
    v.resize(n);
    for (uint32_t i = 0; i < n; ++i)
        if (!read_pod(p, end, v[i])) return false;
    return true;
}

inline bool read_vec_vec_i64(const uint8_t*& p, const uint8_t* end,
                             std::vector<Shape>& vv) {
    uint32_t n;
    if (!read_pod(p, end, n)) return false;
    vv.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t m;
        if (!read_pod(p, end, m)) return false;
        vv[i].resize(m);
        for (uint32_t j = 0; j < m; ++j)
            if (!read_pod(p, end, vv[i][j])) return false;
    }
    return true;
}

inline bool atomic_write(const std::string& path,
                         const std::vector<uint8_t>& data) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
        if (!f) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

inline bool read_all(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    if (size < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));
    f.read(reinterpret_cast<char*>(out.data()), size);
    return static_cast<std::streamsize>(out.size()) == size;
}
}  // namespace

bool save_executable(const CompiledExecutable* exe, const std::string& path) {
    if (exe == nullptr || exe->executable == nil) return false;

    @autoreleasepool {
        const std::string pkg_path = path + ".mpsgraphpackage";
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:
                                                            pkg_path.c_str()]];
        // Apple's ``serializeToMPSGraphPackageAtURL:descriptor:``
        // returns ``void`` and offers no error signal (macOS 14+,
        // SDK header confirms).  Wrap in ``@try`` so a fatal ObjC
        // exception (e.g. read-only filesystem) doesn't propagate
        // into C++ unwinding.  If the file fails to materialise,
        // the meta write below or the subsequent ``load_executable``
        // will catch it as a cache miss.
        @try {
            [exe->executable serializeToMPSGraphPackageAtURL:url
                                                  descriptor:nil];
        } @catch (NSException* e) {
            (void)e;
            return false;
        }

        // Pack metadata.
        std::vector<uint8_t> meta;
        meta.reserve(1024);
        meta.insert(meta.end(), kMetaMagic, kMetaMagic + 4);
        write_pod<uint8_t>(meta, kMetaFormatVersion);
        write_pod<uint8_t>(meta, static_cast<uint8_t>(exe->device));
        write_pod<uint8_t>(meta, exe->dynamic_batch ? 1 : 0);
        write_pod<uint8_t>(meta, 0);  // padding
        write_vec<std::int64_t>(meta, exe->input_ids);
        write_vec<std::int64_t>(meta, exe->output_ids);
        write_vec<std::int64_t>(meta, exe->grad_output_ids);
        write_vec_vec_i64(meta, exe->input_shapes);
        write_vec_vec_i64(meta, exe->output_shapes);
        // Dtypes encoded as uint8 (matches the Dtype enum's
        // underlying width — all entries fit).
        std::vector<uint8_t> in_dt(exe->input_dtypes.size());
        for (std::size_t i = 0; i < in_dt.size(); ++i)
            in_dt[i] = static_cast<uint8_t>(exe->input_dtypes[i]);
        write_vec<uint8_t>(meta, in_dt);
        std::vector<uint8_t> out_dt(exe->output_dtypes.size());
        for (std::size_t i = 0; i < out_dt.size(); ++i)
            out_dt[i] = static_cast<uint8_t>(exe->output_dtypes[i]);
        write_vec<uint8_t>(meta, out_dt);
        std::vector<uint64_t> static_slots(exe->static_feed_slots.begin(),
                                            exe->static_feed_slots.end());
        write_vec<uint64_t>(meta, static_slots);

        const std::string meta_path = path + ".meta";
        return atomic_write(meta_path, meta);
    }
}

CompiledExecutable* load_executable(const std::string& path,
                                    std::string* error_msg) {
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg) *error_msg = std::move(msg);
        return nullptr;
    };

    const std::string meta_path = path + ".meta";
    std::vector<uint8_t> meta;
    if (!read_all(meta_path, meta))
        return fail("load_executable: missing or unreadable .meta sidecar");

    const uint8_t* p = meta.data();
    const uint8_t* end = p + meta.size();

    // Magic + version + flags
    if ((end - p) < 8) return fail("load_executable: truncated header");
    if (std::memcmp(p, kMetaMagic, 4) != 0)
        return fail("load_executable: bad magic — not a Lucid meta file");
    p += 4;
    uint8_t version, device_byte, dynamic_byte, pad;
    if (!read_pod(p, end, version)) return fail("truncated version");
    if (version != kMetaFormatVersion)
        return fail("load_executable: meta format version mismatch");
    if (!read_pod(p, end, device_byte)) return fail("truncated device");
    if (!read_pod(p, end, dynamic_byte)) return fail("truncated dynamic flag");
    if (!read_pod(p, end, pad)) return fail("truncated padding");

    std::vector<std::int64_t> input_ids, output_ids, grad_output_ids;
    std::vector<Shape> input_shapes, output_shapes;
    std::vector<uint8_t> input_dt_raw, output_dt_raw;
    std::vector<uint64_t> static_slots;

    if (!read_vec(p, end, input_ids)) return fail("truncated input_ids");
    if (!read_vec(p, end, output_ids)) return fail("truncated output_ids");
    if (!read_vec(p, end, grad_output_ids))
        return fail("truncated grad_output_ids");
    if (!read_vec_vec_i64(p, end, input_shapes))
        return fail("truncated input_shapes");
    if (!read_vec_vec_i64(p, end, output_shapes))
        return fail("truncated output_shapes");
    if (!read_vec(p, end, input_dt_raw))
        return fail("truncated input_dtypes");
    if (!read_vec(p, end, output_dt_raw))
        return fail("truncated output_dtypes");
    if (!read_vec(p, end, static_slots))
        return fail("truncated static_feed_slots");

    @autoreleasepool {
        const std::string pkg_path = path + ".mpsgraphpackage";
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:
                                                            pkg_path.c_str()]];
        MPSGraphCompilationDescriptor* cdesc = nil;
        if (::lucid::Determinism::is_enabled()) {
            cdesc = [[MPSGraphCompilationDescriptor alloc] init];
            cdesc.optimizationLevel = MPSGraphOptimizationLevel0;
        }
        MPSGraphExecutable* exec = [[MPSGraphExecutable alloc]
            initWithMPSGraphPackageAtURL:url
                   compilationDescriptor:cdesc];
        if (exec == nil)
            return fail("load_executable: MPSGraphExecutable load returned nil");

        auto* result = new CompiledExecutable();
        result->executable = exec;
        result->device = static_cast<Device>(device_byte);
        result->dynamic_batch = (dynamic_byte != 0);
        result->input_ids = std::move(input_ids);
        result->output_ids = std::move(output_ids);
        result->grad_output_ids = std::move(grad_output_ids);
        result->input_shapes = std::move(input_shapes);
        result->output_shapes = std::move(output_shapes);
        result->input_dtypes.resize(input_dt_raw.size());
        for (std::size_t i = 0; i < input_dt_raw.size(); ++i)
            result->input_dtypes[i] = static_cast<Dtype>(input_dt_raw[i]);
        result->output_dtypes.resize(output_dt_raw.size());
        for (std::size_t i = 0; i < output_dt_raw.size(); ++i)
            result->output_dtypes[i] = static_cast<Dtype>(output_dt_raw[i]);
        for (uint64_t s : static_slots)
            result->static_feed_slots.insert(static_cast<std::size_t>(s));
        return result;
    }
}

}  // namespace lucid::compile
