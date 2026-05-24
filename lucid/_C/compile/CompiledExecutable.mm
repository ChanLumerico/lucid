// lucid/_C/compile/CompiledExecutable.mm
//
// Objective-C++ side of the opaque :class:`CompiledExecutable` handle.
// Defines the actual class (with an ARC-managed ``MPSGraphExecutable*``
// member) and the free-function helpers declared in
// :file:`CompiledExecutable.h`.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <mlx/array.h>

#include "../backend/gpu/MlxBridge.h"
#include "../backend/gpu/mps/MpsBridge.h"
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
        desc.waitUntilCompleted = NO;
        MPSCommandBuffer* mps_cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        (void)[exe->executable encodeToCommandBuffer:mps_cb
                                         inputsArray:feeds
                                        resultsArray:results
                                 executionDescriptor:desc];
        [mps_cb commit];

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

        // Allocate a FRESH MTLBuffer per output target — never alias
        // the target's existing buffer.  See the function-header note
        // on the read-write hazard.
        NSMutableArray<MPSGraphTensorData*>* results =
            [NSMutableArray arrayWithCapacity:total_outs];
        for (std::size_t j = 0; j < total_outs; ++j) {
            const std::size_t nbytes =
                detail::shape_nbytes(exe->output_shapes[j], exe->output_dtypes[j]);
            id<MTLBuffer> out_buf =
                [device newBufferWithLength:nbytes
                                    options:MTLResourceStorageModeShared];
            if (!out_buf)
                throw std::runtime_error(
                    "run_executable_inplace: MTLBuffer allocation failed");
            fresh_out_bufs.push_back(out_buf);
            NSArray<NSNumber*>* ns_shape =
                detail::shape_to_nsarray(exe->output_shapes[j]);
            MPSDataType ns_dt = detail::to_mps_dtype(exe->output_dtypes[j]);
            MPSGraphTensorData* td =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                        shape:ns_shape
                                                     dataType:ns_dt];
            [results addObject:td];
        }

        // ``encodeToCommandBuffer:`` + async commit (matching the
        // ``run_executable`` inference path).  See its commentary for
        // the perf rationale.
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = NO;
        MPSCommandBuffer* mps_cb =
            [MPSCommandBuffer commandBufferFromCommandQueue:queue];
        (void)[exe->executable encodeToCommandBuffer:mps_cb
                                         inputsArray:feeds
                                        resultsArray:results
                                 executionDescriptor:desc];
        [mps_cb commit];

        // Swap: wrap each fresh MTLBuffer as a leaf MLX array and
        // overwrite the target's ``GpuStorage::arr`` pointer.  No
        // ``mlx::copy`` runs — the old buffer is released when the
        // last shared_ptr to the previous array drops.
        for (std::size_t j = 0; j < total_outs; ++j) {
            auto& target = const_cast<TensorImplPtr&>(output_targets[j]);
            auto& gs = std::get<GpuStorage>(target->mutable_storage());
            id<MTLBuffer> buf = fresh_out_bufs[j];
            void* raw = (__bridge_retained void*)buf;
            std::vector<int> mlx_shape(target->shape().begin(),
                                       target->shape().end());
            ::mlx::core::array fresh = lucid::gpu::mps::buffer_to_array(
                raw, std::move(mlx_shape), target->dtype());
            gs.arr = std::make_shared<::mlx::core::array>(std::move(fresh));
            target->bump_version();
        }
    }
}

}  // namespace lucid::compile
