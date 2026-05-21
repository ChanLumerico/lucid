// lucid/_C/backend/gpu/mps/MpsKernels.mm
//
// Obj-C++ per-op MPSGraph kernels.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <mlx/array.h>
#include <mlx/dtype.h>

#include "../MlxBridge.h"
#include "MpsBridge.h"
#include "MpsDispatch.h"
#include "MpsKernels.h"

namespace lucid::gpu::mps {

namespace {

MPSDataType to_mps_dtype(Dtype dt) {
    switch (dt) {
        case Dtype::F32:
            return MPSDataTypeFloat32;
        case Dtype::F16:
            return MPSDataTypeFloat16;
        default:
            throw std::runtime_error(
                "lucid::gpu::mps: dtype not supported on MPSGraph path");
    }
}

NSArray<NSNumber*>* shape_to_nsarray(const Shape& shape) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:shape.size()];
    for (std::int64_t d : shape) {
        [out addObject:[NSNumber numberWithLongLong:d]];
    }
    return out;
}

std::size_t shape_nbytes(const Shape& shape, Dtype dt) {
    std::size_t n = 1;
    for (std::int64_t d : shape) n *= static_cast<std::size_t>(d);
    std::size_t itemsize = 4;
    switch (dt) {
        case Dtype::F32: itemsize = 4; break;
        case Dtype::F16: itemsize = 2; break;
        default: itemsize = 4; break;
    }
    return n * itemsize;
}

// Helper: scalar tensor inside the graph, dtype matching the others.
MPSGraphTensor* scalar_tensor(MPSGraph* graph, double value, MPSDataType dt) {
    return [graph constantWithScalar:value dataType:dt];
}

// Process-wide executable cache.  Keyed by op name + shape + dtype.
// MPSGraphExecutable compilation is the bottleneck on first call; for
// every subsequent call with the same (shape, dtype) we just feed
// fresh buffers into the cached executable.
NSMutableDictionary<NSString*, MPSGraphExecutable*>* executable_cache(void) {
    static dispatch_once_t once;
    static NSMutableDictionary* cache;
    dispatch_once(&once, ^{
        cache = [NSMutableDictionary new];
    });
    return cache;
}

NSString* cache_key(NSString* op, NSArray<NSNumber*>* shape, MPSDataType dt) {
    NSMutableString* k = [NSMutableString stringWithFormat:@"%@:%d:", op, (int)dt];
    for (NSNumber* n in shape) [k appendFormat:@"%@,", n];
    return [k copy];
}

}  // namespace

namespace {

// Build an MPSGraphExecutable that computes the tanh-approximation GELU
// for inputs of the given shape + dtype.  Cached across calls.
MPSGraphExecutable* gelu_executable(NSArray<NSNumber*>* nsShape,
                                    MPSDataType mps_dt,
                                    id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = cache_key(@"gelu_fwd", nsShape, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* x_t =
            [graph placeholderWithShape:nsShape dataType:mps_dt name:@"x"];

        // Tanh-approximation GELU — matches Lucid's MLX path so parity
        // tests against the existing implementation are bit-for-bit:
        //   y = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // MPSGraph fuses the 9 elementwise ops into a single Metal
        // kernel at compile time — that's the win vs MLX's per-op dispatch.
        MPSGraphTensor* half  = scalar_tensor(graph, 0.5, mps_dt);
        MPSGraphTensor* one   = scalar_tensor(graph, 1.0, mps_dt);
        MPSGraphTensor* a044  = scalar_tensor(graph, 0.044715, mps_dt);
        MPSGraphTensor* kk    = scalar_tensor(graph, std::sqrt(2.0 / M_PI), mps_dt);
        MPSGraphTensor* x2    = [graph multiplicationWithPrimaryTensor:x_t secondaryTensor:x_t name:nil];
        MPSGraphTensor* x3    = [graph multiplicationWithPrimaryTensor:x2 secondaryTensor:x_t name:nil];
        MPSGraphTensor* a_x3  = [graph multiplicationWithPrimaryTensor:a044 secondaryTensor:x3 name:nil];
        MPSGraphTensor* sum   = [graph additionWithPrimaryTensor:x_t secondaryTensor:a_x3 name:nil];
        MPSGraphTensor* inner = [graph multiplicationWithPrimaryTensor:kk secondaryTensor:sum name:nil];
        MPSGraphTensor* tanh_t = [graph tanhWithTensor:inner name:nil];
        MPSGraphTensor* one_plus = [graph additionWithPrimaryTensor:one secondaryTensor:tanh_t name:nil];
        MPSGraphTensor* half_x = [graph multiplicationWithPrimaryTensor:half secondaryTensor:x_t name:nil];
        MPSGraphTensor* y_t = [graph multiplicationWithPrimaryTensor:half_x secondaryTensor:one_plus name:@"gelu"];

        MPSGraphShapedType* x_typed =
            [[MPSGraphShapedType alloc] initWithShape:nsShape dataType:mps_dt];
        NSDictionary* feedShapes = @{x_t : x_typed};

        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:feedShapes
                       targetTensors:@[y_t]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

Storage gelu_forward(const Storage& x, const Shape& shape, Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    if (!gx.arr) {
        throw std::runtime_error("mps::gelu_forward: input storage has no MLX array");
    }

    const bool dbg = debug_enabled();
    auto t0 = std::chrono::high_resolution_clock::now();
    BufferView in_view = array_to_buffer(*gx.arr);
    id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)in_view.mtl_buffer;
    id<MTLDevice> device = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();
    auto t1 = std::chrono::high_resolution_clock::now();

    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes
                            options:MTLResourceStorageModeShared];
    if (!out_buf) {
        throw std::runtime_error("mps::gelu_forward: MTLBuffer allocation failed");
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    @autoreleasepool {
        NSArray<NSNumber*>* nsShape = shape_to_nsarray(shape);
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe = gelu_executable(nsShape, mps_dt, device);

        MPSGraphTensorData* x_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                    shape:nsShape
                                                 dataType:mps_dt];
        MPSGraphTensorData* y_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                    shape:nsShape
                                                 dataType:mps_dt];

        auto t3 = std::chrono::high_resolution_clock::now();
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_data]
                             resultsArray:@[y_data]
                      executionDescriptor:desc];
        auto t4 = std::chrono::high_resolution_clock::now();

        if (dbg) {
            auto us = [](auto a, auto b) {
                return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
            };
            fprintf(stderr,
                "[mps gelu] sync=%lld μs   alloc=%lld μs   wrap=%lld μs   run=%lld μs\n",
                us(t0, t1), us(t1, t2), us(t2, t3), us(t3, t4));
            fflush(stderr);
        }
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr =
        buffer_to_array(out_buf_raw,
                        std::vector<int>(shape.begin(), shape.end()),
                        dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

// ── GELU tanh-approx backward — fused MPSGraph kernel ─────────────────────

namespace {

MPSGraphExecutable* gelu_bwd_executable(NSArray<NSNumber*>* nsShape,
                                        MPSDataType mps_dt,
                                        id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = cache_key(@"gelu_bwd", nsShape, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* x_t =
            [graph placeholderWithShape:nsShape dataType:mps_dt name:@"x"];
        MPSGraphTensor* g_t =
            [graph placeholderWithShape:nsShape dataType:mps_dt name:@"grad"];

        // dy/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*c1*(1 + 3*c2*x^2)
        // with t = tanh(c1*(x + c2*x^3)).
        MPSGraphTensor* half  = scalar_tensor(graph, 0.5, mps_dt);
        MPSGraphTensor* one   = scalar_tensor(graph, 1.0, mps_dt);
        MPSGraphTensor* three = scalar_tensor(graph, 3.0, mps_dt);
        MPSGraphTensor* c1    = scalar_tensor(graph, 0.7978845608028654, mps_dt);
        MPSGraphTensor* c2    = scalar_tensor(graph, 0.044715, mps_dt);

        MPSGraphTensor* x2     = [graph multiplicationWithPrimaryTensor:x_t  secondaryTensor:x_t  name:nil];
        MPSGraphTensor* x3     = [graph multiplicationWithPrimaryTensor:x2   secondaryTensor:x_t  name:nil];
        MPSGraphTensor* c2x3   = [graph multiplicationWithPrimaryTensor:c2   secondaryTensor:x3   name:nil];
        MPSGraphTensor* sum    = [graph additionWithPrimaryTensor:x_t        secondaryTensor:c2x3 name:nil];
        MPSGraphTensor* inner  = [graph multiplicationWithPrimaryTensor:c1   secondaryTensor:sum  name:nil];
        MPSGraphTensor* t      = [graph tanhWithTensor:inner name:nil];
        MPSGraphTensor* t2     = [graph multiplicationWithPrimaryTensor:t    secondaryTensor:t    name:nil];
        MPSGraphTensor* one_pt = [graph additionWithPrimaryTensor:one        secondaryTensor:t    name:nil];
        MPSGraphTensor* term1  = [graph multiplicationWithPrimaryTensor:half secondaryTensor:one_pt name:nil];

        MPSGraphTensor* one_mt2 = [graph subtractionWithPrimaryTensor:one secondaryTensor:t2 name:nil];
        MPSGraphTensor* c2x2    = [graph multiplicationWithPrimaryTensor:c2   secondaryTensor:x2   name:nil];
        MPSGraphTensor* tc2x2   = [graph multiplicationWithPrimaryTensor:three secondaryTensor:c2x2 name:nil];
        MPSGraphTensor* one_p3  = [graph additionWithPrimaryTensor:one       secondaryTensor:tc2x2 name:nil];
        MPSGraphTensor* dinner  = [graph multiplicationWithPrimaryTensor:c1  secondaryTensor:one_p3 name:nil];
        MPSGraphTensor* x_dinner = [graph multiplicationWithPrimaryTensor:x_t secondaryTensor:dinner name:nil];
        MPSGraphTensor* x_dinner_omt2 = [graph multiplicationWithPrimaryTensor:x_dinner secondaryTensor:one_mt2 name:nil];
        MPSGraphTensor* term2 = [graph multiplicationWithPrimaryTensor:half secondaryTensor:x_dinner_omt2 name:nil];

        MPSGraphTensor* dydx  = [graph additionWithPrimaryTensor:term1 secondaryTensor:term2 name:nil];
        MPSGraphTensor* dx_t  = [graph multiplicationWithPrimaryTensor:dydx secondaryTensor:g_t name:@"gelu_bwd"];

        MPSGraphShapedType* shaped =
            [[MPSGraphShapedType alloc] initWithShape:nsShape dataType:mps_dt];
        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:@{x_t : shaped, g_t : shaped}
                       targetTensors:@[dx_t]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

Storage gelu_backward(const Storage& x,
                      const Storage& grad,
                      const Shape& shape,
                      Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(grad);
    if (!gx.arr || !gg.arr) {
        throw std::runtime_error("mps::gelu_backward: input has no MLX array");
    }
    BufferView x_view = array_to_buffer(*gx.arr);
    BufferView g_view = array_to_buffer(*gg.arr);
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_view.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_view.mtl_buffer;
    id<MTLDevice> device = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes options:MTLResourceStorageModeShared];
    if (!out_buf) {
        throw std::runtime_error("mps::gelu_backward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        NSArray<NSNumber*>* nsShape = shape_to_nsarray(shape);
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe = gelu_bwd_executable(nsShape, mps_dt, device);
        MPSGraphTensorData* x_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf shape:nsShape dataType:mps_dt];
        MPSGraphTensorData* g_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf shape:nsShape dataType:mps_dt];
        MPSGraphTensorData* dx_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf shape:nsShape dataType:mps_dt];
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_data, g_data]
                             resultsArray:@[dx_data]
                      executionDescriptor:desc];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr =
        buffer_to_array(out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

// ── SiLU backward — fused MPSGraph kernel ─────────────────────────────────

namespace {

MPSGraphExecutable* silu_bwd_executable(NSArray<NSNumber*>* nsShape,
                                        MPSDataType mps_dt,
                                        id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = cache_key(@"silu_bwd", nsShape, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* x_t = [graph placeholderWithShape:nsShape dataType:mps_dt name:@"x"];
        MPSGraphTensor* g_t = [graph placeholderWithShape:nsShape dataType:mps_dt name:@"grad"];

        // dy/dx = σ(x) * (1 + x*(1 - σ(x)))
        MPSGraphTensor* one  = scalar_tensor(graph, 1.0, mps_dt);
        MPSGraphTensor* sx   = [graph sigmoidWithTensor:x_t name:nil];
        MPSGraphTensor* oms  = [graph subtractionWithPrimaryTensor:one secondaryTensor:sx name:nil];
        MPSGraphTensor* xoms = [graph multiplicationWithPrimaryTensor:x_t secondaryTensor:oms name:nil];
        MPSGraphTensor* op   = [graph additionWithPrimaryTensor:one secondaryTensor:xoms name:nil];
        MPSGraphTensor* dydx = [graph multiplicationWithPrimaryTensor:sx secondaryTensor:op name:nil];
        MPSGraphTensor* dx_t = [graph multiplicationWithPrimaryTensor:dydx secondaryTensor:g_t name:@"dx"];

        MPSGraphShapedType* typed = [[MPSGraphShapedType alloc] initWithShape:nsShape dataType:mps_dt];
        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:@{x_t : typed, g_t : typed}
                       targetTensors:@[dx_t]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

Storage silu_backward(const Storage& x,
                      const Storage& grad,
                      const Shape& shape,
                      Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(grad);
    if (!gx.arr || !gg.arr) {
        throw std::runtime_error("mps::silu_backward: input has no MLX array");
    }
    BufferView x_v = array_to_buffer(*gx.arr);
    BufferView g_v = array_to_buffer(*gg.arr);
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_v.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_v.mtl_buffer;
    id<MTLDevice> device = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes options:MTLResourceStorageModeShared];
    if (!out_buf) {
        throw std::runtime_error("mps::silu_backward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        NSArray<NSNumber*>* nsShape = shape_to_nsarray(shape);
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe = silu_bwd_executable(nsShape, mps_dt, device);
        MPSGraphTensorData* x_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf shape:nsShape dataType:mps_dt];
        MPSGraphTensorData* g_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf shape:nsShape dataType:mps_dt];
        MPSGraphTensorData* o_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf shape:nsShape dataType:mps_dt];
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_data, g_data]
                             resultsArray:@[o_data]
                      executionDescriptor:desc];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr =
        buffer_to_array(out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

// ── GELU exact (Gaussian-CDF) — fused MPSGraph kernel ──────────────────────

namespace {

MPSGraphExecutable* gelu_exact_executable(NSArray<NSNumber*>* nsShape,
                                          MPSDataType mps_dt,
                                          id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = cache_key(@"gelu_exact_fwd", nsShape, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* x_t =
            [graph placeholderWithShape:nsShape dataType:mps_dt name:@"x"];

        // y = 0.5 * x * (1 + erf(x / sqrt(2)))
        MPSGraphTensor* half      = scalar_tensor(graph, 0.5, mps_dt);
        MPSGraphTensor* one       = scalar_tensor(graph, 1.0, mps_dt);
        MPSGraphTensor* inv_sqrt2 = scalar_tensor(graph, 0.7071067811865476, mps_dt);
        MPSGraphTensor* z         = [graph multiplicationWithPrimaryTensor:x_t
                                                            secondaryTensor:inv_sqrt2
                                                                       name:nil];
        MPSGraphTensor* erf_z     = [graph erfWithTensor:z name:nil];
        MPSGraphTensor* one_pe    = [graph additionWithPrimaryTensor:one
                                                      secondaryTensor:erf_z
                                                                 name:nil];
        MPSGraphTensor* cdf       = [graph multiplicationWithPrimaryTensor:half
                                                            secondaryTensor:one_pe
                                                                       name:nil];
        MPSGraphTensor* y_t       = [graph multiplicationWithPrimaryTensor:x_t
                                                            secondaryTensor:cdf
                                                                       name:@"gelu_exact"];

        MPSGraphShapedType* x_typed =
            [[MPSGraphShapedType alloc] initWithShape:nsShape dataType:mps_dt];
        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:@{x_t : x_typed}
                       targetTensors:@[y_t]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

MPSGraphExecutable* gelu_exact_bwd_executable(NSArray<NSNumber*>* nsShape,
                                              MPSDataType mps_dt,
                                              id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = cache_key(@"gelu_exact_bwd", nsShape, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* x_t =
            [graph placeholderWithShape:nsShape dataType:mps_dt name:@"x"];
        MPSGraphTensor* g_t =
            [graph placeholderWithShape:nsShape dataType:mps_dt name:@"grad"];

        // dy/dx = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2π)
        MPSGraphTensor* half        = scalar_tensor(graph, 0.5, mps_dt);
        MPSGraphTensor* one         = scalar_tensor(graph, 1.0, mps_dt);
        MPSGraphTensor* neg_half    = scalar_tensor(graph, -0.5, mps_dt);
        MPSGraphTensor* inv_sqrt2   = scalar_tensor(graph, 0.7071067811865476, mps_dt);
        MPSGraphTensor* inv_sqrt2pi = scalar_tensor(graph, 0.3989422804014327, mps_dt);
        MPSGraphTensor* z           = [graph multiplicationWithPrimaryTensor:x_t
                                                              secondaryTensor:inv_sqrt2
                                                                         name:nil];
        MPSGraphTensor* erf_z       = [graph erfWithTensor:z name:nil];
        MPSGraphTensor* one_pe      = [graph additionWithPrimaryTensor:one
                                                        secondaryTensor:erf_z
                                                                   name:nil];
        MPSGraphTensor* cdf         = [graph multiplicationWithPrimaryTensor:half
                                                              secondaryTensor:one_pe
                                                                         name:nil];
        MPSGraphTensor* x2          = [graph multiplicationWithPrimaryTensor:x_t
                                                              secondaryTensor:x_t
                                                                         name:nil];
        MPSGraphTensor* nx2_2       = [graph multiplicationWithPrimaryTensor:neg_half
                                                              secondaryTensor:x2
                                                                         name:nil];
        MPSGraphTensor* exp_nx2_2   = [graph exponentWithTensor:nx2_2 name:nil];
        MPSGraphTensor* pdf         = [graph multiplicationWithPrimaryTensor:inv_sqrt2pi
                                                              secondaryTensor:exp_nx2_2
                                                                         name:nil];
        MPSGraphTensor* x_pdf       = [graph multiplicationWithPrimaryTensor:x_t
                                                              secondaryTensor:pdf
                                                                         name:nil];
        MPSGraphTensor* deriv       = [graph additionWithPrimaryTensor:cdf
                                                        secondaryTensor:x_pdf
                                                                   name:nil];
        MPSGraphTensor* dx_t        = [graph multiplicationWithPrimaryTensor:deriv
                                                              secondaryTensor:g_t
                                                                         name:@"gelu_exact_bwd"];

        MPSGraphShapedType* shaped =
            [[MPSGraphShapedType alloc] initWithShape:nsShape dataType:mps_dt];
        // Order matters for runWithMTLCommandQueue:inputsArray:resultsArray:
        // We always feed [x, g] below; compile feeds must match this order.
        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:@{x_t : shaped, g_t : shaped}
                       targetTensors:@[dx_t]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

Storage gelu_exact_forward(const Storage& x, const Shape& shape, Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    if (!gx.arr) {
        throw std::runtime_error("mps::gelu_exact_forward: input has no MLX array");
    }
    BufferView in_view = array_to_buffer(*gx.arr);
    id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)in_view.mtl_buffer;
    id<MTLDevice> device = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes options:MTLResourceStorageModeShared];
    if (!out_buf) {
        throw std::runtime_error("mps::gelu_exact_forward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        NSArray<NSNumber*>* nsShape = shape_to_nsarray(shape);
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe = gelu_exact_executable(nsShape, mps_dt, device);
        MPSGraphTensorData* x_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf shape:nsShape dataType:mps_dt];
        MPSGraphTensorData* y_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf shape:nsShape dataType:mps_dt];
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_data]
                             resultsArray:@[y_data]
                      executionDescriptor:desc];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr =
        buffer_to_array(out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

Storage gelu_exact_backward(const Storage& x,
                            const Storage& grad,
                            const Shape& shape,
                            Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(grad);
    if (!gx.arr || !gg.arr) {
        throw std::runtime_error("mps::gelu_exact_backward: input has no MLX array");
    }
    BufferView x_view = array_to_buffer(*gx.arr);
    BufferView g_view = array_to_buffer(*gg.arr);
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_view.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_view.mtl_buffer;
    id<MTLDevice> device = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes options:MTLResourceStorageModeShared];
    if (!out_buf) {
        throw std::runtime_error("mps::gelu_exact_backward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        NSArray<NSNumber*>* nsShape = shape_to_nsarray(shape);
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe = gelu_exact_bwd_executable(nsShape, mps_dt, device);
        MPSGraphTensorData* x_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf shape:nsShape dataType:mps_dt];
        MPSGraphTensorData* g_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf shape:nsShape dataType:mps_dt];
        MPSGraphTensorData* dx_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf shape:nsShape dataType:mps_dt];
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_data, g_data]
                             resultsArray:@[dx_data]
                      executionDescriptor:desc];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr =
        buffer_to_array(out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

// ── LayerNorm backward — multi-output fused MPSGraph kernel ───────────────
//
// 5 feeds: x, gamma, mean, rstd, grad.  3 targets: dx, dgamma, dbeta.
// Feed order is fixed and must match the inputsArray order in the runner.

namespace {

NSString* layer_norm_bwd_cache_key(std::size_t outer,
                                   std::size_t normalized_size,
                                   MPSDataType dt) {
    return [NSString stringWithFormat:@"layer_norm_bwd:%d:%zu,%zu", (int)dt, outer, normalized_size];
}

MPSGraphExecutable* layer_norm_bwd_executable(std::size_t outer,
                                              std::size_t normalized_size,
                                              MPSDataType mps_dt,
                                              id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = layer_norm_bwd_cache_key(outer, normalized_size, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        NSArray<NSNumber*>* x_shape =
            @[@((long long)outer), @((long long)normalized_size)];
        NSArray<NSNumber*>* stat_shape = @[@((long long)outer), @1];
        NSArray<NSNumber*>* gamma_shape = @[@((long long)normalized_size)];

        MPSGraph* graph = [[MPSGraph alloc] init];
        // Placeholder creation order matches inputsArray order in runner.
        MPSGraphTensor* x      = [graph placeholderWithShape:x_shape   dataType:mps_dt name:@"x"];
        MPSGraphTensor* gamma  = [graph placeholderWithShape:gamma_shape dataType:mps_dt name:@"gamma"];
        MPSGraphTensor* mean   = [graph placeholderWithShape:stat_shape dataType:mps_dt name:@"mean"];
        MPSGraphTensor* rstd   = [graph placeholderWithShape:stat_shape dataType:mps_dt name:@"rstd"];
        MPSGraphTensor* g_in   = [graph placeholderWithShape:x_shape   dataType:mps_dt name:@"grad"];

        // gamma needs broadcast shape (1, normalized_size) for elementwise.
        MPSGraphTensor* gamma_2d =
            [graph reshapeTensor:gamma withShape:@[@1, @((long long)normalized_size)] name:nil];

        // centered = x - mean (broadcast over axis 1)
        MPSGraphTensor* centered =
            [graph subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
        // xnorm = centered * rstd
        MPSGraphTensor* xnorm =
            [graph multiplicationWithPrimaryTensor:centered secondaryTensor:rstd name:nil];
        // dbeta = sum(grad, axis=0)
        MPSGraphTensor* dbeta =
            [graph reductionSumWithTensor:g_in axes:@[@0] name:@"dbeta"];
        // dgamma = sum(grad * xnorm, axis=0)
        MPSGraphTensor* g_xnorm =
            [graph multiplicationWithPrimaryTensor:g_in secondaryTensor:xnorm name:nil];
        MPSGraphTensor* dgamma =
            [graph reductionSumWithTensor:g_xnorm axes:@[@0] name:@"dgamma"];
        // gx_scaled = grad * gamma
        MPSGraphTensor* gx_scaled =
            [graph multiplicationWithPrimaryTensor:g_in secondaryTensor:gamma_2d name:nil];
        // mean1 = mean(gx_scaled, axis=1)
        MPSGraphTensor* mean1 =
            [graph meanOfTensor:gx_scaled axes:@[@1] name:nil];
        // mean2 = mean(gx_scaled * xnorm, axis=1)
        MPSGraphTensor* gx_xnorm =
            [graph multiplicationWithPrimaryTensor:gx_scaled secondaryTensor:xnorm name:nil];
        MPSGraphTensor* mean2 =
            [graph meanOfTensor:gx_xnorm axes:@[@1] name:nil];
        // dx = rstd * (gx_scaled - mean1 - xnorm * mean2)
        MPSGraphTensor* xnorm_mean2 =
            [graph multiplicationWithPrimaryTensor:xnorm secondaryTensor:mean2 name:nil];
        MPSGraphTensor* inner1 =
            [graph subtractionWithPrimaryTensor:gx_scaled secondaryTensor:mean1 name:nil];
        MPSGraphTensor* inner2 =
            [graph subtractionWithPrimaryTensor:inner1 secondaryTensor:xnorm_mean2 name:nil];
        MPSGraphTensor* dx =
            [graph multiplicationWithPrimaryTensor:rstd secondaryTensor:inner2 name:@"dx"];

        MPSGraphShapedType* x_typed     = [[MPSGraphShapedType alloc] initWithShape:x_shape   dataType:mps_dt];
        MPSGraphShapedType* stat_typed  = [[MPSGraphShapedType alloc] initWithShape:stat_shape dataType:mps_dt];
        MPSGraphShapedType* gamma_typed = [[MPSGraphShapedType alloc] initWithShape:gamma_shape dataType:mps_dt];

        // Feeds dict (Obj-C preserves insertion order on Apple Silicon / modern macOS).
        NSDictionary* feeds = @{
            x:     x_typed,
            gamma: gamma_typed,
            mean:  stat_typed,
            rstd:  stat_typed,
            g_in:  x_typed,
        };
        // Target order: dx, dgamma, dbeta — must match resultsArray below.
        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:feeds
                       targetTensors:@[dx, dgamma, dbeta]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

LayerNormBackwardOut layer_norm_backward(const Storage& x,
                                         const Storage& gamma,
                                         const Storage& saved_mean,
                                         const Storage& saved_rstd,
                                         const Storage& grad,
                                         std::size_t outer,
                                         std::size_t normalized_size,
                                         const Shape& x_shape,
                                         const Shape& gamma_shape,
                                         const Shape& beta_shape,
                                         Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);
    if (!gx.arr || !gg.arr || !gm.arr || !gr.arr || !ggrad.arr) {
        throw std::runtime_error("mps::layer_norm_backward: input has no MLX array");
    }
    BufferView x_v   = array_to_buffer(*gx.arr);
    BufferView g_v   = array_to_buffer(*gg.arr);
    BufferView m_v   = array_to_buffer(*gm.arr);
    BufferView r_v   = array_to_buffer(*gr.arr);
    BufferView gr_v  = array_to_buffer(*ggrad.arr);

    id<MTLDevice> device       = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>)shared_mtl_queue();
    id<MTLBuffer> x_buf  = (__bridge id<MTLBuffer>)x_v.mtl_buffer;
    id<MTLBuffer> g_buf  = (__bridge id<MTLBuffer>)g_v.mtl_buffer;
    id<MTLBuffer> m_buf  = (__bridge id<MTLBuffer>)m_v.mtl_buffer;
    id<MTLBuffer> r_buf  = (__bridge id<MTLBuffer>)r_v.mtl_buffer;
    id<MTLBuffer> gr_buf = (__bridge id<MTLBuffer>)gr_v.mtl_buffer;

    const std::size_t dx_nbytes     = shape_nbytes(x_shape, dt);
    const std::size_t dgamma_nbytes = shape_nbytes(gamma_shape, dt);
    const std::size_t dbeta_nbytes  = shape_nbytes(beta_shape, dt);
    id<MTLBuffer> dx_buf = [device newBufferWithLength:dx_nbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dg_buf = [device newBufferWithLength:dgamma_nbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> db_buf = [device newBufferWithLength:dbeta_nbytes options:MTLResourceStorageModeShared];
    if (!dx_buf || !dg_buf || !db_buf) {
        throw std::runtime_error("mps::layer_norm_backward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe = layer_norm_bwd_executable(outer, normalized_size, mps_dt, device);

        NSArray<NSNumber*>* x_shape_ns =
            @[@((long long)outer), @((long long)normalized_size)];
        NSArray<NSNumber*>* stat_shape_ns = @[@((long long)outer), @1];
        NSArray<NSNumber*>* gamma_shape_ns = @[@((long long)normalized_size)];

        MPSGraphTensorData* x_data  = [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf  shape:x_shape_ns    dataType:mps_dt];
        MPSGraphTensorData* g_data  = [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf  shape:gamma_shape_ns dataType:mps_dt];
        MPSGraphTensorData* m_data  = [[MPSGraphTensorData alloc] initWithMTLBuffer:m_buf  shape:stat_shape_ns dataType:mps_dt];
        MPSGraphTensorData* r_data  = [[MPSGraphTensorData alloc] initWithMTLBuffer:r_buf  shape:stat_shape_ns dataType:mps_dt];
        MPSGraphTensorData* gr_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:gr_buf shape:x_shape_ns    dataType:mps_dt];

        // Output shapes for MPSGraphTensorData.  dx is x-shape; dgamma+dbeta are (normalized_size,).
        MPSGraphTensorData* dx_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:dx_buf shape:x_shape_ns     dataType:mps_dt];
        MPSGraphTensorData* dg_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:dg_buf shape:gamma_shape_ns dataType:mps_dt];
        MPSGraphTensorData* db_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:db_buf shape:gamma_shape_ns dataType:mps_dt];

        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_data, g_data, m_data, r_data, gr_data]
                             resultsArray:@[dx_data, dg_data, db_data]
                      executionDescriptor:desc];
    }

    auto wrap = [&](id<MTLBuffer> buf, const Shape& sh) -> Storage {
        void* raw = (__bridge_retained void*)buf;
        auto arr = buffer_to_array(raw, std::vector<int>(sh.begin(), sh.end()), dt);
        return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
    };
    return LayerNormBackwardOut{
        wrap(dx_buf, x_shape),
        wrap(dg_buf, gamma_shape),
        wrap(db_buf, beta_shape),
    };
}

// ── BatchNorm train fwd + bwd — fused MPSGraph kernels ────────────────────
//
// MLX has no fused BN; we lean on MPSGraph's `normalizationWithTensor:...`
// and the matching gradient ops.  Dispatched only for very large
// activations (Phase 0 measurement: 5.5× torch on 32×64×112×112).

namespace {

NSString* bn_cache_key(NSString* tag,
                       const Shape& x_shape,
                       MPSDataType dt,
                       double eps) {
    NSMutableString* k = [NSMutableString stringWithFormat:@"%@:%d:%.6g:", tag, (int)dt, eps];
    for (auto d : x_shape) [k appendFormat:@"%lld,", (long long)d];
    return [k copy];
}

NSArray<NSNumber*>* shape_vec_to_ns(const Shape& s) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:s.size()];
    for (auto d : s) [out addObject:@((long long)d)];
    return out;
}

Shape bn_stat_shape(int channels, int ndim) {
    // (1, C, 1, 1, ...) — ndim spatial 1's after C.
    Shape s;
    s.reserve(2 + ndim);
    s.push_back(1);
    s.push_back(channels);
    for (int i = 0; i < ndim; ++i) s.push_back(1);
    return s;
}

NSArray<NSNumber*>* bn_reduce_axes(int ndim) {
    // BatchNorm reduces over batch + spatial = axis 0 plus axes 2..2+ndim-1.
    NSMutableArray<NSNumber*>* a = [NSMutableArray arrayWithObject:@0];
    for (int i = 0; i < ndim; ++i) [a addObject:@(2 + i)];
    return a;
}

MPSGraphExecutable* bn_train_fwd_executable(const Shape& x_shape,
                                            int channels,
                                            int ndim,
                                            double eps,
                                            MPSDataType mps_dt,
                                            id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = bn_cache_key(@"bn_train_fwd", x_shape, mps_dt, eps);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        NSArray<NSNumber*>* x_shape_ns = shape_vec_to_ns(x_shape);
        Shape stat = bn_stat_shape(channels, ndim);
        NSArray<NSNumber*>* stat_ns = shape_vec_to_ns(stat);
        NSArray<NSNumber*>* gamma_shape_ns = @[@(channels)];
        NSArray<NSNumber*>* axes = bn_reduce_axes(ndim);

        MPSGraph* graph = [[MPSGraph alloc] init];
        // Placeholder order = inputs order at run time: x, gamma, beta.
        MPSGraphTensor* x      = [graph placeholderWithShape:x_shape_ns dataType:mps_dt name:@"x"];
        MPSGraphTensor* gamma  = [graph placeholderWithShape:gamma_shape_ns dataType:mps_dt name:@"gamma"];
        MPSGraphTensor* beta   = [graph placeholderWithShape:gamma_shape_ns dataType:mps_dt name:@"beta"];

        MPSGraphTensor* gamma_view =
            [graph reshapeTensor:gamma withShape:stat_ns name:nil];
        MPSGraphTensor* beta_view =
            [graph reshapeTensor:beta withShape:stat_ns name:nil];

        MPSGraphTensor* mean = [graph meanOfTensor:x axes:axes name:@"mean"];
        MPSGraphTensor* var  = [graph varianceOfTensor:x meanTensor:mean axes:axes name:@"var"];

        // MPSGraph reductions drop the reduced axes; restore the broadcast
        // shape so the saved tensors are usable in backward + downstream.
        MPSGraphTensor* mean_view = [graph reshapeTensor:mean withShape:stat_ns name:nil];
        MPSGraphTensor* var_view  = [graph reshapeTensor:var  withShape:stat_ns name:nil];

        MPSGraphTensor* y = [graph normalizationWithTensor:x
                                                meanTensor:mean_view
                                            varianceTensor:var_view
                                               gammaTensor:gamma_view
                                                betaTensor:beta_view
                                                   epsilon:(float)eps
                                                      name:@"y"];

        // rstd = 1 / sqrt(var + eps)  for the saved-tensor contract.
        MPSGraphTensor* eps_t  = scalar_tensor(graph, eps, mps_dt);
        MPSGraphTensor* v_pe   = [graph additionWithPrimaryTensor:var_view secondaryTensor:eps_t name:nil];
        MPSGraphTensor* sqrt_v = [graph squareRootWithTensor:v_pe name:nil];
        MPSGraphTensor* one_t  = scalar_tensor(graph, 1.0, mps_dt);
        MPSGraphTensor* rstd   = [graph divisionWithPrimaryTensor:one_t secondaryTensor:sqrt_v name:@"rstd"];

        MPSGraphShapedType* x_typed     = [[MPSGraphShapedType alloc] initWithShape:x_shape_ns dataType:mps_dt];
        MPSGraphShapedType* gamma_typed = [[MPSGraphShapedType alloc] initWithShape:gamma_shape_ns dataType:mps_dt];

        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:@{x : x_typed, gamma : gamma_typed, beta : gamma_typed}
                       targetTensors:@[y, mean_view, rstd]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

MPSGraphExecutable* bn_train_bwd_executable(const Shape& x_shape,
                                            int channels,
                                            int ndim,
                                            double eps,
                                            MPSDataType mps_dt,
                                            id<MTLDevice> device) {
    // Uses MPSGraph's canonical `normalizationGradient*` ops, which take
    // mean+variance+eps (not rstd).  Variance is reconstructed from saved
    // rstd: `var = 1/rstd^2 - eps`.  eps is part of the executable cache
    // key since the graph compiles eps as a constant.
    NSMutableDictionary* cache = executable_cache();
    NSString* key = bn_cache_key(@"bn_train_bwd_mps", x_shape, mps_dt, eps);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        NSArray<NSNumber*>* x_shape_ns = shape_vec_to_ns(x_shape);
        Shape stat = bn_stat_shape(channels, ndim);
        NSArray<NSNumber*>* stat_ns = shape_vec_to_ns(stat);
        NSArray<NSNumber*>* gamma_shape_ns = @[@(channels)];
        NSArray<NSNumber*>* axes = bn_reduce_axes(ndim);

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* x     = [graph placeholderWithShape:x_shape_ns   dataType:mps_dt name:@"x"];
        MPSGraphTensor* gamma = [graph placeholderWithShape:gamma_shape_ns dataType:mps_dt name:@"gamma"];
        MPSGraphTensor* mean  = [graph placeholderWithShape:stat_ns    dataType:mps_dt name:@"mean"];
        MPSGraphTensor* rstd  = [graph placeholderWithShape:stat_ns    dataType:mps_dt name:@"rstd"];
        MPSGraphTensor* grad  = [graph placeholderWithShape:x_shape_ns   dataType:mps_dt name:@"grad"];

        // var = 1/rstd^2 - eps
        MPSGraphTensor* one_t   = scalar_tensor(graph, 1.0, mps_dt);
        MPSGraphTensor* eps_t   = scalar_tensor(graph, eps, mps_dt);
        MPSGraphTensor* inv_rstd = [graph divisionWithPrimaryTensor:one_t secondaryTensor:rstd name:nil];
        MPSGraphTensor* inv_sq   = [graph multiplicationWithPrimaryTensor:inv_rstd secondaryTensor:inv_rstd name:nil];
        MPSGraphTensor* var      = [graph subtractionWithPrimaryTensor:inv_sq secondaryTensor:eps_t name:@"var"];

        MPSGraphTensor* gamma_view = [graph reshapeTensor:gamma withShape:stat_ns name:nil];

        MPSGraphTensor* dgamma_t = [graph normalizationGammaGradientWithIncomingGradientTensor:grad
                                                                                  sourceTensor:x
                                                                                    meanTensor:mean
                                                                                varianceTensor:var
                                                                                 reductionAxes:axes
                                                                                       epsilon:(float)eps
                                                                                          name:nil];
        MPSGraphTensor* dbeta_t = [graph normalizationBetaGradientWithIncomingGradientTensor:grad
                                                                                sourceTensor:x
                                                                               reductionAxes:axes
                                                                                        name:nil];
        MPSGraphTensor* dx_t = [graph normalizationGradientWithIncomingGradientTensor:grad
                                                                         sourceTensor:x
                                                                           meanTensor:mean
                                                                       varianceTensor:var
                                                                          gammaTensor:gamma_view
                                                                  gammaGradientTensor:dgamma_t
                                                                   betaGradientTensor:dbeta_t
                                                                        reductionAxes:axes
                                                                              epsilon:(float)eps
                                                                                 name:@"dx"];

        MPSGraphTensor* dgamma_1d = [graph reshapeTensor:dgamma_t withShape:gamma_shape_ns name:@"dgamma"];
        MPSGraphTensor* dbeta_1d  = [graph reshapeTensor:dbeta_t  withShape:gamma_shape_ns name:@"dbeta"];

        MPSGraphShapedType* x_typed     = [[MPSGraphShapedType alloc] initWithShape:x_shape_ns dataType:mps_dt];
        MPSGraphShapedType* gamma_typed = [[MPSGraphShapedType alloc] initWithShape:gamma_shape_ns dataType:mps_dt];
        MPSGraphShapedType* stat_typed  = [[MPSGraphShapedType alloc] initWithShape:stat_ns dataType:mps_dt];

        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:@{x : x_typed, gamma : gamma_typed,
                                       mean : stat_typed, rstd : stat_typed,
                                       grad : x_typed}
                       targetTensors:@[dx_t, dgamma_1d, dbeta_1d]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

BatchNormForwardOut batch_norm_train_forward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& beta,
                                             int channels,
                                             int ndim,
                                             double eps,
                                             const Shape& x_shape,
                                             Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gb = std::get<GpuStorage>(beta);

    BufferView x_v = array_to_buffer(*gx.arr);
    BufferView g_v = array_to_buffer(*gg.arr);
    BufferView b_v = array_to_buffer(*gb.arr);

    id<MTLDevice> device      = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_v.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_v.mtl_buffer;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)b_v.mtl_buffer;

    Shape stat = bn_stat_shape(channels, ndim);
    const std::size_t y_nbytes    = shape_nbytes(x_shape, dt);
    const std::size_t stat_nbytes = shape_nbytes(stat, dt);
    id<MTLBuffer> y_buf  = [device newBufferWithLength:y_nbytes    options:MTLResourceStorageModeShared];
    id<MTLBuffer> m_buf  = [device newBufferWithLength:stat_nbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> r_buf  = [device newBufferWithLength:stat_nbytes options:MTLResourceStorageModeShared];
    if (!y_buf || !m_buf || !r_buf) {
        throw std::runtime_error("mps::batch_norm_train_forward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe =
            bn_train_fwd_executable(x_shape, channels, ndim, eps, mps_dt, device);

        NSArray<NSNumber*>* x_ns  = shape_vec_to_ns(x_shape);
        NSArray<NSNumber*>* st_ns = shape_vec_to_ns(stat);
        NSArray<NSNumber*>* g_ns  = @[@(channels)];

        MPSGraphTensorData* x_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf shape:x_ns dataType:mps_dt];
        MPSGraphTensorData* g_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf shape:g_ns dataType:mps_dt];
        MPSGraphTensorData* b_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:b_buf shape:g_ns dataType:mps_dt];
        MPSGraphTensorData* y_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:y_buf shape:x_ns dataType:mps_dt];
        MPSGraphTensorData* m_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:m_buf shape:st_ns dataType:mps_dt];
        MPSGraphTensorData* r_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:r_buf shape:st_ns dataType:mps_dt];

        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_d, g_d, b_d]
                             resultsArray:@[y_d, m_d, r_d]
                      executionDescriptor:desc];
    }

    auto wrap = [&](id<MTLBuffer> buf, const Shape& sh) -> Storage {
        void* raw = (__bridge_retained void*)buf;
        auto arr = buffer_to_array(raw, std::vector<int>(sh.begin(), sh.end()), dt);
        return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
    };
    return BatchNormForwardOut{
        wrap(y_buf, x_shape),
        wrap(m_buf, stat),
        wrap(r_buf, stat),
    };
}

BatchNormBackwardOut batch_norm_train_backward(const Storage& x,
                                               const Storage& gamma,
                                               const Storage& saved_mean,
                                               const Storage& saved_rstd,
                                               const Storage& grad,
                                               int channels,
                                               int ndim,
                                               const Shape& x_shape,
                                               Dtype dt,
                                               double eps) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gm = std::get<GpuStorage>(saved_mean);
    const auto& gr = std::get<GpuStorage>(saved_rstd);
    const auto& ggrad = std::get<GpuStorage>(grad);

    BufferView x_v  = array_to_buffer(*gx.arr);
    BufferView g_v  = array_to_buffer(*gg.arr);
    BufferView m_v  = array_to_buffer(*gm.arr);
    BufferView r_v  = array_to_buffer(*gr.arr);
    BufferView gr_v = array_to_buffer(*ggrad.arr);

    id<MTLDevice> device      = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();
    id<MTLBuffer> x_buf  = (__bridge id<MTLBuffer>)x_v.mtl_buffer;
    id<MTLBuffer> g_buf  = (__bridge id<MTLBuffer>)g_v.mtl_buffer;
    id<MTLBuffer> m_buf  = (__bridge id<MTLBuffer>)m_v.mtl_buffer;
    id<MTLBuffer> r_buf  = (__bridge id<MTLBuffer>)r_v.mtl_buffer;
    id<MTLBuffer> gr_buf = (__bridge id<MTLBuffer>)gr_v.mtl_buffer;

    Shape stat = bn_stat_shape(channels, ndim);
    Shape gamma_shape{channels};
    const std::size_t dx_nbytes = shape_nbytes(x_shape, dt);
    const std::size_t dg_nbytes = shape_nbytes(gamma_shape, dt);
    id<MTLBuffer> dx_buf = [device newBufferWithLength:dx_nbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> dg_buf = [device newBufferWithLength:dg_nbytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> db_buf = [device newBufferWithLength:dg_nbytes options:MTLResourceStorageModeShared];
    if (!dx_buf || !dg_buf || !db_buf) {
        throw std::runtime_error("mps::batch_norm_train_backward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe =
            bn_train_bwd_executable(x_shape, channels, ndim, eps, mps_dt, device);

        NSArray<NSNumber*>* x_ns  = shape_vec_to_ns(x_shape);
        NSArray<NSNumber*>* st_ns = shape_vec_to_ns(stat);
        NSArray<NSNumber*>* g_ns  = @[@(channels)];

        MPSGraphTensorData* x_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:x_buf  shape:x_ns  dataType:mps_dt];
        MPSGraphTensorData* g_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf  shape:g_ns  dataType:mps_dt];
        MPSGraphTensorData* m_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:m_buf  shape:st_ns dataType:mps_dt];
        MPSGraphTensorData* r_d  = [[MPSGraphTensorData alloc] initWithMTLBuffer:r_buf  shape:st_ns dataType:mps_dt];
        MPSGraphTensorData* gr_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:gr_buf shape:x_ns  dataType:mps_dt];
        MPSGraphTensorData* dx_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:dx_buf shape:x_ns  dataType:mps_dt];
        MPSGraphTensorData* dg_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:dg_buf shape:g_ns  dataType:mps_dt];
        MPSGraphTensorData* db_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:db_buf shape:g_ns  dataType:mps_dt];

        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[x_d, g_d, m_d, r_d, gr_d]
                             resultsArray:@[dx_d, dg_d, db_d]
                      executionDescriptor:desc];
    }

    auto wrap = [&](id<MTLBuffer> buf, const Shape& sh) -> Storage {
        void* raw = (__bridge_retained void*)buf;
        auto arr = buffer_to_array(raw, std::vector<int>(sh.begin(), sh.end()), dt);
        return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
    };
    return BatchNormBackwardOut{
        wrap(dx_buf, x_shape),
        wrap(dg_buf, gamma_shape),
        wrap(db_buf, gamma_shape),
    };
}

// ── Softmax backward — fused MPSGraph chain ───────────────────────────────
//
// Formula: dx = z * (grad - sum(z * grad, axis))   where z is softmax(x).
// MLX expresses this as 4 separate ops; MPSGraph compiler fuses the
// equivalent primitive chain into a single Metal kernel that streams
// through (z, grad) once instead of allocating per-step intermediates.

namespace {

NSString* softmax_bwd_cache_key(const Shape& shape, int axis, MPSDataType dt) {
    NSMutableString* k = [NSMutableString stringWithFormat:@"softmax_bwd:%d:axis=%d:", (int)dt, axis];
    for (auto d : shape) [k appendFormat:@"%lld,", (long long)d];
    return [k copy];
}

MPSGraphExecutable* softmax_bwd_executable(const Shape& shape,
                                           int axis,
                                           MPSDataType mps_dt,
                                           id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key = softmax_bwd_cache_key(shape, axis, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        NSArray<NSNumber*>* shape_ns = shape_vec_to_ns(shape);

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* z    = [graph placeholderWithShape:shape_ns dataType:mps_dt name:@"z"];
        MPSGraphTensor* grad = [graph placeholderWithShape:shape_ns dataType:mps_dt name:@"grad"];

        // MPSGraph's canonical softmax-gradient op.  The `sourceTensor`
        // parameter is the softmax OUTPUT (despite the misleading name);
        // PyTorch MPS calls this exact API with saved-output, and the math
        // produces  dx = z * (grad - sum(z*grad, axis))  which matches
        // Lucid's saved-z convention.
        MPSGraphTensor* result =
            [graph softMaxGradientWithIncomingGradient:grad
                                          sourceTensor:z
                                                  axis:axis
                                                  name:@"dx"];

        MPSGraphShapedType* typed = [[MPSGraphShapedType alloc] initWithShape:shape_ns dataType:mps_dt];
        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:@{z : typed, grad : typed}
                       targetTensors:@[result]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

Storage softmax_backward(const Storage& z,
                         const Storage& grad,
                         int axis,
                         const Shape& shape,
                         Dtype dt) {
    const auto& gz_st = std::get<GpuStorage>(z);
    const auto& gg_st = std::get<GpuStorage>(grad);
    if (!gz_st.arr || !gg_st.arr) {
        throw std::runtime_error("mps::softmax_backward: input has no MLX array");
    }
    BufferView z_v = array_to_buffer(*gz_st.arr);
    BufferView g_v = array_to_buffer(*gg_st.arr);

    id<MTLDevice> device      = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)shared_mtl_queue();
    id<MTLBuffer> z_buf = (__bridge id<MTLBuffer>)z_v.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_v.mtl_buffer;

    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes options:MTLResourceStorageModeShared];
    if (!out_buf) {
        throw std::runtime_error("mps::softmax_backward: MTLBuffer alloc failed");
    }

    @autoreleasepool {
        MPSDataType mps_dt = to_mps_dtype(dt);
        MPSGraphExecutable* exe = softmax_bwd_executable(shape, axis, mps_dt, device);
        NSArray<NSNumber*>* shape_ns = shape_vec_to_ns(shape);
        MPSGraphTensorData* z_d   = [[MPSGraphTensorData alloc] initWithMTLBuffer:z_buf shape:shape_ns dataType:mps_dt];
        MPSGraphTensorData* g_d   = [[MPSGraphTensorData alloc] initWithMTLBuffer:g_buf shape:shape_ns dataType:mps_dt];
        MPSGraphTensorData* out_d = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf shape:shape_ns dataType:mps_dt];
        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:@[z_d, g_d]
                             resultsArray:@[out_d]
                      executionDescriptor:desc];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr =
        buffer_to_array(out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

}  // namespace lucid::gpu::mps
