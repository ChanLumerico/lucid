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
#include <mlx/transforms.h>  // mlx::core::eval(std::vector<array>)

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

// ── GELU tanh-approx — custom Metal compute kernel ────────────────────────
//
// The MPSGraph 9-op composite above produces no measurable speedup vs
// MLX (both compile down to roughly the same multi-op kernel chain on
// M-series).  The reference framework's MPS path reaches ~4× faster
// on (8, 1024, 3072) by dispatching a single hand-tuned Metal kernel.
// We match that with
// a one-pass MSL kernel below: read x once, evaluate GELU in registers,
// write y once.  Threadgroup geometry tuned to 256 threads/group
// (default M-series sweet spot for memory-bound elementwise ops).
//
// Falls back to the MPSGraph build above via the dispatch predicate
// when the env-gated metal path is disabled.

namespace {

// MSL source for the tanh-approximation GELU.  Single kernel, in-register
// arithmetic.  The function symbol is ``gelu_tanh_f32`` (we currently
// only ship the F32 variant; the F16 variant is straight-line if added
// later).
constexpr const char* kGeluTanhF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Vectorised: process 4 elements per thread via float4 loads / stores.
// One float4 = 16 bytes, matches the natural alignment of MTLBuffer
// allocations.  Per-thread elements / vector width = 4, so the dispatch
// grid is shrunk 4× and the thread count drops accordingly.
kernel void gelu_tanh_f32(device const float4* x   [[buffer(0)]],
                          device float4*       y   [[buffer(1)]],
                          constant uint&       n4  [[buffer(2)]],
                          uint                 gid [[thread_position_in_grid]]) {
    if (gid >= n4) return;
    const float k1 = 0.7978845608028654f;
    const float k2 = 0.044715f;
    float4 xv = x[gid];
    float4 x2 = xv * xv;
    float4 x3 = x2 * xv;
    float4 inner = k1 * (xv + k2 * x3);
    float4 t = tanh(inner);
    y[gid] = 0.5f * xv * (1.0f + t);
}
)MSL";

// MSL source for the tanh-approximation GELU backward.  Reads x AND
// grad_out, writes grad_in.
// Exact-GELU (erf-based) variant — the default of ``F.gelu(x)``
// (no ``approximate="tanh"`` arg).  Same per-thread float4 pattern.
// The exact-GELU kernels below inline a polynomial erf approximation
// (Abramowitz & Stegun 7.1.26, max abs error ≈ 1.5e-7) directly —
// MSL has no native ``erf`` builtin.  Inlined per-kernel rather than
// shared via a common header so the MSL compile of each kernel is
// self-contained (libraries are compiled from a single source string).

constexpr const char* kGeluExactF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float4 erf_approx_f4(float4 x) {
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float4 ax = abs(x);
    float4 t  = 1.0f / (1.0f + p * ax);
    float4 t2 = t  * t;
    float4 t3 = t2 * t;
    float4 t4 = t3 * t;
    float4 t5 = t4 * t;
    float4 poly = a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5;
    float4 y = 1.0f - poly * exp(-ax * ax);
    return sign(x) * y;
}

kernel void gelu_exact_f32(device const float4* x   [[buffer(0)]],
                           device float4*       y   [[buffer(1)]],
                           constant uint&       n4  [[buffer(2)]],
                           uint                 gid [[thread_position_in_grid]]) {
    if (gid >= n4) return;
    const float inv_sqrt2 = 0.7071067811865476f;
    float4 xv = x[gid];
    float4 e = erf_approx_f4(xv * inv_sqrt2);
    y[gid] = 0.5f * xv * (1.0f + e);
}
)MSL";

constexpr const char* kGeluExactBwdF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float4 erf_approx_f4(float4 x) {
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float4 ax = abs(x);
    float4 t  = 1.0f / (1.0f + p * ax);
    float4 t2 = t  * t;
    float4 t3 = t2 * t;
    float4 t4 = t3 * t;
    float4 t5 = t4 * t;
    float4 poly = a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5;
    float4 y = 1.0f - poly * exp(-ax * ax);
    return sign(x) * y;
}

// d/dx[0.5*x*(1+erf(x/sqrt(2)))] = Φ(x) + x * φ(x).
kernel void gelu_exact_bwd_f32(device const float4* x   [[buffer(0)]],
                               device const float4* g   [[buffer(1)]],
                               device float4*       dx  [[buffer(2)]],
                               constant uint&       n4  [[buffer(3)]],
                               uint                 gid [[thread_position_in_grid]]) {
    if (gid >= n4) return;
    const float inv_sqrt2 = 0.7071067811865476f;
    const float inv_sqrt_2pi = 0.3989422804014327f;
    float4 xv = x[gid];
    float4 gv = g[gid];
    float4 e = erf_approx_f4(xv * inv_sqrt2);
    float4 cdf = 0.5f * (1.0f + e);
    float4 pdf = inv_sqrt_2pi * exp(-0.5f * xv * xv);
    dx[gid] = gv * (cdf + xv * pdf);
}
)MSL";

// SiLU forward / backward — single-pass float4-vectorised kernels.
// Math:
//   silu(x)   = x * sigmoid(x) = x / (1 + exp(-x))
//   silu'(x)  = σ(x) * (1 + x * (1 - σ(x)))   where σ = sigmoid
constexpr const char* kSiluF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void silu_f32(device const float4* x   [[buffer(0)]],
                     device float4*       y   [[buffer(1)]],
                     constant uint&       n4  [[buffer(2)]],
                     uint                 gid [[thread_position_in_grid]]) {
    if (gid >= n4) return;
    float4 xv = x[gid];
    float4 sig = 1.0f / (1.0f + exp(-xv));
    y[gid] = xv * sig;
}
)MSL";

constexpr const char* kSiluBwdF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void silu_bwd_f32(device const float4* x   [[buffer(0)]],
                         device const float4* g   [[buffer(1)]],
                         device float4*       dx  [[buffer(2)]],
                         constant uint&       n4  [[buffer(3)]],
                         uint                 gid [[thread_position_in_grid]]) {
    if (gid >= n4) return;
    float4 xv = x[gid];
    float4 gv = g[gid];
    float4 sig = 1.0f / (1.0f + exp(-xv));
    // dy/dx = sig * (1 + x * (1 - sig))
    float4 dydx = sig * (1.0f + xv * (1.0f - sig));
    dx[gid] = gv * dydx;
}
)MSL";

constexpr const char* kGeluTanhBwdF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Vectorised float4 version of the GELU-tanh backward.
kernel void gelu_tanh_bwd_f32(device const float4* x   [[buffer(0)]],
                              device const float4* g   [[buffer(1)]],
                              device float4*       dx  [[buffer(2)]],
                              constant uint&       n4  [[buffer(3)]],
                              uint                 gid [[thread_position_in_grid]]) {
    if (gid >= n4) return;
    const float k1 = 0.7978845608028654f;
    const float k2 = 0.044715f;
    float4 xv = x[gid];
    float4 gv = g[gid];
    float4 x2 = xv * xv;
    float4 x3 = x2 * xv;
    float4 inner = k1 * (xv + k2 * x3);
    float4 t = tanh(inner);
    float4 t2 = t * t;
    float4 term1 = 0.5f * (1.0f + t);
    float4 dinner = k1 * (1.0f + 3.0f * k2 * x2);
    float4 term2 = 0.5f * xv * (1.0f - t2) * dinner;
    dx[gid] = gv * (term1 + term2);
}
)MSL";

struct MetalPipelineEntry {
    id<MTLComputePipelineState> pso = nil;
    id<MTLLibrary> lib = nil;
};

// Single per-process cache for our custom Metal kernels — key is the
// MSL function name.  Compilation happens once on first use.
NSMutableDictionary<NSString*, NSValue*>* custom_metal_cache(void) {
    static dispatch_once_t once;
    static NSMutableDictionary* cache;
    dispatch_once(&once, ^{ cache = [NSMutableDictionary new]; });
    return cache;
}

id<MTLComputePipelineState>
compile_custom_metal(const char* msl_src, NSString* fn_name,
                     id<MTLDevice> device) {
    NSMutableDictionary* cache = custom_metal_cache();
    @synchronized(cache) {
        NSValue* hit = cache[fn_name];
        if (hit) {
            return (__bridge id<MTLComputePipelineState>)hit.pointerValue;
        }
        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:msl_src];
        id<MTLLibrary> lib =
            [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            throw std::runtime_error(
                std::string("mps::compile_custom_metal: ") +
                [fn_name UTF8String] + ": " +
                (err ? [err.localizedDescription UTF8String] : "no library"));
        }
        id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
        if (!fn) {
            throw std::runtime_error(
                std::string("mps::compile_custom_metal: function not found: ") +
                [fn_name UTF8String]);
        }
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            throw std::runtime_error(
                std::string("mps::compile_custom_metal: pipeline state failed: ") +
                (err ? [err.localizedDescription UTF8String] : "unknown"));
        }
        // Retain so the cache holds a live reference.  Using NSValue
        // pointerValue with __bridge_retained keeps the pso alive until
        // process exit (acceptable — cache is process-lifetime).
        cache[fn_name] = [NSValue valueWithPointer:(__bridge_retained void*)pso];
        return pso;
    }
}

// Shared single-input / single-output unary launcher for our custom
// GELU forward kernel.  Zero-copy input via array_to_buffer.
Storage gelu_metal_forward_impl(const Storage& x, const Shape& shape,
                                Dtype dt) {
    if (dt != Dtype::F32) {
        // Fall back to MPSGraph composite for non-F32 (F16 path not yet
        // ported to MSL; caller checks should_dispatch_gelu and routes
        // accordingly).
        return gelu_forward(x, shape, dt);
    }
    const auto& gx = std::get<GpuStorage>(x);
    if (!gx.arr)
        throw std::runtime_error("mps::gelu_metal_forward: input has no MLX array");
    BufferView in_view = array_to_buffer(*gx.arr);
    id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)in_view.mtl_buffer;
    id<MTLDevice> device =
        (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    std::size_t numel = 1;
    for (auto d : shape) numel *= static_cast<std::size_t>(d);
    // Vectorised kernel needs numel divisible by 4.  Fall back to the
    // MPSGraph composite when this constraint fails (rare for typical
    // transformer shapes where the trailing dim is a multiple of 4).
    if (numel % 4 != 0) {
        return gelu_forward(x, shape, dt);
    }
    const std::size_t n4 = numel / 4;
    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes
                            options:MTLResourceStorageModeShared];
    if (!out_buf)
        throw std::runtime_error("mps::gelu_metal_forward: MTLBuffer alloc failed");

    @autoreleasepool {
        id<MTLComputePipelineState> pso =
            compile_custom_metal(kGeluTanhF32MSL, @"gelu_tanh_f32", device);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:in_buf offset:0 atIndex:0];
        [enc setBuffer:out_buf offset:0 atIndex:1];
        uint32_t n4_u32 = static_cast<uint32_t>(n4);
        [enc setBytes:&n4_u32 length:sizeof(uint32_t) atIndex:2];
        // 256 threads/threadgroup — M-series default sweet spot.  Each
        // thread processes one float4 = 4 elements.
        const NSUInteger tpg = std::min(
            (NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroup = MTLSizeMake(tpg, 1, 1);
        MTLSize grid = MTLSizeMake((n4 + tpg - 1) / tpg, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr = buffer_to_array(
        out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

Storage gelu_metal_backward_impl(const Storage& x, const Storage& grad,
                                 const Shape& shape, Dtype dt) {
    if (dt != Dtype::F32) {
        return gelu_backward(x, grad, shape, dt);
    }
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(grad);
    if (!gx.arr || !gg.arr)
        throw std::runtime_error("mps::gelu_metal_backward: input has no MLX array");
    BufferView x_view = array_to_buffer(*gx.arr);
    BufferView g_view = array_to_buffer(*gg.arr);
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_view.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_view.mtl_buffer;
    id<MTLDevice> device =
        (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    std::size_t numel = 1;
    for (auto d : shape) numel *= static_cast<std::size_t>(d);
    if (numel % 4 != 0) {
        return gelu_backward(x, grad, shape, dt);
    }
    const std::size_t n4 = numel / 4;
    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> dx_buf =
        [device newBufferWithLength:out_nbytes
                            options:MTLResourceStorageModeShared];
    if (!dx_buf)
        throw std::runtime_error("mps::gelu_metal_backward: MTLBuffer alloc failed");

    @autoreleasepool {
        id<MTLComputePipelineState> pso =
            compile_custom_metal(kGeluTanhBwdF32MSL, @"gelu_tanh_bwd_f32", device);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:g_buf offset:0 atIndex:1];
        [enc setBuffer:dx_buf offset:0 atIndex:2];
        uint32_t n4_u32 = static_cast<uint32_t>(n4);
        [enc setBytes:&n4_u32 length:sizeof(uint32_t) atIndex:3];
        const NSUInteger tpg = std::min(
            (NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroup = MTLSizeMake(tpg, 1, 1);
        MTLSize grid = MTLSizeMake((n4 + tpg - 1) / tpg, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    void* out_buf_raw = (__bridge_retained void*)dx_buf;
    ::mlx::core::array out_arr = buffer_to_array(
        out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

}  // namespace

// Public wrappers — pick up the env-gated custom-Metal path.
Storage gelu_metal_forward(const Storage& x, const Shape& shape, Dtype dt) {
    return gelu_metal_forward_impl(x, shape, dt);
}
Storage gelu_metal_backward(const Storage& x, const Storage& grad,
                            const Shape& shape, Dtype dt) {
    return gelu_metal_backward_impl(x, grad, shape, dt);
}

namespace {

// Generic single-input unary dispatcher.  Used by the exact-GELU
// variant; mirrors gelu_metal_forward_impl's structure but parameterised
// by the MSL function name.
Storage metal_unary_f32(const Storage& x, const Shape& shape, Dtype dt,
                        const char* msl_src, NSString* fn_name) {
    if (dt != Dtype::F32) {
        return Storage{};  // caller must guard with should_dispatch_gelu_metal
    }
    const auto& gx = std::get<GpuStorage>(x);
    if (!gx.arr)
        throw std::runtime_error("mps::metal_unary_f32: input has no MLX array");
    BufferView in_view = array_to_buffer(*gx.arr);
    id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)in_view.mtl_buffer;
    id<MTLDevice> device =
        (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    std::size_t numel = 1;
    for (auto d : shape) numel *= static_cast<std::size_t>(d);
    if (numel % 4 != 0) return Storage{};  // caller must guard
    const std::size_t n4 = numel / 4;
    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes
                            options:MTLResourceStorageModeShared];
    if (!out_buf)
        throw std::runtime_error("mps::metal_unary_f32: MTLBuffer alloc failed");

    @autoreleasepool {
        id<MTLComputePipelineState> pso =
            compile_custom_metal(msl_src, fn_name, device);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:in_buf offset:0 atIndex:0];
        [enc setBuffer:out_buf offset:0 atIndex:1];
        uint32_t n4_u32 = static_cast<uint32_t>(n4);
        [enc setBytes:&n4_u32 length:sizeof(uint32_t) atIndex:2];
        const NSUInteger tpg = std::min(
            (NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroup = MTLSizeMake(tpg, 1, 1);
        MTLSize grid = MTLSizeMake((n4 + tpg - 1) / tpg, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr = buffer_to_array(
        out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

// Binary input + single output (x, grad → dx).
Storage metal_binary_f32(const Storage& x, const Storage& g,
                         const Shape& shape, Dtype dt,
                         const char* msl_src, NSString* fn_name) {
    if (dt != Dtype::F32) return Storage{};
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(g);
    if (!gx.arr || !gg.arr)
        throw std::runtime_error("mps::metal_binary_f32: input has no MLX array");
    BufferView x_view = array_to_buffer(*gx.arr);
    BufferView g_view = array_to_buffer(*gg.arr);
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_view.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_view.mtl_buffer;
    id<MTLDevice> device =
        (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    std::size_t numel = 1;
    for (auto d : shape) numel *= static_cast<std::size_t>(d);
    if (numel % 4 != 0) return Storage{};
    const std::size_t n4 = numel / 4;
    const std::size_t out_nbytes = shape_nbytes(shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes
                            options:MTLResourceStorageModeShared];
    if (!out_buf)
        throw std::runtime_error("mps::metal_binary_f32: MTLBuffer alloc failed");

    @autoreleasepool {
        id<MTLComputePipelineState> pso =
            compile_custom_metal(msl_src, fn_name, device);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:g_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf offset:0 atIndex:2];
        uint32_t n4_u32 = static_cast<uint32_t>(n4);
        [enc setBytes:&n4_u32 length:sizeof(uint32_t) atIndex:3];
        const NSUInteger tpg = std::min(
            (NSUInteger)256, pso.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroup = MTLSizeMake(tpg, 1, 1);
        MTLSize grid = MTLSizeMake((n4 + tpg - 1) / tpg, 1, 1);
        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr = buffer_to_array(
        out_buf_raw, std::vector<int>(shape.begin(), shape.end()), dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

}  // namespace

Storage gelu_exact_metal_forward(const Storage& x, const Shape& shape, Dtype dt) {
    Storage out = metal_unary_f32(x, shape, dt, kGeluExactF32MSL,
                                  @"gelu_exact_f32");
    // Guard fall-through: when dt != F32 or numel % 4 != 0, the helper
    // returns an empty Storage to signal "caller please fall back."
    if (std::holds_alternative<GpuStorage>(out) &&
        std::get<GpuStorage>(out).arr) {
        return out;
    }
    return gelu_exact_forward(x, shape, dt);
}

Storage gelu_exact_metal_backward(const Storage& x, const Storage& grad,
                                  const Shape& shape, Dtype dt) {
    Storage out = metal_binary_f32(x, grad, shape, dt,
                                   kGeluExactBwdF32MSL, @"gelu_exact_bwd_f32");
    if (std::holds_alternative<GpuStorage>(out) &&
        std::get<GpuStorage>(out).arr) {
        return out;
    }
    return gelu_exact_backward(x, grad, shape, dt);
}

// ── SiLU (Swish) custom Metal — uses metal_unary_f32/metal_binary_f32 ──

Storage silu_metal_forward(const Storage& x, const Shape& shape, Dtype dt) {
    Storage out = metal_unary_f32(x, shape, dt, kSiluF32MSL, @"silu_f32");
    if (std::holds_alternative<GpuStorage>(out) &&
        std::get<GpuStorage>(out).arr) {
        return out;
    }
    // Fall through: caller already validated dt; this path is taken only
    // when numel % 4 != 0.  MLX produces silu via a 2-op composite
    // (sigmoid * x); the dispatch predicate ought to have screened the
    // shape, but defend with a safe return.
    throw std::runtime_error(
        "mps::silu_metal_forward: numel % 4 != 0; caller must gate via "
        "should_dispatch_silu_metal which already requires numel % 4 == 0");
}

Storage silu_metal_backward(const Storage& x, const Storage& grad,
                            const Shape& shape, Dtype dt) {
    Storage out = metal_binary_f32(x, grad, shape, dt,
                                   kSiluBwdF32MSL, @"silu_bwd_f32");
    if (std::holds_alternative<GpuStorage>(out) &&
        std::get<GpuStorage>(out).arr) {
        return out;
    }
    return silu_backward(x, grad, shape, dt);
}

// ── BatchNorm train forward — custom 2-pass Metal kernel ──────────────────
//
// Strategy: separate the reduction from the normalize / affine pass.
//
//   Pass 1 (`bn_train_fwd_reduce_f32`):
//     - One threadgroup per channel (so the C32-aligned slot fits in
//       SIMD-shared memory).
//     - 256 threads/group stride over (N * H * W) elements.
//     - simd_sum → threadgroup_barrier → simd_sum across the 8
//       simdgroups → thread 0 writes ``mean[c]`` + ``rstd[c]``.
//
//   Pass 2 (`bn_train_fwd_normalize_f32`):
//     - Plain elementwise: ``y[n,c,h,w] = (x - mean[c]) * rstd[c] *
//       gamma[c] + beta[c]``.
//
// The MPSGraph ``normalizationWithTensor:`` route (above, ~1500 LOC
// down) was measured at noise level vs MLX on M4 Max
// (perf-baseline-rebench-2026-05-25 dispatch audit).  This custom
// kernel matches the reference framework's MPS hand-tuned approach
// and reclaims a fraction of the 2.5–2.8× gap.

namespace {

constexpr const char* kBnFwdReduceF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void bn_train_fwd_reduce_f32(
    device const float* x      [[buffer(0)]],   // [N, C, H*W]
    device       float* mean   [[buffer(1)]],   // [C]
    device       float* rstd   [[buffer(2)]],   // [C]
    constant uint& N           [[buffer(3)]],
    constant uint& C           [[buffer(4)]],
    constant uint& HW          [[buffer(5)]],
    constant float& eps        [[buffer(6)]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint tgid                  [[threadgroup_position_in_grid]],
    uint tg_size               [[threads_per_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_id               [[simdgroup_index_in_threadgroup]])
{
    const uint c = tgid;
    if (c >= C) return;
    const uint N_total = N * HW;

    // Per-thread strided accumulation along the channel's flattened
    // (n, hw) axis.  Layout is x[n, c, hw] → idx = (n*C + c)*HW + hw.
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (uint i = tid; i < N_total; i += tg_size) {
        uint n = i / HW;
        uint hw = i - n * HW;
        uint idx = (n * C + c) * HW + hw;
        float v = x[idx];
        local_sum   += v;
        local_sumsq += v * v;
    }

    // Reduce within the SIMD group (32 lanes).
    local_sum   = simd_sum(local_sum);
    local_sumsq = simd_sum(local_sumsq);

    // Stage per-simdgroup partials into threadgroup memory, then reduce
    // across the 8 simdgroups (assumes ≤ 32 simdgroups / tg, which is
    // the M-series cap).
    threadgroup float tg_sum[32];
    threadgroup float tg_sumsq[32];
    if (simd_lane == 0) {
        tg_sum[simd_id]   = local_sum;
        tg_sumsq[simd_id] = local_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        const uint n_simd_groups = (tg_size + 31u) / 32u;
        float s  = simd_lane < n_simd_groups ? tg_sum[simd_lane]   : 0.0f;
        float ss = simd_lane < n_simd_groups ? tg_sumsq[simd_lane] : 0.0f;
        s  = simd_sum(s);
        ss = simd_sum(ss);
        if (simd_lane == 0) {
            float inv_n = 1.0f / float(N_total);
            float m = s * inv_n;
            float var = ss * inv_n - m * m;
            mean[c] = m;
            rstd[c] = rsqrt(var + eps);
        }
    }
}
)MSL";

constexpr const char* kBnFwdNormalizeF32MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void bn_train_fwd_normalize_f32(
    device const float* x     [[buffer(0)]],   // [N, C, H*W]
    device const float* mean  [[buffer(1)]],   // [C]
    device const float* rstd  [[buffer(2)]],   // [C]
    device const float* gamma [[buffer(3)]],   // [C]
    device const float* beta  [[buffer(4)]],   // [C]
    device       float* y     [[buffer(5)]],   // [N, C, H*W]
    constant uint& C          [[buffer(6)]],
    constant uint& HW         [[buffer(7)]],
    constant uint& total      [[buffer(8)]],
    uint gid                  [[thread_position_in_grid]])
{
    if (gid >= total) return;
    // gid = (n * C + c) * HW + hw   →   c = (gid / HW) % C
    uint n_c = gid / HW;
    uint c   = n_c - (n_c / C) * C;
    float xv = x[gid];
    y[gid] = (xv - mean[c]) * rstd[c] * gamma[c] + beta[c];
}
)MSL";

// Compute the per-channel stat shape (1, C, 1, ..., 1) for ndim
// spatial dims — matches the existing MPSGraph ``batch_norm_train_forward``
// layout so callers don't need a separate reshape.
Shape bn_stat_full_shape(int channels, int ndim) {
    Shape s;
    s.reserve(2 + static_cast<std::size_t>(ndim));
    s.push_back(1);
    s.push_back(channels);
    for (int i = 0; i < ndim; ++i) s.push_back(1);
    return s;
}

}  // namespace

BatchNormForwardOut bn_train_metal_forward(const Storage& x,
                                           const Storage& gamma,
                                           const Storage& beta,
                                           int channels,
                                           int ndim,
                                           double eps,
                                           const Shape& x_shape,
                                           Dtype dt) {
    if (dt != Dtype::F32) {
        // F16 path: delegate to the MPSGraph composite (which already
        // handles both dtypes).  F16 reduction needs higher-precision
        // accumulators and a separate kernel pair — not in scope here.
        return batch_norm_train_forward(x, gamma, beta, channels, ndim,
                                        eps, x_shape, dt);
    }

    const auto& gx = std::get<GpuStorage>(x);
    const auto& gg = std::get<GpuStorage>(gamma);
    const auto& gb = std::get<GpuStorage>(beta);
    if (!gx.arr || !gg.arr || !gb.arr)
        throw std::runtime_error(
            "mps::bn_train_metal_forward: input storage has no MLX array");

    BufferView x_view  = array_to_buffer(*gx.arr);
    BufferView g_view  = array_to_buffer(*gg.arr);
    BufferView b_view  = array_to_buffer(*gb.arr);
    id<MTLBuffer> x_buf = (__bridge id<MTLBuffer>)x_view.mtl_buffer;
    id<MTLBuffer> g_buf = (__bridge id<MTLBuffer>)g_view.mtl_buffer;
    id<MTLBuffer> b_buf = (__bridge id<MTLBuffer>)b_view.mtl_buffer;

    id<MTLDevice> device =
        (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)shared_mtl_queue();

    // N = x_shape[0]; HW = product of spatial dims.
    const std::uint32_t N32 = static_cast<std::uint32_t>(x_shape[0]);
    const std::uint32_t C32 = static_cast<std::uint32_t>(channels);
    std::uint32_t HW32 = 1;
    for (int i = 0; i < ndim; ++i) {
        HW32 *= static_cast<std::uint32_t>(x_shape[2 + i]);
    }
    const std::size_t total_numel = static_cast<std::size_t>(N32) *
                                     static_cast<std::size_t>(C32) *
                                     static_cast<std::size_t>(HW32);

    // Output buffers: y same size as x, stat buffers shape (C,) but
    // exposed as (1, C, 1, …, 1) via reshape on the MLX side.
    const std::size_t y_bytes    = shape_nbytes(x_shape, dt);
    const std::size_t stat_bytes = static_cast<std::size_t>(C32) * sizeof(float);

    id<MTLBuffer> y_buf    =
        [device newBufferWithLength:y_bytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> mean_buf =
        [device newBufferWithLength:stat_bytes
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> rstd_buf =
        [device newBufferWithLength:stat_bytes
                            options:MTLResourceStorageModeShared];
    if (!y_buf || !mean_buf || !rstd_buf)
        throw std::runtime_error("mps::bn_train_metal_forward: MTLBuffer alloc failed");

    const float eps_f = static_cast<float>(eps);

    @autoreleasepool {
        id<MTLComputePipelineState> pso_reduce = compile_custom_metal(
            kBnFwdReduceF32MSL, @"bn_train_fwd_reduce_f32", device);
        id<MTLComputePipelineState> pso_normalize = compile_custom_metal(
            kBnFwdNormalizeF32MSL, @"bn_train_fwd_normalize_f32", device);

        id<MTLCommandBuffer> cb = [queue commandBuffer];

        // Pass 1: per-channel reduction → mean[C], rstd[C].
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_reduce];
            [enc setBuffer:x_buf    offset:0 atIndex:0];
            [enc setBuffer:mean_buf offset:0 atIndex:1];
            [enc setBuffer:rstd_buf offset:0 atIndex:2];
            [enc setBytes:&N32 length:sizeof(N32)   atIndex:3];
            [enc setBytes:&C32 length:sizeof(C32)   atIndex:4];
            [enc setBytes:&HW32 length:sizeof(HW32) atIndex:5];
            [enc setBytes:&eps_f length:sizeof(eps_f) atIndex:6];
            // 1024 threads/group on the reduce path: each per-channel
            // tile in large workloads (8 × 256² = 524288 elements)
            // benefits from 4× more parallel SIMD lanes per channel,
            // bringing the per-thread inner loop from 2048 → 512
            // elements.
            const NSUInteger tpg = std::min(
                (NSUInteger)1024, pso_reduce.maxTotalThreadsPerThreadgroup);
            MTLSize threadgroup = MTLSizeMake(tpg, 1, 1);
            // One threadgroup per channel.
            MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(C32), 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
            [enc endEncoding];
        }

        // Pass 2: per-element normalize + affine.
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso_normalize];
            [enc setBuffer:x_buf    offset:0 atIndex:0];
            [enc setBuffer:mean_buf offset:0 atIndex:1];
            [enc setBuffer:rstd_buf offset:0 atIndex:2];
            [enc setBuffer:g_buf    offset:0 atIndex:3];
            [enc setBuffer:b_buf    offset:0 atIndex:4];
            [enc setBuffer:y_buf    offset:0 atIndex:5];
            [enc setBytes:&C32 length:sizeof(C32)   atIndex:6];
            [enc setBytes:&HW32 length:sizeof(HW32) atIndex:7];
            std::uint32_t total_u32 = static_cast<std::uint32_t>(total_numel);
            [enc setBytes:&total_u32 length:sizeof(total_u32) atIndex:8];
            const NSUInteger tpg = std::min(
                (NSUInteger)256, pso_normalize.maxTotalThreadsPerThreadgroup);
            MTLSize threadgroup = MTLSizeMake(tpg, 1, 1);
            MTLSize grid = MTLSizeMake(
                (total_numel + tpg - 1) / tpg, 1, 1);
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];
    }

    // Wrap buffers as MLX arrays.  ``y`` keeps the full x_shape; stat
    // buffers get a broadcast shape (1, C, 1, …, 1) so downstream
    // backward / normalisation reuses them without reshape.
    void* y_raw    = (__bridge_retained void*)y_buf;
    void* mean_raw = (__bridge_retained void*)mean_buf;
    void* rstd_raw = (__bridge_retained void*)rstd_buf;

    ::mlx::core::array y_arr = buffer_to_array(
        y_raw, std::vector<int>(x_shape.begin(), x_shape.end()), dt);

    const Shape stat_shape = bn_stat_full_shape(channels, ndim);
    ::mlx::core::array mean_arr = buffer_to_array(
        mean_raw, std::vector<int>(stat_shape.begin(), stat_shape.end()), dt);
    ::mlx::core::array rstd_arr = buffer_to_array(
        rstd_raw, std::vector<int>(stat_shape.begin(), stat_shape.end()), dt);

    return BatchNormForwardOut{
        Storage{gpu::wrap_mlx_array(std::move(y_arr), dt)},
        Storage{gpu::wrap_mlx_array(std::move(mean_arr), dt)},
        Storage{gpu::wrap_mlx_array(std::move(rstd_arr), dt)},
    };
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
// activations (Phase 0 measurement: 5.5× reference framework on
// 32×64×112×112).

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
        // the reference framework's MPS path calls this exact API with
        // saved-output, and the math produces
        // ``dx = z * (grad - sum(z*grad, axis))`` which matches Lucid's
        // saved-z convention.
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

// ── Embedding backward — scatter-add via MPSGraph ─────────────────────
//
// Lucid's MLX path composes ``broadcast → reshape → scatter_add_axis``
// (with an optional ``multiply`` mask for ``padding_idx``).  MPSGraph
// has a native ``MPSGraphScatterModeAdd`` primitive that performs the
// same reduction in a single fused kernel — measured 28× faster on
// GPT-2-scale inputs.

namespace {

NSString* embedding_bwd_cache_key(std::int64_t N,
                                  std::int64_t D,
                                  std::int64_t M_total,
                                  bool has_padding,
                                  MPSDataType mps_dt) {
    return [NSString stringWithFormat:
                @"embedding_bwd:N=%lld:D=%lld:M=%lld:pad=%d:dt=%d",
                (long long)N, (long long)D, (long long)M_total,
                has_padding ? 1 : 0, (int)mps_dt];
}

// Build (and cache) the MPSGraph executable that scatter-adds
// ``grad_flat`` ([M, D]) into a zero-initialised ``dW`` ([N, D]) at
// rows given by ``idx_flat`` ([M]).
//
// When ``has_padding`` is true the executable also takes a 0-D
// ``pad_idx`` int32 input and zeroes out rows of ``grad_flat`` whose
// index equals ``pad_idx`` BEFORE the scatter.
MPSGraphExecutable* embedding_bwd_executable(std::int64_t N,
                                             std::int64_t D,
                                             std::int64_t M_total,
                                             bool has_padding,
                                             MPSDataType mps_dt,
                                             id<MTLDevice> device) {
    NSMutableDictionary* cache = executable_cache();
    NSString* key =
        embedding_bwd_cache_key(N, D, M_total, has_padding, mps_dt);
    @synchronized(cache) {
        MPSGraphExecutable* hit = cache[key];
        if (hit) return hit;

        NSArray<NSNumber*>* grad_shape =
            @[ @((long long)M_total), @((long long)D) ];
        NSArray<NSNumber*>* idx_shape = @[ @((long long)M_total) ];
        NSArray<NSNumber*>* pad_shape = @[];

        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* grad =
            [graph placeholderWithShape:grad_shape
                               dataType:mps_dt
                                   name:@"grad"];
        // ``MPSGraphScatterMode`` requires int32 indices on macOS 14+;
        // the Lucid GpuBackend caller casts the int64 indices it
        // receives down to int32 before invoking us.
        MPSGraphTensor* idx =
            [graph placeholderWithShape:idx_shape
                               dataType:MPSDataTypeInt32
                                   name:@"idx"];
        MPSGraphTensor* pad_idx_t = nil;
        if (has_padding) {
            pad_idx_t = [graph placeholderWithShape:pad_shape
                                           dataType:MPSDataTypeInt32
                                               name:@"pad_idx"];
        }

        MPSGraphTensor* updates = grad;
        if (has_padding) {
            // mask = (idx != pad_idx) → bool [M] → cast to ``mps_dt``
            // [M] → reshape [M, 1] → broadcast [M, D] → grad * mask.
            //
            // ``neq_t`` (not ``not_eq``) — the latter is a C++ alternate
            // operator token from <ciso646>, banned as an identifier.
            MPSGraphTensor* neq_t =
                [graph notEqualWithPrimaryTensor:idx
                                 secondaryTensor:pad_idx_t
                                            name:@"pad_mask_bool"];
            MPSGraphTensor* mask_t =
                [graph castTensor:neq_t
                           toType:mps_dt
                             name:@"pad_mask_cast"];
            MPSGraphTensor* mask_2d =
                [graph reshapeTensor:mask_t
                           withShape:@[ @((long long)M_total), @1 ]
                                name:@"pad_mask_2d"];
            updates =
                [graph multiplicationWithPrimaryTensor:grad
                                       secondaryTensor:mask_2d
                                                  name:@"masked_grad"];
        }

        // Zero-initialised dW [N, D] — MPSGraph constant lets the
        // scheduler skip allocating a separate buffer for this base.
        MPSGraphTensor* zeros =
            [graph constantWithScalar:0.0
                                shape:@[ @((long long)N), @((long long)D) ]
                             dataType:mps_dt];

        MPSGraphTensor* dW =
            [graph scatterWithDataTensor:zeros
                           updatesTensor:updates
                           indicesTensor:idx
                                    axis:0
                                    mode:MPSGraphScatterModeAdd
                                    name:@"dW"];

        MPSGraphShapedType* grad_ty =
            [[MPSGraphShapedType alloc] initWithShape:grad_shape
                                             dataType:mps_dt];
        MPSGraphShapedType* idx_ty =
            [[MPSGraphShapedType alloc] initWithShape:idx_shape
                                             dataType:MPSDataTypeInt32];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
            [NSMutableDictionary dictionary];
        feeds[grad] = grad_ty;
        feeds[idx] = idx_ty;
        if (has_padding) {
            MPSGraphShapedType* pad_ty =
                [[MPSGraphShapedType alloc] initWithShape:pad_shape
                                                 dataType:MPSDataTypeInt32];
            feeds[pad_idx_t] = pad_ty;
        }

        MPSGraphExecutable* compiled =
            [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                               feeds:feeds
                       targetTensors:@[ dW ]
                    targetOperations:nil
               compilationDescriptor:nil];
        cache[key] = compiled;
        return compiled;
    }
}

}  // namespace

Storage embedding_backward(const Storage& grad_out,
                           const Storage& indices,
                           std::int64_t N,
                           std::int64_t D,
                           std::int64_t M_total,
                           int padding_idx,
                           Dtype dt) {
    const auto& g_st = std::get<GpuStorage>(grad_out);
    const auto& i_st = std::get<GpuStorage>(indices);
    if (!g_st.arr || !i_st.arr) {
        throw std::runtime_error(
            "mps::embedding_backward: input storage has no MLX array");
    }

    // Cast indices to int32 + flatten to [M_total].  MPSGraph
    // ScatterMode requires int32 indices (macOS 26 SDK still rejects
    // int64 for the scatter index path).  ``flatten`` is a reshape, no
    // copy.
    ::mlx::core::array idx_i32 =
        ::mlx::core::astype(*i_st.arr, ::mlx::core::int32);
    ::mlx::core::array idx_flat =
        ::mlx::core::reshape(idx_i32, {static_cast<int>(M_total)});
    ::mlx::core::array grad_flat = ::mlx::core::reshape(
        *g_st.arr, {static_cast<int>(M_total), static_cast<int>(D)});
    // ``eval()`` forces the MLX side to materialise the freshly
    // reshaped/cast arrays before we hand their MTLBuffers to MPS — the
    // bridge expects ready-to-read buffers.
    std::vector<::mlx::core::array> _to_eval = {idx_flat, grad_flat};
    ::mlx::core::eval(_to_eval);

    BufferView grad_v = array_to_buffer(grad_flat);
    BufferView idx_v = array_to_buffer(idx_flat);

    id<MTLDevice> device = (__bridge id<MTLDevice>)shared_mtl_device();
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)shared_mtl_queue();
    id<MTLBuffer> grad_buf = (__bridge id<MTLBuffer>)grad_v.mtl_buffer;
    id<MTLBuffer> idx_buf = (__bridge id<MTLBuffer>)idx_v.mtl_buffer;

    const Shape out_shape = {N, D};
    const std::size_t out_nbytes = shape_nbytes(out_shape, dt);
    id<MTLBuffer> out_buf =
        [device newBufferWithLength:out_nbytes
                            options:MTLResourceStorageModeShared];
    if (!out_buf) {
        throw std::runtime_error(
            "mps::embedding_backward: MTLBuffer alloc failed");
    }

    // For ``padding_idx``: allocate a 1-element int32 buffer holding the
    // pad value.  Compiled once into the executable, so the buffer is
    // alive only for this dispatch.
    id<MTLBuffer> pad_buf = nil;
    if (padding_idx >= 0) {
        pad_buf = [device newBufferWithLength:sizeof(std::int32_t)
                                      options:MTLResourceStorageModeShared];
        if (!pad_buf) {
            throw std::runtime_error(
                "mps::embedding_backward: padding-idx MTLBuffer alloc failed");
        }
        std::int32_t v = static_cast<std::int32_t>(padding_idx);
        std::memcpy([pad_buf contents], &v, sizeof(v));
    }

    @autoreleasepool {
        MPSDataType mps_dt = to_mps_dtype(dt);
        const bool has_padding = (padding_idx >= 0);
        MPSGraphExecutable* exe = embedding_bwd_executable(
            N, D, M_total, has_padding, mps_dt, device);

        NSArray<NSNumber*>* grad_shape =
            @[ @((long long)M_total), @((long long)D) ];
        NSArray<NSNumber*>* idx_shape = @[ @((long long)M_total) ];
        NSArray<NSNumber*>* out_shape_ns =
            @[ @((long long)N), @((long long)D) ];

        MPSGraphTensorData* grad_d =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:grad_buf
                                                    shape:grad_shape
                                                 dataType:mps_dt];
        MPSGraphTensorData* idx_d =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:idx_buf
                                                    shape:idx_shape
                                                 dataType:MPSDataTypeInt32];
        MPSGraphTensorData* out_d =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                    shape:out_shape_ns
                                                 dataType:mps_dt];
        NSMutableArray<MPSGraphTensorData*>* inputs =
            [NSMutableArray arrayWithObjects:grad_d, idx_d, nil];
        if (has_padding) {
            MPSGraphTensorData* pad_d =
                [[MPSGraphTensorData alloc] initWithMTLBuffer:pad_buf
                                                        shape:@[]
                                                     dataType:MPSDataTypeInt32];
            [inputs addObject:pad_d];
        }

        MPSGraphExecutableExecutionDescriptor* desc =
            [[MPSGraphExecutableExecutionDescriptor alloc] init];
        desc.waitUntilCompleted = YES;
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:inputs
                             resultsArray:@[ out_d ]
                      executionDescriptor:desc];
    }

    void* out_buf_raw = (__bridge_retained void*)out_buf;
    ::mlx::core::array out_arr = buffer_to_array(
        out_buf_raw,
        std::vector<int>{static_cast<int>(N), static_cast<int>(D)}, dt);
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

}  // namespace lucid::gpu::mps
