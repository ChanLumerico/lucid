// lucid/_C/backend/gpu/mps/MpsDispatch.cpp
//
// Plain-C++ policy layer.  No Obj-C; safe to include from anywhere.

#include "MpsDispatch.h"
#include "MpsBridge.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace lucid::gpu::mps {

namespace {

bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    // Treat "0" / "false" / "off" / "no" as false; anything else as true.
    if (std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0) return false;
    if (std::strcmp(v, "off") == 0 || std::strcmp(v, "OFF") == 0) return false;
    if (std::strcmp(v, "no") == 0 || std::strcmp(v, "NO") == 0) return false;
    return true;
}

std::once_flag g_init_once;
bool g_enabled = true;
bool g_debug = false;

void init_once() {
    std::call_once(g_init_once, []() {
        if (env_truthy("LUCID_MPS_DISABLE")) g_enabled = false;
        if (env_truthy("LUCID_MPS_DEBUG")) g_debug = true;
    });
}

}  // namespace

bool enabled() {
    init_once();
    if (!g_enabled) return false;
    // Defer to bridge_available so any device-init failure cleanly disables
    // dispatch without requiring callers to check separately.
    return bridge_available();
}

bool debug_enabled() {
    init_once();
    return g_debug;
}

bool should_dispatch_gelu(std::int64_t numel, Dtype dt) {
    (void)numel;
    (void)dt;
    // Phase 0 measurement: GELU is 13–31× slower on MLX across every shape
    // (transformer_acts, ffn-big, rn18 activations).  Universal dispatch
    // until Phase 4 measurement shows a regime where MLX wins.
    return enabled();
}

bool should_dispatch_gelu_exact(std::int64_t numel, Dtype dt) {
    (void)numel;
    (void)dt;
    // Exact (erf-based) GELU is the default Python F.gelu path — same
    // universal-win profile as the tanh-approx variant.  Phase 0 baseline
    // measured this path at 13-30× vs torch MPS.
    return enabled();
}

bool should_dispatch_layer_norm_backward(std::int64_t outer,
                                         std::int64_t normalized_size,
                                         Dtype dt) {
    (void)dt;
    // Llama-scale (normalized_size=4096, outer~4096) sees the biggest gap
    // (~2.4× torch).  Smaller transformer Q/K/V projections (normalized=768)
    // have a smaller gap (~2× torch) but dispatch overhead bites more.
    // Threshold = normalized_size >= 512 AND outer >= 256 ≈ "real layer".
    if (!enabled()) return false;
    return normalized_size >= 512 && outer >= 256;
}

bool should_dispatch_batch_norm_train(std::int64_t numel, Dtype dt) {
    (void)dt;
    // ResNet shapes (≤ 2M numel) already at torch parity via MLX; the
    // dispatch overhead would hurt.  ImageNet-scale large_acts (~26M
    // numel) is 5.5× torch — that's the case we dispatch.  Threshold
    // at 8M numel cleanly separates these regimes.
    if (!enabled()) return false;
    return numel >= 8 * 1024 * 1024;
}

bool should_dispatch_silu_backward(std::int64_t numel, Dtype dt) {
    (void)dt;
    // Phase 4 measurement (Mac Studio M4 Max):
    //   1M numel:  MLX 0.27 ms → MPS 0.44 ms (regression, dispatch ohead bites)
    //   3M numel:  MLX 0.53 ms → MPS 0.63 ms (mild regression)
    //   12.5M numel: MLX 2.63 ms → MPS 1.70 ms (1.55× win, ratio 5.75→3.68× torch)
    // Dispatch overhead (~150 µs/call) only amortizes on large activations
    // — FFN-scale (B*L*D ≥ ~6M) gets the benefit, CNN activation shapes don't.
    if (!enabled()) return false;
    return numel >= 6 * 1024 * 1024;
}

bool should_dispatch_softmax_backward(std::int64_t axis_size, Dtype dt) {
    (void)axis_size;
    (void)dt;
    // Tried Phase 4 — disabled.  Measurement on dev box (4096, 50257) F32:
    //   MLX chain: 51 ms  MPSGraph canonical: 67 ms (0.76× slower)
    //   MPSGraph hand-rolled: 64 ms (0.71× slower)  torch: 19 ms
    // MPSGraph framework overhead (executable lookup + 800 MB MTLBuffer
    // alloc + run) exceeds the per-call kernel saving for very large axes.
    // MLX's lazy chain is already memory-bandwidth-bound and competitive.
    // Closing the residual 3× torch gap would need a custom Metal compute
    // kernel — out of MPSGraph's wheelhouse.  Kernel kept (in MpsKernels)
    // as reference for the canonical signature.
    return false;
}

}  // namespace lucid::gpu::mps
