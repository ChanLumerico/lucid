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
    //
    // 2026-05-25 rebench correction: the 13-31× gap was an artifact of
    // MLX lazy graph (perf-baseline-rebench-2026-05-25).  Real gap is
    // 3.83× on ffn_big, but the existing MPSGraph 9-op composite below
    // **doesn't help** (1.07× vs MLX = noise; 0.91× = mild regression
    // on fwd).  Route through ``should_dispatch_gelu_metal`` instead —
    // the custom Metal kernel matches the reference framework's MPS
    // path.  Keep this predicate returning false so the MPSGraph build
    // is no longer used by default.
    return false;
}

bool should_dispatch_gelu_metal(std::int64_t numel, Dtype dt) {
    if (!enabled()) return false;
    if (dt != Dtype::F32) return false;
    // Small inputs: the compute-pass setup cost dominates the per-element
    // gain.  Threshold at 128K elements (≈ 1 transformer head's
    // sequence on a small model) — below this the MLX composite is
    // already fine.  Tune in Phase 4 if needed.
    return numel >= (1LL << 17);
}

bool should_dispatch_gelu_exact_metal(std::int64_t numel, Dtype dt) {
    (void)numel;
    (void)dt;
    // Disabled by default — the inlined Abramowitz polynomial erf
    // (~14 ops + exp) makes the per-call compute 2× heavier than the
    // MLX erf chain, and the bench shows a regression vs MLX on
    // ffn_big-scale shapes.  Kernel kept for future SDK versions
    // where Metal exposes a native ``erf`` intrinsic.
    return false;
}

bool should_dispatch_gelu_exact(std::int64_t numel, Dtype dt) {
    (void)numel;
    (void)dt;
    // 2026-05-25 rebench: the "13-30×" gap was measuring MLX lazy
    // graph construction (perf-baseline-rebench-2026-05-25.md).  Real
    // gap on properly-evaled bench is ~1.7×, and our MPSGraph 10-op
    // erf composite doesn't measurably help vs MLX.  Off by default;
    // kept compilable for future SDK reactivation.
    return false;
}

bool should_dispatch_layer_norm_backward(std::int64_t outer,
                                         std::int64_t normalized_size,
                                         Dtype dt) {
    (void)outer;
    (void)normalized_size;
    (void)dt;
    // 2026-05-25 rebench: Lucid MLX LayerNorm fwd is 1.5× **faster**
    // than the reference framework's MPS path on llama-scale
    // (perf-baseline-rebench-2026-05-25) and the bwd dispatch shows
    // noise-level differences vs MLX (1.02×) per the dispatch audit.
    // Off by default; kept compilable.
    return false;
}

bool should_dispatch_batch_norm_train(std::int64_t numel, Dtype dt) {
    (void)numel;
    (void)dt;
    // 2026-05-25 rebench: the 5.5× gap was lazy-graph artifact; real
    // gap is 2.8× but our MPSGraph dispatch shows noise vs MLX (1.00×
    // per dispatch audit).  See perf-bn-train-gap-deferred-2026-05-25
    // — closing the gap needs a custom Metal kernel, not MPSGraph.
    // Off by default; kept compilable for future SDK or BNNS route.
    return false;
}

bool should_dispatch_bn_train_metal(std::int64_t per_channel_numel,
                                    Dtype dt) {
    (void)per_channel_numel;
    (void)dt;
    if (!enabled()) return false;
    // Measurement note (2026-05-25, M4 Max F32):
    //
    //   shape            custom Metal   MLX-only   verdict
    //   (8,64,256²)      11.81 ms       10.50 ms   regression
    //   (32,64,56²)      2.25 ms        2.28 ms    parity
    //
    // The 2-pass design (reduce → normalize) pays an unavoidable
    // 2× kernel-dispatch + sync overhead.  Combined with MLX's
    // own simdgroup-optimised reduction primitive, my custom path
    // loses on the big-tile shape that this work was intended to
    // help.  Higher thread counts (256 → 1024) didn't recover the
    // gap.  Closing it likely needs a 1-pass cooperative kernel
    // using float atomic-add accumulators (Metal 3+) — out of
    // scope for now.
    //
    // Kernel pair kept in MpsKernels.mm as reference; default OFF.
    return false;
}

bool should_dispatch_silu_backward(std::int64_t numel, Dtype dt) {
    (void)numel;
    (void)dt;
    // 2026-05-25 rebench: silu_bwd 2.7× vs reference framework (real
    // but mild); the existing dispatch shows 1.11× vs MLX (noise) per
    // dispatch audit.
    // Off by default; pattern same as silu_fwd / gelu_fwd — needs
    // custom Metal kernel to close further, MPSGraph route is dead.
    return false;
}

bool should_dispatch_silu_metal(std::int64_t numel, Dtype dt) {
    (void)numel;
    (void)dt;
    if (!enabled()) return false;
    // Measurement (2026-05-25, M4 Max F32 ffn-scale):
    //   (8,1024,768)   fwd  Custom 1.82 ms  MLX 1.14 ms  (1.6× slower)
    //   (8,1024,3072)  fwd  Custom 6.84 ms  MLX 3.67 ms  (1.9× slower)
    //   bwd numbers parallel (custom 1.5-1.6× slower than MLX).
    //
    // Pattern note: custom Metal beats MLX on **deep** elementwise
    // composites (GELU has 9 ops, custom Metal wins 1.9×).  SiLU is
    // only 2 ops (sigmoid * x); MLX kernel-fuses both into a single
    // kernel that's well-tuned for M-series, leaving no room for our
    // single-pass custom kernel to amortise its dispatch overhead.
    //
    // Kernels kept in MpsKernels.mm as reference; default OFF.
    return false;
}

bool should_dispatch_softmax_backward(std::int64_t axis_size, Dtype dt) {
    (void)axis_size;
    (void)dt;
    // Tried Phase 4 — disabled.  Measurement on dev box (4096, 50257) F32:
    //   MLX chain: 51 ms  MPSGraph canonical: 67 ms (0.76× slower)
    //   MPSGraph hand-rolled: 64 ms (0.71× slower)  reference: 19 ms
    // MPSGraph framework overhead (executable lookup + 800 MB MTLBuffer
    // alloc + run) exceeds the per-call kernel saving for very large axes.
    // MLX's lazy chain is already memory-bandwidth-bound and competitive.
    // Closing the residual 3× reference-framework gap would need a custom
    // Metal compute kernel — out of MPSGraph's wheelhouse.  Kernel kept
    // (in MpsKernels) as reference for the canonical signature.
    return false;
}

bool should_dispatch_embedding_backward(std::int64_t M_total,
                                        std::int64_t D,
                                        Dtype dt) {
    (void)M_total;
    (void)D;
    (void)dt;
    if (!enabled()) return false;
    // Phase 4 fresh measurement (2026-05-25, M4 Max, gpt2-input
    // 8×1024×768) found the MPSGraph ``scatterWithDataTensor:`` path
    // 3–10× **slower** than MLX's ``scatter_add_axis`` once the
    // benchmark forces gradient consumption per iteration:
    //
    //   gpt2-input         MLX 1.72 ms   MPSGraph 6.72 ms   (3.9× slower)
    //   gpt2-input-pad     MLX 1.98 ms   MPSGraph 7.31 ms   (3.7× slower)
    //   vocab-input        MLX 2.30 ms   MPSGraph 18.09 ms  (7.9× slower)
    //   large_emb (50K×1K) MLX 3.35 ms   MPSGraph 36.05 ms  (10.7× slower)
    //
    // The Phase 0 baseline (perf-mlx-op-baseline-2026-05.md) that
    // reported MLX 28× slower than the reference framework did NOT
    // force grad eval, so MLX's lazy graph skipped the scatter
    // entirely.  Once eval is forced the MLX path is already faster
    // than the reference framework's MPS path (ref 1.25 ms for
    // gpt2-input).
    //
    // Kernel kept in MpsKernels for two reasons: (a) opt-in
    // ``LUCID_MPS_EMBEDDING_BWD=1`` for future SDK regressions, (b)
    // documentation of the canonical MPSGraph scatter pattern.  Default
    // dispatch is OFF.
    const char* force = std::getenv("LUCID_MPS_EMBEDDING_BWD");
    return force && std::string(force) == "1";
}

}  // namespace lucid::gpu::mps
