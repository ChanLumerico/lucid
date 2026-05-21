// lucid/_C/backend/gpu/mps/MpsDispatch.h
//
// Per-op dispatch policy.  Each `should_dispatch_*` returns true iff the
// MPSGraph kernel should handle the call, false iff the existing MLX path
// should run.  Heuristics start as either "always-true" (for ops with
// universal wins like GELU) or "shape-gated" (for ops where the gap
// matters only on large activations).  Tuned via Phase 4 measurements.
//
// All functions are plain C++ — no Obj-C dependency.

#pragma once

#include <cstdint>
#include "../../../core/Dtype.h"

namespace lucid::gpu::mps {

// One-time init: reads LUCID_MPS_DISABLE.  If set, all dispatch returns
// false.  Defaults to enabled.  Thread-safe.
bool enabled();

// One-time init: reads LUCID_MPS_DEBUG.  When true, dispatch decisions are
// logged to stderr.  Defaults to false.
bool debug_enabled();

// Universal-win ops — no shape gate needed because the kernel-quality
// difference is large across all measured shapes.
bool should_dispatch_gelu(std::int64_t numel, Dtype dt);
bool should_dispatch_gelu_exact(std::int64_t numel, Dtype dt);

// LayerNorm backward — dispatched when `normalized_size` is large enough
// to amortize the MPSGraph dispatch overhead.  Sub-1ms small-normalize
// cases (transformer Q/K/V) should keep using MLX since the kernel
// launch cost dominates.  Phase 4 will tune the threshold from measurement.
bool should_dispatch_layer_norm_backward(std::int64_t outer,
                                         std::int64_t normalized_size,
                                         Dtype dt);

// BatchNorm train fwd + bwd — MLX has no fused BN primitive; for very
// large activations MPSGraph's `normalizationWithTensor:...` is the only
// way to close the gap.  ResNet shapes (≤ 2M numel) hit MLX's reduction
// kernels at roughly torch parity, so dispatch only large activations
// (e.g. ImageNet-scale 16×64×112×112).  Phase 0 baseline: 5.5× torch on
// `large_acts`, 1× on ResNet shapes.
bool should_dispatch_batch_norm_train(std::int64_t numel, Dtype dt);

// Softmax backward — the 4-op MLX chain (gz / sum / diff / result) is
// memory-bound on large axes (CE @ vocab=50257 allocates ~800MB).  Phase 0
// measured 3× torch on ce_gpt2_logits, ~1× on small (attention).  Gate
// at axis_size >= 1024 to skip small / attention-shape softmaxes.
bool should_dispatch_softmax_backward(std::int64_t axis_size, Dtype dt);

// SiLU backward — same 7-op chain shape as GELU; benefits from the same
// MPSGraph fusion.  Universal dispatch (Phase 0 showed gap across all
// measured shapes).
bool should_dispatch_silu_backward(std::int64_t numel, Dtype dt);

// Shape-gated ops — only dispatch when the gap exceeds dispatch overhead.
// Heuristics come from obsidian/perf/perf-mpsgraph-shortlist-2026-05.md
// and tighten in Phase 4.

}  // namespace lucid::gpu::mps
