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

// Return whether MPS dispatch is globally enabled this process.
//
// Reads ``LUCID_MPS_DISABLE`` from the environment exactly once on
// first call.  When set (to any non-empty value), every
// :func:`should_dispatch_*` heuristic short-circuits to ``false`` and
// all ops fall back to MLX.  Defaults to enabled.
//
// Returns
// -------
// bool
//     ``true`` iff MPS dispatch is enabled.
//
// Notes
// -----
// Thread-safe; the env-read is guarded by a one-shot flag.
bool enabled();

// Return whether MPS dispatch decisions should be logged to ``stderr``.
//
// Reads ``LUCID_MPS_DEBUG`` from the environment once.  When ``true``,
// each ``should_dispatch_*`` call emits a line indicating whether the
// MPS or MLX path was selected (and why, for shape-gated ops).
//
// Returns
// -------
// bool
//     ``true`` iff MPS debug logging is enabled.
bool debug_enabled();

// Always dispatch GELU (tanh approximation) through MPSGraph.
//
// The MLX path for tanh-approximation GELU is a 7-op composite that
// is memory-bound on every measured shape.  MPSGraph fuses these
// into a single activation node — universal win.
//
// Parameters
// ----------
// numel : std::int64_t
//     Total element count of the input.  Currently unused (universal
//     dispatch) but kept in the signature for future gating.
// dt : Dtype
//     Element dtype.  Currently unused.
//
// Returns
// -------
// bool
//     ``true`` if MPS dispatch is enabled at process scope.
//
// See Also
// --------
// :func:`should_dispatch_gelu_exact` — Gaussian-CDF variant.
bool should_dispatch_gelu(std::int64_t numel, Dtype dt);

// Dispatch the GELU (tanh approximation) through a custom Metal
// compute kernel instead of the 9-op MPSGraph composite.  The
// composite produces no measurable speedup vs MLX on M-series; a
// single-pass MSL kernel matches the reference framework's MPS path
// (~4× win on transformer-scale activations).
//
// Parameters
// ----------
// numel : std::int64_t
//     Total element count.  Small inputs (under ~128K elements) skip
//     dispatch because the compute-pass setup cost dominates.
// dt : Dtype
//     Element dtype.  Only ``F32`` is currently supported by the MSL
//     kernel; other dtypes return ``false`` and fall back to the MLX
//     composite.
//
// Returns
// -------
// bool
//     ``true`` if the custom Metal path should be used.
//
// See Also
// --------
// :func:`gelu_metal_forward` / :func:`gelu_metal_backward` — the
// kernels.
bool should_dispatch_gelu_metal(std::int64_t numel, Dtype dt);

// Same as :func:`should_dispatch_gelu_metal` but for the exact
// (erf-based) GELU variant — the default ``F.gelu(x)`` path.
bool should_dispatch_gelu_exact_metal(std::int64_t numel, Dtype dt);

// Always dispatch GELU exact (Gaussian-CDF) through MPSGraph.
//
// The MLX path composes erf via a 10-op approximation chain; MPSGraph
// has a single fused node.  Universal win.
//
// Parameters
// ----------
// numel : std::int64_t
//     Total element count of the input.  Currently unused.
// dt : Dtype
//     Element dtype.  Currently unused.
//
// Returns
// -------
// bool
//     ``true`` if MPS dispatch is enabled at process scope.
bool should_dispatch_gelu_exact(std::int64_t numel, Dtype dt);

// Dispatch LayerNorm backward through MPSGraph for large normalize sizes.
//
// The fused MPSGraph executable producing ``(dx, dgamma, dbeta)``
// from saved tensors is large enough that its launch overhead
// dominates for small ``normalized_size`` (e.g. transformer
// ``Q/K/V`` projections), so the heuristic shape-gates on that
// dimension.  Phase-4 tuning will lock in the exact threshold.
//
// Parameters
// ----------
// outer : std::int64_t
//     Product of leading (non-normalised) dimensions.
// normalized_size : std::int64_t
//     Size of the trailing (normalised) dimension.  The gate.
// dt : Dtype
//     Element dtype.  Reserved for future dtype-specific tuning.
//
// Returns
// -------
// bool
//     ``true`` iff MPS dispatch is enabled and ``normalized_size`` is
//     large enough to amortise the dispatch cost.
//
// See Also
// --------
// :func:`layer_norm_backward` — the kernel that gets dispatched.
bool should_dispatch_layer_norm_backward(std::int64_t outer,
                                         std::int64_t normalized_size,
                                         Dtype dt);

// Dispatch BatchNorm train (fwd + bwd) through MPSGraph for large activations.
//
// MLX has no fused BatchNorm primitive; for very large activations
// (e.g. ImageNet-scale 16×64×112×112 ≈ 12.8M elements) the
// MPSGraph ``normalizationWithTensor:`` kernel is the only way to
// close the parity gap.  ResNet-scale shapes (≤ ~2M numel) hit
// MLX's per-axis reduction kernels at roughly reference-framework
// parity, so they keep the MLX path.  Phase 0 measured 5.5× the
// reference on ``large_acts``, 1× on ResNet shapes.
//
// Parameters
// ----------
// numel : std::int64_t
//     Total element count of the activation tensor.  The gate.
// dt : Dtype
//     Element dtype.  Reserved for future dtype-specific tuning.
//
// Returns
// -------
// bool
//     ``true`` iff MPS dispatch is enabled and ``numel`` clears the
//     large-activation threshold.
//
// See Also
// --------
// :func:`batch_norm_train_forward`, :func:`batch_norm_train_backward`.
bool should_dispatch_batch_norm_train(std::int64_t numel, Dtype dt);

// Dispatch the BN training forward through a 2-pass custom Metal
// kernel pair (reduce + normalize) instead of the MPSGraph
// composite.  Targets the 2.5–2.8× gap to the reference framework's
// MPS path on ImageNet-scale activations
// (perf-baseline-rebench-2026-05-25).
//
// Parameters
// ----------
// per_channel_numel : std::int64_t
//     ``N × H × W`` — elements reduced into each channel.  Threshold
//     here so tiny channel-tiles stay on MLX where the dispatch
//     setup cost dominates.
// dt : Dtype
//     Element dtype.  Only ``F32`` is wired by the custom kernel;
//     other dtypes fall back to the MPSGraph composite.
//
// Returns
// -------
// bool
//     ``true`` iff the custom Metal path should be used.
//
// See Also
// --------
// :func:`bn_train_metal_forward` — the kernel pair.
bool should_dispatch_bn_train_metal(std::int64_t per_channel_numel, Dtype dt);

// Dispatch Softmax backward through MPSGraph for large reduction axes.
//
// The MLX path is a 4-op chain (``gz / sum / diff / result``) that
// is memory-bound on large axes (e.g. cross-entropy at GPT-2's
// vocab=50257 allocates ~800 MB of intermediate tensors).  Phase 0
// measured 3× the reference on ``ce_gpt2_logits`` and ~1× on
// attention-shape softmaxes.  The heuristic gates at
// ``axis_size >= 1024`` to skip small / attention shapes.
//
// Parameters
// ----------
// axis_size : std::int64_t
//     Size of the softmax reduction axis.  The gate.
// dt : Dtype
//     Element dtype.  Reserved for future dtype-specific tuning.
//
// Returns
// -------
// bool
//     ``true`` iff MPS dispatch is enabled and ``axis_size`` clears
//     the threshold.
//
// See Also
// --------
// :func:`softmax_backward` — the kernel that gets dispatched.
bool should_dispatch_softmax_backward(std::int64_t axis_size, Dtype dt);

// Always dispatch SiLU backward through MPSGraph.
//
// The MLX backward composes ``sigmoid + multiply + add + multiply
// + ...`` (~7 ops) with the same memory-bound profile as GELU; the
// MPSGraph fusion is a universal win across all measured shapes.
//
// Parameters
// ----------
// numel : std::int64_t
//     Total element count of the input.  Currently unused.
// dt : Dtype
//     Element dtype.  Currently unused.
//
// Returns
// -------
// bool
//     ``true`` if MPS dispatch is enabled at process scope.
//
// See Also
// --------
// :func:`silu_backward` — the kernel that gets dispatched.
bool should_dispatch_silu_backward(std::int64_t numel, Dtype dt);

// Dispatch SiLU forward / backward through a custom Metal compute
// kernel (float4-vectorised, single pass).  Same shape gate as the
// GELU Metal path: F32 only, numel ≥ 128K and numel % 4 == 0.
bool should_dispatch_silu_metal(std::int64_t numel, Dtype dt);

// Dispatch the embedding backward (``scatter_add`` along axis 0) to the
// MPSGraph ``MPSGraphScatterModeAdd`` primitive instead of MLX's
// ``scatter_add_axis`` composition.  MLX's path is 14 – 28× slower on
// GPT-style inputs (``perf-mlx-op-baseline-2026-05.md`` row
// ``embedding / gpt2_input / bwd``: 34.73 ms vs ref 1.25 ms).  Gate
// keeps tiny embeddings on MLX where MPS dispatch overhead would
// dominate the scatter cost.
//
// Parameters
// ----------
// M_total : std::int64_t
//     Number of scatter rows (``product(indices_shape)``).
// D : std::int64_t
//     Embedding dimension — second axis of the weight table.
// dt : Dtype
//     Element dtype of the gradient.
//
// Returns
// -------
// bool
//     ``true`` to route through :func:`embedding_backward`.  Heuristic:
//     enabled iff ``M_total × D >= 1M`` elements (covers GPT-2 input
//     bwd 8 × 1024 × 768 ≈ 6.3M and falls back to MLX for small
//     embedding tables where dispatch overhead exceeds the win).
//
// See Also
// --------
// :func:`embedding_backward` — the kernel that gets dispatched.
bool should_dispatch_embedding_backward(std::int64_t M_total, std::int64_t D, Dtype dt);

// Shape-gated ops — only dispatch when the gap exceeds dispatch overhead.
// Heuristics come from obsidian/perf/perf-mpsgraph-shortlist-2026-05.md
// and tighten in Phase 4.

}  // namespace lucid::gpu::mps
