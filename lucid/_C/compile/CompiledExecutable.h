// lucid/_C/compile/CompiledExecutable.h
//
// Opaque C++ handle wrapping an MPSGraph executable + its I/O ordering
// plan.  The wrapped Objective-C object (``MPSGraphExecutable*``) lives
// in :file:`CompiledExecutable.mm` and is hidden behind ``void*`` so
// pure-C++ headers (e.g. :file:`ExecutableCache.h`) can manipulate
// ``CompiledExecutable*`` without pulling in MPSGraph headers or the
// Objective-C runtime.
//
// Lifetime
// --------
// The :class:`ExecutableCache` is the canonical owner: every executable
// inserted into the cache is destroyed via :func:`destroy_executable`
// when the cache evicts the entry (LRU overflow) or clears.  Direct
// callers that bypass the cache (e.g. the Phase 1.2 smoke harness) must
// call :func:`destroy_executable` themselves.

#pragma once

#include <cstddef>
#include <vector>

#include "../api.h"
#include "../core/fwd.h"  // TensorImplPtr (used by run_executable)
#include "TraceIR.h"

namespace lucid::compile {

// Opaque handle.  Full definition lives in
// :file:`CompiledExecutable.mm` so this header stays MPSGraph-free.
class CompiledExecutable;

// Number of inputs the executable expects, in feed order.
//
// Returns ``0`` when ``exe`` is ``nullptr``.  Useful for binding-layer
// validation before invoking :func:`run_executable`.
LUCID_API std::size_t executable_num_inputs(const CompiledExecutable* exe);

// Number of outputs the executable produces, in target order.
//
// Returns ``0`` when ``exe`` is ``nullptr``.
LUCID_API std::size_t executable_num_outputs(const CompiledExecutable* exe);

// Feed-order :type:`TensorId` list (returns a copy so callers can keep
// the result alive past the executable's lifetime).  Empty when
// ``exe`` is ``nullptr``.
LUCID_API std::vector<TensorId> executable_input_ids(const CompiledExecutable* exe);

// Target-order :type:`TensorId` list — see :func:`executable_input_ids`.
LUCID_API std::vector<TensorId> executable_output_ids(const CompiledExecutable* exe);

// Phase 1.3: when the executable was built via :func:`compile_trace_with_backward`
// it also holds the auto-derived gradient tensors (one per parameter in the
// order they were passed in).  Returns an empty vector for forward-only
// executables.  The grad ids are minted by the builder so they do not
// overlap with anything in the trace's own id space.
LUCID_API std::vector<TensorId> executable_grad_output_ids(const CompiledExecutable* exe);

// Release a previously-built executable.  Safe to call with
// ``nullptr``.  Defined in :file:`CompiledExecutable.mm`; releases the
// ARC-strong reference held on the wrapped ``MPSGraphExecutable*``.
LUCID_API void destroy_executable(CompiledExecutable* exe);

// Run a compiled executable.
//
// Parameters
// ----------
// exe : CompiledExecutable*
//     Executable returned by :func:`compile_trace`; must be non-null.
// input_feeds : const std::vector<TensorImplPtr>&
//     Inputs in the executable's ``input_ids`` order.  Each feed must
//     live on the same device as the executable; the GPU path
//     materialises the underlying ``MTLBuffer`` via the MLX bridge
//     and binds it as an ``MPSGraphTensorData``.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     Freshly-allocated output tensors in ``output_ids`` order.  Each
//     output's ``GpuStorage`` wraps a new ``MTLBuffer`` allocated by
//     the executable run — no copy back to MLX.
//
// Notes
// -----
// Implementation lives in :file:`CompiledExecutable.mm` (Objective-C++)
// so all MPSGraph types stay in one translation unit.
LUCID_API std::vector<TensorImplPtr> run_executable(CompiledExecutable* exe,
                                                    const std::vector<TensorImplPtr>& input_feeds);

// In-place variant: instead of allocating fresh MTLBuffers per output
// and returning new TensorImpls, write each output directly into the
// corresponding pre-existing TensorImpl's GPU buffer.  Used by the
// compiled-optimizer step path where every output is a parameter (or
// optimizer-state buffer) that already has an allocated MTLBuffer;
// skipping the alloc + copy_ round-trip cuts the wrapper's overhead
// by ~50%.
//
// Parameters
// ----------
// exe : CompiledExecutable*
//     The compiled forward (or forward+backward) graph.
// input_feeds : vector<TensorImplPtr>
//     Inputs in ``input_ids`` order — same as :func:`run_executable`.
// output_targets : vector<TensorImplPtr>
//     Pre-existing tensors whose buffers receive the executable's
//     outputs.  Must equal ``output_ids.size() + grad_output_ids.size()``
//     in length, and each target must match its slot's shape and dtype
//     exactly (no broadcasting).
//
// Notes
// -----
// Not supported in dynamic-batch mode (output shapes vary per call,
// no stable target buffers).  Each target's ``bump_version`` is
// called after the write so autograd's mutation tracker sees the
// update.  The target's MLX array reference is replaced with a
// fresh one wrapping the same buffer to invalidate any cached MLX
// dependency state.
LUCID_API void run_executable_inplace(CompiledExecutable* exe,
                                      const std::vector<TensorImplPtr>& input_feeds,
                                      const std::vector<TensorImplPtr>& output_targets);

// ── Disk cache (Tier 1-A) ───────────────────────────────────────────
//
// Persist a compiled :class:`CompiledExecutable` to disk so subsequent
// process invocations can skip the per-signature MPSGraph compile step
// (typically 30-100ms per executable on M-series).  Two files are
// written:
//
//   * ``<path>.mpsgraphpackage`` — the Apple-native serialised
//     :class:`MPSGraphExecutable`.
//   * ``<path>.meta`` — Lucid-side I/O metadata (input_ids, shapes,
//     dtypes, device, dynamic_batch flag, static_feed_slots,
//     output_ids, grad_output_ids).  Hand-rolled binary format —
//     length-prefixed fields, little-endian, no compression.  Format
//     version byte lives in the first byte of the file so future
//     Lucid releases can detect + skip incompatible caches.
//
// Cache lifecycle is the caller's responsibility — these primitives
// just write / read.  Higher-level orchestration (filename hashing,
// invalidation, eviction) lives in the binding layer.
//
// Notes
// -----
// * ``save_executable`` overwrites any existing files at the same
//   path without warning.  Atomicity: writes to ``.tmp`` siblings
//   first, then renames into place.  Failure leaves the previous
//   cache (if any) untouched.
// * ``load_executable`` returns ``nullptr`` on any I/O / parse /
//   ABI-mismatch error, setting ``error_msg`` to a one-line
//   diagnostic.  Callers should treat this as a cache miss and fall
//   through to a fresh compile.

// Serialise ``exe`` to ``<path>.mpsgraphpackage`` + ``<path>.meta``.
// Returns ``true`` on success, ``false`` on any failure (filesystem
// error, MPSGraph serialise failure, null exe).
LUCID_API bool save_executable(const CompiledExecutable* exe, const std::string& path);

// Reload an executable previously saved via :func:`save_executable`.
// Returns a fresh ``CompiledExecutable*`` (caller owns) on success,
// or ``nullptr`` on miss / corruption.  Callers MUST treat any
// failure as a cache miss + compile path fallback.
LUCID_API CompiledExecutable* load_executable(const std::string& path,
                                              std::string* error_msg = nullptr);

}  // namespace lucid::compile
