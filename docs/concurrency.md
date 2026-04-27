# Lucid Concurrency Contract

This document defines the thread-safety guarantees of the Lucid C++ engine.
It is part of Phase 2.5 (production foundations) and applies to every op,
optimizer, and engine path landed in Phase 3 onward.

## TL;DR

- **Forward**: thread-safe IFF each thread holds its own `Generator` and
  shares no `TensorImpl` writes.
- **Backward**: single-threaded per root tensor. Concurrent `.backward()`
  calls are safe IFF they operate on disjoint computation graphs.
- **`GradMode` (`no_grad()`)**: thread-local. Setting on thread A does not
  affect thread B.
- **`Determinism` (`set_deterministic(True)`)**: process-global; intentional.
- **`MemoryStats`**: atomic counters; always safe to query from any thread.
- **Python GIL**: pybind11 holds the GIL on entry to every binding. The C++
  engine itself does not assume the GIL — long-running kernels should release
  it via `py::gil_scoped_release` (Phase 3 op base does this).

## Detailed invariants

### TensorImpl

A single `TensorImpl` instance is **not** thread-safe under concurrent writes.
Concurrent reads of `data_as_python()` / `grad_as_python()` are safe as long
as no other thread is mutating the underlying storage. In practice:

- Forward in thread A while another thread writes `tensor.data` → undefined.
- Two threads reading the same tensor's `.shape` / `.dtype` → safe.
- Two threads each running independent forward graphs → safe.

This matches PyTorch's contract.

### Autograd Engine

`Engine::backward(root)` walks the topological order of `root`'s graph and
invokes each Node's `apply()`. The engine is **not** internally parallel
(Phase 2 ships single-threaded backward; Phase 6 JIT may add parallel
execution within fused subgraphs, gated by `set_deterministic`).

Two threads each calling `Engine::backward` on **disjoint** roots is safe.
Two threads on the same root is undefined (and would double-accumulate
gradients into shared leaves).

### GradMode

`thread_local bool` — `no_grad()` on thread A does not propagate to thread B.
Each thread's RAII `NoGradGuard` restores its own thread's previous value.

### Determinism

Process-global atomic flag. Setting `set_deterministic(True)` on any thread
takes effect immediately for all subsequent op invocations on every thread.
The flag is a user-intent declaration, not an execution mode — flipping it
mid-computation is allowed but not recommended.

### Generator

A single `Generator` instance is **not** thread-safe. The `mutex()` accessor
exists for the unusual case where threads must share one. Standard pattern:
each thread allocates its own `Generator` with its own seed.

The `default_generator()` returned for `lucid.random.seed(n)` is shared
across threads — accesses are serialized via its embedded mutex
automatically by every random op in Phase 3.8.

### MemoryStats

Per-device `std::atomic<size_t>` counters, updated from `Allocator`'s alloc
and shared_ptr deleter paths. Always safe to query. The peak update uses a
CAS loop (lock-free, monotone).

## What the engine does NOT promise

- **Determinism without `set_deterministic(True)`**: when the flag is off,
  ops are free to use non-associative parallel reductions, atomic-add
  scatter, etc. Bit-for-bit reproducibility requires the flag.
- **Cross-process determinism**: the `Generator` produces the same sequence
  for the same seed on the same machine, but cross-architecture (e.g. CPU vs
  GPU) bit-identity is not guaranteed.
- **Lock-free op execution**: ops may use locks internally (e.g. random ops
  on `default_generator()`); contention is the caller's problem.

## Debug-build assertions

In `LUCID_BUILD_MODE=debug` / `debug-asan` / `debug-ubsan` builds, the
engine adds runtime assertions:

- `Engine::backward` asserts no re-entrant call on the same thread for the
  same root.
- `TensorImpl` debug builds may add a single-writer detector (Phase 4+).

These do not run in release builds — production has zero overhead from the
contract.
