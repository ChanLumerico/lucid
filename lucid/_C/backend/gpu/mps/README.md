# lucid/_C/backend/gpu/mps/

MPSGraph dispatch layer.  Coexists with the MLX path inside the same
`GpuBackend`; per-op `should_dispatch_*` decides which path runs.

## Files

- `MpsBridge.h` / `MpsBridge.mm` — process-wide MTLDevice + MTLCommandQueue;
  `array_to_buffer` / `buffer_to_array` round-trip primitives.  The `.h` is
  plain C++ (opaque `void*` for Obj-C types); the `.mm` uses Obj-C++.
- `MpsDispatch.h` / `MpsDispatch.cpp` — per-op heuristics.  Pure C++.
  Reads `LUCID_MPS_DISABLE` / `LUCID_MPS_DEBUG` env vars.
- `MpsKernels.h` / `MpsKernels.mm` — one function per shortlisted op.

## Rules

- **No Obj-C in `.h`** — only `.mm` files include MetalPerformanceShadersGraph.
  Headers expose `void*` opaque handles; callers cast via `__bridge` in `.mm`.
- **Same MTLDevice as MLX** — `MpsBridge::shared_mtl_device()` returns
  whatever `mlx::core::metal::device(Device::gpu).mtl_device()` returned.
  Both stacks operate on the same buffers; no cross-device copies needed.
- **Distinct MTLCommandQueue** — we run on `g_queue` (label
  "lucid-mps-dispatch"), MLX runs on its own.  Cross-queue safety is
  handled by `arr.wait()` before extracting MTLBuffer (forces MLX kernel
  completion) and by `[mpsCB waitUntilCompleted]` before returning a buffer
  to MLX (forces MPSGraph kernel completion).  Optimisable in Phase 5.3
  via MTLSharedEvent.
- **ARC for `.mm` files** — set via `-fobjc-arc` in CMakeLists.txt.

## Design references

- `obsidian/engine/engine-mps-bridge-2026-05.md` — feasibility spike.
- `obsidian/perf/perf-mpsgraph-shortlist-2026-05.md` — which ops graduate.
- 3.4 plan in `~/.claude/plans/recursive-scribbling-moore.md`.
