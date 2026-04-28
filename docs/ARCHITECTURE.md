# Lucid C++ Engine — Architecture

This is the load-bearing design document. Every PR that adds a header,
moves a file, or changes a layer dependency must update this file.

## Layer diagram

```
┌──────────────────────────────────────────────────────┐
│ Python API (lucid.Tensor / lucid.nn / lucid.optim)   │  ← Phase 5
├──────────────────────────────────────────────────────┤
│ pybind11 boundary (lucid._C.engine)                  │
├──────────────────────────────────────────────────────┤
│ bindings/  ←  jit/                                   │  Phase 6: jit/
│            ↖                                         │
│ optim/                                                │  Phase 4: optim/
│   ↑                                                   │
│ backend/cpu/, backend/gpu/                            │  Phase 3.0: backend/
│   ↑                                                   │
│ autograd/                                             │  Phase 2: autograd/
│   ↑                                                   │
│ core/                                                 │  Phase 1+1.5+2.5: core/
└──────────────────────────────────────────────────────┘
```

**Dependency rule**: layer `X` may include from layer `Y` only if `X` is
**above** `Y` in this diagram. There are no upward includes. If you find
yourself wanting one, the abstraction probably belongs in a lower layer.

## Modules

### core/  (Phase 1, 1.5, 2.5)

The vocabulary types — no logic, just data + tiny helpers.

| Header | Purpose |
|---|---|
| `Allocator.h` | 64-byte aligned `posix_memalign` + `MemoryTracker` integration |
| `Determinism.h` | process-global `set_deterministic` flag |
| `Device.h` | `enum Device { CPU, GPU }` |
| `Dtype.h` | `enum Dtype` (9 values) + size/name helpers |
| `Error.h` | `LucidError` hierarchy |
| `fwd.h` | forward declarations for all public types |
| `Generator.h` | Philox-4x32-10 RNG |
| `GradMode.h` | thread-local grad-enabled + `NoGradGuard` |
| `MemoryStats.h` | per-device alloc/peak counters |
| `Shape.h` | `Shape`, `Stride` typedefs + `shape_numel`, `contiguous_stride` |
| `Storage.h` | `CpuStorage`, `GpuStorage` (Phase 3 fills GPU), `Storage = variant` |
| `TensorImpl.h` | the load-bearing tensor: storage + shape + dtype + grad_fn |

### autograd/  (Phase 2)

The dependency graph and its traversal.

| Header | Purpose |
|---|---|
| `Node.h` | abstract Node + `Edge { weak_ptr<Node>, input_nr }` |
| `AccumulateGrad.h` | leaf sentinel: writes grad into `TensorImpl::grad_storage_` |
| `Engine.h` | `Engine::backward(root, retain_graph)` — DFS topo sort + reverse traversal |
| `Helpers.h` | `make_zero_storage`, `make_ones_storage`, `reduce_grad_to_shape`, `accumulate_into` |
| `FuncOp.h` (Phase 3.0) | CRTP base for ops: `template<class D, int N_IN> struct FuncOp : Node` |
| `ops/binary/`, `ops/unary/`, `ops/reduce/`, `ops/shape/`, `ops/nn/`, `ops/norm/`, `ops/sparse/` | concrete op subclasses |

### backend/  (Phase 3.0)

Math kernels. Thin wrappers over Accelerate / MLX that take and return
`Storage`. **No autograd here** — these are pure functions.

| Subdir | Purpose |
|---|---|
| `backend/cpu/Blas.h` | `cblas_sgemm`, `cblas_sgemv` |
| `backend/cpu/Vdsp.h` | `vDSP_vadd`, `vDSP_vmul`, … |
| `backend/cpu/Vforce.h` | `vvexpf`, `vvlogf`, `vvsqrtf`, `vvtanhf` |
| `backend/cpu/Im2Col.h` | im2col / col2im for conv2d |
| `backend/cpu/CpuOps.h` | top-level dispatch (dtype switch) |
| `backend/gpu/MlxOps.h` | `mlx::core::*` wrappers |

### optim/  (Phase 4)

Optimizers as native C++ objects holding their state in `Storage`.

| Header | Purpose |
|---|---|
| `Optimizer.h` | abstract base (`step(params, grads)`) |
| `SGD.h`, `Adam.h`, `AdamW.h`, `RMSProp.h`, … | concrete optimizers |
| `GradScaler.h` (Phase 3.5) | fp16 loss scaling |

### jit/  (Phase 6)

Graph capture and replay. Built on the autograd Node hierarchy.

| Header | Purpose |
|---|---|
| `Tracer.h` | `TraceGuard` RAII; records ops as they execute |
| `IRGraph.h` | flatbuffers-friendly IR mirroring the Python IR |
| `CompiledPlan.h` | cache key, plan layout |
| `ForwardExecutor.h`, `BackwardExecutor.h` | execute IR without re-entering Python |

### bindings/

pybind11 glue. Each `register_*` function exposes one module's API.
Bindings depend on **everything** but nothing depends on bindings — they're
the leaf of the layering DAG.

## Key invariants

1. **`TensorImpl::storage_` is the single source of truth** for tensor data.
   Anything reading it must go through `data_as_python()` (zero-copy view) or
   `copy_from()` (mutation).
2. **Autograd nodes don't own tensors.** They hold `weak_ptr<TensorImpl>`
   (via `AccumulateGrad`) and `weak_ptr<Node>` (via `Edge`). Tensors stay
   alive only as long as the user / Python wrapper holds them.
3. **`Storage` is moveable, never reference-counted directly.** Reference
   counting happens at the `TensorImpl` level. `Storage` is moved into
   `TensorImpl`'s constructor and out via `std::move`.
4. **Every op declares its `OpSchema`** at module-init via a static initializer
   (Phase 3.0). Schemas are immutable post-init.
5. **No layer above `core/` may store `py::object`.** Pybind types live only
   in `bindings/` and the boundary functions of `core/TensorImpl` (`data_as_python`).

## Adding a new op (Phase 3+)

Checklist for every PR that adds an op:

- [ ] Header in `autograd/ops/<family>/<name>.h` with full Doxygen contract block.
- [ ] CRTP subclass `<Name>Backward : <Family>Op<<Name>Backward>`.
- [ ] Implementation in `backend/cpu/<Family>Ops.cpp` and (if applicable)
      `backend/gpu/MlxOps.cpp`.
- [ ] `OpSchema` registered with `name`, `version=1`, dtype matrix, attrs.
- [ ] `AmpPolicy` declared (`Promote` / `KeepInput` / `ForceFP32`).
- [ ] Determinism tag (`deterministic` or `nondeterministic-fast` with rationale).
- [ ] Pybind11 free function in `bindings/bind_ops_<family>.cpp`.
- [ ] Parity test in `lucid/test/parity/cpp_engine/test_<name>.py` against the
      corresponding Python Lucid op.
- [ ] Numerical-grad test (atol=1e-3) for backward.
- [ ] Entry in `CHANGELOG.md` under "Unreleased / Added".

## Why the layering is strict

Without it:
- 50+ op files start including `bind_*.cpp` for convenience helpers, locking
  the engine to Python.
- The C ABI (Phase 7) becomes impossible without major rewrites — anything
  that reaches into pybind types stops being callable from C.
- Build times explode as headers fan-in.
- Refactor blast radius is unpredictable.
