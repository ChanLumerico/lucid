# Lucid C++ Engine — Architecture v2

This is the load-bearing design document. Every PR that adds a header,
moves a file, or changes a layer dependency must update this file.

Last updated: Phases 0–9 complete.

---

## Layer diagram

```
┌──────────────────────────────────────────────────────────────┐
│  Python API  (lucid.Tensor / lucid.nn / lucid.optim)         │  Phase 11
├──────────────────────────────────────────────────────────────┤
│  pybind11 boundary  (lucid._C.engine)                        │
├──────────────────────────────────────────────────────────────┤
│  bindings/                                                    │
│    bind_bfunc, bind_ufunc, bind_linalg, bind_nn, bind_optim  │
│    bind_einops, bind_gfunc, bind_utils, bind_random …        │
│    BindingGen.h  ← auto-binding helpers (Phase 6)            │
├──────────────────────────────────────────────────────────────┤
│  ops/                                                         │
│    bfunc/   — binary ops   (Add, Sub, Mul, Div, Pow, …)      │
│    ufunc/   — unary + reduce ops (Activation, Arith, …)      │
│    linalg/  — Det, Inv, Norm, Solve                          │
│    einops/  — einsum, einops_reduce                          │
│    utils/   — Sort, Tri, View, Contiguous                    │
│    gfunc/   — gather, scatter, index ops                     │
│    nn/      — Linear, Dropout, Conv, Pool, BN, LN, Attention │
├──────────────────────────────────────────────────────────────┤
│  kernel/                                                      │
│    BinaryKernel<D>    — 2-input element-wise CRTP base       │
│    UnaryKernel<D>     — 1-input element-wise CRTP base       │
│    ReduceKernel<D>    — reduction op CRTP base               │
│    NaryKernel<D,N>    — fixed-N-input op CRTP base           │
│    VariadicKernel<D>  — variadic-input op CRTP base          │
├──────────────────────────────────────────────────────────────┤
│  autograd/                                                    │
│    Engine              — backward(root, seed, retain_graph)  │
│    Node / Edge         — abstract graph node + edge          │
│    AutogradNode<D,N>   — CRTP backward-node base (FuncOp)    │
│    AccumulateGrad      — leaf sentinel, writes grad          │
│    Helpers             — make_zero/ones, accumulate_into, …  │
├──────────────────────────────────────────────────────────────┤
│  backend/                                                     │
│    IBackend            — pure-virtual device interface       │
│    Dispatcher          — Device → IBackend* singleton router │
│    cpu/CpuBackend      — Apple Accelerate (vDSP/vForce/BLAS) │
│    gpu/GpuBackend      — MLX / Metal                        │
├──────────────────────────────────────────────────────────────┤
│  core/                                                        │
│    TensorImpl          — storage + shape + dtype + grad_fn   │
│    Storage             — variant<CpuStorage, GpuStorage>     │
│    OpSchema            — versioned op metadata               │
│    OpRegistry          — process-wide schema map             │
│    GradMode            — thread-local grad-enabled toggle    │
│    SchemaGuard         — AMP dtype resolution + det. gate    │
│    MemoryStats         — per-device alloc/peak counters      │
│    Allocator           — 64-byte aligned posix_memalign      │
│    Dtype / Device / Shape / Error / Generator / Profiler …   │
└──────────────────────────────────────────────────────────────┘
```

**Dependency rule**: a layer may only `#include` headers from the same
layer or a layer **below** it. There are no upward includes. The one
allowed exception is `kernel/AutogradNode.h`, which re-exports
`autograd/AutogradNode.h` so kernel headers can reference it without
an upward include chain.

---

## Modules

### core/  (Phases 1, 1.5, 2.5, 5)

Vocabulary types — data and tiny helpers only. No autograd, no backend
math, no pybind11.

| Header | Purpose |
|---|---|
| `TensorImpl.h` | Load-bearing tensor: Storage + shape + dtype + grad_fn + version |
| `Storage.h` | `CpuStorage` (aligned ptr) + `GpuStorage` (mlx::array shared_ptr); `Storage = variant` |
| `Allocator.h` | 64-byte `posix_memalign` + MemoryTracker integration |
| `MemoryStats.h` | Per-device atomic alloc/peak/free counters |
| `OpSchema.h` | Versioned op metadata: name, version, AmpPolicy, determinism, arity, internal flag |
| `OpRegistry.h` | Process-wide `shared_mutex` map; `LUCID_REGISTER_OP` static initializer |
| `SchemaGuard.h` | Phase 5: AMP dtype resolution, F16 CPU fallback, determinism gate |
| `GradMode.h` | Thread-local grad-enabled flag + `NoGradGuard` RAII |
| `AmpPolicy.h` | `AmpPolicy` enum (Promote / KeepInput / ForceFP32) + `AutocastGuard` |
| `Dtype.h` | `Dtype` enum (9 values) + `dtype_size` / `dtype_name` helpers |
| `Device.h` | `enum Device { CPU, GPU }` |
| `Shape.h` | `Shape` / `Stride` typedefs + `shape_numel`, `contiguous_stride` |
| `Error.h` | `LucidError` hierarchy (ShapeMismatch, DtypeMismatch, VersionMismatch, …) |
| `Generator.h` | Philox-4×32-10 counter RNG; seedable, thread-safe |
| `Profiler.h` | Thread-local profiler pointer + `OpScopeFull` RAII |
| `Result.h` | `Result<T>` / `Ok` / `Err` — no-throw error handling for shape ops |
| `fwd.h` | Forward declarations for all public types |

### autograd/  (Phase 2, 3)

The dependency graph and its traversal. No backend math here.

| Header | Purpose |
|---|---|
| `Node.h` | Abstract `Node` + `Edge { shared_ptr<Node>, input_nr }` |
| `AutogradNode.h` | CRTP base `AutogradNode<Derived, N_IN>`: saved state, version checks, `release_saved()` |
| `AccumulateGrad.h` | Leaf sentinel — writes grad into `TensorImpl::grad_storage_` |
| `Engine.h` | `Engine::backward(root, seed, retain_graph)` — DFS topo sort + reverse traversal |
| `Helpers.h` | `make_zero/ones_storage`, `reduce_grad_to_shape`, `accumulate_into`, `check_version_match` |
| `FuncOp.h` | Backward-compat alias: `template<class D, N> using FuncOp = AutogradNode<D,N>` |

### kernel/  (Phase 3.5)

CRTP kernel bases. Each base owns the entire `forward()` → autograd-wiring
path so concrete op files implement only math (cpu_kernel / grad_formula).

| Header | Purpose |
|---|---|
| `BinaryKernel<D>` | 2-input element-wise: validates shapes/dtypes, broadcasts, dispatches, wires autograd |
| `UnaryKernel<D>` | 1-input element-wise: SchemaGuard, optional Dispatcher path, wires autograd |
| `ReduceKernel<D>` | Reduction: handles axes normalization, keepdims, wires autograd |
| `NaryKernel<D,N>` | Fixed-N-input ops (e.g., Conv: x, W, b) |
| `VariadicKernel<D>` | Ops with runtime-variable input count (e.g., einsum) |
| `AutogradNode.h` | Re-export of `autograd/AutogradNode.h` to avoid upward includes |
| `Contig.h` | Forward-declaration of `contiguous_op` used by kernels |
| `primitives/` | Storage-level compute primitives shared by backward passes |

### backend/  (Phase 4, 4.5)

Pure math kernels behind a device-agnostic interface. No autograd here.

| Component | Purpose |
|---|---|
| `IBackend` | Pure-virtual interface: `zeros`, `ones`, `clone`, 7 binary ops, 35 unary ops, 4 reductions, `matmul`, `broadcast`, `cast` |
| `Dispatcher` | Singleton `Device → IBackend*` router. `for_device(d)` called by kernels |
| `cpu/CpuBackend` | Apple Accelerate: vDSP (vectorized arithmetic), vForce (transcendentals), BLAS (matmul), LAPACK (linalg) |
| `gpu/GpuBackend` | MLX/Metal: wraps `mlx::core::*` ops; `MlxBridge.h` for dtype/shape conversions and CPU↔GPU upload/download |

**Backend split rule**: the CPU stream uses only Apple Accelerate (never
MLX). The GPU stream uses only MLX. Linalg (inv/det/solve/norm) dispatches
through MLX on both streams because MLX is CPU-backed there; results are
wrapped back into the appropriate Storage type.

### ops/  (Phases 3.1–3.8, 7)

Concrete op implementations. Each file is short: `schema_v1` declaration,
`cpu_kernel` (calls a backend or Accelerate primitive), optional
`gpu_kernel` or `dispatch()`, and `grad_formula`.

| Directory | Contents |
|---|---|
| `bfunc/` | Add, Sub, Mul, Div, Pow, Maximum, Minimum, Matmul, Dot, Inner, Outer, Bitwise, Compare, Floordiv, Tensordot, Inplace |
| `ufunc/` | Activation, Arith, Discrete, Exponential, Hyperbolic, Reductions, Softmax, Trig, Transpose, UnaryGpu |
| `linalg/` | Det, Inv, Norm, Solve |
| `einops/` | Einsum, EinopsReduce |
| `utils/` | Sort, Tri, View, Contiguous |
| `gfunc/` | Gather, scatter, index ops |
| `nn/` | Linear, Dropout, Conv{1,2,3}d, ConvTranspose{1,2,3}d, MaxPool/AvgPool/AdaptivePool, BatchNorm, GroupNorm, LayerNorm, RMSNorm, Attention |

### bindings/  (Phases 3–6)

pybind11 glue. Each `register_*` function exposes one module family.
Bindings depend on everything; nothing depends on bindings — they are the
leaf of the layering DAG.

`BindingGen.h` (Phase 6) provides `bind_unary<D>`, `bind_binary<D>`,
`bind_unary_extra<D>` helpers that read the op name from `schema_v1`,
ensuring the Python name always tracks the canonical schema name.

---

## Key design decisions

### 1. CRTP kernels (BinaryKernel / UnaryKernel / ReduceKernel)

**Why**: Before Phase 3.5, every op had ~40 lines of identical boilerplate
(null checks, dtype/device validation, SchemaGuard, contiguity ensure, Op
Scope, autograd wiring, edge construction, version saving). Across ~100
ops this was thousands of lines of copy-paste that diverged under
maintenance.

**Solution**: CRTP places all boilerplate in the base class `forward()` and
`apply()`. The derived class implements only `schema_v1`, `cpu_kernel`,
`grad_formula` — typically 20–30 lines. The static `dispatch()` hook
(Phase 4.5) allows opting into the Dispatcher without touching the kernel
base.

**Trade-off**: CRTP means the entire `forward()` body is instantiated once
per derived type (code size). Empirically this is acceptable; each
instantiation is small and the compiler inlines the tiny dispatch.

### 2. IBackend / Dispatcher split

**Why**: Before Phase 4, every op had `if (device == GPU) { mlx::… } else
{ cpu::… }` branches scattered throughout op files. Adding a new backend
(or changing a backend primitive) required touching every op.

**Solution**: `IBackend` defines the device contract. `Dispatcher` routes
`for_device(d)` to the registered singleton at startup. Op files call
`backend_for(device_).exp(storage, shape, dt)` — zero device awareness.
Adding a new backend (e.g., CUDA) = implement `IBackend`, register with
`Dispatcher`. Zero op-file changes.

**Note**: `CpuBackend` = Apple Accelerate only. `GpuBackend` = MLX only.
This is intentional: mixing Accelerate into the GPU path or MLX into the
CPU path would create debugging and portability problems.

### 3. Thread-local GradMode

**Why**: `torch.no_grad()` context managers must be thread-safe without
locking. A process-global `grad_enabled` would require a mutex on every
forward pass, or break when two threads run inference and training
simultaneously.

**Solution**: `GradMode::is_enabled()` reads a `thread_local bool`. Each
thread gets its own stack. `NoGradGuard` saves + restores on destruction.
Zero overhead (single `thread_local` read per forward call).

### 4. OpSchema / OpRegistry

**Why**: Op metadata (AMP policy, determinism flag, arity, internal flag)
was duplicated between op headers and binding files before Phase 3/6. Any
mismatch caused silent wrong-dtype math or untested code paths.

**Solution**: Every op declares `static const OpSchema schema_v1` as its
single source of truth. `LUCID_REGISTER_OP` at static init populates the
process-wide `OpRegistry`. The registry drives:
- `BindingGen.h` — Python binding name comes from `schema_v1.name`
- `SchemaGuard` — AMP and determinism behavior
- `check_registry_coverage.py` — CI gate ensuring all ops have a parity test
- `schema_hash()` — checkpoint compatibility validation

**Thread safety** (Phase 9): `OpRegistry` uses `shared_mutex` — concurrent
reads (all normal op dispatch) take shared lock; registration (startup
only) takes exclusive lock.

---

## Layer import rules

| Layer | May include from |
|---|---|
| `core/` | Standard library only |
| `autograd/` | `core/` |
| `backend/` | `core/` |
| `kernel/` | `autograd/`, `backend/`, `core/` |
| `ops/` | `kernel/`, `autograd/`, `backend/`, `core/` |
| `bindings/` | `ops/`, `kernel/`, `autograd/`, `backend/`, `core/`, pybind11 |

**Forbidden**: `core/` including `autograd/` or `backend/`. `autograd/`
including `backend/`. `backend/` including `autograd/`. Any layer
including `bindings/`.

---

## Extension recipes

All recipes are completable in under 30 minutes.

### Recipe A: Add a new unary op

**Example**: adding `cube_root(x)` = x^(1/3).

1. **Create the header** `lucid/_C/ops/ufunc/Arith.h` (or a new file if
   the op doesn't fit an existing family). Declare:
   ```cpp
   struct CubeRootBackward : UnaryKernel<CubeRootBackward> {
       static const OpSchema schema_v1;
       static CpuStorage cpu_kernel(const CpuStorage& a, const Shape& s, Dtype dt);
       Storage grad_formula(const Storage& g);
   };
   TensorImplPtr cube_root_op(const TensorImplPtr& a);
   ```

2. **Create the .cpp**. Implement:
   - `schema_v1` with name, version=1, AmpPolicy, deterministic=true.
   - `cpu_kernel`: call the appropriate vForce wrapper (e.g., `vforce::vpow_f32`
     with exponent 1/3) or a scalar loop for other dtypes.
   - `grad_formula`: return `g / (3 * cbrt(x)^2)` using saved_inputs_.
   - `cube_root_op`: call `CubeRootBackward::forward(a)`.
   - `LUCID_REGISTER_OP(CubeRootBackward)` at file scope.

3. **Optionally add `dispatch()`** if you want IBackend routing:
   ```cpp
   static Storage dispatch(backend::IBackend& be, Storage a, Shape s, Dtype dt) {
       return be.cube_root(a, s, dt);  // requires IBackend + CpuBackend + GpuBackend
   }
   ```
   Skip this for a quick CPU-only op; add `gpu_kernel` for a direct MLX path.

4. **Bind** in `lucid/_C/bindings/bind_ufunc.cpp`:
   ```cpp
   #include "../ops/ufunc/Arith.h"
   // in register_ufunc():
   bind_unary<CubeRootBackward>(m, &cube_root_op, "Cube root x^(1/3).");
   ```

5. **Write a parity spec** in `tests/parity/test_cube_root.py`:
   ```python
   def test_cube_root_fwd():
       x = np.array([1., 8., 27.], dtype=np.float32)
       ref = np.cbrt(x)
       out = eng.cube_root(make_tensor(x))
       np.testing.assert_allclose(to_numpy(out), ref, atol=1e-6)
   ```

### Recipe B: Add a new binary op

**Example**: adding `hypot(a, b)` = sqrt(a² + b²).

1. **Create** `lucid/_C/ops/bfunc/Hypot.h` / `Hypot.cpp`:
   ```cpp
   struct HypotBackward : BinaryKernel<HypotBackward> {
       static const OpSchema schema_v1;
       static CpuStorage cpu_kernel(const CpuStorage& a, const CpuStorage& b,
                                    const Shape& s, Dtype dt);
       std::pair<Storage, Storage> grad_formula(const Storage& g);
   };
   TensorImplPtr hypot_op(const TensorImplPtr& a, const TensorImplPtr& b);
   ```

2. **Implement** `cpu_kernel` (scalar loop or vForce path),
   `grad_formula` (`da = g*a/hypot`, `db = g*b/hypot` using saved_inputs_),
   `hypot_op` calling `HypotBackward::forward(a, b)`, and
   `LUCID_REGISTER_OP(HypotBackward)`.

3. **Bind** in `bind_bfunc.cpp`:
   ```cpp
   bind_binary<HypotBackward>(m, &hypot_op, "Euclidean distance sqrt(a^2+b^2).");
   ```

4. **Write a parity spec**.

### Recipe C: Add a new backend

**Example**: adding a CUDA backend for Linux.

1. **Create** `lucid/_C/backend/cuda/CudaBackend.h` and `CudaBackend.cpp`.
   Inherit from `IBackend` and implement all pure-virtual methods.
   The `device()` override returns `Device::GPU` (or add a new enum value
   if you want CPU+GPU simultaneously).

2. **Register** in `BackendInit.cpp`:
   ```cpp
   #if LUCID_CUDA
   Dispatcher::register_backend(Device::GPU, std::make_unique<CudaBackend>());
   #endif
   ```

3. **No op changes needed.** Every op that calls
   `backend_for(device_).exp(...)` will automatically route to
   `CudaBackend::exp` when device is GPU.

4. **Wire CMake**: add `cuda/` sources under `LUCID_BACKEND_CUDA` option.

### Recipe D: Add a new dtype

**Example**: adding `BFloat16`.

1. **`core/Dtype.h`**: add `BF16` to the `Dtype` enum.
2. **`core/Dtype.cpp`**: add `BF16` entry in `dtype_size()` (2 bytes) and
   `dtype_name()`.
3. **`backend/cpu/CpuBackend.cpp`**: add `BF16` cases to all unary/binary
   dispatch switches. On Apple Silicon, use vDSP's bfloat16 path or
   upcast-to-F32-then-downcast.
4. **`backend/gpu/GpuBackend.cpp`**: map `Dtype::BF16 → mlx::core::bfloat16`
   in `to_mlx_dtype` / `from_mlx_dtype` in `MlxBridge.h`.
5. **`core/SchemaGuard.cpp`**: update AMP promotion rules if BF16 should
   promote to F32 for ForceFP32 ops.
6. **Parity test**: add a `test_bf16_round_trip.py` spec.

---

## Key invariants

1. **`TensorImpl::storage_` is the single source of truth** for tensor data.
   Reads go through `data_as_python()` (zero-copy numpy view on CPU) or
   `copy_from()` for mutation.

2. **Autograd nodes don't own tensors.** `AutogradNode::input_tensors_`
   holds `weak_ptr<TensorImpl>` for version validation. `AccumulateGrad`
   also holds `weak_ptr`. Tensors stay alive only as long as user code
   (Python wrapper) holds them.

3. **`Storage` is move-only, never reference-counted directly.** RC happens
   at `TensorImpl` level. `Storage` is moved into the `TensorImpl`
   constructor and out via `std::move`.

4. **Every op declares `OpSchema schema_v1`** at module init via
   `LUCID_REGISTER_OP`. Schemas are immutable post-init. The registry is
   the authoritative source for AMP, determinism, binding names, and
   parity-test coverage.

5. **No layer above `core/` may store `py::object`.** Pybind types live
   only in `bindings/` and the boundary functions of `TensorImpl`
   (`data_as_python`, `grad_as_python`).

6. **`release_saved()` is called after `apply()`** (Phase 9) to drop saved
   activation buffers from backward nodes once they have been consumed,
   preventing long-lived graphs from pinning large tensors.

7. **`OpRegistry` uses `shared_mutex`** (Phase 9): concurrent reads (normal
   dispatch) are lock-free in the shared sense; registration (static init)
   takes exclusive lock. The thread-local memory pool (Phase 9) further
   reduces per-op allocation overhead under high concurrency.
