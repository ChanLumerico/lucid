# Changelog

All notable changes to **Lucid** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Scope.** Lucid is an Apple Silicon-only ML framework with PyTorch-compatible
> Python surface, MLX/Accelerate-native backend, and a custom C++ engine.
> Categories below follow Keep-a-Changelog plus two project-specific buckets:
> **Performance** (measured speed/memory wins) and **Tooling** (dev-only changes
> that don't affect runtime — CI, lints, scaffolding).

---

## [Unreleased]

_No changes yet._

---

## [3.2.2] — 2026-05-20

Codebase-wide inefficiency sweep prompted by a deep audit after the
3.2.1 BN-leak hotfix.  Six independent improvements grouped by cost
(critical leak fix + five medium-ROI hot-path optimisations) + a
broader contiguous-removal sweep that extends 3.1.0's work.

### Fixed

- **`nn.InstanceNorm{1,2,3}d._update_running_stats` lazy-graph leak.**
  Same root cause as 3.2.1's BatchNorm fix: the running-stats update
  `(1 - m) * running_mean + m * batch_mean` builds a lazy MLX
  expression that holds the entire forward graph (conv weights +
  activations) as parents through the `batch_mean` parent.
  `.detach()` clears autograd but not the MLX lazy chain, and
  `loss.item()` only evaluates the loss path — running stats
  accumulate one full forward graph per step.  Trigger condition:
  `InstanceNorm(track_running_stats=True)` on Metal (default is
  False, so most users are unaffected, but the opt-in path used by
  RNN / time-series models was vulnerable).  Fix mirrors the 3.2.1
  one-liner: force-eval running stats after update via
  `_eval_running_stats_metal()`.

### Performance

- **`_REGISTRY` linear scan → O(1) dict lookup** in
  `lucid._ops.__init__._make_free_fn`.  Each free-function bind
  previously walked all ~500–1000 ops; with the new
  `_REGISTRY_BY_FREE_FN` index the lookup is constant-time.  Module-
  load cost ~1 ms saved, and the path is also hit by late-bound name
  resolution.
- **`nn.Module.__call__` early-exit on hookless modules.**  Skips
  the four `OrderedDict` iterations (`_GLOBAL_FORWARD_PRE_HOOKS`,
  `self._forward_pre_hooks`, the post-fwd equivalents, and the
  backward-hook check) when no hooks are registered.  Saves ~15–20 µs
  per `forward()` call on the 99 % case; over a ResNet-18 forward
  (50+ module calls) that's ~1 ms / batch.
- **Default dtype/device cache in `lucid._dispatch.normalize_factory_kwargs`.**
  Process-lifetime cache of `to_engine_dtype(get_default_dtype())`
  and `_parse_device(get_default_device())`.  Invalidated by
  `lucid.set_default_dtype` / `lucid.set_default_device` (rare in
  practice).  Was costing ~0.5–1.2 µs per op call before; now ~0.1 µs.
  Mirrors the same pattern that `lucid._factories.converters`
  already used for the ndarray-fast-path device cache.
- **Conditional `Optimizer.step()` wrapper.**  Previously every
  optimizer subclass had its `step()` unconditionally wrapped in
  `_step_with_eval` that checked `AUTO_EVAL_AFTER_STEP` at runtime —
  even though the default has been False since 3.0.3.  Now the
  wrapper is installed only when `AUTO_EVAL_AFTER_STEP = True` at
  subclass declaration; the default path runs the user's raw
  `step()`.  Saves ~0.7 µs / step.  *Behaviour note*: toggling
  `AUTO_EVAL_AFTER_STEP` at runtime no longer enables auto-flush for
  subclasses declared with the default.  To opt back in, set the
  class attribute before the subclass is defined.
- **Contiguous-sweep follow-up (3.1.0 extension).**  21 more
  `wrap_mlx_array(::mlx::core::contiguous(<expr>), dt)` sites
  identified as safe to drop (operands produce fresh contiguous
  output):
    * `lucid/_C/backend/gpu/GpuBackend.h`: 16 sites — `zeros`, `ones`,
      `reverse_along_axis`, `trace`, `trace_backward`, `where`
      forward + backward, `masked_fill`, `gather`,
      `scatter_add_axis_backward`, `pad`, `concatenate`, `stack`,
      `topk` (values), `argsort`.
    * `lucid/_C/ops/ufunc/Scan.cpp`: 4 sites — `cummax_backward` /
      `cummin_backward` F32 / F64 paths.
    * `lucid/_C/ops/utils/Nextafter.cpp`: 1 site — `nextafter` view cast.
  Cumulative effect on training loops that pass through these ops:
  modest individually (~0.1–0.5 % each); over a full forward
  graph the deferred-materialization wins add up to ~1–2 % on
  workloads that use these ops.

### Cumulative estimated impact

| Change | Per-call overhead saved | Per-epoch effect (LeNet-5/MNIST) |
|---|---|---|
| `_REGISTRY` dict lookup | ~2 µs (one-shot at module load) | negligible runtime |
| `Module.__call__` early-exit | ~15–20 µs / forward | ~+1 % throughput |
| dispatch dtype/device cache | ~1 µs / op | ~+0.5 % |
| optimizer wrapper conditional | ~0.7 µs / step | ~+0.3 % |
| Contiguous sweep (21 sites) | varies | ~+1 % on relevant workloads |
| **Total** | | **~+2–3 % LeNet, +OOM safety for InstanceNorm** |

### Tests

501 tests pass (factories, autograd, device, metal, nn unit, nn parity,
ops parity, optim parity, data utils parity).  No public-API
regression; all 3.2.1 behaviour preserved except for the documented
``AUTO_EVAL_AFTER_STEP`` runtime-toggle edge case.

### Migration

`Optimizer.AUTO_EVAL_AFTER_STEP` runtime toggle: if you previously
relied on `Adam.AUTO_EVAL_AFTER_STEP = True` *after* declaration to
enable auto-flush, set it inside the subclass body instead.  All
other changes are transparent.

---

## [3.2.1] — 2026-05-20

BatchNorm running-stats lazy-graph leak hotfix.  Found during a
CIFAR-10 / ResNet-18 measurement on Mac Studio: training consistently
OOM'd at ~batch 400 (bs=32) regardless of memory pressure or cache
clearing.  Bisected to:

  * Simple Linear-only training: no leak ✓
  * Conv2d-only training: no leak ✓
  * BatchNorm2d-only training: no leak ✓
  * Conv2d + BatchNorm2d (CBR pattern): **+4 MB/iter** leak 🔥
  * Residual block (2× CBR + skip): **+8 MB/iter** leak 🔥
  * Full ResNet-18: **+37 MB/iter** leak → OOM after few hundred batches

### Root cause

`nn.BatchNorm{1,2,3}d._update_running_stats` constructs the new
``running_mean`` / ``running_var`` as a *lazy MLX expression*:

```python
new_rm = (1 - eff) * running_mean + eff * batch_mean
self._buffers["running_mean"] = new_rm.detach()
```

`.detach()` clears the autograd graph but **leaves the MLX lazy
expression intact**.  The new buffer holds the old running_mean,
batch_mean, and indirectly the entire forward graph that produced the
batch's input (conv weights + activations) as graph parents.

The training loop's only force-evaluation point is ``loss.item()``,
which materialises the path connected to the loss.  BN running stats
are **not** connected to the loss (they're statistics, not gradients
flow targets), so the running-stats lazy chain accumulates one full
forward graph per training step and is never collected.

Pure-MLX experiments verified that MLX *does* release parents after
``mx.eval()`` — so the fix is just to eval the running stats
explicitly after the update.

### Fix

`nn.BatchNorm{1,2,3}d._update_running_stats` now calls
`_eval_running_stats_metal(self._buffers)` after assigning the new
running_mean / running_var.  That helper dispatches a single
`engine.eval_tensors([running_mean, running_var, num_batches_tracked])`
which forces materialisation and detaches the lazy expression's
parents.

### Measurement (M4 Max Mac Studio, ResNet-18 / bs=16)

| Pattern | Active memory growth | 30-iter total |
|---|---|---|
| Before fix | +8 MB/iter (residual block) | 228 MB after 20 iter |
| **After fix** | **stable** | **64 MB across all iters** |

For full ResNet-18 the growth was +37 MB/iter → 5-epoch projection 286 GB (impossible).  After fix: stable across full training.

### Performance

No measurable throughput regression — the eval call adds a sync
barrier for ~tens of small tensors per BN layer, which is the same
work MLX would have done anyway when the running stats are finally
read.  CPU-only path is unaffected (the helper skips when buffers are
on CPU).

### Tests

44 BN + norm parity tests pass on local M1 Pro.  Mac Studio full
ResNet-18 / CIFAR-10 5-epoch training now completes (was OOM on 3.2.0
and earlier).

### Migration

No user code change needed — fix is transparent.  Existing models
using `nn.BatchNorm{1,2,3}d` benefit immediately on `pip install --upgrade`.

---

## [3.2.0] — 2026-05-17

Training-pipeline overhead pass.  After 3.1.0's GPU-kernel fusion sweep
brought Lucid's forward to within +3.9 % of raw MLX, the next layer of
the LeNet-5 / MNIST profile pointed at Python-side hot loops: 48 % of
per-epoch time in `engine.item()`, 30 % in `Dataset.__getitem__` →
`lucid.tensor(np_array)`, and small per-call overhead in `.to(device)`
/ `.long()`.  3.2.0 collapses each of those.

A new isolated raw-MLX vs PyTorch-MPS measurement (Mac Studio M4 Max,
LeNet-5, varying batch sizes) also corrected the framing of the gap:
**MLX matches or beats PyTorch MPS at the GPU-kernel level** for
forward+backward on training-scale workloads (0.52× at BS=16, 0.60× at
BS=64).  The ~2.2× wall-clock gap on the full training script is
non-GPU pipeline overhead; 3.2.0 targets it directly.

### New (small) public API

- **`lucid.nn.functional.accuracy(logits, target, *, dim=-1)`** — fused
  `(argmax == target).float().mean()`, returns a 0-d Tensor in
  `[0, 1]`.
- **`lucid.nn.functional.correct_count(logits, target, *, dim=-1)`** —
  fused `(argmax == target).long().sum()`, returns a 0-d int64 Tensor.
  Pairs naturally with the `running_correct += ... .item()` training
  pattern: one Python wrap instead of four.
- **`Dataset.__getitems__(indices) -> already-batched`** — optional
  protocol method.  When present on a dataset, `DataLoader` skips the
  per-sample `__getitem__` loop + `collate_fn` and passes the result
  through directly.  Backward-compatible: datasets without
  `__getitems__` keep working unchanged.
- **`TensorDataset.__getitems__`** — vectorised override using fancy
  indexing.  When the wrapped tensors live on Metal, the resulting
  batch tensors stay on Metal — no per-batch `.to(device)` round-trip
  in the training loop.

### Performance

- **`TensorImpl::item()` direct memory read.**  Old path:
  `to_bytes()` → `download_gpu_to_cpu()` → fresh `CpuStorage` +
  `py::bytes` allocation + extract + decode.  New path: pointer-offset
  into the storage buffer (CPU) or `mx::array::data<T>()` after `eval()`
  (GPU), then decode the single scalar.  cProfile of LeNet-5/MNIST
  training counted `engine.item()` as 1.63 s of 3.34 s per epoch (48 %).
  Measured on M4 Max:
    * CPU 0-d item: ~5 µs → **0.25 µs** (20× faster)
    * Metal item (already-evaluated): ~870 µs → **0.29 µs** (3000× faster — old path's bytes-roundtrip was the dominant cost)
    * Metal item-after-compute: **221 µs**, which is now the genuine
      `mx::array::eval()` sync cost; PyTorch MPS's `.item()` is in the
      same ballpark.  Lucid overhead is effectively zero on this path.
- **`Tensor.to(device=...)` no-op fast path.**  When the kwarg is the
  only argument and the tensor is already on the requested device, the
  whole arg-parse + dtype/device normalisation walk is bypassed via a
  string→engine.Device lookup table.  3 µs/call → 1.12 µs/call (M4 Max).
- **dtype shortcut methods no-op fast paths** (`.long() .float()
  .double() .half() .int() .bool() .cpu() .metal()`).  All eight now
  short-circuit to `return self` when the source already matches the
  target dtype/device — the docstrings already documented this as
  no-op semantics, but the implementation went through `to(...)` 's
  full machinery (~2.5 µs).  Now ~0.96 µs.
- **`lucid.tensor(np.ndarray)` fast path.**  For the hot
  `lucid.tensor(np_array)` case with no dtype/device override, skip
  `normalize_factory_kwargs`, the `_try_numpy_free_to_impl` isinstance
  gauntlet, `_np_dtype_to_engine`'s `np.dtype.name` lookup, and
  `np.ascontiguousarray` (gated on the array's `C_CONTIGUOUS` flag).
  Microbench: 9.0 µs/call → 2.17 µs/call (4.1× faster).
- **`TensorDataset` + vectorised batch fetch.**  With pre-tensorised
  data, the DataLoader path goes from 60 k `__getitem__` calls +
  64-element stack per batch to one fancy index per wrapped tensor.
  Measured per-batch cost on (60 000, 1, 28, 28) MNIST-shape:
    * old `NumpyMNIST` per-sample pattern: **793 µs/batch**
    * `TensorDataset` (CPU): 722 µs/batch (−9 %)
    * **`TensorDataset` (Metal, dataset already on GPU): 246 µs/batch
      (−69 %, 3.2× faster)**
  The Metal-resident variant also makes the per-batch `.to(device=)` in
  user code a no-op (already on Metal) — both effects compose.

### Notes on what *didn't* help (investigated, then deferred)

A pipeline-overhead profile evaluated four hypothetical wins.  Two of
them lost; only one of the wins generalises cleanly to all users.

| Mode | 1-epoch wall | vs baseline | Verdict |
|---|---|---|---|
| BASELINE (per-sample tensor + 2 `.item()` / batch) | 2.50 s | — | reference |
| Lazy GPU metric accumulation (no per-batch `.item()`) | 2.68 s | +7.1 % | **regression** — lazy graph bloats |
| Batched-collate (`__getitem__` returns numpy slice) | 1.89 s | −24.5 % | win, ships as the `TensorDataset` pattern above |
| Multi-worker DataLoader (nw=2) | 2.52 s | +0.8 % | neutral on MNIST-sized data |
| Multi-worker DataLoader (nw=4) | 2.66 s | +6.4 % | regression on small data |

Net: per-batch `.item()` sync is *not* a bottleneck once the dataset
path is fast — it actually acts as natural backpressure that keeps the
MLX lazy graph manageable.  Multi-worker DataLoader stays on the
roadmap for ImageNet-scale workloads but adds no value here.

### Tests

361 tests pass: 142 nn unit + 73 nn parity + 118 ops/optim/autograd
parity + 11 data utils parity + integration data-pipeline + factory /
device / metal regression.  No public-API regression.  All 3.1.0
behaviour preserved.

### Migration note

No code change required to benefit from the `.item()` / `.to()` /
`lucid.tensor(np_array)` fast paths — they're internal.

For the `TensorDataset` win, the recommended pattern is:

```python
# 3.2.0+ recommended pattern for in-memory datasets
import lucid
from lucid.utils.data import TensorDataset, DataLoader

# Pre-load entire dataset to GPU once
X = lucid.tensor(train_x_np).to("metal")
y = lucid.tensor(train_y_np).to("metal")
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# Training loop — batches are already on Metal
for x, y in loader:
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    opt.step()
```

The per-sample `__getitem__ → lucid.tensor(np_array)` pattern continues
to work for streaming or larger-than-memory datasets.

---

## [3.1.1] — 2026-05-17

DataLoader-side Python overhead pass.  Deep cProfile of LeNet-5/MNIST
training revealed that `lucid.tensor(np_array)` is the single hottest
Python call in a real training loop (≈ 120 k calls / epoch via
per-sample tensorisation inside `Dataset.__getitem__`), accounting for
**~33 % of per-epoch CPU time** at BS=64.  The non-GPU side of Lucid's
PyTorch gap — not GPU kernel speed — is what 3.1.1 targets.

### Performance

- **`lucid.tensor(np_array)` fast path.**  For the hot case
  `lucid.tensor(np_array)` with no dtype / device / requires_grad
  override, skip:
    * `normalize_factory_kwargs` (dtype + device parsing — ~150 ns/call)
    * `_try_numpy_free_to_impl` isinstance gauntlet (ndarray bails out)
    * `_np_dtype_to_engine` (`.name` lookup → string formatting — measured
      at ~120 ns/call on M1 Pro just for ``np.dtype.name``)
    * `np.ascontiguousarray` (gated on the array's `C_CONTIGUOUS` flag —
      a no-op when the array is already contiguous, but the call itself
      cost ~1 µs)
  When the array is already C-contiguous, the path collapses to a
  single `_C_engine.TensorImpl(arr, default_device, False)` constructor
  call.  Microbench on M1 Pro:
    * Before: 9.0 µs / call (60 k iter median, fp32 (1, 28, 28) array)
    * After: **2.17 µs / call (−76 %, 4.1× faster)**
- **Cached default-device enum.**  Resolving the default device through
  `get_default_device() → _parse_device` cost ~50 ns × 120 k = ~6 ms /
  epoch.  Cached now in `_CACHED_DEFAULT_DEVICE_ENUM`; invalidated by
  `lucid.set_default_device` (rare in practice — set once per process).

### Notes on what *didn't* help

A deep training-pipeline profile evaluated four hypothetical wins:

| Mode | Wall-clock (1 epoch) | vs baseline | Verdict |
|---|---|---|---|
| BASELINE (per-sample `lucid.tensor` + 2 `.item()` / batch) | 2.50 s | — | reference |
| Lazy GPU metric accumulation (no per-batch `.item()`) | 2.68 s | **+7.1 %** | **regression** — lazy graph bloats |
| Batched-collate (`__getitem__` returns numpy slice) | 1.89 s | −24.5 % | win |
| Batched-collate + lazy metric (combined) | 1.65 s | −34.1 % | best |
| Multi-worker DataLoader (nw=2) | 2.52 s | +0.8 % | neutral |
| Multi-worker DataLoader (nw=4) | 2.66 s | +6.4 % | regression on small data |

Key takeaways folded into the 3.1.1 release:

- `.item()` per-batch sync is **not** the bottleneck — removing it
  causes lazy-graph bloat that costs more than the saved sync time.
  Per-batch sync acts as natural backpressure.
- Multi-worker DataLoader **does not help** for cheap-to-load datasets
  (MNIST-size).  Subprocess spawn + IPC overhead exceeds the loading
  cost.  Reserve `num_workers > 0` for ImageNet-scale data.
- The biggest realisable win — batched `__getitem__` returning numpy
  slices instead of pre-tensorised samples — is a *user-side* pattern,
  not a Lucid internal change.  The 3.1.1 fast path makes even the
  legacy pre-tensorised pattern 4× faster, so existing code benefits
  without modification.

### Tests

171 factory + ops parity tests pass (same coverage as 3.1.0).
No public API surface change; drop-in replacement for 3.1.0.

---

## [3.1.0] — 2026-05-16

Metal performance pass — focus on lazy-graph fusion.  Earned by a
deep per-layer profile of LeNet-5 that revealed Lucid's GPU forward
was **+57.9 % slower** than the raw MLX equivalent (M1 Pro, BS=64,
fp32, fwd-only, model output).  Root cause: a defensive
`mlx::core::contiguous(...)` wrap on the return path of many GPU
backend ops, applied as a habit when the operation was already
guaranteed to produce contiguous output.  Each redundant
`contiguous()` materialises the lazy graph at that point, breaking
MLX's ability to fuse adjacent kernels and forcing a memcpy that
the next op would otherwise have folded into its own kernel
launch.

After the sweep, M1 Pro LeNet-5 forward is **+3.9 % vs raw MLX**
(essentially parity — was +57.9 %).

### Performance

- **Conv2d / ConvTranspose2d input contiguous-before-conv_general.**
  Symmetric counterpart to 3.0.3's *output* contig removal — force a
  contiguous NHWC buffer for x and W before invoking
  `mlx::core::conv_general()`.  When the kernel receives a strided
  transpose-view, MLX dispatches a slower stride-aware path; with a
  contiguous input it picks the fastest contiguous-NHWC kernel.
  Microbench (W mutates every iter, simulating training pattern):
  484 µs → 446 µs per call (**−7.8 %**).  Applied to all 5 conv-kernel
  call sites (forward + backward dx + backward dW + conv_transpose
  forward + conv_transpose backward).
- **`matmul` — drop trailing `contiguous(out)`.**  The MLX matmul
  kernel always produces a fresh contiguous buffer; the defensive
  wrap was forcing a redundant memcpy and breaking fusion with
  downstream activation / bias-add.
- **`linear` forward + backward — drop trailing `contiguous()` on
  all four outputs** (forward out, backward dx, dW, db).  The single
  biggest find of the 3.1 sweep — Lucid's Linear was **+12 to +25 %
  slower** than raw MLX Linear (fc1/fc2/fc3 in the LeNet-5 profile)
  for this exact reason.  After fix:
    * Lucid fc1 solo: 383 µs → **232 µs** (**−39 %**; now **−2.7 %
      vs raw MLX**)
    * Lucid fc2 solo: 364 µs → 234 µs (**−36 %**)
    * Lucid fc3 solo: 307 µs → 242 µs (**−21 %**)
  The fix also lets the surrounding chain fuse better — Conv2d solo
  costs dropped from 466 µs → 336 µs (conv1) and 596 µs → 385 µs
  (conv2) on the same M1 Pro measurement, even though Conv2d itself
  wasn't touched in this change — the lazy graph extends past the
  matmul/linear boundary now.
- **`softmax` / `log_softmax` (forward + backward) — drop redundant
  `contiguous()` wrap.**  Both are computed-fresh ops; the wrap was
  forcing materialisation right where loss kernels want to fuse the
  result.
- **`cross_entropy_loss` forward + backward — drop redundant
  `contiguous()` on saved softmax / valid_count / dx.**  Saved
  tensors used in autograd were being re-materialised at save time
  then re-read in the backward; both round-trips removed.
- **`variance`, `cumsum`, `cumprod`, `cummax`, `cummin` — drop
  redundant `contiguous()` wrap.**  All MLX-native reductions /
  scans that produce contiguous output naturally.

### Measurement summary (M1 Pro, BS=64, fp32)

LeNet-5 model forward (model output only, no loss):

| Build | Lucid fwd µs | MLX fwd µs | Δ vs MLX |
|---|---|---|---|
| 3.0.3 (pre-sweep) | 1537 | 973 | +57.9 % |
| 3.1.0 (this) | 770 | 741 | **+3.9 %** |

Per-layer fusion benefit (sum of solo-eval / chain-eval):
- 3.0.3: 60 % (3805 µs solo sum → 1537 µs chained)
- 3.1.0: 72 % (3456 µs solo sum → **1169** µs chained)

### Tests

All conv unit + parity tests pass (37 conv tests).  73 nn parity,
118 ops / optim / autograd parity, 19 vision-model parity (LeNet /
AlexNet / ConvNeXt / EfficientNet / DenseNet / GoogLeNet /
InceptionV3) — no numerical regression.

### Backward compatibility

No API changes.  Internal-only optimisation — every public op
signature, every Tensor method, every Module shape contract is
unchanged.  Drop-in replacement for 3.0.3.

### What got skipped (was on the 3.1 wish list, deferred to 3.2+)

- **W-NHWC sidecar cache** at the `nn.Conv2d` module level — the
  user's original 3.1 request.  Investigation revealed that the
  cache *can't* help training workloads (every `optimizer.step()`
  bumps the parameter version, invalidating any cache; cache miss
  rate = 100 % in training) and the kernel-selection benefit it
  would have provided is already captured by the
  contiguous-before-conv_general change above.  The Python-module-
  level cache would still benefit inference loops (W reused across
  many forwards without mutation) — kept on the 3.2 backlog for the
  inference-perf milestone.
- **Fused CrossEntropy** — the engine's `cross_entropy_loss` is
  already a single MLX expression chain (softmax →
  take_along_axis → log → negate → multiply → reduce); not two
  separate log_softmax + nll passes.  The 3.1 contiguous removal
  on the loss path is the actually-useful optimisation.
- **Fused Adam** — the C++ Adam step is already expressed as one
  MLX expression chain per parameter (~14 lazy ops fused into one
  or two kernel launches per param at eval time).  The 730 µs / 10
  params = 73 µs / param measured cost is at the Metal kernel-
  launch floor; cross-parameter fusion isn't possible since each
  param has a different shape.

---

## [3.0.3] — 2026-05-16

Correctness + Metal-perf pass.  Found during a real LeNet-5 / MNIST
training smoke on M4 Max Mac Studio: training accuracy was stuck at
exactly 1.56 % (= 1 / batch) every step despite loss decreasing
identically to PyTorch.  Root cause was a silent `bool.sum()` bug;
fixing it surfaced two more never-implemented integer dispatch paths
and two redundant Metal sync points that together cost ~5–37 % of
step throughput.

### Fixed

- **`bool` / `int` reductions** — `lucid._C.engine.sum` /
  `engine.prod` now auto-promote `Bool` / `I8` / `I16` / `I32` inputs
  to `I64` before reducing, matching PyTorch's `bool.sum() → int64`
  semantics.  Pre-3.0.3 behaviour was: CPU raised
  `NotImplementedError: cpu_backend::reduce: dtype not supported`,
  Metal silently returned a 0-d `bool` (acting like `any()`) — that
  was the source of the 1.56 % stuck-accuracy training bug.  Caller
  code like ``(pred == y).sum().item()`` now reports the real count
  on every supported dtype.
- **`Tensor.astype` Cartesian-product cast** — `CpuBackend::astype`
  now covers every {F16, F32, F64, I8, I16, I32, I64, Bool} ×
  {same} pair.  Previously several pairs (notably `Bool → I64`,
  `I64 → Bool`, `I16 → I64`, `F64 → I8`, `Bool → F64`) were
  `NotImplementedError`, which broke ``Tensor.long()`` /
  ``Tensor.bool()`` chains on integer / bool inputs.  F16 is handled
  via a two-step F32 bridge so the dispatch table stays simple.
- **Native I64 reduction kernel** — added to
  `CpuBackend.reduce_axes` so the promoted bool/int reduce path runs
  on integer math directly, not round-tripped through F64 (which
  would have lost precision past `2^53`).

### Changed

- **`Optimizer.step()` no longer auto-flushes Metal params.**  Every
  concrete `step()` was historically wrapped to call
  `_metal_eval_params()` after the update, forcing
  `mlx.core.eval()` on every parameter tensor.  Metal profiling
  (LeNet-5 / MNIST, M4 Max, May 2026) showed this shattered the MLX
  lazy-graph pipeline that would otherwise chain
  forward → backward → step into one submission, and cost between
  5 % and 37 % of step throughput depending on batch size.  Default
  is now lazy — the new class-level flag
  ``AUTO_EVAL_AFTER_STEP: ClassVar[bool] = False`` controls the old
  behaviour.  Set it to ``True`` per-class (or per-instance) to
  restore the synchronous flush when you need ``step()`` to act as a
  sync point.  Matches PyTorch, which never auto-eval's after
  ``step()``.
- **`Tensor.backward()` no longer pre-evals the forward graph.**
  Historical docstring claimed a `self._impl.eval()` before the
  backward pass gave a ~2× speedup; current measurement on M4 Max
  shows it's neutral-to-negative because the MLX backward kernel
  triggers the necessary evaluation on its own.  The explicit
  pre-eval was redundant.  Removed.  No correctness or autograd
  semantics change.

### Performance

- **Conv2d / ConvTranspose2d / conv backward — drop redundant
  `mlx::core::contiguous()` after the final NHWC→NCHW transpose.**
  Lucid's GPU conv path is `transpose(x, NCHW→NHWC) → transpose(W,
  NCHW→NHWC) → conv_general → add(bias) → transpose(NHWC→NCHW)` —
  the trailing `contiguous()` was a defensive copy under the
  assumption that downstream ops needed a row-major buffer.  MLX
  ops are stride-aware, so the call was forcing a memory copy that
  the next op (relu / pool / batchnorm / next conv) would normally
  fuse with its own kernel.  Removing it lets the transpose stay as
  a lazy view all the way down to the final `mx.eval()`.  Touches
  4 sites: `conv_nd_forward`, `conv_nd_backward (dx)`,
  `conv_transpose_nd_forward`, `conv_transpose_nd_backward (dx)`.
  Measured impact (M1 Pro, BS=64, fp32):
    * Conv2d microbench (single conv layer, LeNet shape):
      `transpose+conv+transpose+contig` 740 µs → `transpose+conv+transpose` 624 µs (**−15.7 %**).
    * LeNet-5 full forward: 2680 µs → 2219 µs (**−17.2 %**).
    * LeNet-5 full step (fwd + bwd + opt): 6947 µs → 6610 µs (5-run median, **−4.8 %**).
  All 20 conv unit + parity tests still pass.

### Tooling

- New profiling baseline note in obsidian:
  - 5-epoch LeNet-5 / MNIST on M4 Max Metal: 27.9 s (Lucid 3.0.2) →
    26.4 s (this release).  PyTorch MPS reference: 12.1 s.  Remaining
    ~2.2× gap is structural MLX small-op kernel-launch overhead,
    closable only by ``lucid.compile()`` (graph capture / fusion) —
    tracked separately as the 3.1 Tier-S item.

---

## [3.0.2] — 2026-05-16

Standalone-mode hotfix.  The core Lucid lifecycle — `import lucid`,
tensor construction, forward, backward, optimiser step, device
transfer, RNN pack/unpack, autograd.grad, lucid.func transforms,
register_hook — now runs without numpy installed.  NumPy is reduced
to a strict opt-in extra (`pip install lucid-dl[numpy]`) used only at
its documented bridge methods: `Tensor.numpy()`, `from_numpy`,
`from_dlpack`, and `lucid.tensor(np_array)` with an actual ndarray
input.  Also drops the BNNS scalar API + legacy CBLAS / CLAPACK
deprecations the 3.0.x line had been silencing under
`-Wno-deprecated-declarations`.

### Fixed

- **`Tensor.to(device=...)` no longer requires numpy.**  The 3.0.x
  ``_to.py`` path called ``data_as_python()`` (returns a numpy view)
  then ``TensorImpl(np.ndarray, ...)`` to round-trip across devices,
  which forced ``import numpy`` on every device transfer.  Replaced
  with a new C++ method ``TensorImpl.transfer_to_device(target,
  requires_grad)`` that runs the copy inside the engine via
  ``mlx::core::copy()`` (CPU→GPU) or ``gpu::download_gpu_to_cpu()``
  (GPU→CPU).  SharedStorage tensors still use ``transfer_storage``
  for zero-copy relabelling.
- **`lucid.tensor([list])` no longer requires numpy.**  Pure-Python
  scalars / lists / tuples now build a TensorImpl directly via
  ``struct.pack`` + ``TensorImpl.from_bytes``, with dtype inference
  matching numpy semantics (`float → F32`, `int → I64`, `bool → Bool`).
  Ragged sequences, BF16 / complex64 target dtypes, and ``ndarray``
  inputs still go through the numpy bridge (with the existing
  ``pip install lucid[numpy]`` ImportError guidance when missing).
- **`lucid.autograd.{grad, backward}`, `Tensor.register_hook`, and the
  `lucid.func.{grad, jacrev, jacfwd, hessian}` family** no longer pull
  numpy in.  Internal grad accessors switched from
  ``grad_as_python()`` (returns numpy ndarray) to the existing
  ``grad_as_impl()`` (graph-mode grad) / ``grad_to_tensor()`` (detached
  grad) pair, which produce TensorImpls directly.
- **`lucid.nn.utils.rnn.pack_padded_sequence` /
  `pad_packed_sequence`** read `lengths` / `batch_sizes` /
  `unsorted_indices` via a new module-private helper that
  `struct.unpack`s the integer tensor's raw bytes — no numpy.
- **Wheel `LC_RPATH` dual-entry layout.**  3.0.1 set
  `INSTALL_RPATH "@loader_path/../../mlx/lib"` and
  `BUILD_WITH_INSTALL_RPATH ON`, which broke editable installs
  (`pip install -e .`): the build artifact's lone RPATH pointed at a
  wheel-style site-packages layout that doesn't exist in the source
  tree.  3.0.2 lists `@loader_path/../../mlx/lib` *first* (correct for
  wheels) and ``${LUCID_MLX_LIBRARY_DIR}`` *second* (correct for
  editable installs), tried in order by dyld.

### Changed

- **macOS Accelerate modernisation.**  The 3.0.x compile options
  carried `-Wno-deprecated-declarations` to silence the BNNS scalar API
  deprecation Apple introduced in macOS 15 SDK.  That flag was
  simultaneously hiding the *separate* CBLAS / CLAPACK legacy
  Fortran-name interface deprecation Apple introduced in macOS 13.3,
  which would surface the moment we removed the BNNS workaround.
  Resolved both in one pass:
  - **BNNS scalar API**: removed the Conv2d / BatchNorm2d / LSTM fast
    paths that called `BNNSFilter*`, `BNNSLayerParameters*`, and
    `BNNSDirectApplyLSTM*`.  Conv and BatchNorm fall through to the
    existing CPU im2col / column reduction paths.  LSTM inference now
    delegates to ``CpuBackend::lstm_forward_train`` and trims the
    returned tuple to ``[out, hn, cn]`` (the proj_size > 0 branch
    already used this pattern).  Side-effect: F64 / bidirectional /
    multi-layer / no-bias LSTM inference, which previously failed the
    fast-path guards and threw ``not_implemented``, now works correctly.
  - **CBLAS / LAPACK new interface**: defined
    ``ACCELERATE_NEW_LAPACK`` globally on
    ``lucid_compile_options`` so all ``cblas_*`` and ``*_`` calls
    route to the new symbol layout, and switched ``Lapack.cpp``'s
    ``using i32 = __CLPK_integer`` to ``__LAPACK_int`` (the new
    typedef, ABI-identical on LP64 macOS).
  - With both fixed, ``-Wno-deprecated-declarations`` is dropped —
    future Accelerate deprecations now surface immediately.

### Tooling

- **Smoke step doesn't silently rebuild after host MLX strip.**
  `scripts/ci_publish.sh` detects an already-built wheel in `dist/`
  and reuses it instead of running `pip wheel .` again.  Required by
  `publish.yml`'s new flow: build → strip host MLX from runner →
  smoke against the artefact in a fresh venv (catches RPATH absolute-
  path regressions like 3.0.0's).

### Documentation

- `obsidian/api/api-cpp-tree.md` lists `TensorImpl.transfer_to_device`.
- `obsidian/api/api-python-toplevel.md` notes the `lucid.tensor()`
  numpy-free fast path.

---

## [3.0.1] — 2026-05-16

Hotfix for a dylib RPATH bug in 3.0.0 that made the wheel unusable on
fresh installs.

### Fixed

- **`engine.cpython-*-darwin.so` RPATH baking** — 3.0.0 baked the
  build machine's absolute MLX library path
  (`/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/mlx/lib`)
  as the wheel's only `LC_RPATH` entry. At runtime `dyld` followed
  that absolute path verbatim, ignoring the user's venv-installed MLX
  in `site-packages/mlx/lib/`. On any machine whose framework Python
  had a different MLX version (or no MLX at all), `import lucid`
  failed with `Symbol not found: __ZN3mlx4core10as_strided...`.
  Fixed by switching `lucid/_C/CMakeLists.txt` to
  `INSTALL_RPATH "@loader_path/../../mlx/lib"` +
  `BUILD_WITH_INSTALL_RPATH ON` +
  `INSTALL_RPATH_USE_LINK_PATH OFF`, so the .so resolves libmlx
  relative to its own location inside `site-packages/lucid/_C/`.
  Works in venv and system Python equivalently.

### Tooling

- **`release-testpypi.yml` smoke hardening** — the editable install
  in the build-deps step (`pip install -e ".[test]"`) was masking
  RPATH regressions because it kept the build env's MLX 1:1 with the
  baked path. Smoke now `pip uninstall -y mlx` after the wheel is
  built, then re-installs MLX into a clean venv and imports — exactly
  what a real user sees. Any future RPATH absolute-path leak fails
  the workflow at this step.

---

## [3.0.0] — 2026-05-16

First production release. Lucid is now PyTorch-compatible across the public
surface (~100% parity in every measured module) and runs natively on Apple
Silicon via MLX (GPU) and Apple Accelerate (CPU). The C++ engine has been
fully rewritten under a new OOP architecture (IBackend / Dispatcher / OpSchema
/ kernel framework) and is the single source of truth for numerics.

**Platform support:** macOS 26 (Tahoe) or later on Apple Silicon (M1–M4),
Python 3.14. Wheels are published as `cp314-cp314-macosx_26_0_arm64`. MLX
0.31+ is bundled as a hard runtime dependency (engine.so links against
libmlx.dylib with RPATH baked in at build time).

### Added — New Modules

- **`lucid.fft`** — full 22-function module: `fft`/`ifft`/`fft2`/`ifft2`/`fftn`/`ifftn`,
  `rfft`/`irfft`/`rfft2`/`irfft2`/`rfftn`/`irfftn`, `hfft`/`ihfft`/`hfft2`/`ihfft2`/
  `hfftn`/`ihfftn`, `fftshift`/`ifftshift`/`fftfreq`/`rfftfreq`. Backward through
  `fft`/`ifft`/`rfft`/`irfft` etc. is implemented; `norm` ∈ {`'backward'`, `'ortho'`,
  `'forward'`} matches PyTorch semantics.
- **`lucid.signal.windows`** — 12 spectral windows: `bartlett`, `blackman`,
  `cosine`, `exponential`, `gaussian`, `general_cosine`, `general_hamming`,
  `hamming`, `hann`, `kaiser`, `nuttall`, `triangular`. All composite, no
  engine work.
- **`lucid.special`** — sub-package with 33 functions: `erf`/`erfc`/`erfinv`/
  `erfcx`, `i0`/`i0e`/`i1`/`i1e`, `ndtr`/`ndtri`/`log_ndtr`, `xlog1py`/`xlogy`/
  `entr`, `digamma`/`polygamma{0,1,2,3}`/`multigammaln`, `lgamma`,
  `spherical_bessel_j0`, plus Bessel J/Y/K (arbitrary order via Miller's
  algorithm), Hurwitz ζ, and orthogonal polynomials (Hermite, Legendre,
  Chebyshev, Laguerre).
- **`lucid.distributions`** — 26 distributions, 9 transforms, 10 KL-pair
  closed forms, MC fallback in `kl_divergence`. Includes `Distribution` /
  `ExponentialFamily` bases, `Independent`, `TransformedDistribution`, full
  `constraints` registry, `kl_divergence` registry. Univariate continuous:
  Normal, LogNormal, Uniform, Exponential, Laplace, Cauchy, Gamma, Chi2, Beta,
  StudentT, Pareto, Weibull, HalfNormal, HalfCauchy, FisherSnedecor.
  Univariate discrete: Bernoulli, Geometric, Categorical, OneHotCategorical,
  Poisson, Binomial, NegativeBinomial. Multivariate: Dirichlet,
  MultivariateNormal, Wishart, LKJCholesky, MixtureSameFamily,
  RelaxedBernoulli, RelaxedOneHotCategorical (Concrete).
- **`lucid.amp`** — `autocast` context manager + `GradScaler` for mixed-precision
  training (fp16 / bfloat16 forward, fp32 master).
- **`lucid.profiler`** — `profile()` context manager + `record_function`,
  CPU and GPU timing, kernel-level breakdown.
- **`lucid.metal`** — public Metal escape hatches: `run_kernel()` for custom
  Metal shaders, `shared_tensor()` / `to_shared()` / `is_shared()` for
  zero-copy CPU↔GPU `MTLResourceStorageModeShared` buffers, `is_available()`,
  `synchronize()`.
- **`lucid.einops`** — `rearrange`, `reduce`, `repeat`, `pack`, `unpack`,
  `EinopsError`. (Sub-package canonical path only — no top-level alias.)
- **`lucid.serialization`** — `save` / `load` (PyTorch-compatible
  `weights_only=True` default), `save_sharded` / `load_sharded` (multi-file
  checkpoints with `index.json`), `map_location`.
- **`lucid.func`** — functional transforms: `vmap`, `grad`, `grad_and_value`,
  `vjp`, `jvp`, `jacrev`, `jacfwd`, `hessian`, `linearize`. `vmap` Stage 2
  adds element isolation for `vmap(jacrev/jacfwd/hessian)` via the
  `_ISOLATION_ATTR` marker; explicit `strategy='auto'|'isolated'|'vectorized'`
  dispatch; `randomness='error'` enforced through a `_vmap_ctx` thread-local;
  `chunk_size` respected in isolation mode for bounded peak autograd-graph
  memory; `linear_fn` from `linearize` auto-tagged for isolation so
  `vmap(lin)` uses correct per-tangent jvp dispatch. H9-compliant
  `lucid/func/__init__.pyi` covers all 9 public transforms.
- **`lucid.models`** — model zoo with config / registry / Auto / Hub /
  pretrained-checkpoint infrastructure. See _Added — Model Zoo_ below.

### Added — Engine Surface

- **Complex dtype**: `complex64` end-to-end (`real` / `imag` / `complex` / `conj`
  engine ops on both CPU=vDSP and GPU=mlx), plus composites `angle` / `polar` /
  `view_as_real` / `view_as_complex`. C64 backend extensions for `full` / `ones` /
  `mul`.
- **DLPack interop**: `from_dlpack` / `to_dlpack` + `Tensor.__dlpack__` /
  `Tensor.__dlpack_device__` (zero-copy when device + dtype match, NumPy bridge
  fallback otherwise).
- **NumPy independence**: import / repr / serialize / grad paths are all
  NumPy-free. NumPy is now an _optional_ extra used only at the 6 documented
  bridge boundaries.
- **`Generator` + RNG state**: `seed`, `initial_seed`, `manual_seed`,
  `get_rng_state`, `set_rng_state`. Philox-4x32-10 counter-based PRNG with
  external mutex for shared use.
- **Bitwise shifts**: `bitwise_left_shift`, `bitwise_right_shift` on both CPU
  and MLX.
- **`nextafter`** (CPU-only with GPU round-trip).
- **Index ops**: `put` / `index_put` / `index_put_` (composite via `scatter` +
  flat-index reduction).
- **Sampling**: `poisson` (Knuth for λ<30, Normal-approx for λ≥30, threaded
  through Lucid Philox).
- **Histogram**: `histogram2d`, `histogramdd` composites.
- **Engine ops**: `erf`, `erfinv`, `cummax`, `cummin`, `scatter_amax/amin/prod`,
  `clip` / `clamp` with scalar bounds.

### Added — `torch.nn`, `torch.nn.functional`, `torch.linalg`, etc.

- **`nn` modules** (≥30 new classes): MaxUnpool1d/2d/3d, FractionalMaxPool2d/3d,
  ReflectionPad3d, CircularPad1d/2d/3d, ChannelShuffle, SoftMarginLoss,
  MultiLabelSoftMarginLoss, TripletMarginWithDistanceLoss, Threshold, Hardtanh,
  LogSigmoid, ConstantPad1d/2d/3d, Transformer / TransformerEncoder /
  TransformerDecoder, FusedLinear, lazy variants of Conv* / ConvTranspose* /
  BatchNorm* / InstanceNorm*, MultiheadAttention with full attention contract.
- **`nn.functional`** (≥13 new): hardtanh, logsigmoid, softsign, threshold,
  lp_pool1d/2d, max_unpool1d/2d/3d, local_response_norm, soft_margin_loss,
  multilabel_soft_margin_loss, channel_shuffle, pdist, fused_linear_relu/gelu,
  pixel_shuffle / pixel_unshuffle, multi_head_attention_forward,
  `fractional_max_pool2d` / `fractional_max_pool3d`.
- **`nn.utils`** — 100% parity: `clip_grad_norm_`, `clip_grad_value_`,
  `parameters_to_vector`, `vector_to_parameters`, `weight_norm` /
  `remove_weight_norm`, `parametrize` framework, RNN utils
  (`pack_sequence` / `pad_sequence` / `pack_padded_sequence` /
  `pad_packed_sequence`), `prune` package, `copy_parameters_and_buffers`,
  `fusion.fuse_conv_bn_eval`.
- **`nn.init`** — 100% parity (13 functions including `trunc_normal_`,
  `kaiming_*`, `xavier_*`, `orthogonal_`, `dirac_`, etc.).
- **`linalg`** — 100% parity (37 functions). New: `cholesky_ex`/`inv_ex`/
  `solve_ex` (info-flag variants), `lu` (P/L/U extraction from `lu_factor`),
  `ldl_solve` (1×1 pivot), `diagonal`. Backward implemented for `cholesky`,
  `eigh`, `svd`, `qr`, `pinv`, `matrix_power` (25 gradcheck tests pass).
- **`autograd`** — `set_detect_anomaly` / `is_anomaly_enabled`,
  `autograd.profiler` namespace, `autograd.graph.allow_mutation_on_saved_tensors`
  (engine-backed), `autograd.graph.save_on_cpu` (stub), `Tensor.register_hook` +
  `RemovableHandle`, `checkpoint`, `enable_grad` fix.
- **`utils.data`** — 100% parity: `default_convert`, `collate`, `ChainDataset`,
  `StackDataset`, `DistributedSampler`.
- **`optim`** — proper `state_dict` round-trip including LBFGS state buffers.

### Added — Tensor / Top-level Polish

- Tensor PyTorch parity APIs: `itemsize`, `stride`, `data_ptr`, `storage_offset`,
  `H`, `type()`, `get_device`, `pin_memory`, `is_cuda` (always False on Apple
  Silicon), `reshape_as`, `untyped_storage`, `expand(-1)` correctness fix.
- Tensor convenience: `lerp`, `diff`, `scatter_`, `index_*` family,
  `register_hook`, `__iter__`, `__format__`, `new_*` factories,
  `element_size`.
- Top-level composite gap closure: `randperm`, `count_nonzero`, `frexp`,
  `tril_indices` / `triu_indices` / `combinations`, `finfo` / `iinfo`,
  `flip` / `fliplr` / `flipud`, threading stubs, determinism aliases,
  `relu` / `sigmoid` top-level.

### Added — Apple Silicon Native Path

- **Memory pool** — thread-local slab allocator with 23 size classes,
  `kMaxDepth=32`, automatic free-list reuse for ≤ 4 MB allocations
  (`Allocator.cpp`).
- **MetalAllocator + SharedStorage** — `MTLResourceStorageModeShared` buffers
  exposed via `lucid.metal.shared_tensor()` / `to_shared()`. Zero memcpy when
  cross-device transfer is on a SharedStorage tensor.
- **MetalKernelRunner** — `lucid.metal.run_kernel(source, inputs, outputs,
  threadgroups)` allows arbitrary user-supplied Metal compute kernels with full
  argument marshaling and output tensor allocation.
- **FusionPass** — `nn.FusedLinear` + `F.fused_linear_relu` /
  `fused_linear_gelu`. Inference path is a fused C++ kernel; training falls back
  to standard autograd for gradient correctness.
- **BNNS fast paths** — Conv1d/2d, BatchNorm1d/2d use Apple BNNS when
  applicable; LSTM uses BNNS for inference (proj_size supported).

### Added — Model Zoo

- **Foundation** — `ModelConfig` / `PretrainedModel` / `ModelOutput` /
  `Registry` / `Auto*` / `Hub` / mixins; 30 dedicated tests, mypy --strict
  clean. `models._registry.ModelFactory` is a `Protocol` with explicit
  `__name__` + `__call__` signature; `_RegistryEntry.model_class` +
  `default_config` fast-path fields. `AutoConfig.from_pretrained` returns
  `default_config` instantly when pre-registered (avoids full instantiation);
  `load_from_pretrained_entry` validates `entry.config.model_type ==
  model.config.model_type` before downloading weights.
- **Auto-classes** for every task tag: `AutoModel`, `AutoModelForCausalLM`,
  `AutoModelForMaskedLM`, `AutoModelForSeq2SeqLM`,
  `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`,
  `AutoModelForQuestionAnswering`, `AutoModelForImageClassification`,
  `AutoModelForObjectDetection`, `AutoModelForSemanticSegmentation`,
  `AutoModelForImageGeneration`.
- **Vision — image classification (~156 registered variants):** LeNet-5
  (original tanh+avg + modern relu+max), AlexNet (paper-faithful 96/256/384/
  384/256 + LRN), VGG 11/13/16/19 + BN (VGG-16 138,357,544 params paper-exact),
  GoogLeNet / Inception v1 (with auxiliary classifiers, 0.3× weighted, 13.4M
  params), ResNet 18/34/50/101/152, DenseNet 121/169/201/264 (DenseNet-121
  7,978,856 params), Inception v3/v4/Inception-ResNet, Xception, MobileNet
  v1/v2/v3/v4, EfficientNet B0–B7 (B0 5,288,548 params reference-exact),
  ResNeXt, SENet, SKNet, ResNeSt, CSPNet, ConvNeXt T/S/B/L/XL (ConvNeXt-T
  28,589,128 params), ViT B/16 B/32 L/16 L/32 H/14 (ViT-B/16 86,567,656
  params), Swin T/S/B/L (Swin-T 28,288,354 params), CoAtNet, CvT, CrossViT,
  PVT, EfficientFormer, MaxViT, InceptionNeXt, ZFNet.
- **Vision — object detection:** R-CNN (AlexNet warped crop, class-specific
  bbox regression), Fast R-CNN (VGG16 + RoI Pool 7×7 + 2-FC head; smooth-L1
  σ=1), Faster R-CNN (VGG16 + RPN + RoI Pool; smooth-L1 σ=3 RPN; 9 anchors/
  cell), Mask R-CNN (ResNet-50-FPN + RoI Align + mask FCN 14→28 deconv),
  DETR R50/R101 (ResNet + transformer encoder-decoder + Hungarian set-
  prediction; 100 queries), EfficientDet D0–D7 (EfficientNet-B0 + BiFPN
  fast-normalised weighted fusion + focal + smooth-L1), YOLO v1/v2/v3/v4 +
  tiny (custom Darknet / Darknet-19 / Darknet-53 / CSPDarknet-53; YOLOv4 uses
  Mish per paper §3.4).
- **Vision — segmentation:** FCN (dilated ResNet-50/101 stride 8 + FCN head +
  aux head 0.4 weight), U-Net 2D/3D + ResUNet 2D/3D (`dim` switches Conv2d↔
  Conv3d / bilinear↔trilinear; `block` toggles residual DoubleConv),
  Attention U-Net (additive Wx+Wg→ReLU→ψ→sigmoid gates), MaskFormer (ResNet
  18/34/50/101 + FPN pixel decoder + N mask queries + Hungarian mask-cls
  loss), Mask2Former (ResNet 18/34/50/101 _or_ Swin T/S/B/L + multi-scale
  FPN + masked cross-attention with per-layer FPN level cycling).
- **Text (39 registered variants):** Transformer (Vaswani et al., 2017) base/
  large + seq2seq + cls + token-cls heads; BERT (Devlin et al., 2019) tiny/
  mini/small/medium/base/large + MLM/SequenceCls/TokenCls/QA heads; GPT
  (Radford et al., 2018) base + LM/Cls heads; GPT-2 (Radford et al., 2019)
  small/medium/large/xlarge + LM/Cls heads; RoFormer (Su et al., 2021) base +
  MLM/SequenceCls/TokenCls heads. Shared `_utils._text` infra (causal masks,
  position ids, RoPE).
- **Generative (16 registered variants):** VAE (Kingma & Welling, 2013)
  vanilla + hierarchical Sønderby/Ladder + image-gen heads; DDPM (Ho et al.,
  2020) CIFAR-10 / LSUN-256 / ImageNet-64 + image-gen heads; NCSN/NCSNv2
  (Song & Ermon, 2019) CIFAR-10 / CelebA-64 + image-gen heads. Shared
  `_utils._generative` infra (β / σ schedule helpers, `DiffusionScheduler`
  base, `DDPMScheduler` ancestral sampler, annealed Langevin dynamics).
- **Output dataclasses:** `ObjectDetectionOutput` (with optional `proposals`
  field so `postprocess(output)` works without re-running the RPN),
  `InstanceSegmentationOutput`, `SemanticSegmentationOutput`.
- **Shared utilities (`models._utils`):** `_common` (`make_divisible`
  canonicalised across 7 model families), `_classification` (`LayerScale`,
  `DropPath` with linear schedule across the trunk), `_detection` (box ops,
  NMS, AnchorGenerator, roi_align/roi_pool, FPN, RPN, RoIHead shared modules),
  `_kuhn_munkres_rectangular` (textbook Hungarian/JV implementation shared by
  DETR / MaskFormer / Mask2Former matchers; verified against
  `scipy.optimize.linear_sum_assignment` on 100+ random matrices).
- **safetensors round-trip:** `save_safetensors` / `load_safetensors` (optional
  `pip install lucid-dl[test]` brings in `safetensors`; clear `ImportError` if
  missing). `lucid.load()` auto-detects `.safetensors` extension and delegates.
  `PretrainedModel.save_pretrained(safe_serialization=True)` saves
  `model.safetensors` instead of `weights.lucid`. 0-d tensors (BatchNorm
  `num_batches_tracked`) round-trip via a `(1,)` + metadata tag, squeezed back
  to `()` on load.

### Added — Build / Distribution

- **macOS 26 (Tahoe) build target** — `MACOSX_DEPLOYMENT_TARGET=26.0` baked
  into `setup.py` default and the publish workflow. Wheels are tagged
  `cp314-cp314-macosx_26_0_arm64`.
- **PEP 561 typed package** — `lucid/py.typed` marker shipped in the wheel so
  mypy / pyright recognise lucid as a typed package. `pyproject.toml`
  `[tool.setuptools.package-data]` extended to bundle `py.typed`, all `*.pyi`
  stubs, and registry `*.json` files.
- **Trusted-publishing pipeline** — `publish.yml` rewritten to use PyPI OIDC
  trusted publishing (no API token), `python -m build --no-isolation` to
  preserve the libmlx.dylib RPATH, version-derived-from-tag with three-way
  consistency check against `pyproject.toml` and `lucid/version.py`. Test PyPI
  staging via `release-testpypi.yml` on the same `v*` tag push.

### Changed

- **`axis` → `dim`** — engine-wide rename to match PyTorch. Old `axis` /
  `axes` kwargs accepted via explicit `__signature__` shim where the engine
  function name still uses `axis` internally.
- **Sub-package canonical paths (H8)** — `linalg` ops are accessible only via
  `lucid.linalg.*`, einops only via `lucid.einops.*`. Top-level shortcuts
  (`lucid.norm`, `lucid.cross`, `lucid.einsum`, `lucid.vander`, etc.) and
  Tensor method aliases (`tensor.norm()`, `tensor.cross()`) **removed** —
  every op now has exactly one path.
- **Strict typing (no `Any` in stubs)** — `.pyi` files have zero `Any`. All
  function annotations use `lucid._types` aliases or fall back to `object`.
  `_types_base.py` was merged into `_types.py`.
- **No string type hints** — `from __future__ import annotations` removed
  globally; `TYPE_CHECKING` block + bare names used (Python 3.14 lazy
  annotations).
- **NumPy demoted to optional** — `pip install lucid-dl` no longer requires
  NumPy. Use `pip install lucid-dl[numpy]` for `from_numpy` / `.numpy()` /
  `from_dlpack` via NumPy. Six sanctioned bridge boundaries documented in
  `CLAUDE.md` H4.
- **`state_dict` v2** — `_load_from_state_dict` matches PyTorch signature;
  `_metadata` round-trip; `_version` keys preserved; `assign=` parameter
  supported.
- **Tier-1 namespace hygiene** — `Module` / `Parameter` / `Linear` / `Adam` are
  no longer accessible under the top-level `lucid.*` namespace; they live
  under their proper sub-package (`lucid.nn.*`, `lucid.optim.*`).
- **Builtin shadowing fixed** — `from lucid import *` no longer pollutes
  `float` / `int` / `bool` / `bytes`.
- **MLX dependency pin** — `mlx>=0.29` → `mlx>=0.31`. 0.31 is the first
  release that ships the native `macosx_26_0_arm64` MLX wheel and the
  `mlx-metal` split package matching our build target.
- **`ModelConfig.from_dict`** — unknown fields now warn+ignore instead of
  raising (forward-compatible checkpoint loading).
- **`PretrainedModel.config_class`** — default changed from `ModelConfig` to
  `None`; concrete subclasses that forget to set it now get a clear
  `TypeError`.
- **`_load_from_directory`** — no longer instantiates the model twice; uses
  `model_class` fast path when registered, else one factory call.

### Fixed

- **Cholesky `upper=True` backward** — gradient was using `tril` projection
  unconditionally; now correctly switches to `triu` when `upper=True` (Murray's
  formula).
- **`Conv*(bias=False)`** — engine binding now accepts `None` for the bias
  parameter; `Module.__setattr__` shadow fix prevents the attribute from
  leaking back into `_parameters`.
- **MaxPool backward + LSTM training** — both now run fully Metal-native
  (no GPU→CPU fallback during the backward pass).
- **GPU `scatter_add`** — wired correctly to MLX `scatter_add_axis`; previously
  fell back to CPU.
- **All engineering-fixable GPU→CPU fallbacks eliminated** — only true
  data-dependent ops (e.g. `nonzero`) round-trip through CPU, by design.
- **`flip` backward** — was silently returning `None`; now properly inverted.
- **`det` backward (batched)** — GPU was reducing over wrong axes for batched
  input; broadcast fix matches reference framework.
- **0-d `reduce_axes` recursion** — fixed infinite recursion when reducing a
  scalar tensor.
- **`expand(-1)`** — `-1` now correctly preserves the existing dimension size
  (was being treated as an error).
- **`upload_cpu_to_gpu()`** — uses `mlx::core::copy(external)` to schedule a
  Metal blit into a GPU-private buffer rather than wrapping as a SharedStorage
  external array. After the first eval, the array is fully native and avoids
  the ~131 µs/op external-array bandwidth penalty.
- **H5/H7 Hard Rule violations in `lucid.func` and parity tests** — purged.
- **`lucid.func.jvp` scalar output shape** — α gradient was `(1,)` instead of
  `()` for scalar primal outputs.
- **`CosineAnnealingWarmRestarts`** — reset `T_cur` / `T_i` before computing
  LR so restart epoch returns `base_lr` (not `eta_min`).
- **`ReduceLROnPlateau`** — patience check `>=` → `>` to match reference (was
  reducing one epoch too early).
- **`OneCycleLR`** — warmup end = `total_steps*pct_start - 1` (not floor);
  `init_lr = max_lr/div_factor` regardless of optimizer LR.
- **`nn.Transformer`** — added final `LayerNorm` to encoder and decoder by
  default; was missing 4 parameters vs the reference.
- **R-CNN family — RPN anchor ordering** — `Conv2d` output `(B, A, H, W)` was
  being flattened anchor-major while `AnchorGenerator` emits spatial-major
  `(G·A, 4)`. Fixed by permuting predictions to spatial-major before flatten.
- **YOLOv3 detection head — channel-count mismatch** — `_Darknet53` returns
  `p3_raw=128ch` / `p4_raw=256ch` but the head was built for 256/512. Rewired
  to use actual backbone widths.
- **`.clamp()` is positional-only in Lucid** — replaced every `clamp(min=...)`
  / `clamp(max=...)` single-kwarg call (in `_detection.py`, YOLOv1) with
  `clamp(low, high)`.
- **`lucid.tensor([int_list])` defaults to float32** — added `.long()` to all
  index-tensor construction sites in `_detection.py` + 4 R-CNN family models.
- **Device propagation across 30+ sites in detection train / postprocess
  paths** (R-CNN, Fast/Faster/Mask R-CNN, EfficientDet): every
  `lucid.zeros(...)`, `lucid.tensor([...])`, `lucid.full(...)` in the loss
  helpers / postprocessors now derives `device=` from input tensors so the
  models work on Metal training.
- **DETR / MaskFormer / Mask2Former Hungarian matcher** — custom JV variant
  iterated over the wrong axis and returned non-optimal assignments even on
  trivial inputs (a 5×3 trivial match returned 1/2/4 instead of 0/1/2).
  Replaced with a textbook rectangular Kuhn-Munkres implementation that
  cross-checks against `scipy.optimize.linear_sum_assignment`.
- **MaskFormer pixel decoder** — `out3` / `out4` / `out5` 3×3 smoothing convs
  were declared but never applied in forward (dead parameters + silent paper-
  fidelity deviation). Every FPN level now passes through its own smoothing
  conv per paper §3.2.
- **MaskFormer / Mask2Former `_binary_mask_iou`** — vectorised; the per-pixel
  Python double-loop with `.item()` (O(H·W) device→host syncs per call) was
  replaced with `(p>0.5).float() * (g>0.5).float()` + a single `.sum().item()`.
- **Swin `rel_pos_idx`** — re-registered as a non-persistent buffer (was a raw
  attribute via `object.__setattr__`, so `.to(device=...)` left it on CPU and
  broke metal-side `rel_pos_bias[idx]`).
- **Swin `_attn_mask`** — takes `device=x.device.type` so the shifted-window
  mask is built on the same device as activations.
- **MaxViT `_MaxViTBlock`** — pad spatial dims to `window_size` multiple before
  grid/window partition to handle non-divisible resolutions (e.g. 28×28 with
  `ws=7`).
- **MaxViT docstring** — replaced "Standard PyTorch padding=1" with framework-
  neutral wording (H5).
- **Paper-faithful audit pass on the model zoo** (closes the remaining ⚠️
  deviations flagged in the Wave-3 retrospective):
  - **EfficientDet BiFPN** — removed `.item()` round-trip in fast-normalised
    weighted fusion (per-step host sync removed).
  - **CoAtNet `_rel_idx`** — registered as non-persistent buffer so
    `.to(device=...)` works.
  - **EfficientNet stochastic depth** — was applied unconditionally; now
    respects `training` flag and per-block survival-probability schedule
    (Tan & Le 2019 §3.3).
  - **R-CNN family class-specific decode** — Fast / Faster R-CNN now decode
    bbox deltas with the predicted top-class deltas (paper §3.3) instead of
    class-0 / argmax-of-bg-included.
  - **ResNeSt `is_first` flag** — first block of each stage receives the
    correct `is_first=True` to drop the redundant 1×1 down-projection.
  - **MaskFormer / Mask2Former dice loss** — corrected denominator from
    `|p|·|g|` (cosine-style) to `|p|+|g|` per Milletari 2016.
  - **YOLOv1 w/h decoding** — paper §2 / Eq.1 uses sigmoid-bounded direct
    prediction (`sigmoid(raw)·{W,H}`); was incorrectly using YOLOv2's
    `exp(raw)·{W,H}` anchor formulation. Loss term updated to MSE on
    `√w_norm`, `√h_norm`.
  - **CvT `stride_kv`** — paper Table 1 specifies stride=2 for K/V conv-
    projection in *all* three stages; was only stage 0.
  - **CrossViT classification head** — paper §3.3 averages two per-branch
    classifier logits; was concat → single FC.
  - **MobileNetV2 `last_ch`** — `last_ch = make_divisible(1280·max(1,
    width_mult))` per paper §3.4 / torchvision; was hard-coded 1280 for all
    width multipliers.
  - **DDPM `learn_sigma=True`** — now raises `NotImplementedError` (Improved-
    DDPM hybrid `L_simple + L_vlb` loss not yet implemented) instead of
    silently emitting an unusable variance head.
  - **Inception v3 auxiliary classifier** — moved from after `inception_c1`
    (35×35) to after `inception_c3` (last 17×17 = Mixed_6e) per paper §6 /
    Fig.10.
  - **SKNet `_SKAttentionGate`** — `AdaptiveAvgPool2d(1)` lifted into
    `__init__` (was instantiated each forward call).
  - **EfficientFormer LayerScale + DropPath** — added per-residual-branch γ
    (init 1e-5) and linear stochastic-depth schedule per paper §4.1 (max-rate
    0.0 / 0.1 / 0.2 for L1 / L3 / L7).

### Performance

- **GPU `relu`** — 78 % overhead removed: `zeros_like(x)` (full-tensor
  allocation) replaced with broadcast scalar `array(0.0, dtype)`. Same fix
  applied to `elu_backward` (1.0 scalar instead of `ones_like`).
- **MLX template overhead** — removed redundant `::mlx::core::contiguous()`
  calls from `mlx_unary` / `mlx_binary` / `mlx_reduce` (every op was paying
  for an extra MLX graph node it didn't need). Added `mlx_unary_contiguous()`
  variant for ops that genuinely require contiguity.
- **`eval_gpu()` single-tensor fast path** — `_C_engine.eval_gpu(impl)` skips
  the ~25 µs Python list-construction overhead of `eval_tensors([impl])`.
- **SharedStorage zero-copy CPU↔GPU** — for SharedStorage-backed tensors,
  `.to('metal')` and `.to('cpu')` are now zero memcpy (relabel via
  `transfer_storage()`).
- **`.to('metal')` for regular tensors** — single Metal blit to GPU-private
  memory (was 2 copies via Python round-trip). Subsequent ops pay no
  external-array penalty.
- **NMS** — vectorised per-row IoU: replaced O(N²) pairwise
  `box_iou(box_i, box_j)` allocations with K vectorised
  `box_iou(boxes[idx:idx+1], boxes)` rows (K = number of survivors). Sort is
  now a single device-side `argsort` instead of N `.item()` calls inside
  Python `sorted`.
- **Anchor assignment** — Faster R-CNN / Mask R-CNN / EfficientDet RPN +
  RoI losses: replaced 2·A·M nested `.item()` loops with a single
  `argmax(dim=...)` / `max(dim=...)` reduction per axis and bulk
  materialisation. ~10× fewer device→host syncs.
- **Wave-3d unit test suite (CPU + Metal)** — 62 s → 53 s end-to-end as a
  result of the NMS / anchor-assignment vectorisation.

### Removed

- **Top-level shortcuts for sub-packages** — see _Changed_ above (H8).
- **`from __future__ import annotations`** — see _Changed_ above (H7).
- **scipy dependency** — `trunc_normal_` reimplemented without scipy.
- **`torch` / `PyTorch` literals from production code** — only allowed in
  `lucid/test/_fixtures/ref_framework.py` (test infra opt-in).
- **`cuda` references** — Apple Silicon only; `metal` is the GPU device name
  throughout.

### Tooling

- **`tools/new_op.py`** — op scaffolding CLI. Generates 9 boilerplate files
  (`.h` / `.cpp` + IBackend / CpuBackend / GpuBackend stubs + binding +
  CMake entry + `__init__.py` export + `_registry.py` `OpEntry`) in ~1 second.
  Supports `--kind unary|binary|composite`, `--save-input` / `--save-output`,
  `--amp keep|promote|fp32`, `--dry-run`. Auto-runs `gen_pyi.py` after apply.
- **`tools/gen_pyi.py`** — regenerates `engine.pyi`, `tensor.pyi`, and
  `__init__.pyi` from live runtime introspection. Strict typing, zero `Any`,
  `*args`/`**kwargs` only for genuinely variadic APIs (H9). `lucid.load()`
  stub now includes the `weights_only` parameter.
- **`tools/check_doxygen.py`** / `check_stubs.py` / `check_op_api.py` /
  `check_layers.py` / `check_op_template.py` / `check_kernel_template.py` /
  `check_phase1.py` — automated CI checks.
- **`tools/changelog.py`** — Keep-a-Changelog helper (add / propose / release
  / check).
- **Git hooks** — `.githooks/{post-commit,commit-msg}` for CHANGELOG hygiene.
- **`mypy --strict` baseline** — 0 errors locked in `mypy.ini`; only
  `operator` / `index` disabled. New code must pass before commit.
- **Test infrastructure rebuild (Phases 1-11)** — full from-scratch test
  layer in `lucid/test/`. 1574 unit tests pass (61 skipped). Cross-product
  CPU+Metal fixtures, lazy reference-framework loader, parity gating, golden
  numerical checks, integration train-loops (MLP / CNN / RNN / Transformer),
  microbench / e2e / memory perf tests, CI wiring. Adds 19 nn parity tests
  (LayerNorm / RMSNorm / GroupNorm / BatchNorm / InstanceNorm / LRN /
  MultiheadAttention / TransformerEncoderLayer), 4 transformer-decoder
  parity tests, 21 `optim.lr_scheduler` parity tests across 15 schedulers,
  25 `lucid.func` parity tests, 69 model-zoo detection + segmentation tests
  (including 6 scipy-cross-checked Hungarian correctness tests).
- **C++ Google Test suite** — 108 tests. Includes `Concurrency.*` stress
  tests covering thread-local allocator hammer, `MemoryTracker` counter
  consistency, and `Generator` mutex serialization.
- **Performance baseline suite (`benchmarks/`)** — A (self-regression with
  threshold guard) + B (vs. raw MLX) for ops, transfer, and training loops.
  `run_all.py --save` records baseline; `--check --threshold 15` fails if any
  result regresses by more than 15 %.
- **Hard Rules H1–H10** — fully enforced across the codebase. Verified by
  AST scan (zero violations).
- **Release pipeline** — `publish.yml` rewritten for tag-based trusted-
  publishing (PyPI OIDC, no API token); `release-testpypi.yml` gates against
  Test PyPI on the same `v*` tag push; both use `macos-26` Apple Silicon
  runners with `python -m build --no-isolation` to preserve libmlx RPATH.

### Documentation

- **Doxygen** — 184/184 = 100.0 % coverage of the public C++ engine surface.
- **`.pyi` stubs** — `engine.pyi`, `tensor.pyi`, `__init__.pyi`,
  `func/__init__.pyi` all up to date; verified by `tools/check_stubs.py`.
- **PEP 561** — `lucid/py.typed` marker shipped so external type checkers
  (mypy / pyright) recognise lucid as a typed package.
- **Obsidian vault (`obsidian/`)** — git-ignored team knowledge base
  documenting architecture decisions, engine quirks, op contracts, debugging
  recipes, performance numbers, retros, and roadmaps. Updated in real-time
  alongside code changes per `CLAUDE.md`.

---

## [Pre-3.0]

The 3.0 release is the project's first stable, externally consumable
release. Prior commits (~1300+) span the framework's iterative development
under the working titles _alpha-0.1_ through _alpha-0.14_, the lucid-1.x and
lucid-2.x experimental lines, and the lucid-3.0 OOP rewrite. No stable APIs
were guaranteed during that period and no semver was applied; users
upgrading from a pre-3.0 working copy should expect breaking changes
across every public surface.

---

[Unreleased]: https://github.com/ChanLumerico/lucid/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/ChanLumerico/lucid/releases/tag/v3.0.0
