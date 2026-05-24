# `lucid.compile` — User Guide

A pragmatic, honest guide to when `lucid.compile()` helps, when it
doesn't, and what its current limits are.  Numbers in this document
come from `python -m tools.bench_compile` on an Apple Silicon M-series
Mac running macOS 26 + MLX ≥ 0.31 + MPSGraph from the macOS 26 SDK.

---

## TL;DR

| Workload | Use compile? | Why |
|---|---|---|
| Inference on a **small CNN** (LeNet, MobileNet at 224²) | ❌ | Eager (MLX) is already fast; compile dispatch overhead dominates. |
| Inference on a **large transformer** (ViT-Base 224²) | ✅ | ~1.6× speedup; the larger the matmuls, the more compile wins. |
| Inference on **LSTM** | ◯ | Neutral — LSTM emits via the same MLX-backed Metal kernel that eager uses, so there's nothing left to fuse. |
| Training step on **small models** (< 5 ms eager step) | ❌ | `fused_step` Python overhead (~0.5 ms/step) wipes out compile gains. |
| Training step on **medium / large models** | ◯ | Theoretically wins for compute-bound steps; benchmark before adopting. |
| Stochastic dropout / data augmentation in training | ❌ (RNG falls back) | Compile-mode RNG is deterministic-per-executable; dropout in `.train()` mode auto-falls back to eager. |

**One-line guidance**: try `lucid.compile(model)` for inference-heavy
serving of large transformers, measure both paths, only adopt where
the speedup justifies the recompile cost (~50 ms per signature).

---

## 1. What gets compiled

Calling `lucid.compile(model)` wraps the model in a `CompiledModule`.
On the first `__call__` with a given input signature (shape × dtype ×
device × training-flag), it:

1. Traces the model's forward pass under `no_grad()` — every op
   that enters `OpScopeFull` is recorded.
2. Emits an MPSGraph for the recorded trace via the per-op emitter
   in `lucid/_C/compile/OpEmitters/`.
3. Compiles the graph into an `MPSGraphExecutable` and caches it
   under the input signature.

Subsequent calls with the same signature run the cached executable
directly, skipping Python dispatch entirely for the inner ops.

### Supported op surface

**141 of 173 traceable ops emit directly** (≈ 82 %).  The
deliberately-stubbed 32 ops fall back to eager — they fall into
five categories:

| Category | Ops |
|---|---|
| Iterative linalg (no MPSGraph primitive) | `cholesky`, `eig`, `eigh`, `svd`, `qr`, `solve`, `pinv`, `erfinv` |
| FFT (no MPSGraph primitive) | `fftn`, `ifftn`, `rfftn`, `irfftn` |
| 3-D pool / conv / interpolate (Apple SDK is 2-D only) | `max_pool3d`, `avg_pool3d`, `conv_transpose3d`, `interpolate_trilinear`, `interpolate_nearest_3d` |
| Dynamic gather / segment-reduce | `grid_sample`, `rotate`, `embedding_bag` |
| Other | `complex`, `nonzero`, `unique`, `histogram*`, `arange`/`eye`/etc. (factory) |

When the trace contains a stubbed op, the whole signature is
blacklisted as `eager_only` for the cache and routed to eager —
the result is still correct, just not compiled.

### Supported optimizers

`compile_optimizer` / `fused_step` accept 8 optimizer families:
**SGD** (incl. momentum + Nesterov), **Adam**, **AdamW**, **RMSprop**,
**Adagrad**, **Adadelta**, **Adamax**, **NAdam**.

These raise `NotImplementedError` with a structural reason:

- **LBFGS** — line search has data-dependent iteration count.
- **SparseAdam** — index-driven update needs runtime nonzero/scatter.
- **Rprop** — sign-based per-element conditional branching.
- **RAdam** — `ρ_t > 4` branch can't be a static MPSGraph op.
- **ASGD** — averaging coefficient depends on iteration count.

---

## 2. Required device

The compile path is **Metal-only**.  Both the model and inputs must be
on the `metal` device:

```python
import lucid

model = MyModel().to("metal")
x = lucid.randn(...).to("metal")

cm = lucid.compile(model)
y = cm(x)
```

If the model stays on CPU, `lucid.compile()` silently returns a wrapper
that runs eager — the trace records nothing because the CPU dispatch
path doesn't enter the `OpScopeFull` Metal hook.  This is
intentional (compile gracefully degrades) but easy to miss; if you
expected compile-mode speedup and see eager wall-clock, check
`model.parameters()`'s `.device`.

---

## 3. Numerical accuracy

Compile mode is **not** bit-exact with eager in general — both paths
lower to Metal, but with different fusion + reduction ordering.
Expected drift:

| Path | Typical relative diff |
|---|---|
| Inference (CNN / Transformer) | `1e-7` – `1e-5` |
| LSTM (shares the MLX backend kernel) | `0.0` (truly bit-exact) |
| Training step (forward + backward + update) | `1e-5` – `1e-3` cumulative over 100 steps |

Anything > `1e-3` rel-diff after a single step indicates a real bug —
file an issue with the trace structure and input shapes.

---

## 4. Dropout, RNG, and training mode

The compile-path RNG (`rand` / `randn` / `randint` / `bernoulli` /
`uniform` / `normal` / `dropout(training=True)`) is
**deterministic-per-executable**: every invocation of the same
compiled executable produces the same random sequence, baked in via
the MPSGraph descriptor's `seed` field at compile time.

This means:

- **Inference dropout** (`p=0` or `model.eval()`): compiles cleanly,
  fully fused with surrounding ops.
- **Training dropout** (`p>0` and `model.train()`): the emitter
  returns `nullptr` → builder aborts → trace falls back to eager.
  Dropout's regularising randomness is preserved (eager applies the
  RNG correctly).

If you depend on per-step stochasticity (training loops, data
augmentation, Monte-Carlo sampling), keep those parts in eager.

---

## 5. Cache behaviour

The compile path uses two cache layers:

1. **Python-side `CompiledModule._cache`** — unbounded `dict`
   keyed by `(args_signature, kwargs_signature, training,
   param_fingerprint)`.  Dropped on `.train()` / `.eval()` /
   `.to()` / `.load_state_dict()` / `.clear_cache()`.
2. **C++ `ExecutableCache::session()`** — process-global LRU
   bounded by `LUCID_COMPILE_MAX_CACHE` (default 32).

The C++ cache key includes per-op attribute payloads as of the
2026-05-24 fix — without this, two traces with identical structure
but different attributes (e.g. `dropout(p=0)` vs `dropout(p=0.5)`)
collided and one caller silently received the other's executable.

### Long-running services

If your workload compiles many distinct signatures (variable-length
NLP, dataloader with remainder batches), tune the LRU cap up:

```bash
LUCID_COMPILE_MAX_CACHE=128 python serve.py
```

Each executable holds a Metal command buffer + kernel state; budget
roughly 1–10 MB per cached signature for typical models.

### Inspecting cache state

```python
cm = lucid.compile(model)
cm(x)
info = cm.cache_info()
# {'entries': 1, 'keys': (CacheKey(...),), 'eager_only': (), 'n_calls': {...}}

cm.timing()
# [{'key': ..., 'compile_ms': 47.5, 'last_run_ms': 0.25, 'n_hits': 1}]
```

---

## 6. Performance (M-series Mac, macOS 26, MLX 0.31)

### Inference

| Model | Eager (ms) | Compile (ms) | Speedup |
|---|---:|---:|---:|
| LeNet-5 (1×1×32²) | 0.05 | 0.56 | **0.08×** |
| ResNet-18 (1×3×224²) | 0.53 | 3.71 | **0.14×** |
| MobileNet v1 (1×3×224²) | 0.60 | 1.94 | **0.31×** |
| EfficientNet-B0 (1×3×224²) | 1.75 | 4.37 | **0.40×** |
| DenseNet-121 (1×3×224²) | 3.04 | 9.20 | **0.33×** |
| **ViT-Base/16 (1×3×224²)** | 29.28 | 18.26 | **1.60×** ✅ |
| LSTM (50×16×64) | 0.39 | 0.40 | 0.97× |
| LSTM-large (200×32×64) | 1.44 | 1.42 | 1.01× |

**Reading the table**: compile mode currently wins only for the
largest transformer.  Small / medium CNNs lose to MLX-eager because
Python dispatch + Metal command submission is already O(0.1 ms).  This
is the inverse of the `torch.compile` story on CUDA, where eager
Python overhead is closer to O(1 ms) per op.

### Training step (`fused_step`)

| Model | Eager step (ms) | Fused step (ms) | Speedup |
|---|---:|---:|---:|
| MLP 64→128→10 (BS=32) | 0.06 | 0.72 | 0.08× |
| Deep MLP ×4×256 (BS=64) | 0.12 | 1.20 | 0.10× |

Training compile currently has ~0.5–1 ms of Python orchestration
overhead per step (feed-list construction, output unpacking).  For
small models the eager Adam step is already so fast that this
overhead dominates.  Larger models or batched gradient accumulation
should see the trade-off flip — benchmark your actual workload
before adopting.

---

## 7. Known limitations

- **Dynamic shapes**: `lucid.compile(..., dynamic=True)` raises
  `NotImplementedError`.  MPSGraph's symbolic-shape lowering is too
  unstable to be production-safe (MLIR aborts on conv-heavy graphs).
  Use static shapes and let the per-batch-size cache populate.
- **Control flow**: `if` / `while` inside `forward()` is traced as
  the branch taken on the first call.  Models that need
  data-dependent control flow can't compile.
- **Single-thread tracing**: don't share a `Tracer` across threads;
  the underlying TLS slot is per-thread by design.
- **Backward in training compile**: `fused_step` uses MPSGraph's
  `gradientForPrimaryTensor:` autodiff — gradient values are
  numerically identical to eager's hand-rolled `*Backward` autograd
  to ~`1e-5` rel-diff, not bit-exact.

---

## 8. When to use compile

**Yes, use it for**:

- Inference serving of large transformers (ViT, GPT-style).
- Repeated inference on a small set of shapes (where amortising the
  ~50 ms first-compile cost makes sense).
- Educational / reproducibility cases that want a deterministic
  Philox stream baked per executable.

**Don't bother for**:

- One-off training experiments — eager wins on dispatch overhead.
- Workloads with constantly-changing shapes (every shape pays the
  recompile cost; cache thrashes once you pass the LRU cap).
- Anything where `model.training=True` *and* you depend on per-step
  RNG variation (dropout, augmentation, MC sampling).

---

## 9. Debugging compile failures

If `lucid.compile(model)(x)` produces results that match eager but
seems unexpectedly slow, the trace probably aborted to eager.  Check:

```python
cm = lucid.compile(model)
cm(x)
print(cm.cache_info()["eager_only"])  # non-empty ⇒ trace fell back
```

To see which op the builder rejected, set:

```bash
LUCID_COMPILE_VERBOSE=1 python my_script.py 2>&1 | grep '\[compile\]'
```

You'll see `[compile] op[i] <name> ...` for each emitted op; the
missing `→ emitted` line marks the rejecting op.

---

## 10. Reporting issues

When you hit a compile bug, please include:

1. Output of `cm.cache_info()` and `cm.timing()`.
2. The model's `__init__` (or the smallest repro).
3. Input shapes + dtypes + device.
4. The full `LUCID_COMPILE_VERBOSE=1` trace output.

The most useful repro is usually the smallest model that still
triggers the same `[compile] op[i] <name>` failure point.
