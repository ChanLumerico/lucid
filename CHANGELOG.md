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

### Added

- Git hooks (.githooks/{post-commit,commit-msg}) for CHANGELOG hygiene

- fractional_max_pool2d and fractional_max_pool3d

- lucid.func module: vmap, grad, grad_and_value, vjp, jvp, jacrev, jacfwd, hessian, linearize

### Tooling

- tools/changelog.py — Keep-a-Changelog helper (add/propose/release/check)
- CHANGELOG.md — initial 3.0.0 release notes
- mypy --strict baseline (0 errors) locked in mypy.ini

### Fixed

- H5/H7 Hard Rule violations in lucid.func + parity tests

---

## [3.0.0] — 2026-05-10

First production release. Lucid is now PyTorch-compatible across the public
surface (~100% parity in every measured module) and runs natively on Apple
Silicon via MLX (GPU) and Apple Accelerate (CPU). The C++ engine has been
fully rewritten under a new OOP architecture (IBackend / Dispatcher / OpSchema
/ kernel framework) and is the single source of truth for numerics.

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
  pixel_shuffle / pixel_unshuffle, multi_head_attention_forward.
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
  `RemovableHandle`, `checkpoint`, `enable_grad` fix. _Deferred_: `vmap`.
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
- **NumPy demoted to optional** — `pip install lucid` no longer requires NumPy.
  Use `pip install lucid[numpy]` for `from_numpy` / `.numpy()` / `from_dlpack`
  via NumPy. Six sanctioned bridge boundaries documented in `CLAUDE.md` H4.
- **`state_dict` v2** — `_load_from_state_dict` matches PyTorch signature;
  `_metadata` round-trip; `_version` keys preserved; `assign=` parameter
  supported.
- **Tier-1 namespace hygiene** — `Module` / `Parameter` / `Linear` / `Adam` are
  no longer accessible under the top-level `lucid.*` namespace; they live
  under their proper sub-package (`lucid.nn.*`, `lucid.optim.*`).
- **Builtin shadowing fixed** — `from lucid import *` no longer pollutes
  `float` / `int` / `bool` / `bytes`.

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

### Performance

- **GPU `relu`** — 78 % overhead removed: `zeros_like(x)` (full-tensor
  allocation) replaced with broadcast scalar `array(0.0, dtype)`.
  Same fix applied to `elu_backward` (1.0 scalar instead of `ones_like`).
- **MLX template overhead** — removed redundant `::mlx::core::contiguous()`
  calls from `mlx_unary` / `mlx_binary` / `mlx_reduce` (every op was paying
  for an extra MLX graph node it didn't need). Added `mlx_unary_contiguous()`
  variant for ops that genuinely require contiguity.
- **`eval_gpu()` single-tensor fast path** — `_C_engine.eval_gpu(impl)` skips
  the ~25 µs Python list-construction overhead of `eval_tensors([impl])`.
  Used by the Lucid GPU benchmark harness.
- **SharedStorage zero-copy CPU↔GPU** — for SharedStorage-backed tensors,
  `.to('metal')` and `.to('cpu')` are now zero memcpy (relabel via
  `transfer_storage()`).
- **`.to('metal')` for regular tensors** — single Metal blit to GPU-private
  memory (was 2 copies via Python round-trip). Subsequent ops pay no
  external-array penalty.

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
  `*args`/`**kwargs` only for genuinely variadic APIs (H9).
- **`tools/check_doxygen.py`** / `check_stubs.py` / `check_op_api.py` /
  `check_layers.py` / `check_op_template.py` / `check_kernel_template.py` /
  `check_phase1.py` — automated CI checks.
- **Test infrastructure rebuild (Phases 1-11)** — full from-scratch test layer
  in `lucid/test/`. 1574 unit tests pass (61 skipped). Cross-product
  CPU+Metal fixtures, lazy reference-framework loader, parity gating, golden
  numerical checks, integration train-loops (MLP / CNN / RNN / Transformer),
  microbench / e2e / memory perf tests, CI wiring.
- **C++ Google Test suite** — 108 tests (was 105 prior to this release).
  Includes new `Concurrency.*` stress tests covering thread-local allocator
  hammer, `MemoryTracker` counter consistency, and `Generator` mutex
  serialization.
- **Performance baseline suite** (`benchmarks/`) — A (self-regression with
  threshold guard) + B (vs. raw MLX) for ops, transfer, and training loops.
  `run_all.py --save` records baseline; `--check --threshold 15` fails if any
  result regresses by more than 15 %.
- **Hard Rules H1–H9** — fully enforced across the codebase. Verified by
  AST scan (zero violations).

### Documentation

- **Doxygen** — 184/184 = 100.0 % coverage of the public C++ engine surface.
- **`.pyi` stubs** — `engine.pyi`, `tensor.pyi`, `__init__.pyi` all up to
  date; verified by `tools/check_stubs.py`.
- **Obsidian vault** (`obsidian/`) — git-ignored team knowledge base
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
