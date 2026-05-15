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

### Fixed

- numpy-free Tensor.tolist() + H4 enforcement + hook installer
- close numpy-free gaps (C64 .tolist + .tensor) + doc/tooling cleanup

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
