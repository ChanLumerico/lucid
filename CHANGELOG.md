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

- lucid.func parity test suite (25 tests) against reference framework for vmap / grad / vjp / jvp / jacrev / jacfwd / hessian / linearize

- nn parity tests: LayerNorm, RMSNorm, GroupNorm, BatchNorm, InstanceNorm, LRN, MultiheadAttention, TransformerEncoderLayer (19 tests)

- optim.lr_scheduler parity test suite: 15 schedulers × 21 tests against reference framework

- lucid/func/__init__.pyi: H9-compliant type stub for all 9 public functional transforms

- TransformerDecoderLayer / TransformerDecoder / Transformer parity tests (4 new tests)

- vmap Stage 2: element isolation for vmap(jacrev/jacfwd/hessian) — _ISOLATION_ATTR marker + _isolated_vmap per-element loop; fixes (B,out,B,in)→(B,out,in) shape

- vmap: strategy='auto'|'isolated'|'vectorized' parameter for explicit dispatch control

- vmap: randomness='error' now enforced — all random factories check _vmap_ctx thread-local state and raise RuntimeError immediately

- vmap: chunk_size respected in isolation mode — partial stacking after each chunk bounds peak autograd-graph memory

- linearize: linear_fn tagged with _ISOLATION_ATTR so vmap(lin) auto-uses isolation (correct per-tangent jvp dispatch)

- lucid.models Phase 0: foundation infrastructure — ModelConfig/PretrainedModel/ModelOutput/Registry/Auto/Hub/Mixins (30 tests, mypy --strict clean)

- lucid/__init__.pyi + gen_pyi.py: lucid.load() stub now includes weights_only parameter

- models._registry.ModelFactory: Protocol with explicit __name__ + __call__ signature; _RegistryEntry.model_class + default_config fast-path fields

- save_safetensors / load_safetensors: SafeTensors format support (optional pip install safetensors; clear ImportError if missing)

- lucid.load() auto-detects .safetensors extension and delegates to load_safetensors()

- PretrainedModel.save_pretrained(safe_serialization=True): saves model.safetensors instead of weights.lucid

- lucid.models.vision.resnet: ResNet 18/34/50/101/152 backbone (task=base) + classifier (task=image-classification); 10 registry entries; AutoModel/AutoModelForImageClassification compatible

- lucid.models.vision.lenet: LeNet-5 (LeCun et al., 1998) — original tanh+avg and modern relu+max variants; backbone + classifier; 4 registry entries

- lucid.models.vision.alexnet: AlexNet (Krizhevsky et al., 2012) — original paper architecture (96/256/384/384/256 channels + LRN); backbone + classifier

- lucid.models.vision.vgg: VGG 11/13/16/19 + BN variants (Simonyan & Zisserman, 2014); 16 registry entries; VGG-16 classifier 138,357,544 params (paper-exact)

- lucid.models.vision.googlenet: GoogLeNet / Inception v1 (Szegedy et al., 2014) — parallel Inception modules, auxiliary classifiers (train only, 0.3× weighted), 13.4M params paper-exact

- lucid.models.vision.densenet: DenseNet 121/169/201/264 (Huang et al., 2016) — dense connections with growth_rate/bn_size config; DenseNet-121 classifier 7,978,856 params paper-exact

- lucid.models.vision.mobilenet: MobileNet v1 (Howard et al., 2017) — depthwise separable convolutions; width_mult 1.0/0.75/0.5/0.25; 4.2M params paper-exact

- lucid.models.vision.efficientnet: EfficientNet B0–B7 (Tan & Le, 2019) — compound scaling, MBConv+SE blocks; B0 classifier 5,288,548 params reference-exact; 16 registry entries

- lucid.models.vision.vit: ViT B/16 B/32 L/16 L/32 H/14 (Dosovitskiy et al., 2020) — patch embedding + transformer encoder; ViT-B/16 86,567,656 params reference-exact; 10 registry entries

- lucid.models.vision.swin: Swin Transformer T/S/B/L (Liu et al., 2021) — shifted-window self-attention, hierarchical feature maps; Swin-T 28,288,354 params reference-exact; 8 registry entries

- lucid.models.vision.convnext: ConvNeXt T/S/B/L/XL (Liu et al., 2022) — patchify stem, DWConv-7×7 + inverted-bottleneck MLP, layer scale; ConvNeXt-T 28,589,128 params reference-exact; 10 registry entries

- 20 new vision model families: ZFNet, Inception v3/v4/ResNet, Xception, ResNeXt, SENet, SKNet, MobileNet v2/v3/v4, ResNeSt, CSPNet, CoAtNet, CvT, CrossViT, PVT, EfficientFormer, MaxViT, InceptionNeXt (156 total registered variants)

- lucid.models.vision.rcnn: R-CNN (Girshick et al., CVPR 2014) — AlexNet-style CNN per warped proposal crop, class-specific bbox regression (1 factory)

- lucid.models.vision.fast_rcnn: Fast R-CNN (Girshick, ICCV 2015) — VGG16 backbone + RoI Pool 7×7 + 2-FC head; smooth-L1(σ=1) + cross-entropy loss (1 factory)

- lucid.models.vision.faster_rcnn: Faster R-CNN (Ren et al., NeurIPS 2015) — VGG16 + RPN + RoI Pool; smooth-L1(σ=3) RPN loss + ROI head loss; 9 anchors per cell (1 factory)

- lucid.models.vision.mask_rcnn: Mask R-CNN (He et al., ICCV 2017) — ResNet-50-FPN backbone, RoI Align, mask FCN head 14→28 via deconv; instance-segmentation output (1 factory)

- lucid.models.vision.detr: DETR (Carion et al., ECCV 2020) — ResNet-50/101 + nn.Transformer encoder-decoder + Hungarian set-prediction loss; 100 object queries (2 factories)

- lucid.models.vision.efficientdet: EfficientDet D0–D7 (Tan et al., CVPR 2020) — EfficientNet-B0 backbone, BiFPN with fast-normalised weighted fusion, focal+smooth-L1 loss (8 factories)

- lucid.models.vision.yolo: YOLO v1/v2/v3/v4 + tiny variants (Redmon et al., Bochkovskiy et al., 2016–2020) — custom Darknet / Darknet-19 / Darknet-53 / CSPDarknet-53 backbones; YOLOv4 uses Mish in backbone per paper §3.4 (7 factories)

- lucid.models.vision.fcn: FCN (Long et al., CVPR 2015) — dilated ResNet-50/101 (stride 8) + FCN head + auxiliary head with 0.4 weight (2 factories)

- lucid.models.vision.unet: U-Net 2D/3D + ResUNet 2D/3D (Ronneberger et al., MICCAI 2015) — `dim` config field switches Conv2d↔Conv3d / bilinear↔trilinear; `block` field toggles residual DoubleConv (6 factories)

- lucid.models.vision.attention_unet: Attention U-Net (Oktay et al., MIDL 2018) — additive Wx+Wg→ReLU→ψ→sigmoid attention gates on skip connections (2 factories)

- lucid.models.vision.maskformer: MaskFormer (Cheng et al., NeurIPS 2021) — ResNet-18/34/50/101 backbone, FPN pixel decoder, N learnable mask queries + Hungarian mask-classification loss (4 factories)

- lucid.models.vision.mask2former: Mask2Former (Cheng et al., CVPR 2022) — ResNet-18/34/50/101 or Swin-T/S/B/L backbone, multi-scale FPN pixel decoder, masked cross-attention with per-layer FPN level cycling (8 factories)

- lucid.models.text — first NLP family wave (5 architectures, 39 registry entries): Transformer (Vaswani et al., NeurIPS 2017) base/large + seq2seq + cls + token-cls heads (7 factories); BERT (Devlin et al., NAACL 2019) tiny/mini/small/medium/base/large + MLM/SequenceCls/TokenCls/QA heads (13 factories); GPT (Radford et al., 2018) base + LM/Cls heads (4 factories); GPT-2 (Radford et al., 2019) small/medium/large/xlarge + LM/Cls heads (10 factories); RoFormer (Su et al., 2021) base + MLM/SequenceCls/TokenCls heads (5 factories). Shared `_utils._text` infra (causal masks, position ids, RoPE).

- lucid.models.generative — first generative family wave (3 architectures, 16 registry entries): VAE (Kingma & Welling, 2013) vanilla + hierarchical Sønderby/Ladder VAE + image-gen heads (4 factories); DDPM (Ho et al., NeurIPS 2020) CIFAR-10 / LSUN-256 / ImageNet-64 backbones + image-gen heads (7 factories); NCSN (Song & Ermon, NeurIPS 2019 + NCSNv2) CIFAR-10 / CelebA-64 backbones + image-gen heads (5 factories). Shared `_utils._generative` infra (β / σ schedule helpers, `DiffusionScheduler` base class, `DDPMScheduler` ancestral sampler, annealed Langevin dynamics).

- AutoModelForCausalLM / AutoModelForMaskedLM / AutoModelForSeq2SeqLM / AutoModelForSequenceClassification / AutoModelForTokenClassification / AutoModelForQuestionAnswering / AutoModelForImageGeneration auto-classes for the seven new task tags.

- DropPath helper (Huang et al., 2016 stochastic depth) added to `models._utils._classification`. Wired into ConvNeXt, Swin, EfficientFormer with linear schedule across the trunk per the original recipes.

- ObjectDetectionOutput / InstanceSegmentationOutput / SemanticSegmentationOutput dataclasses in models._output

- ObjectDetectionOutput.proposals: optional per-image RoI tuple emitted by R-CNN / Fast R-CNN / Faster R-CNN forward, so `postprocess(output)` works without re-running the RPN.

- lucid.models._utils package: _common (make_divisible canonicalised across 7 model families), _classification (LayerScale), _detection (box ops, NMS, AnchorGenerator, roi_align/roi_pool, FPN, RPN, RoIHead shared modules)

- _kuhn_munkres_rectangular: textbook Hungarian/JV implementation in models.vision.detr — shared by DETR / MaskFormer / Mask2Former matchers; verified against `scipy.optimize.linear_sum_assignment` on 100+ random matrices

- AutoModelForObjectDetection / AutoModelForSemanticSegmentation auto-classes for the two new task tags

- unit tests: test_models_detection.py (10 detection model groups × CPU+Metal), test_models_segmentation.py (13 segmentation model groups × CPU+Metal), test_hungarian.py (6 scipy-cross-checked correctness tests) — 69 / 69 pass

### Tooling

- tools/changelog.py — Keep-a-Changelog helper (add/propose/release/check)
- CHANGELOG.md — initial 3.0.0 release notes
- mypy --strict baseline (0 errors) locked in mypy.ini

### Fixed

- H5/H7 Hard Rule violations in lucid.func + parity tests
- lucid.func.jvp scalar output shape — alpha gradient was (1,) instead of () for scalar primal outputs
- CosineAnnealingWarmRestarts: reset T_cur/T_i before computing LR — restart epoch now correctly returns base_lr (not eta_min)
- ReduceLROnPlateau: patience check changed >= → > to match reference (was reducing one epoch too early)
- OneCycleLR: warmup end = total_steps*pct_start-1 (not floor); init_lr = max_lr/div_factor regardless of optimizer LR
- nn.Transformer: add final LayerNorm to encoder and decoder by default — matches reference (was missing 4 parameters)
- models.ModelConfig.from_dict: unknown fields now warn+ignore instead of raising (forward-compatible checkpoint loading)
- models.PretrainedModel: config_class default changed from ModelConfig to None — concrete subclasses that forget to set it now get a clear TypeError
- models._load_from_directory: no longer instantiates the model twice — uses model_class fast path when registered, else one factory call
- models.AutoConfig.from_pretrained: returns default_config instantly when pre-registered, avoiding full model instantiation
- models.load_from_pretrained_entry: validates entry.config.model_type == model.config.model_type before downloading weights
- safetensors: 0-d tensors (BatchNorm num_batches_tracked) now round-trip correctly — saved as (1,) with metadata tag, squeezed back to () on load
- MaxViT _MaxViTBlock: pad spatial dims to window_size multiple before grid/window partition to handle non-divisible resolutions (e.g. 28×28 with ws=7)

- Faster R-CNN / Mask R-CNN / `_utils._detection` RPN: anchor ordering bug — Conv2d output `(B, A, H, W)` was being flattened anchor-major while AnchorGenerator emits spatial-major `(G·A, 4)`. Fixed by permuting predictions to spatial-major before flatten.

- YOLOv3 detection head: channel-count mismatch — `_Darknet53` returns p3_raw=128ch / p4_raw=256ch (residual blocks keep input channels) but the head was built for 256/512. Rewired to use actual backbone widths.

- `.clamp()` is positional-only in Lucid — replaced every `clamp(min=...)` / `clamp(max=...)` single-kwarg call (in `_detection.py`, YOLOv1) with `clamp(low, high)`.

- `lucid.tensor([int_list])` returns float32 but `index_select` requires int — added `.long()` to all index-tensor construction sites in `_detection.py` + 4 R-CNN family models.

- Device propagation across 30+ sites in detection training / postprocess paths (R-CNN, Fast/Faster/Mask R-CNN, EfficientDet): every `lucid.zeros(...)`, `lucid.tensor([...])`, and `lucid.full(...)` in the loss helpers / postprocessors now derives `device=` from input tensors so the models work on Metal training.

- DETR / MaskFormer / Mask2Former Hungarian matcher: custom JV variant iterated over the wrong axis and returned non-optimal assignments even on trivial inputs (5×3 trivial match returned 1/2/4 instead of 0/1/2). Replaced with a textbook rectangular Kuhn-Munkres implementation that cross-checks against `scipy.optimize.linear_sum_assignment`.

- MaskFormer pixel decoder: `out3`/`out4`/`out5` 3×3 smoothing convs were declared but never applied in forward — dead parameters and a silent paper-fidelity deviation. Now every FPN level passes through its own smoothing conv per paper §3.2.

- MaskFormer / Mask2Former `_binary_mask_iou`: vectorised — replaced the per-pixel Python double-loop with `.item()` (O(H·W) device→host syncs per call) with `(p>0.5).float() * (g>0.5).float()` form + a single `.sum().item()`.

- Swin `rel_pos_idx`: re-registered as a non-persistent buffer (was a raw attribute via `object.__setattr__`, so `.to(device=...)` left it on CPU and broke metal-side `rel_pos_bias[idx]`).

- Swin `_attn_mask`: takes `device=x.device.type` so the shifted-window mask is built on the same device as activations.

- MaxViT docstring: replaced "Standard PyTorch padding=1" with framework-neutral wording (H5).

- move CHANGELOG auto-injection from prepare-commit-msg → post-commit

- Paper-faithful audit pass on the model zoo (closes the remaining ⚠️ deviations flagged in the Wave-3 retrospective):
  - **EfficientDet BiFPN** — removed `.item()` round-trip in fast-normalised weighted fusion (was forcing a per-step host sync).
  - **CoAtNet `_rel_idx`** — registered as non-persistent buffer so `.to(device=...)` works.
  - **EfficientNet stochastic depth** — was applied unconditionally; now respects `training` flag and per-block survival probability schedule (Tan & Le 2019 §3.3).
  - **R-CNN family class-specific decode** — Fast / Faster R-CNN now decode bbox deltas with the predicted top-class deltas (paper §3.3) instead of class 0 / argmax-of-bg-included.
  - **ResNeSt `is_first` flag** — first block of each stage receives the correct `is_first=True` to drop the redundant 1×1 down-projection.
  - **MaskFormer / Mask2Former dice loss** — corrected denominator from `|p|·|g|` (cosine-style) to `|p|+|g|` per Milletari 2016.
  - **YOLOv1 w/h decoding** — paper §2 / Eq.1 uses sigmoid-bounded direct prediction (`sigmoid(raw)·{W,H}`); was incorrectly using YOLOv2's `exp(raw)·{W,H}` anchor formulation. Loss term updated to MSE on `√w_norm, √h_norm`.
  - **CvT `stride_kv`** — paper Table 1 specifies stride=2 for K/V conv-projection in *all* three stages; was only stage 0.
  - **CrossViT classification head** — paper §3.3 averages two per-branch classifier logits; was concat → single FC.
  - **MobileNetV2 `last_ch`** — `last_ch = make_divisible(1280·max(1, width_mult))` per paper §3.4 / torchvision; was hard-coded 1280 for all width multipliers.
  - **DDPM `learn_sigma=True`** — now raises `NotImplementedError` (Improved-DDPM hybrid `L_simple + L_vlb` loss not yet implemented) instead of silently emitting an unusable variance head.
  - **Inception v3 auxiliary classifier** — moved from after `inception_c1` (35×35) to after `inception_c3` (last 17×17 = Mixed_6e) per paper §6 / Fig.10.
  - **SKNet `_SKAttentionGate`** — `AdaptiveAvgPool2d(1)` lifted into `__init__` (was instantiated each forward call).
  - **EfficientFormer LayerScale + DropPath** — added per-residual-branch γ (init 1e-5) and linear stochastic-depth schedule per paper §4.1 (max-rate 0.0 / 0.1 / 0.2 for L1 / L3 / L7).

### Performance

- NMS: vectorised per-row IoU computation — replaced O(N²) pairwise `box_iou(box_i, box_j)` allocations with K vectorised `box_iou(boxes[idx:idx+1], boxes)` rows (where K is the number of survivors). Sort is now a single device-side `argsort` instead of N `.item()` calls inside Python `sorted`.

- Anchor assignment in Faster R-CNN / Mask R-CNN / EfficientDet RPN + RoI losses: replaced 2·A·M nested `.item()` loops with a single `argmax(dim=...)` / `max(dim=...)` reduction per axis and bulk materialisation. ~10× fewer device→host syncs.

- Wave 3d unit test suite (CPU + Metal): 62 s → 53 s end-to-end as a result of the NMS / anchor-assignment vectorisation.

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
