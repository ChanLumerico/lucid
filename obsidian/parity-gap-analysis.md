# Lucid vs. PyTorch — Package-Level Parity Gap Analysis

> **Last updated:** 2026-05-08  
> **Branch at time of writing:** `lucid-3.0`  
> **Lucid commit:** `05b52ca52`  
> **Lucid test count:** 966 passed, 2 skipped, 6 xfailed  

---

## What this document is

A precise, name-by-name accounting of what PyTorch exposes publicly versus what Lucid currently implements, organized by sub-package. Use it to:

1. Pick the next parity target without re-auditing the whole codebase.
2. Give Claude (or any contributor) an instant state-of-the-world without reading source.
3. Track progress over time — check off rows as they land.

**Update discipline:** whenever a gap is closed, strike through or delete the row and update the "Parity %" column in the [Overview](#overview-summary).

---

## Overview Summary

| Package | Lucid exports | PyTorch exports | Missing count | Parity |
|---------|:-------------:|:---------------:|:-------------:|:------:|
| `torch` top-level | 165 | ~210 | ~45 | **~79%** |
| `torch.nn.functional` | 70 | ~88 | ~18 | **~80%** |
| `torch.nn` (modules) | ~151 | ~165 | ~14 | **~92%** |
| `torch.nn.init` | 13 | 13 | 0 | **100% ✅** |
| `torch.nn.utils` | 14 | 16 | ~2 | **~88%** |
| `torch.optim` | 23 | 21 | 0 | **100% ✅** |
| `torch.linalg` | 31 | 37 | 6 | **~84%** |
| `torch.autograd` | 11 | ~16 | ~5 | **~69%** |
| `torch.utils.data` | 18 | ~22 | ~4 | **~82%** |
| `torch.fft` | 0 | 20 | 20 | **0% ❌** |
| `torch.special` | 0 (scattered) | ~60 | ~50 | **~15% ⚠️** |
| `torch.distributions` | 0 | ~50 | ~50 | **0% ❌** |
| `torch.signal` | 0 | 12 | 12 | **0% ❌** |

---

## Priority Queue (Next Work Order)

| Pri | Target | Effort | Why High Impact |
|-----|--------|--------|-----------------|
| **P1** | `torch.fft` | Low — wrap `mlx.core.fft.*` | Used in transformers, convolutions, signal proc |
| **P2** | `torch` top-level gaps | Low–Med | Basic everyday ops (`randperm`, `index_put`, etc.) |
| **P3** | `torch.nn.functional` gaps | Low | Fills holes in activation / pooling surface |
| **P4** | `torch.nn` module gaps | Medium | MaxUnpool, FractionalMaxPool, CircularPad |
| **P5** | `torch.special` submodule | Medium | Scientific computing, stats |
| **P6** | `torch.linalg` completion | Low | 6 missing ex-variants + `lu` |
| **P7** | `torch.signal.windows` | Very low | Rarely critical |
| **P8** | `torch.distributions` | High | Probabilistic models — largest remaining gap |

---

## 1. `torch` top-level

### What Lucid has (165 names)

<details>
<summary>Expand full list</summary>

**Core types:** `Tensor`, `dtype`, `dtypes`, `device`

**Dtypes (14):** `float16`, `bfloat16`, `float32`, `float64`, `int8`, `int16`, `int32`, `int64`, `bool_`, `complex64`, `half`, `double`, `short`, `long`

**Device / defaults (6):** `set_default_dtype`, `get_default_dtype`, `set_default_device`, `get_default_device`

**Deterministic factories (14):** `tensor`, `as_tensor`, `from_numpy`, `zeros`, `ones`, `empty`, `full`, `eye`, `arange`, `linspace`, `logspace`, `zeros_like`, `ones_like`, `empty_like`, `full_like`

**Random factories (8):** `rand`, `randn`, `randint`, `bernoulli`, `normal`, `rand_like`, `randn_like`, `manual_seed`

**Unary ops (30+):** `abs`, `neg`, `sign`, `exp`, `exp2`, `log`, `log2`, `log10`, `log1p`, `sqrt`, `square`, `reciprocal`, `rsqrt`, `floor`, `ceil`, `round`, `trunc`, `frac`, `sin`, `cos`, `tan`, `arcsin`/`asin`, `arccos`/`acos`, `arctan`/`atan`, `sinh`, `cosh`, `tanh`, `clip`/`clamp`, `isinf`, `isnan`, `isfinite`, `nan_to_num`

**New engine ops (4):** `erf`, `erfinv`, `cummax`, `cummin`

**Binary ops (20+):** `add`, `sub`, `mul`, `div`, `pow`, `matmul`, `mm`, `bmm`, `tensordot`, `einsum`, `kron`, `atan2`, `fmod`, `remainder`, `hypot`, `logaddexp`, `maximum`, `minimum`, `eq`/`equal`, `ne`/`not_equal`, `lt`/`less`, `le`/`less_equal`, `gt`/`greater`, `ge`/`greater_equal`, `isclose`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`

**Reductions (14):** `sum`, `mean`, `max`, `min`, `prod`, `argmax`, `argmin`, `cumsum`, `cumprod`, `std`, `var`, `trace`, `any`, `all`, `logsumexp`

**Shape ops (30+):** `reshape`, `view`, `permute`, `transpose`, `unsqueeze`, `squeeze`, `flatten`, `unflatten`, `narrow`, `movedim`, `expand`, `broadcast_to`, `repeat`, `repeat_interleave`, `tile`, `cat`/`concat`, `stack`, `hstack`, `vstack`, `split`, `chunk`, `unbind`, `tril`, `triu`, `roll`, `flip`, `fliplr`, `flipud`, `pad`, `sort`, `argsort`, `topk`, `kthvalue`, `nonzero`, `unique`, `meshgrid`, `contiguous`, `detach`, `clone`

**Index ops:** `gather`, `scatter`, `scatter_add`, `index_select`, `masked_select`, `take`, `where`, `masked_fill`

**Stats/search:** `searchsorted`, `bucketize`, `histc`, `cartesian_prod`

**Composites — elementwise specials:** `absolute`, `negative`, `positive`, `subtract`, `multiply`, `divide`, `true_divide`, `rsub`, `arctan2`, `arccosh`/`acosh`, `arcsinh`/`asinh`, `arctanh`/`atanh`, `expm1`, `sinc`, `heaviside`, `xlogy`, `logit`, `signbit`, `float_power`, `fmax`, `fmin`, `erfc`, `copysign`, `ldexp`, `gcd`, `lcm`, `lgamma`, `digamma`, `i0`

**Composites — NaN-safe reductions:** `nansum`, `nanmean`, `nanmedian`

**Composites — statistics:** `quantile`, `nanquantile`, `cov`, `corrcoef`, `cdist`, `bincount`, `histogram`, `multinomial`

**Composites — index ops:** `index_fill`, `index_add`, `index_copy`, `scatter_reduce`, `masked_scatter`

**Composites — BLAS:** `addmm`, `addbmm`, `baddbmm`, `addmv`, `addr`, `addcmul`, `addcdiv`, `mv`, `ger`, `vdot`, `block_diag`

**Composites — shape:** `swapaxes`, `swapdims`, `moveaxis`, `adjoint`, `t`, `column_stack`, `row_stack`, `dstack`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `vsplit`, `hsplit`, `dsplit`, `tensor_split`, `take_along_dim`, `vander`, `rot90`

**Composites — predicates:** `numel`, `is_storage`, `is_nonzero`, `is_same_size`, `is_neg`, `is_conj`, `isin`, `isneginf`, `isposinf`, `isreal`, `conj`, `conj_physical`, `resolve_conj`, `resolve_neg`

**Composites — type promotion:** `result_type`, `promote_types`, `can_cast`

**Constants:** `pi`, `e`, `inf`, `nan`, `newaxis`

**Tensor method aliases:** `lerp`, `diff`

**Grad control:** `no_grad`, `enable_grad`, `is_grad_enabled`, `set_grad_enabled`, `inference_mode`

**Type predicates:** `is_tensor`, `is_floating_point`, `is_complex`, `is_signed`

**Serialization:** `save`, `load`

</details>

### What is missing from `torch` top-level

| Category | Missing name(s) | Notes |
|---|---|---|
| **Random factories** | `randperm` | returns shuffled `arange` |
| | `poisson` | Poisson-sampled tensor |
| | `Generator` | RNG state object |
| | `get_rng_state`, `set_rng_state`, `initial_seed`, `seed` | RNG state management |
| **Complex numbers** | `real`, `imag` | real/imaginary part views |
| | `angle` | complex argument |
| | `complex`, `polar` | construct complex tensor |
| | `view_as_real`, `view_as_complex` | reinterpret memory |
| **Math unary** | `nextafter` | next float towards y |
| | `polygamma` | higher-order digamma |
| | `frexp` | decompose to mantissa+exp |
| | `bitwise_left_shift`, `bitwise_right_shift` | integer bit shifts |
| **Reductions** | `count_nonzero` | counts non-zero elements |
| **Index / scatter** | `index_put` / `index_put_` | advanced in-place write |
| | `put` | flat-index scatter |
| **Shape** | `tril_indices`, `triu_indices` | index helpers for triangular |
| | `combinations` | n-choose-k combinations |
| | `histogramdd` | N-D histogram |
| **Top-level aliases** | `dot`, `inner`, `outer` | present in `lucid.linalg` but NOT at `lucid.*` — PyTorch has both |
| | `cross`, `norm` | same — linalg-only in Lucid |
| **Activation aliases** | `relu`, `sigmoid`, `tanh` | PyTorch exposes these as top-level ufuncs; Lucid removed them (in `nn.functional` only) |
| **DType info** | `finfo`, `iinfo` | float/int dtype info objects |
| **Threading** | `set_num_threads`, `get_num_threads`, `get_num_interop_threads` | |
| **Determinism** | `use_deterministic_algorithms`, `are_deterministic_algorithms_enabled` | |
| **Interop** | `from_dlpack`, `to_dlpack` | DLPack protocol |
| **JIT / compile** | `compile`, `jit.script`, `jit.trace` | N/A for Lucid's design |

---

## 2. `torch.nn.functional`

### What Lucid has (70 names)

**Activations (23):** `relu`, `leaky_relu`, `elu`, `celu`, `selu`, `gelu`, `silu`, `mish`, `hardswish`, `hardsigmoid`, `sigmoid`, `tanh`, `softmax`, `log_softmax`, `softplus`, `relu6`, `softmin`, `glu`, `prelu`, `hardshrink`, `tanhshrink`, `softshrink`, `normalize`, `cosine_similarity`, `pairwise_distance`

**Linear (2):** `linear`, `bilinear`

**Convolution (6):** `conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`

**Normalization (5):** `batch_norm`, `layer_norm`, `group_norm`, `rms_norm`, `instance_norm`

**Pooling (12):** `max_pool1d`, `max_pool2d`, `max_pool3d`, `avg_pool1d`, `avg_pool2d`, `avg_pool3d`, `adaptive_avg_pool1d`, `adaptive_avg_pool2d`, `adaptive_avg_pool3d`, `adaptive_max_pool1d`, `adaptive_max_pool2d`, `adaptive_max_pool3d`

**Dropout (5):** `dropout`, `dropout2d`, `dropout3d`, `alpha_dropout`, `feature_alpha_dropout`

**Attention (2):** `scaled_dot_product_attention`, `multi_head_attention_forward`

**Losses (18):** `mse_loss`, `l1_loss`, `smooth_l1_loss`, `huber_loss`, `cross_entropy`, `nll_loss`, `binary_cross_entropy`, `binary_cross_entropy_with_logits`, `kl_div`, `triplet_margin_loss`, `cosine_embedding_loss`, `margin_ranking_loss`, `hinge_embedding_loss`, `poisson_nll_loss`, `gaussian_nll_loss`, `ctc_loss`, `multi_margin_loss`, `multilabel_margin_loss`

**Sparse (2):** `embedding`, `one_hot`

**Sampling / misc (9):** `interpolate`, `grid_sample`, `affine_grid`, `pad`, `unfold`, `fold`, `embedding_bag`, `pixel_shuffle`, `pixel_unshuffle`

### What is missing from `torch.nn.functional`

| Category | Missing name(s) | Notes |
|---|---|---|
| **Activations** | `hardtanh` | clamped linear; `Hardtanh` module exists |
| | `logsigmoid` | log(sigmoid(x)); `LogSigmoid` module exists |
| | `softsign` | x/(1+\|x\|); `Softsign` module exists |
| | `threshold` | threshold + replace; `Threshold` module exists |
| **Pooling** | `lp_pool1d`, `lp_pool2d` | Lp-norm pooling; modules exist |
| | `max_unpool1d`, `max_unpool2d`, `max_unpool3d` | inverse of max pool |
| | `fractional_max_pool2d`, `fractional_max_pool3d` | random fractional pooling |
| **Normalization** | `local_response_norm` | LRN; `LocalResponseNorm` module exists |
| **Losses** | `multilabel_soft_margin_loss` | multi-label sigmoid loss |
| | `soft_margin_loss` | logistic-style binary loss |
| **Misc** | `channel_shuffle` | ShuffleNet operation |
| | `pdist` | pairwise L-p distances |

---

## 3. `torch.nn` (Module classes)

### What Lucid has (~151 names)

**Base (3):** `Module`, `Parameter`, `RemovableHandle`

**Hooks (6):** `register_module_full_backward_hook`, `register_module_full_backward_pre_hook`, `register_module_forward_hook`, `register_module_forward_pre_hook`, `register_module_load_state_dict_pre_hook`, `register_module_load_state_dict_post_hook`

**Linear (4):** `Linear`, `Identity`, `Bilinear`, `LazyLinear`

**Conv (12):** `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`, `LazyConv1d`, `LazyConv2d`, `LazyConv3d`, `LazyConvTranspose1d`, `LazyConvTranspose2d`, `LazyConvTranspose3d`

**Activations (26):** `ReLU`, `LeakyReLU`, `ELU`, `CELU`, `SELU`, `GELU`, `SiLU`, `Mish`, `Softplus`, `Hardswish`, `Hardsigmoid`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `ReLU6`, `PReLU`, `Threshold`, `Hardtanh`, `LogSigmoid`, `Softsign`, `Softmin`, `GLU`, `Hardshrink`, `Tanhshrink`, `Softshrink`

**Normalization (16):** `LayerNorm`, `RMSNorm`, `GroupNorm`, `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`, `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d`, `LocalResponseNorm`, `LazyBatchNorm1d`, `LazyBatchNorm2d`, `LazyBatchNorm3d`, `LazyInstanceNorm1d`, `LazyInstanceNorm2d`, `LazyInstanceNorm3d`

**Pooling (13):** `MaxPool1d`, `MaxPool2d`, `MaxPool3d`, `AvgPool1d`, `AvgPool2d`, `AvgPool3d`, `AdaptiveAvgPool1d`, `AdaptiveAvgPool2d`, `AdaptiveAvgPool3d`, `AdaptiveMaxPool1d`, `AdaptiveMaxPool2d`, `AdaptiveMaxPool3d`, `LPPool1d`, `LPPool2d`

**Dropout (6):** `Dropout`, `Dropout1d`, `Dropout2d`, `Dropout3d`, `AlphaDropout`, `FeatureAlphaDropout`

**Embedding (2):** `Embedding`, `EmbeddingBag`

**Attention + RNN (9):** `MultiheadAttention`, `LSTM`, `GRU`, `RNN`, `LSTMCell`, `GRUCell`, `RNNCell`

**Losses (18):** `MSELoss`, `L1Loss`, `CrossEntropyLoss`, `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, `HuberLoss`, `SmoothL1Loss`, `KLDivLoss`, `TripletMarginLoss`, `CosineEmbeddingLoss`, `MarginRankingLoss`, `HingeEmbeddingLoss`, `PoissonNLLLoss`, `GaussianNLLLoss`, `CTCLoss`, `MultiMarginLoss`, `MultilabelMarginLoss`

**Containers (5):** `Sequential`, `ModuleList`, `ModuleDict`, `ParameterList`, `ParameterDict`

**Shape (2):** `Flatten`, `Unflatten`

**Unfold/Fold (2):** `Unfold`, `Fold`

**Padding (11):** `ConstantPad1d`, `ConstantPad2d`, `ConstantPad3d`, `ZeroPad1d`, `ZeroPad2d`, `ZeroPad3d`, `ReflectionPad1d`, `ReflectionPad2d`, `ReplicationPad1d`, `ReplicationPad2d`, `ReplicationPad3d`

**Upsampling (5):** `Upsample`, `UpsamplingNearest2d`, `UpsamplingBilinear2d`, `PixelShuffle`, `PixelUnshuffle`

**Transformer (5):** `TransformerEncoderLayer`, `TransformerEncoder`, `TransformerDecoderLayer`, `TransformerDecoder`, `Transformer`

**Subpackages (3):** `functional`, `init`, `utils`

### What is missing from `torch.nn`

| Category | Missing name(s) | Notes |
|---|---|---|
| **Pooling** | `MaxUnpool1d`, `MaxUnpool2d`, `MaxUnpool3d` | Inverse of MaxPool (needs saved indices) |
| | `FractionalMaxPool2d`, `FractionalMaxPool3d` | Random pooling regions |
| **Padding** | `CircularPad1d`, `CircularPad2d`, `CircularPad3d` | Wrap-around padding |
| | `ReflectionPad3d` | 3-D reflection padding |
| **Normalization** | `SyncBatchNorm` | Multi-GPU sync BN (N/A for Apple Silicon) |
| | `CrossMapLRN2d` | Deprecated LRN variant |
| **Misc layers** | `ChannelShuffle` | ShuffleNet channel permutation |
| **Losses** | `TripletMarginWithDistanceLoss` | Custom distance variant |
| | `SoftMarginLoss`, `MultiLabelSoftMarginLoss` | Sigmoid-based multi-label losses |

---

## 4. `torch.nn.init`

### Status: **100% parity ✅**

All 13 functions present: `calculate_gain`, `constant_`, `dirac_`, `eye_`, `kaiming_normal_`, `kaiming_uniform_`, `normal_`, `ones_`, `orthogonal_`, `sparse_`, `trunc_normal_`, `uniform_`, `xavier_normal_`, `xavier_uniform_`, `zeros_`

---

## 5. `torch.nn.utils`

### What Lucid has (14 names)

`clip_grad_norm_`, `clip_grad_value_`, `parameters_to_vector`, `vector_to_parameters`, `weight_norm`, `remove_weight_norm`, `spectral_norm`, `remove_spectral_norm`, `PackedSequence`, `pack_padded_sequence`, `pack_sequence`, `pad_packed_sequence`, `pad_sequence`, `parametrize` (submodule), `parametrizations` (submodule), `prune` (submodule)

### What is missing

| Missing | Notes |
|---|---|
| `nn.utils.fusion` | Conv+BN folding, layer fusion for inference |
| `copy_parameters_and_buffers` | Utility for model copying/merging |

---

## 6. `torch.optim`

### Status: **100% parity ✅** (plus extras)

**Optimizers (13):** `SGD`, `Adam`, `AdamW`, `LBFGS`, `RMSprop`, `Adagrad`, `Adadelta`, `Adamax`, `RAdam`, `NAdam`, `ASGD`, `Rprop`, `SparseAdam`

**LR Schedulers (14 + 1 Lucid-specific):** `StepLR`, `ExponentialLR`, `MultiStepLR`, `CosineAnnealingLR`, `LambdaLR`, `CyclicLR`, `ReduceLROnPlateau`, `MultiplicativeLR`, `LinearLR`, `ConstantLR`, `PolynomialLR`, `CosineAnnealingWarmRestarts`, `OneCycleLR`, `SequentialLR`, `ChainedScheduler`, `NoamScheduler` *(Lucid-only — transformer LR)*

---

## 7. `torch.linalg`

### What Lucid has (31 names)

`inv`, `det`, `solve`, `cholesky`, `norm`, `qr`, `svd`, `svdvals`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `lu_factor`, `ldl_factor`, `matrix_power`, `pinv`, `slogdet`, `matrix_rank`, `cond`, `multi_dot`, `solve_triangular`, `vector_norm`, `cross`, `vecdot`, `dot`, `inner`, `outer`, `matrix_norm`, `lstsq`, `lu_solve`, `householder_product`, `vander`

### What is missing

| Missing | Notes |
|---|---|
| `cholesky_ex` | Cholesky + `info` singularity flag (LAPACK `_potrf`) |
| `inv_ex` | `inv` + `info` flag |
| `solve_ex` | `solve` + `info` flag |
| `ldl_solve` | Back-substitution using LDL factorization |
| `lu` | Full LU decomposition (`P`, `L`, `U` — different from `lu_factor`) |
| `diagonal` | `linalg.diagonal` as a batched view (different from `torch.diagonal`) |

---

## 8. `torch.autograd`

### What Lucid has (11 names)

`no_grad`, `enable_grad`, `set_grad_enabled`, `is_grad_enabled`, `inference_mode`, `backward`, `grad`, `Function`, `FunctionCtx`, `gradcheck`, `gradgradcheck`, `detect_anomaly`, `jacobian`, `hessian`, `vjp`, `jvp`

### What is missing

| Missing | Notes |
|---|---|
| `vmap` / `torch.vmap` | Vectorised map (functorch-style) |
| `graph.allow_mutation_on_saved_tensors` | For in-place ops in custom backward |
| `profiler.profile` | `autograd.profiler` context manager |
| `set_detect_anomaly` | Programmatic anomaly control |
| `save_on_cpu` | Memory-efficient backward hook |

---

## 9. `torch.utils.data`

### What Lucid has (18 names)

**Datasets (8):** `Dataset`, `IterableDataset`, `TensorDataset`, `ConcatDataset`, `ChainDataset`, `StackDataset`, `Subset`, `random_split`

**Samplers (7):** `Sampler`, `SequentialSampler`, `RandomSampler`, `SubsetRandomSampler`, `WeightedRandomSampler`, `BatchSampler`, `DistributedSampler`

**DataLoader (3):** `DataLoader`, `default_collate`, `get_worker_info`, `WorkerInfo`

### What is missing

| Missing | Notes |
|---|---|
| `default_convert` | Converts numpy/list to tensor without copying |
| `collate` | Composable collate function builder |
| `IterDataPipe`, `MapDataPipe` | Deprecated in PyTorch 2.x — low priority |
| `dataloader2` | New-style DataLoader (experimental in PyTorch) |

---

## 10. `torch.fft` — **0% — Entire module missing ❌**

> **Implementation path:** MLX already provides `mlx.core.fft.*`. Wrap each in a thin `lucid.fft` module under `lucid/fft/__init__.py` using the same `_wrap`/`_unwrap` dispatch pattern. Autograd: FFT backward is `ifft(g)` scaled — implement via `FunctionMeta` or as engine ufuncs.

| Missing | Description |
|---|---|
| `fft` | 1-D discrete Fourier transform |
| `ifft` | 1-D inverse DFT |
| `fft2` | 2-D DFT |
| `ifft2` | 2-D inverse DFT |
| `fftn` | N-D DFT |
| `ifftn` | N-D inverse DFT |
| `rfft` | 1-D real DFT (output has `n//2+1` bins) |
| `irfft` | 1-D inverse real DFT |
| `rfft2` | 2-D real DFT |
| `irfft2` | 2-D inverse real DFT |
| `rfftn` | N-D real DFT |
| `irfftn` | N-D inverse real DFT |
| `hfft` | 1-D Hermitian DFT |
| `ihfft` | 1-D inverse Hermitian DFT |
| `hfft2` | 2-D Hermitian DFT |
| `ihfft2` | 2-D inverse Hermitian DFT |
| `hfftn` | N-D Hermitian DFT |
| `ihfftn` | N-D inverse Hermitian DFT |
| `fftfreq` | DFT sample frequencies |
| `rfftfreq` | Real DFT sample frequencies |
| `fftshift` | Shift zero-frequency to center |
| `ifftshift` | Inverse of fftshift |

---

## 11. `torch.special` — **~15% — No submodule, basics scattered ⚠️**

> Many basics are already reachable at `lucid.*` (composites/engine), but there is no `lucid.special` namespace, and ~50 more-advanced functions are absent.

### Already covered (in engine or composites, but NOT under `lucid.special`)

`erf`, `erfc`, `erfinv`, `exp2`, `expm1`, `log1p`, `log2`, `log10`, `xlogy`, `logit`, `sinc`, `digamma`, `lgamma`, `i0`

### Missing entirely

| Missing | Notes |
|---|---|
| `erfcx` | Scaled complementary error function: `exp(x²)*erfc(x)` |
| `i0e` | Exponentially scaled `i0`: `exp(-|x|)*i0(x)` |
| `i1` | Modified Bessel function of first kind, order 1 |
| `i1e` | Exponentially scaled `i1` |
| `iv` | Modified Bessel function, arbitrary order |
| `ive` | Exponentially scaled `iv` |
| `ndtr` | Normal CDF: `Φ(x)` |
| `ndtri` | Inverse normal CDF (probit) |
| `log_ndtr` | `log(Φ(x))` (numerically stable) |
| `xlog1py` | `x * log1p(y)`, 0 * log1p(0) = 0 |
| `entr` | Entropy: `-x * log(x)`, 0 if x=0 |
| `polygamma` | Higher-order derivatives of digamma |
| `multigammaln` | Log of multivariate gamma function |
| `zeta` | Hurwitz zeta function |
| `bessel_j0`, `bessel_j1` | Bessel functions of the first kind, order 0/1 |
| `bessel_y0`, `bessel_y1` | Bessel functions of the second kind, order 0/1 |
| `spherical_bessel_j0` | Spherical Bessel function order 0 |
| `modified_bessel_k0`, `modified_bessel_k1` | Modified Bessel K functions |
| `scaled_modified_bessel_k0`, `scaled_modified_bessel_k1` | Scaled variants |
| **Orthogonal polynomials** | `chebyshev_polynomial_t`, `chebyshev_polynomial_u`, `chebyshev_polynomial_v`, `chebyshev_polynomial_w`, `legendre_polynomial_p`, `shifted_legendre_polynomial_p`, `associated_legendre_polynomial_p`, `hermite_polynomial_h`, `hermite_polynomial_he`, `laguerre_polynomial_l`, `generalized_laguerre_polynomial_l`, `jacobi_polynomial_p` |

> **Note:** To get `lucid.special`, create `lucid/special/__init__.py` that re-exports the basics from composites/engine and adds the missing ones.

---

## 12. `torch.distributions` — **0% — Entire module missing ❌**

> **Implementation path:** Distributions are pure-Python probability objects. Can be built as a standalone `lucid/distributions/` package layered on `lucid.*` tensor ops. No C++ needed. Gradients flow through `rsample()` (reparameterization) or `log_prob()` differentiable paths automatically.

### Univariate discrete

| Missing | Notes |
|---|---|
| `Bernoulli` | p(x=1) = p |
| `Binomial` | sum of Bernoullis |
| `Categorical` | discrete with logits/probs |
| `Geometric` | number of trials until first success |
| `NegativeBinomial` | |
| `OneHotCategorical` | one-hot version of Categorical |
| `Poisson` | λ-parameterized |

### Univariate continuous

| Missing | Notes |
|---|---|
| `Normal` / `LogNormal` | Gaussian and log-Gaussian |
| `Uniform` | U(a,b) |
| `Beta` | Beta(α,β) |
| `Gamma` / `Chi2` | Gamma-family |
| `Exponential` | Exp(λ) |
| `Cauchy` / `HalfCauchy` | |
| `HalfNormal` | |
| `StudentT` | t-distribution |
| `Laplace` | Double exponential |
| `Pareto` | Power-law |
| `Weibull` | |
| `Kumaraswamy` | Beta-like, simpler reparameterization |
| `FisherSnedecor` | F-distribution |
| `Gumbel` | |
| `VonMises` | Circular distribution |
| `ContinuousBernoulli` | |
| `RelaxedBernoulli` | Concrete/Gumbel-softmax |
| `RelaxedOneHotCategorical` | |

### Multivariate

| Missing | Notes |
|---|---|
| `MultivariateNormal` | Full covariance Gaussian |
| `LowRankMultivariateNormal` | Low-rank + diagonal covariance |
| `Dirichlet` | |
| `Wishart` | Matrix distribution |
| `Independent` | Treats batch dims as event dims |
| `MixtureSameFamily` | GMM-style |

### Infrastructure

| Missing | Notes |
|---|---|
| `Distribution` | Base class |
| `ExponentialFamily` | Base for natural-parameter distributions |
| `TransformedDistribution` | Applies bijective transforms |
| `constraints` | `positive`, `simplex`, `real`, etc. |
| `transforms` | `ExpTransform`, `SigmoidTransform`, `AffineTransform`, etc. |
| `kl_divergence` | Registered KL dispatch |

---

## 13. `torch.signal` — **0% — Entire module missing ❌**

> **Implementation path:** All window functions are closed-form formulas over `lucid.arange`. Can be added as `lucid/signal/windows.py` with ~12 functions, all ~5 lines each.

| Missing | Formula |
|---|---|
| `windows.bartlett` | Triangular taper |
| `windows.blackman` | 3-term cosine sum |
| `windows.cosine` | `sin(π(n+0.5)/N)` |
| `windows.exponential` | `exp(-|n - center| / τ)` |
| `windows.gaussian` | `exp(-0.5 * ((n-M/2) / σ)²)` |
| `windows.general_cosine` | Weighted cosine sum |
| `windows.general_hamming` | `α - (1-α)*cos(2πn/(N-1))` |
| `windows.general_gaussian` | Generalized Gaussian |
| `windows.hamming` | Hamming (α=0.54) |
| `windows.hann` | Hann (α=0.5) |
| `windows.kaiser` | Kaiser-Bessel |
| `windows.nuttall` | 4-term Blackman-Nuttall |

---

## Implementation Notes (Architecture Reminders)

### Rules to follow (always)
- **No numpy at the Python composite level.** Use `.item()` loops, engine ops, or `random` stdlib.
- **`from __future__ import annotations` is forbidden** in all files.
- **`lucid._C` imports must alias as `_C_engine`** (never `_ce`, `_engine`, `_e`, `e`).
- Composites live in `lucid/_ops/composite/<category>.py` and register via `__all__` + `COMPOSITE_NAMES`.
- New submodules (e.g. `lucid.fft`) go under `lucid/<name>/__init__.py` and are added to `_SUBPKG_NAMES` in `lucid/__init__.py`.

### CPU vs GPU backend split
- CPU stream: **Apple Accelerate** (`vDSP`/`vForce`/`BLAS`/`LAPACK`) via `backend/cpu/*`.
- GPU stream: **MLX** via `backend/gpu/*`.
- `linalg` ops dispatch through MLX (which is CPU-backed on Apple Silicon) — wrap result back as GPU tensor.
- For ops MLX lacks natively (e.g. `scatter_max`), use **CPU round-trip**: evaluate GPU tensor → compute on CPU → rewrap as MLX array.

### FFT implementation path (P1)
```
lucid/fft/__init__.py
    → wraps mlx.core.fft.{fft, ifft, fft2, ifft2, fftn, ifftn,
                           rfft, irfft, rfft2, irfft2, rfftn, irfftn,
                           hfft, ihfft}
    → fftfreq, rfftfreq: pure-Python formula over lucid.arange
    → fftshift, ifftshift: roll-based composite
    → autograd: fft backward = ifft(g); rfft backward = irfft(g)
```
Add to `_SUBPKG_NAMES` in `lucid/__init__.py`.

### distributions implementation path (P8)
```
lucid/distributions/
    __init__.py         — re-exports all
    distribution.py     — Distribution base class
    exponential_family.py
    normal.py           — Normal (reparameterizable)
    uniform.py
    bernoulli.py
    categorical.py
    ...
    transforms.py       — bijective transforms
    constraints.py      — constraint objects
    kl.py               — kl_divergence registry
```
No C++ needed. All math via `lucid.*` tensor ops.
