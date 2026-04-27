# Changelog

All notable changes to the Lucid C++ engine rebuild are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The rebuild lives on branch `claude/beautiful-cannon-440988`. Current
public Lucid (Python-backed) is `lucid-3.0`. The first rebuild release will
be tagged `v3.1.0-rc1` after Phase 5.

## [Unreleased]

### Added — Phase 4b (full optimizer + LR scheduler parity with Lucid Python package)
Brings the C++ optimizer set to **1:1 parity** with the existing
`lucid.optim` Python package. Was missing 8 optimizers + 4 schedulers
after the initial Phase 4 ship; now all 12 optimizers and 8 schedulers
are present in the C++ engine.

- **`optim/MoreOptimizers.{h,cpp}`** — eight new optimizer classes:
  - **`ASGD`** (Averaged SGD): tracks an `ax` running average of
    parameters once `step >= t0`; matches Lucid Python's flavor (note:
    PyTorch's ASGD uses a different dynamic eta/mu schedule — we use
    the simpler Lucid-Python form intentionally).
  - **`NAdam`**: Nesterov-accelerated Adam with `momentum_decay`
    schedule for μ; matches PyTorch torch.optim.NAdam.
  - **`RAdam`**: Rectified Adam (Liu et al. 2020); falls back to
    plain SGD-with-momentum when `rho_t ≤ 5`.
  - **`RMSprop`**: standard RMSprop with optional `centered` variance
    and momentum; uses PyTorch's `sqrt(avg) + eps` denominator.
  - **`Rprop`**: resilient backprop with per-element step-size
    adaptation, clip to `(step_min, step_max)`. Native MLX path uses
    `where`/`clip`/`sign` (no CPU fallback).
  - **`Adagrad`**: per-parameter accumulator of squared grads;
    optional `initial_accumulator_value`.
  - **`Adadelta`**: parameter-free adaptive LR (Zeiler 2012); two
    running averages (sq_avg, accumulated_update).
  - **`Adamax`**: Adam variant using infinity-norm exp_inf instead of
    L2 v.
  - All eight: full CPU (typed loop F32/F64) and GPU (functional MLX)
    paths.
- **`optim/LRScheduler.{h,cpp}`** — four new schedulers:
  - **`LambdaLR`**: `lr = base · lambda(epoch)`. The lambda is held as
    a `std::function<double(int64_t)>` so a Python callable passes
    through transparently (pybind11/functional).
  - **`ReduceLROnPlateau`**: metric-driven scheduler with `step(metric)`
    signature (not `step()`). Mode={Min, Max}, threshold_mode={Rel, Abs},
    cooldown, min_lr, eps. Doesn't fit the LRScheduler closed-form base
    (it's a stateful FSM) — registered as its own class.
  - **`CyclicLR`** (Smith 2017): triangular wave between `base_lr` and
    `max_lr` with mode in {Triangular, Triangular2, ExpRange}.
  - **`NoamScheduler`** (Attention-Is-All-You-Need): warmup-then-decay
    `lr = factor · model_size^(-0.5) · min(step^(-0.5), step·warmup^(-1.5))`.
- **`bindings/bind_optim.cpp`** — extended with all 8 new optimizer
  classes and all 4 new scheduler classes. CyclicLR.Mode and
  ReduceLROnPlateau.{Mode, ThresholdMode} exposed as nested enums.
  Python callables passed to LambdaLR survive thanks to
  `pybind11/functional.h`.

Full Lucid optimizer surface in C++ now:
| | Optimizers | LR schedulers |
| --- | --- | --- |
| Phase 4 | SGD, Adam, AdamW | StepLR, ExponentialLR, MultiStepLR, CosineAnnealingLR |
| Phase 4b | ASGD, NAdam, RAdam, RMSprop, Rprop, Adagrad, Adadelta, Adamax | LambdaLR, ReduceLROnPlateau, CyclicLR, NoamScheduler |

Verified vs PyTorch (5-step rollouts on 20-element parameter):
- **NAdam, RAdam, Rprop, Adagrad, Adadelta, Adamax**: max\|d\| ≤ 1.8e-7
  (most bit-exact 0.0).
- **RMSprop, RMSprop-centered**: max\|d\| ≤ 6e-8.
- **ASGD**: max\|d\| ≤ 5.8e-6 vs PyTorch ASGD (loose tolerance because
  the algorithms intentionally differ — Lucid Python flavor).
- **GPU spot-checks** (NAdam, RMSprop, Adagrad, Adamax): max\|d\| ≤ 3e-8.
- **All 4 new schedulers** (LambdaLR, CyclicLR-Triangular, NoamScheduler,
  ReduceLROnPlateau): bit-exact vs PyTorch (or vs the formula-derived
  reference where PyTorch lacks a built-in like Noam).

CPU baseline regression: `pytest lucid/test/ -m "not slow"` still
**411 passed / 1 skipped** (zero regressions).

### Added — Phase 4 (Optimizers: SGD / Adam / AdamW + LR schedulers)
First training-loop primitive — parameter updates that work natively on the
new C++ engine, in-place on `TensorImpl::storage_` (CPU + GPU), with
PyTorch-equivalent semantics.

- **`optim/Optimizer.{h,cpp}`** — base class:
  - `step()`: iterate parameters, dispatch to subclass `update_one(slot,
    param, grad)`, bump `param->version_` to invalidate any saved_inputs
    captured by autograd nodes (Item #9 contract).
  - `zero_grad()`: clear `grad_storage_` on every managed parameter.
  - Lazy state allocation: `init_state_slot()` is called on first sight of
    a slot, so device memory for momentum / Adam moments isn't pre-allocated.
  - `set_lr() / lr() / state_dict_id()` virtuals for LR schedulers and
    future checkpoint restore.
- **`optim/SGD.{h,cpp}`** — SGD with momentum, dampening, weight_decay,
  Nesterov. Update formula matches PyTorch torch.optim.SGD bit-for-bit.
  - CPU: typed loop (F32/F64).
  - GPU: functional MLX path — replace `param.storage_.arr` with the
    result of one MLX expression per step. Same for momentum buffer.
- **`optim/Adam.{h,cpp}`** — Adam (L2 weight decay) and AdamW (decoupled
  weight decay) sharing the same internal step kernel parameterized by a
  `decoupled_wd` flag. Bias correction uses `1 - β^t` factors computed
  from the optimizer-global step counter (matches PyTorch). State per
  parameter: two buffers (m, v), zero-initialized on first step.
- **`optim/LRScheduler.{h,cpp}`** — base class + 4 concrete schedulers:
  - `StepLR(step_size, gamma)` — drop by `gamma` every `step_size` epochs.
  - `ExponentialLR(gamma)` — `lr = base · gamma^epoch`.
  - `MultiStepLR(milestones, gamma)` — drop by `gamma` at each milestone.
  - `CosineAnnealingLR(T_max, eta_min)` — continuous cosine
    `lr = eta_min + 0.5(base − eta_min)(1 + cos(π·epoch/T_max))`.
    The cosine continues past T_max (matches PyTorch — no clamp).
  - All schedulers compute LR as a closed-form function of `epoch_`, so
    `set_epoch(n)` resumes from any point (checkpoint restore).
- **`bindings/bind_optim.cpp`** — pybind11 surface:
  `eng.{Optimizer, SGD, Adam, AdamW, LRScheduler, StepLR, ExponentialLR,
  MultiStepLR, CosineAnnealingLR}`.

Verified vs PyTorch (5-step rollouts on a 20-element parameter, then
multi-step LR schedule comparisons):
- **SGD vanilla** CPU: bit-exact (0.0); GPU: 6e-8.
- **SGD + momentum + Nesterov + wd**: CPU 8.9e-8, GPU 2.8e-8.
- **Adam** (lr=1e-3, default betas/eps): CPU bit-exact (0.0), GPU bit-exact.
- **Adam + L2 weight decay**: CPU bit-exact (0.0).
- **AdamW** (decoupled wd=1e-2): CPU 1.8e-7, GPU bit-exact.
- **StepLR** schedule [1.0, 1.0, 0.5, 0.5, 0.5, 0.25, ...]: bit-exact vs
  PyTorch.
- **ExponentialLR** (γ=0.95): max\|d\|=1.1e-16 (machine precision).
- **MultiStepLR** [milestones=[3,7]]: bit-exact.
- **CosineAnnealingLR** (T_max=10): max\|d\|=2.2e-15.
- **`version_` bump after step()**: confirmed (0 → 1 after one step,
  protects against stale backward).
- **Multi-param Adam** end-to-end (linear forward → backward → step over
  5 iterations): x/W/b match PyTorch within 1.2e-7.

CPU baseline regression: `pytest lucid/test/ -m "not slow"` still
**411 passed / 1 skipped** (zero regressions).

### Added — Phase 3.8 (Random tensor ops + determinism enforcement)
User-facing samplers backed by Lucid's Philox-4x32-10 generator. All draws
go through the Lucid `Generator` so a seeded generator produces bit-exact
identical output across CPU and GPU.

- **Storage-level helpers** in `autograd/Helpers.{h,cpp}`:
  - `random_uniform_storage(shape, lo, hi, dt, device, gen)` — F32/F64.
  - `random_normal_storage(shape, mean, std, dt, device, gen)` — Box-Muller
    transform over uniform pairs (with `eps=1e-7` clamp on `u1` to keep
    `log(u1)` finite).
  - `random_bernoulli_storage(shape, p, dt, device, gen)` — uniform draw
    compared against `p`.
  - `random_randint_storage(shape, low, high, dt, device, gen)` — uniform
    integer in [low, high). For ranges > 2³², draws two `uint32` and
    composes a `uint64` to avoid bias from modulo.
  - All four use the same CPU-side fill + optional `upload_cpu_to_gpu`
    path so CPU and GPU outputs are bit-identical given the same
    Generator state.
- **`autograd/ops/random/RandomOps.{h,cpp}`** — user-facing wrappers:
  - `rand_op(shape, dt, device, gen?)` → uniform [0, 1)
  - `uniform_op(shape, low, high, dt, device, gen?)`
  - `randn_op(shape, dt, device, gen?)` → standard normal N(0, 1)
  - `normal_op(shape, mean, std, dt, device, gen?)`
  - `randint_op(shape, low, high, dt, device, gen?)` → I32/I64
  - `bernoulli_op(shape, p, dt, device, gen?)` → 1.0 or 0.0 cells
  - All ops are no-grad (do not enter the autograd graph). Optional
    `Generator*` argument; `nullptr` falls back to the process-default
    generator.
- **`bindings/bind_random.cpp`** — exposes
  `eng.{rand, uniform, randn, normal, randint, bernoulli}` with default
  args `dtype=F32, device=CPU, generator=None`.

**Determinism contract** (now uniformly enforced):
- All random ops route through Lucid Philox; given seed `s`, two `Generator(s)`
  instances produce bit-exact identical sequences.
- CPU and GPU paths produce bit-exact identical output for the same seed
  (same generator advance = same uniform u32 stream = same downstream
  transform). Verified on 200+ samples.
- `set_deterministic(True/False)` is a process-global atomic toggle that
  any future non-deterministic op path can consult; current op kernels are
  deterministic by construction (Philox CPU-side, MLX MPS conv has a
  deterministic default), so the flag is observable but no op currently
  branches on it.
- `Generator.counter` is monotonically advanced by every draw and is
  observable for checkpointing / resume.

Verified (28 checks):
- Shape / dtype correctness for all 6 ops.
- Same seed → bit-exact output (CPU): rand, randn, bernoulli, randint
  all identical to byte level.
- CPU vs GPU bit-exactness for the same seed: randn[200] and uniform[300]
  both bit-identical.
- Different seeds: 1000/1000 randn samples differ.
- Statistical sanity (100k samples each):
  - `rand`: mean=0.5001, range ⊂ [0, 1)
  - `randn`: mean=8e-4, std=0.9977
  - `bernoulli(p=0.3)`: mean=0.3005
  - `uniform[-10, 10)`: mean=0.012, range valid
  - `randint[0, 10)`: histogram bins in [9866, 10163] (uniform within ~1%).
- Default generator (gen=None) works.
- `Generator.counter` advances after each op.
- Error paths: `uniform` low≥high, `bernoulli` p∉[0,1], `randint` high≤low
  all raise clear `LucidError` messages.

OpRegistry: still 69 explicitly registered ops (random ops are
no-graph-node free functions; they don't need autograd registration).

Baseline regression: `pytest lucid/test/ -m "not slow"` still
**411 passed / 1 skipped** (zero regressions). All prior phase
verifications (3.7.4 / 3.7.5a / 3.7.5b / 3.7.5c PyTorch parity +
stress + numerical stability) still pass.

### Verified — Phase 3.7.5c (PyTorch parity + stress + numerical stability)
End-to-end correctness verification suite. No code changes; closes the
"do we agree with PyTorch" question that was outstanding through 3.7.5b.

**PyTorch reference parity** (100+ checks, CPU + GPU):
- conv1d / conv2d / conv3d: forward bit-exact on CPU; GPU within 5e-7 fwd,
  1e-6 dx, 2e-6 dW, 0 db. ResNet-style stride=2 7×7 conv: 6.6e-7 fwd,
  1.9e-6 dx, 1.2e-4 dW.
- conv_transpose2d: CPU bit-exact; GPU within 1.2e-7 fwd, 2.4e-7 dx,
  1.2e-7 dW.
- max_pool2d / avg_pool2d: bit-exact CPU & GPU.
- batch_norm (2D): 4.8e-7 fwd, 7.2e-7 dx, 2.2e-6 dgamma, 0 dbeta.
  ResNet-scale (B=32, C=64, H=W=16): 1.9e-6 fwd, 5.7e-8 dx, 2.6e-4 dgamma.
- group_norm, layer_norm: 4.8e-7 fwd, ≤6e-7 dx/dgamma, 0 dbeta.
- linear: bit-exact (0.0) CPU; 1.2e-7 GPU.
- softmax: 6e-8 fwd, 5.5e-8 dx.
- relu / sigmoid / tanh / silu / softplus: ≤2.4e-7 forward, ≤1.2e-7 dx.

**End-to-end ResNet block** (conv → BN → relu → conv → BN → add → relu):
- CPU vs PyTorch: 9.5e-7 fwd, 2.6e-6 dx, 7.6e-6 dW1.
- GPU vs PyTorch: 1.9e-6 fwd, 4.8e-6 dx, 1.4e-5 dW1.

**Numerical stability**:
- softmax([1000, 1000, 1000, 1000]) → [0.25, 0.25, 0.25, 0.25] on both
  CPU and GPU (no NaN/Inf — max-subtract stability).
- sigmoid([1000, -1000, 0, 50, -50]) → finite values, no overflow.
- batch_norm on a constant feature map (var=0 numerically) returns
  bit-exact zero (with γ=1, β=0) — no NaN/Inf.

**F64 CPU path**:
- linear / batch_norm in F64 match PyTorch at machine precision
  (4.4e-16 forward, 8.9e-16 dx).

**F64 GPU rejection** (MLX-Metal limitation): clear `NotImplementedError`
with actionable message ("Cast to float32 first, or keep the tensor on
CPU") raised at upload time, before any compute.

**Dropout reproducibility**:
- Same seed → bit-exact same mask (Philox-4x32-10 determinism preserved
  through CPU mask generation + optional GPU upload).
- Different seed → ~51/100 positions differ (independence sanity check).

**AMP schema metadata** (33 ops checked):
- Promote: add/sub/mul/div, maximum/minimum, matmul, linear, conv1d/2d/3d,
  conv_transpose1d/2d/3d.
- ForceFP32: pow, softmax, gelu, softplus, layer_norm, rms_norm,
  batch_norm{1d,2d,3d}, group_norm, pow_scalar, rpow_scalar.
- KeepInput: max_pool / avg_pool 1d/2d/3d, clip.

**AutocastGuard (thread-local RAII)**:
- Inactive by default; activates inside guard scope; restored on exit.
- Nesting works correctly (inner scope overrides; outer restores).

**Determinism toggle**: `set_deterministic(True)` → observable via
`is_deterministic()`; cleanly restored on toggle off.

**OpRegistry inventory**: 69 unique ops registered. Spot-check across
1D/2D/3D variants of conv, conv_transpose, pool, batch_norm passes; all
binary/unary/reduce ops still present.

**Limitation noted**: Runtime auto-cast insertion (the actual fp32→fp16
cast inside autocast scope) is currently a no-op. Op-level schemas are
correct and the guard mechanism works; the runtime hook that consults
`amp::active_dtype()` and inserts cast ops is deferred to a later phase
(no concept-level work; ~1 day to wire when an actual cast op family
exists in the autograd graph).

**Baseline regression**: `pytest lucid/test/ -m "not slow"` still
**411 passed / 1 skipped** (zero regressions).

### Added — Phase 3.7.5b (ConvTranspose 1D/2D/3D + AdaptivePool 1D/2D/3D)
Continues the N-D generalization started in 3.7.5a. The new ops sit on
top of the same N-D infrastructure (`ConvNd`, `PoolNd`, im2col/col2im
kernels) and reuse it heavily.

- **`autograd/ops/nn/ConvTransposeNd.{h,cpp}`** — `ConvTransposeNdBackward<N>`
  template instantiated for N ∈ {1, 2, 3}. Weight shape follows PyTorch
  convention `(C_in, C_out, *K)` (C_in first; the spatial inverse of conv).
  Output shape:
  `O[i] = (S[i]-1)·stride[i] − 2·pad[i] + K[i] + output_padding[i]`.
  - **CPU forward**: per-batch `sgemm + col2im`. `cols = W_2d^T @ x_2d` builds
    the `(C_out·prod(K), prod(S))` column matrix; `col2im` scatters it into
    `y_b` of shape `(C_out, *O)` using the standard
    `o = s·stride + k − pad` mapping. Same im2col/col2im kernels as ConvNd —
    no new backend code.
  - **GPU forward**: `mlx::core::conv_transpose{1,2,3}d` after permuting W
    `(C_in, C_out, *K) → (C_out, *K, C_in)` (MLX's transpose-conv weight
    format).
  - **Backward dx**: ConvTranspose's adjoint is regular convolution, so dx
    routes through `mlx::core::conv{1,2,3}d` (GPU) or `im2col + sgemm` of
    grad with W reshaped (CPU). The W permutation for the dx-conv is
    `(C_in, C_out, *K) → (C_in, *K, C_out)` — different from the forward
    permutation; this was the key fix during verification.
  - **Backward dW**: same `conv_general` dilation trick as ConvNd's dW but
    with x and grad swapped roles. Output `(C_out, *K, C_in)` permuted to
    `(C_in, C_out, *K)`.
  - **Backward db**: `sum(grad, axes={0, 2..N+1})`.
- **`autograd/ops/nn/AdaptivePool.{h,cpp}`** — `adaptive_max_pool{1,2,3}d`
  and `adaptive_avg_pool{1,2,3}d`. Implementation strategy: when input is
  divisible by output along every spatial axis (the common case —
  `(224,224)→(7,7)`, `(32,32)→(1,1)`, etc.), adaptive pool is exactly
  regular pool with `kernel = stride = S/O, padding=0`. The op delegates
  to the corresponding `{max,avg}_pool{N}d` — no new state, no new
  kernels, full forward/backward inherited. Non-uniform case raises
  `NotImplementedError` with an actionable message ("Pad input to a
  divisible size or use a regular pool").
- **Bindings**: `bind_nn.cpp` exposes `conv_transpose1d/2d/3d` and
  `adaptive_max/avg_pool1d/2d/3d`.

OpRegistry: now **73 unique ops** (was 70, +3 from
`conv_transpose1d/2d/3d`; AdaptivePool reuses pool registrations).

Verified — 25 forward + backward parity checks all pass:
- conv_transpose1d (B=2, Ci=2, Co=3, S=4, K=3, stride=2, pad=1, opad=1):
  CPU vs numpy ref 6.0e-8; GPU vs CPU 6.0e-8; dx 6.0e-8, dW 1.2e-7, db 0.
- conv_transpose2d (B=1, Ci=2, Co=3, H=W=4, K=3, stride=2, pad=1, opad=1):
  CPU vs ref 6.0e-8; GPU vs CPU 8.9e-8; dx 4.8e-7, dW 2.4e-7.
- conv_transpose3d (B=1, Ci=2, Co=2, D=H=W=3, K=2, stride=2): GPU vs CPU
  fwd bit-exact (0.0); dx/dW 2.4e-7.
- adaptive_avg_pool / adaptive_max_pool 1D/2D/3D: forward + backward
  bit-exact (0.0). cpu vs reg-pool ref bit-exact (delegation works).
- Non-uniform error path raises clear `NotImplementedError`.
- CPU baseline regression: **411 passed / 1 skipped** (zero regressions).

### Added — Phase 3.7.5a (N-D generalization: 1D/2D/3D variants of Conv/Pool/BatchNorm/GroupNorm)
The op layer is now uniformly templated on spatial rank. A single class
template per family handles 1D / 2D / 3D — autograd wiring, MLX dispatch,
shape derivation, and grad formulas are all shared. CPU kernels stay
specialized per rank for performance. Code reuse is maximal at the op
layer; backend kernels are 1D/2D/3D specializations sharing the same
column / pool layout convention.

- **`backend/cpu/Im2Col.{h,cpp}`** — adds 1D and 3D im2col / col2im
  alongside existing 2D. Same channel-major-then-flat-output column
  layout, same memcpy access pattern. Kept in one file; 1D = ~30 LoC,
  3D = ~70 LoC additions.
- **`backend/cpu/Pool.{h,cpp}`** — adds `max_pool{1,3}d` and
  `avg_pool{1,3}d` forward/backward kernels with identical argmax index
  semantics (linear input index in spatial extent: `l`, `h*W+w`,
  `d*H*W+h*W+w`).
- **`autograd/ops/nn/ConvNd.{h,cpp}`** — replaces `Conv2d.{h,cpp}`. A
  single class template `ConvNdBackward<int N>` instantiated for
  N ∈ {1, 2, 3}. Specialized `schema_v1` per rank. CPU forward/backward
  dispatch into rank-specific im2col/col2im via thin specialization
  helpers (`im2col_dispatch<N, T>`, `col2im_dispatch<N, T>`). GPU paths
  share the NCHW↔NHWC bridge logic via `nchw_to_nhwc_perm<N>()` and
  call `mlx::core::conv1d` / `conv2d` / `conv3d` (and their transposed
  siblings) through `mlx_conv_nd<N>` and `mlx_conv_transpose_nd<N>`
  trampolines. dW backward via `mlx::core::conv_general` (rank-agnostic)
  with the dilation trick — same recipe for all ranks. Output cropping,
  `output_padding` for stride/floor ambiguity, and contiguous() to avoid
  lazy-stride read corruption all carry over from the 2D version.
- **`autograd/ops/nn/PoolNd.{h,cpp}`** — replaces `Pool.{h,cpp}`.
  Templates `MaxPoolNdBackward<N>` and `AvgPoolNdBackward<N>`. GPU
  forward uses `mlx::core::pad` + a generalized `as_strided` that builds
  a `(B, C, *O, *K)` window view via per-rank stride computation; reduce
  with `max` / `mean` over the trailing kernel axes. MaxPool backward
  scatters via `mlx::core::scatter_add_axis` — argmax → `(ki..., kn)`
  multi-index → flat padded index built by an N-iteration accumulator
  loop in MLX index arithmetic. AvgPool backward similarly distributes
  `g/prod(K)` across the kernel positions via scatter_add.
- **`autograd/ops/nn/BatchNorm.{h,cpp}`** — `BatchNormNdBackward<N>`
  template; reduction axes computed as `{0, 2, …, N+1}` per rank. CPU
  kernel parameterized by `spatial = prod(*S)`. GPU path identical
  algorithm, axes vector built dynamically from N. Three op
  registrations: `batch_norm1d`, `batch_norm` (kept as 2D for API
  compat), `batch_norm3d`.
- **`autograd/ops/nn/GroupNorm.{h,cpp}`** — single dynamic-rank op (no
  template) since the user-facing API is rank-agnostic. Stores
  `spatial_dims_: vector<int>` and reshapes `(B, C, *S) → (B, G, C/G, *S)`
  on the fly. CPU kernel parameterized by `spatial`. GPU path constructs
  reshape / reduce-axes vectors from the runtime spatial rank. Single
  op registration: `group_norm` (handles 1D / 2D / 3D / etc inputs).
- **Bindings**: `bind_nn.cpp` exposes `conv1d`/`conv3d`,
  `max_pool1d`/`max_pool3d`, `avg_pool1d`/`avg_pool3d`,
  `batch_norm1d`/`batch_norm3d`. The existing 2D names (`conv2d`,
  `max_pool2d`, `avg_pool2d`, `batch_norm`) are preserved.

OpRegistry now contains **70 unique ops** (was 58):
- conv1d, conv2d, conv3d (3 entries; was 1)
- max_pool1d/2d/3d, avg_pool1d/2d/3d (6 entries; was 2)
- batch_norm1d, batch_norm, batch_norm3d (3 entries; was 1)
- (group_norm stays 1; rank-dispatched dynamically)
- net change: +12

Verified (28 forward + backward parity checks, in addition to the
existing 25 from 3.7.4):
- conv1d (B=2, Ci=3, Co=4, L=8, KL=3, stride=2, pad=1): CPU vs numpy
  ref 1.2e-7; GPU vs CPU forward 6e-8; dx/dW/db ≤ 2.4e-7.
- conv3d (B=1, Ci=2, Co=3, D=H=W=4, K=3): CPU vs ref 2.4e-7;
  GPU forward 3.0e-7; dx 9.5e-7, dW 1.2e-6.
- max_pool1d/3d, avg_pool1d/3d: forward + backward bit-exact (0.0).
- batch_norm1d (B=4, C=3, L=8): CPU vs numpy ref 4.8e-7; GPU forward
  4.8e-7; dx 3.9e-7, dgamma 7.2e-7.
- batch_norm3d (B=2, C=3, D=H=W=4): CPU vs ref 4.8e-7; GPU forward
  4.8e-7; dx 1.2e-6.
- group_norm 1D and 3D: forward + dx parity ≤ 2.7e-6.
- CPU baseline regression: **411 passed / 1 skipped** (zero regressions).

Deferred to Phase 3.7.5b follow-up (still in N-D scope but smaller in
practice): conv_transpose1d/2d/3d, adaptive_max/avg_pool1d/2d/3d.
Determinism + AMP + integration verification moved to Phase 3.7.5c.

### Added — Phase 3.7.4 (Conv2d / MaxPool / AvgPool / BatchNorm / GroupNorm on GPU — native MLX, no CPU fallback)
The full conv-family now runs natively on the GPU via MLX primitives.
No download/upload trips for any forward or backward path.

- **`autograd/ops/nn/Conv2d.cpp`** — forward NHWC bridge, backward via
  `conv_transpose2d` (dx) + `conv_general` (dW):
  - Forward: `transpose(x, {0,2,3,1}) → mlx::conv2d → + bias → transpose
    {0,3,1,2} → contiguous`. The trailing `contiguous()` materializes the
    NCHW result, avoiding lazy-stride read corruption that produced wrong
    Co > 1 values when downstream ops indexed into a transposed view.
  - Backward dx: weight permutation for transposed conv is
    `(Co, Ci, KH, KW) → (Ci, KH, KW, Co)` (axes `{1, 2, 3, 0}`), then
    `mlx::conv_transpose2d(grad_nhwc, W_t_nhwc, stride, padding,
    {1, 1}, output_padding)`. `output_padding` resolves the
    floor-division ambiguity when the forward stride does not evenly
    divide the spatial dims (`opad = H + 2p − K − (OH−1)·s`).
  - Backward dW: dilation trick via `conv_general(x_perm, grad_perm,
    stride={1,1}, padding=p, kernel_dilation=stride)` where x is
    permuted `(Ci, H, W, B)` and grad as kernel `(Co, OH, OW, B)`.
    Output is sliced to `(Ci, KH, KW, Co)` (since conv_general's output
    spatial extent can exceed `KH × KW` when the forward stride doesn't
    divide evenly), then transposed to NCHW `(Co, Ci, KH, KW)`.
  - db = `sum(grad, axes={0, 2, 3})`.
- **`autograd/ops/nn/Pool.cpp`** — fully native MLX (no CPU fallback):
  - Forward (both max and avg): `pad` with `−inf` (max) or `0` (avg) on
    the spatial axes, `as_strided` to extract `(B, C, OH, OW, KH, KW)`
    overlapping windows at stride `(sh, sw)`, then `max` / `mean` over
    the kernel axes.
  - MaxPool2d also computes `argmax` over the flattened `(KH·KW)` axis
    and saves it (cast to int32) as `saved_argmax_` — same contract as
    the CPU op, but the saved tensor lives on the GPU.
  - MaxPool2d backward: convert `argmax` → `(ki, kj)` via floor_divide /
    remainder, build a flat index over the padded `Hp·Wp` per `(b, c)`
    slice (`flat = (oh·sh + ki)·Wp + (ow·sw + kj)`), then a single
    `mlx::scatter_add_axis` accumulates `g` into a zero-padded buffer.
    Crop padding back via `slice`.
  - AvgPool2d backward: build a `(B, C, OH, OW, K)` updates tensor
    `g/K` and a matching `(B, C, OH, OW, K)` flat index tensor, then
    one `mlx::scatter_add_axis` distributes uniformly.
- **`autograd/ops/nn/BatchNorm.cpp`** — raw MLX forward (mean / var
  across `{0, 2, 3}`, rstd, broadcast affine) and raw MLX backward
  using the standard `dx = γ·rstd·(g − mean(g) − xnorm·mean(g·xnorm))`
  formula. Saved `mean` / `rstd` are MLX arrays at shape `(1, C, 1, 1)`
  for direct broadcast.
- **`autograd/ops/nn/GroupNorm.cpp`** — raw MLX. Reshape to
  `(B, G, C/G, H, W)` for grouped reduction, then back to NCHW. Saved
  `mean` / `rstd` at `(B, G)` (broadcast via reshape to
  `(B, G, 1, 1, 1)` in backward).
- **`backend/gpu/MlxBridge.cpp`** — `upload_cpu_to_gpu` now eagerly
  copies the void-buffer-backed array into MLX-owned memory via
  `mlx::core::copy()`. The void-pointer constructor produces an
  "external" array whose graph-level reasoning about strides isn't
  always sound for downstream `transpose`/`conv2d` chains; an eager
  copy gives canonical row-major MLX storage at the cost of one
  host→device replication. Also adds an early `NotImplementedError` for
  `Dtype::F64` on GPU (MLX-Metal limitation; clear actionable message).

Verified (25 forward + backward parity checks):
- conv2d stride=1/pad=1 and stride=2/pad=0: forward ≤ 1.4e-6, dx ≤ 4.8e-7,
  dW ≤ 9.5e-7, db = 0.
- max_pool2d / avg_pool2d (kernel=2, stride=2): forward + dx bit-exact
  (0.0).
- batch_norm: forward 4.8e-7, dx 1.2e-6, dgamma/dbeta = 0.
- group_norm G=2: forward 4.8e-7, dx 7.2e-7, dgamma 1.9e-6, dbeta = 0.
- Composite `conv2d → batch_norm → relu → max_pool2d → backward`:
  forward 1.2e-6, dx 2e-6, dW 1.1e-5.
- 50-cycle GPU leak check: zero growth.
- CPU baseline regression: **411 passed / 1 skipped** (zero regressions).

### Added — Phase 3.7.3 (Matmul / Linear / LayerNorm / RMSNorm on GPU)
- **`autograd/ops/linalg/Matmul.cpp`** — GPU branch added inside the
  custom `forward` / `apply`. Forward calls `mlx::core::matmul`; backward
  uses `transpose` + `matmul` (`dA = grad @ B^T`, `dB = A^T @ grad`).
  Empty-case (M/N/K = 0) returns zero arrays of the right shape via
  `mlx::core::zeros`, mirroring the CPU empty-case behavior.
- **`autograd/ops/nn/Linear.cpp`** — GPU forward: `matmul(x, transpose(W))
  + b` (broadcast). Backward reshapes the leading batch dims of `x` and
  `grad` to flat `(M, K)` / `(M, N)` so the matmul has the right rank,
  then reshapes `dx` back to `input_shapes_[0]`. `dW = g_2d^T @ x_2d`,
  `db = sum(g_2d, axis=0)`. Multi-batch (ndim > 2) inputs handled
  correctly.
- **`autograd/ops/nn/LayerNorm.cpp`** — GPU forward built from raw MLX
  ops so we can save mean/rstd as MLX arrays for backward (MLX's
  `fast::layer_norm` doesn't expose them):
  `mean = mean(x_2d, axis=1, keepdims)`, `var = mean(centered², ...)`,
  `rstd = rsqrt(var + eps)`. Backward uses the standard 3-term formula
  `dx = rstd · (gx − mean(gx) − xnorm · mean(gx · xnorm))`,
  `dgamma = sum(g · xnorm, axis=0)`, `dbeta = sum(g, axis=0)`. Saved
  mean/rstd at `(outer, 1)` shape so broadcast against `(outer, N)`
  centered is implicit.
- **`autograd/ops/nn/RMSNorm.cpp`** — GPU forward / backward analogous
  to LayerNorm but without mean subtraction or β.
  `rstd = rsqrt(mean(x², axis=1, keepdims) + eps)`,
  `xnorm = x · rstd`, `y = xnorm · gamma`. Backward:
  `dx = rstd · (gx − xnorm · mean(gx · xnorm, axis=1, keepdims))`,
  `dgamma = sum(g · xnorm, axis=0)`.
- All four ops keep their existing CPU paths intact; the GPU branch is
  selected at the top of forward / apply by `device_ == GPU`.

Verified (24 forward + backward parity checks):
- `matmul` forward bit-exact (0.0); dA at 9.5e-7, dB bit-exact.
- `linear` forward at 4.8e-7 (single-precision); dx/dW/db bit-exact.
  3-D batched `linear` forward at 9.5e-7; gradients bit-exact.
- `layer_norm` (1-axis γ and multi-axis γ): forward/dx/dgamma/dbeta all
  ≤ 7.2e-7.
- `rms_norm` forward/dx/dgamma all ≤ 4.8e-7.
- Composite chain `linear → layer_norm → relu → linear → backward`:
  forward + dx/dW1/dW2 all match between CPU and GPU at ≤ 4.8e-7.
- 50-cycle leak check on `linear → layer_norm → backward`: zero growth.
- CPU baseline regression: **411 passed / 1 skipped** (zero regressions).

### Added — Phase 3.7.2 (Binary / Unary / Reduce families on GPU)
- **`autograd/ops/unary/UnaryOp.h`** — device dispatch via concept
  `detail::HasUnaryGpuKernel<Derived>`, mirroring BinaryOp.
- **`autograd/ops/reduce/ReduceOp.h`** — device dispatch via concept
  `detail::HasReduceGpuKernel<Derived>` (the GPU kernel takes the same
  `(GpuStorage, Shape, axes, keepdims, Dtype)` signature as the CPU one).
- **All 7 binary ops** (Add, Sub, Mul, Div, Pow, Maximum, Minimum) get
  `gpu_kernel` static methods that wrap `mlx::core::{add,subtract,multiply,
  divide,power,maximum,minimum}` and route results through
  `gpu::wrap_mlx_array` for accounting.
- **All 22 unary ops** get GPU paths. Standard ones (Neg/Abs/Sign/
  Reciprocal/Square/Cube/Exp/Log/Log2/Sqrt/Sin/Cos/Tan/Asin/Acos/Atan/
  Sinh/Cosh/Tanh/Relu/Sigmoid/Silu/Gelu/Softplus/Round/Floor/Ceil/Invert)
  are centralized in a new `autograd/ops/unary/UnaryGpu.cpp` to keep
  per-op .cpp files focused on CPU kernels and grad formulas. Special
  cases:
  - `cube(x)` = `multiply(square(x), x)` (no MLX `cube` primitive).
  - `gelu` = tanh approximation, all coefficients constructed inline as
    MLX scalar arrays.
  - `softplus` = numerically stable `max(x,0) + log1p(exp(-|x|))`.
  - `relu` = `maximum(x, 0)` (MLX has no dedicated relu primitive).
  - `invert` = `bitwise_invert` for integers, `logical_not` for Bool.
- **Custom-forward unary ops** (LeakyReLU, PowScalar, RPowScalar, Clip,
  Softmax) get inline GPU branches inside their bespoke `forward`
  methods, computing via MLX while preserving each op's saved-state
  contract.
- **Softmax backward** GPU path: `dx = z · (g − sum(g·z, axis,
  keepdims))` directly via MLX ops; CPU keeps its three-pass kernel.
- **All 5 reduce ops** (Sum, Mean, Prod, Max, Min) get `gpu_kernel`
  defined inline in `Reductions.cpp`, calling
  `mlx::core::{sum,mean,prod,max,min}(arr, axes, keepdims)`.
- **`Helpers.cpp`** — every storage-level math primitive used by
  backwards now has a GPU branch:
  `negate`, `multiply`, `divide`, `square`, `clone`, `log`, `pow`,
  `ge_mask`, `lt_mask`, `add_scalar`, `mul_scalar`, `exp`, `sqrt`,
  `abs`, `sign`, `reciprocal`, `sin/cos/tan/asin/acos/atan`,
  `sinh/cosh/tanh`, `in_range_mask`, `positive_mask`, `leaky_mask`,
  `sigmoid`, `bernoulli_mask` (CPU-generated then uploaded to preserve
  the deterministic Philox contract identically across devices), and
  `broadcast_back_for_reduce` (`reshape` + `broadcast_to` + `copy` to
  produce an owned contiguous result). Two small in-file helpers
  `gpu_unary` / `gpu_binary` handle the boilerplate of unwrap → call MLX
  op → wrap with tracker.
- **`backend/cpu/.../UnaryGpu.cpp`** added to the engine sources list in
  `.extensions/cpp_extensions.json`.

Verified (49 forward + chain-backward parity checks):
- All 7 binary ops: max\|d\|=0.0 for {add,sub,mul,maximum,minimum}; ~2e-6
  for {div,pow}.
- All 22 unary ops: max\|d\| at most 2.4e-7 (single-precision rounding).
- 4 scalar-param ops + softmax: max\|d\| ≤ 9.5e-7.
- All 5 reduce ops (all-axis and per-axis): max\|d\| ≤ 4.8e-7.
- `chain = sum(relu(a*b + b))` end-to-end: forward and `dA`/`dB` match
  bit-exactly (max\|d\|=0.0) between CPU and GPU.
- GPU MemoryTracker: zero leak across 50 cycles of
  `tanh(add(mul(A,B),B)) → backward`.
- CPU baseline regression: **411 passed / 1 skipped** (zero regressions).

### Added — Phase 3.7.1 (MLX bridge + GpuStorage + first GPU op)
First GPU sub-phase. Establishes the bridge so subsequent op families can
add `gpu_kernel` static methods without further plumbing.

- **`core/Storage.h`** — `GpuStorage` is now a real holder:
  `std::shared_ptr<mlx::core::array> arr; size_t nbytes; Dtype dtype`. The
  shared_ptr's deleter notifies `MemoryTracker(GPU)` on free, mirroring the
  CPU side's accounting. The header forward-declares `mlx::core::array` so
  it doesn't transitively pull MLX into every translation unit.
- **`backend/gpu/MlxBridge.{h,cpp}`** — central conversion layer:
  - `to_mlx_dtype` / `from_mlx_dtype` — full Lucid↔MLX dtype mapping
    (Bool, I8/16/32/64, F16/32/64, C64; rejects bfloat16 and uint* until
    Lucid grows them).
  - `to_mlx_shape` — Lucid `int64` shape → MLX `int32` shape (throws if a
    dim exceeds INT32_MAX).
  - `upload_cpu_to_gpu(CpuStorage, Shape) → GpuStorage` — wraps the host
    buffer as an MLX array; the CpuStorage's shared_ptr is captured into
    the MLX deleter so the host buffer outlives any zero-copy use.
  - `download_gpu_to_cpu(GpuStorage, Shape) → CpuStorage` — eval + memcpy
    back to a freshly-allocated aligned CPU buffer.
  - `wrap_mlx_array(array&&, Dtype) → GpuStorage` — the canonical way GPU
    op kernels return their result; ensures every MLX array allocation is
    counted by `MemoryTracker(GPU)`.
- **`core/TensorImpl.cpp`** — `from_numpy` accepts `Device::GPU` (uploads
  via the bridge); `data_as_python` and `grad_as_python` evaluate + copy
  back through the bridge; `copy_from` clones the source MLX array via
  `mlx::core::copy()`.
- **`autograd/Helpers.cpp`** — GPU paths for the four primitives needed
  by the engine + binary backward:
  - `make_zero_storage(GPU)` → `mlx::core::zeros`
  - `make_ones_storage(GPU)` → `mlx::core::ones` (used for the implicit
    `loss.backward()` seed when the root is on GPU)
  - `accumulate_into(GPU)` → in-place `dst = mlx::core::add(dst, src)`
    (functional MLX; we replace dst's array, refcount handles cleanup)
  - `reduce_grad_to_shape(GPU)` → identity-clone fast path for equal
    shapes; multi-axis broadcast-back via `mlx::core::sum` + `reshape`.
- **`autograd/ops/binary/BinaryOp.h`** — device dispatch via a C++20
  concept `detail::HasGpuKernel<Derived>`. When `device_ == GPU`:
  - if Derived defines `static GpuStorage gpu_kernel(...)`, route to it;
  - else throw `NotImplementedError` with a clear "Phase 3.7.x in progress"
    message. Subsequent sub-phases just add `gpu_kernel` per op.
  - The non-contiguous guard is suppressed for GPU because contiguity is
    internal to MLX, not exposed via Lucid's `stride_`.
- **`autograd/Engine.cpp`** — `storage_is_empty` now also recognizes a
  default-constructed `GpuStorage` (null `arr`, `nbytes==0`), so the
  implicit ones-seed kicks in correctly when backward is called from a
  GPU root with no explicit seed.
- **`autograd/ops/binary/Add.{h,cpp}`** — first op with a GPU path. New
  `static GpuStorage gpu_kernel(...)` calls `mlx::core::add` and wraps
  the result. Backward unchanged (uses the now-GPU-capable Helpers).
- **`setup.py`** — `link_mlx` now resolves the install dir via
  `mlx.core.__file__` (mlx is a namespace package on newer wheels;
  `mlx.__file__` is `None`).
- **`.extensions/cpp_extensions.json`** — `link_mlx: true` on the engine
  extension; `MlxBridge.cpp` added to sources.

Verified:
- numpy → GPU TensorImpl → numpy round-trip is bit-exact (max\|d\|=0.0).
- `eng.add(a_gpu, b_gpu)` matches numpy reference and matches the CPU
  result bit-exactly (max\|d\|=0.0 for a 8×16 tensor).
- `engine_backward(c_gpu)` populates `a_gpu.grad` and `b_gpu.grad` as ones
  (correct for `c = a + b`); CPU and GPU grads match bit-exactly.
- GPU MemoryTracker shows zero leak across 50 cycles of
  `(64×64) add → backward` (current_bytes flat after warmup).
- CPU baseline regression: **411 passed / 1 skipped** (zero regressions).

Sub-phases 3.7.2 (binary/unary/reduce GPU), 3.7.3 (matmul/linear/softmax/
norm GPU), 3.7.4 (conv2d/pool/BN/GN GPU), 3.7.5 (determinism + AMP +
integration verification) follow.

### Added — Phase 3.5b + 3.6b (Conv2d / pooling / BatchNorm / GroupNorm)
Five NN ops added in one combined sprint. All backends F32+F64, schemas
registered, AMP policy declared, Python bindings exposed in `bind_nn.cpp`.

- **`backend/cpu/Im2Col.{h,cpp}`** — `im2col_<dt>` and `col2im_<dt>` for
  F32/F64. Layout: `(C·KH·KW, OH·OW)` cols matrix per batch; column index
  picks output spatial position, row index picks (channel, kernel-h,
  kernel-w). `col2im` accumulates with `+=` so backward dx is summed across
  overlapping receptive fields. Handles arbitrary stride and padding.
- **`backend/cpu/Pool.{h,cpp}`** — `max_pool2d_forward_<dt>` (writes int32
  argmax into a flat-spatial index per output cell), `max_pool2d_backward`
  (scatter-add g into argmax positions), `avg_pool2d_forward/backward_<dt>`
  (`g/(KH·KW)` distributed uniformly across each window).
- **`autograd/ops/nn/Conv2d.{h,cpp}`** — 3-input op `(x, W, b)`. Forward:
  per-batch `im2col` then `cblas_sgemm` with W reshaped to `(C_out, C_in·KH·KW)`,
  output `(C_out, OH·OW)` written directly into the (B, C_out, OH, OW) tensor
  (no final reshape). Backward: per-batch grad-cols accumulation for dW,
  `col2im` for dx, spatial sum for db. Stride and padding fully wired.
  AMP `Promote`.
- **`autograd/ops/nn/Pool.{h,cpp}`** — `MaxPool2dBackward` saves an int32
  argmax tensor (shape (B, C, OH, OW)) for backward; `AvgPool2dBackward`
  saves only kernel/stride/pad scalars (no tensor saved). Both AMP `KeepInput`.
  `stride=0` sentinel means "stride equals kernel" (PyTorch convention).
- **`autograd/ops/nn/BatchNorm.{h,cpp}`** — pure-function BatchNorm2d on
  `(B, C, H, W)`. Train-mode statistics: per-channel `μ` and `σ²` across
  `(B, H, W)`. Saves `mean_per_c` and `rstd_per_c` (shape (C,)) for backward.
  No running_mean / running_var update — that lives in the Module wrapper
  (Phase 5) which calls `_copy_from` on the running buffers. Eval-mode is
  just elementwise ops, so no separate eval op needed. AMP `ForceFP32`.
  Backward uses the standard 3-term formula
  `dx = (γ·rstd/N) · [N·dxn − sum(dxn) − xn·sum(dxn·xn)]` summed across
  the (B, H, W) reduction axes.
- **`autograd/ops/nn/GroupNorm.{h,cpp}`** — splits C into `num_groups`
  contiguous chunks; computes mean/var per (B, group) across `(C/G, H, W)`.
  Saves `mean_bg` and `rstd_bg` of shape (B, G). Per-channel γ/β affine
  applied after normalization (matching PyTorch). InstanceNorm is just
  GroupNorm with `num_groups=C`. AMP `ForceFP32`.
- Python bindings: `lucid._C.engine.{conv2d, max_pool2d, avg_pool2d,
  batch_norm, group_norm}`.
- OpRegistry now contains **58 unique ops**.

Verified: all 18 forward + backward parity checks pass.
- Forward parity vs numpy reference at atol=1e-4 across stride=1/pad=1,
  stride=2/pad=0 (conv2d), 2x2 stride-2 (pool), B=4/C=3/H=W=5 (BN), and
  G=2 + G=C (GN).
- Backward parity via central-difference numerical grad at atol=2e-2:
  conv2d dx/dW/db, max_pool2d dx, avg_pool2d dx, batch_norm dx/dγ/dβ,
  group_norm dx/dγ/dβ. Best gap 0.0e+00 (pool fwd), worst 3.14e-3 (conv dW).
- Baseline regression: `pytest lucid/test/ -m "not slow"` still **411 passed,
  1 skipped** (zero regressions vs Phase 3.6).

Phase 3.7 (deferred dedicated sprint): MLX GPU integration — all op
families get GPU paths simultaneously to keep correctness debugging
focused on backend dispatch rather than mixed with op semantics.

### Changed — Norm ops moved into `nn/` (consolidation)
The standalone `autograd/ops/norm/` directory created in Phase 3.6 was
merged into `autograd/ops/nn/`. Reasoning: norm ops are NN-specific
(no use outside neural networks), so the separation was inconsistent with
the `nn/` family containing Linear/Dropout/etc. The dedicated
`bind_norm.cpp` was merged into `bind_nn.cpp`.

Affected paths:
- `autograd/ops/norm/{LayerNorm,RMSNorm}.{h,cpp}` →
  `autograd/ops/nn/{LayerNorm,RMSNorm}.{h,cpp}`
- `bindings/bind_norm.cpp` → merged into `bindings/bind_nn.cpp` (deleted)

`backend/cpu/Norm.{h,cpp}` stays where it is — kernels are organized by
math category (Vdsp / Vforce / Blas / Reduce / Shape / Norm), independent
of which op family consumes them.

Python API surface unchanged.

### Added — Phase 3.6 (norm ops — LayerNorm + RMSNorm)
- **`backend/cpu/Norm.{h,cpp}`** — kernels: `layer_norm_forward/backward_<dt>`,
  `rms_norm_forward/backward_<dt>` (F32+F64). Operates on (outer, N)
  flattened layout. LayerNorm backward uses the standard combined formula
  `dx = (1/N)·rstd·[N·dxn − sum(dxn) − xn·sum(dxn·xn)]`; RMSNorm uses
  `dx = rstd·(γg − x·rstd²·mean(γg·x))`.
- **`autograd/ops/nn/LayerNorm.{h,cpp}`** — 3-input op `(x, γ, β)`. γ/β
  shape must match trailing dims of x (multi-axis normalize supported).
  Saves `mean` and `rstd` for backward. AMP `ForceFP32`.
- **`autograd/ops/nn/RMSNorm.{h,cpp}`** — 2-input op `(x, γ)`, no β. Common
  in modern LLMs (LLaMA, T5). Saves `rstd`. AMP `ForceFP32`.
- Python bindings (in `bind_nn.cpp`): `lucid._C.engine.{layer_norm, rms_norm}`.
- OpRegistry now contains **53 unique ops**.

Verified: forward matches numpy reference at atol=1e-4 (single-axis and
multi-axis normalized_shape); backward dx/dγ/dβ match numerical grad at
atol=2e-2; transformer-like composite chain
`linear → layer_norm → silu → linear → backward` populates all parameter
grads correctly; 50-cycle LN+linear loop has bounded memory.

Phase 3.6b deferred (next sprint): BatchNorm (train/eval split + running
stats), GroupNorm, InstanceNorm.

Phase 3.7 deferred (dedicated sprint): MLX GPU integration. Norm + MLX
together would have made debugging norm vs MLX integration impossible to
separate — handled as a focused Phase 3.7 with all op families getting
GPU paths simultaneously.

### Added — Phase 3.5 (NN ops — 8 ops; activations + softmax + Linear + Dropout)
- **Activations** (extend `unary/Activation.{h,cpp}`):
  - `sigmoid` (saves output; numerically stable formula avoiding `exp(-x)` overflow)
  - `silu` / swish (saves input; grad = σ(1 + x(1-σ)))
  - `gelu` (tanh approximation; AMP `ForceFP32`)
  - `leaky_relu(slope)` (saves input + scalar slope param)
  - `softplus` (numerically stable `max(x,0) + log1p(exp(-|x|))`; grad = σ(x))
- **Softmax** (`unary/Softmax.{h,cpp}`) — axis-aware, numerically stable
  (subtract max along axis). Saves output; backward formula
  `dx = z(g - sum(g·z, axis, keepdim))`. AMP `ForceFP32`.
- **`nn/Linear.{h,cpp}`** — fused 3-input op `y = x @ W^T + b`. Forward via
  `cblas_sgemm/dgemm`, supports arbitrary leading batch dims (flattens to
  2-D internally). Backward returns `(dx, dW, db)`.
- **`nn/Dropout.{h,cpp}`** — inverted dropout. Training: Bernoulli mask via
  `Generator`, scaled by `1/(1-p)`. Inference: identity. Reproducible under
  seeded `Generator`.
- **Helpers extended**: `sigmoid_storage`, `leaky_mask_storage`,
  `bernoulli_mask_storage` (uses `Generator`).
- Python bindings: `lucid._C.engine.{sigmoid, silu, gelu, leaky_relu,
  softplus, softmax, linear, dropout}`.
- OpRegistry now contains **51 unique ops**.

### Fixed — Edge.node strong reference (autograd inline-composition bug)
Previously `Edge::node` was `weak_ptr<Node>`. For chains like
`relu(linear(x, W, b))` where the intermediate `linear(...)` TensorImpl is
released immediately (Python doesn't bind a name), its `LinearBackward`
shared_ptr count fell to zero — the relu's edge then expired before
backward could traverse it, silently dropping gradients to x/W/b. Switched
to `shared_ptr<Node>` (PyTorch's design). Cycle prevention is intact:
grad_fns hold no strong refs back to TensorImpls (AccumulateGrad and
FuncOp's `input_tensors_` are weak). Detected by a multi-layer MLP
verification test where `x.grad` came back `None`.

### Added — Phase 3.4 (shape ops — 9 ops, materialized-copy strategy)
- **`backend/cpu/Shape.{h,cpp}`** — `permute_copy_<dtype>` for F32/F64/I32/I64.
  Iterates output flat indices, computes input flat via inverse-permutation
  multi-index decomposition. Single read + single write per element.
- **`autograd/ops/shape/Permute.{h,cpp}`** — five permutation ops sharing one
  `PermuteBackward` node:
  - `permute(t, dims)` — general permutation
  - `transpose(t)`, `T(t)` — reverse all axes
  - `mT(t)` — swap last two axes (matrix transpose; ndim ≥ 2)
  - `swapaxes(t, a1, a2)` — swap two named axes
  Backward: applies the inverse permutation via the same kernel.
- **`autograd/ops/shape/View.{h,cpp}`** — three "metadata-only" ops sharing
  one `ViewBackward`. Phase 3.4 v1 still memcpy-clones (preserves contiguous
  invariant). Backward = inverse memcpy with input shape's metadata.
  - `reshape(t, new_shape)` — `-1` wildcard supported
  - `squeeze(t, dim)`, `squeeze_all(t)` — remove size-1 dim(s)
  - `unsqueeze(t, dim)` — insert size-1 dim
- **`autograd/ops/shape/Contiguous.{h,cpp}`** — `contiguous(t)` op. Currently
  identity (clone), becomes load-bearing when zero-copy view ops land.
  Backward: identity (clone of grad).
- **FuncOp visibility change**: protected fields → public. Shape-op factories
  are free functions (not static class methods like BinaryOp::forward), so
  they need to populate `bwd->input_shapes_` etc. directly. Compute-op
  forwards (BinaryOp/UnaryOp/ReduceOp/Matmul) are unaffected.
- Python bindings (`bind_shape.cpp`) expose `lucid._C.engine.{permute,
  transpose, T, mT, swapaxes, reshape, squeeze, squeeze_all, unsqueeze,
  contiguous}`.
- OpRegistry now contains **43 unique ops**.

Verification: forward matches numpy across all permutations and views;
backward via numerical grad matches at atol=2e-2 across 11 test cases;
empty-tensor `permute`/`T`/`reshape` produce correct empty shapes; in-place
version check still fires after a transpose; 50-cycle composite chain
(permute → reshape → matmul → sum → backward) shows zero per-iteration leak.

### Added — Phase 3.3.6 (edge-case retrofits — items #7, #8, #9)
Three corner-case bugs identified during a self-audit, fixed before Phase 3.4
introduces stride-only view ops (which would have triggered #8 silently).

- **Item #9 — `VersionMismatch` enforcement.** `Node` gains a virtual
  `validate_versions()` (default no-op); `FuncOp<Derived, N_IN>` overrides to
  walk `input_tensors_` weak refs and compare against `saved_versions_`.
  `Engine::backward` invokes it immediately before `apply()`, throwing
  `lucid::VersionMismatch` when a user did an in-place op on an input
  between forward and backward. Verified across all op families: binary,
  unary, reduce, matmul, scalar-parameterized.
- **Item #8 — non-contiguous tensor guard.** Every op family forward
  (`BinaryOp::forward`, `UnaryOp::forward`, `ReduceOp::forward`,
  `MatmulBackward::forward`, `PowScalarBackward::forward`,
  `RPowScalarBackward::forward`, `ClipBackward::forward`) checks
  `is_contiguous()` on each tensor input and throws
  `NotImplementedError("op X: non-contiguous input not supported (call
  .contiguous() first)")` otherwise. Currently dormant (no path produces a
  non-contiguous TensorImpl yet); Phase 3.4 view ops will activate it.
- **Item #7 — empty tensor handling.** Verified via smoke test that all
  op families correctly handle `numel=0` inputs: element-wise ops are no-ops
  (kernels safe with N=0), reductions return identity (`sum=0`, `prod=1`),
  reduce-on-empty-axis produces correctly-shaped zero output. **`Matmul`
  required explicit fix**: `cblas_sgemm` rejects M/N/K=0; forward and
  backward now bypass BLAS for empty cases and zero-fill the output.
- **`Helpers::check_version_match` helper** — non-template function that
  does the `weak_ptr.lock() + version compare + throw` pattern, called from
  `FuncOp<>::validate_versions`. Keeps `FuncOp.h` light (no full
  `TensorImpl.h` include).

411 passed, 1 skipped after retrofit. Zero regressions.

### Added — Phase 3.3 (reduce ops — Sum/Mean/Prod/Max/Min on `ReduceOp<D>` CRTP)
- **`ReduceOp<Derived>` CRTP base** (`autograd/ops/reduce/ReduceOp.h`) with
  reduce-specific saved state (`reduce_axes_`, `keepdims_`, `full_input_shape_`).
  Forward signature `(tensor, axes, keepdims)`; multi-axis handled by
  sequential descending-order single-axis reduction.
- **Reductions** (`Reductions.{h,cpp}`):
  - `Sum` — backward = broadcast(g, input_shape).
  - `Mean` — sum + scalar divide; backward = broadcast(g)/N.
  - `Prod` — saves both input and output; backward = broadcast(g·out)/x.
  - `Max`, `Min` — saves output; backward routes g to argmax/argmin positions
    via equality mask (ties: every tied element gets g, matching PyTorch).
- **`backend/cpu/Reduce.{h,cpp}`** — single-axis kernels for sum (uses
  `vDSP_sve`/`vDSP_sveD` when inner=1), max/min/prod (templated scalar loops).
  Conventional `[outer, reduce_dim, inner]` layout.
- **Helpers extended** with `normalize_axes`, `reduce_output_shape`,
  `broadcast_back_for_reduce` — generic axis utilities reusable by
  Phase 3.4 (shape ops) too.
- Python bindings via `lucid._C.engine.{sum,mean,prod,max,min}` accept
  `axes: list[int]` (empty = reduce all) and `keepdims: bool`.
- OpRegistry now contains **40 ops** (8 binary + 22 unary + 5 reduce + 5 no-grad).

Deferred to a follow-up sprint: var/std, cumsum/cumprod, argmax/argmin,
trace, norm. (Distinct backward semantics; will share `ReduceOp` and
`backend/cpu/Reduce.cpp` infrastructure already in place.)

### Added — Phase 3.2 (unary op family — 22 ops on `UnaryOp<D>` CRTP)
- **`UnaryOp<Derived>` CRTP base** (`autograd/ops/unary/UnaryOp.h`) — same
  pattern as `BinaryOp` but with three policy flags: `kSavesInput` (default
  true), `kSavesOutput` (false), `kHasGradient` (true). When
  `kHasGradient=false`, forward skips graph wiring entirely (used by
  sign/round/floor/ceil/invert).
- **Arithmetic** (`Arith.{h,cpp}`): neg, abs, sign (no-grad), reciprocal,
  square, cube. Backward via `sign_storage`, `multiply_storages`,
  `mul_scalar_storage`, etc.
- **Exponential** (`Exponential.{h,cpp}`): exp (saves output), log, log2,
  sqrt (saves output). vForce-backed kernels (`vvexpf`/`vvlogf`/`vvlog2f`/
  `vvsqrtf`).
- **Trigonometric** (`Trig.{h,cpp}`): sin, cos, tan, arcsin, arccos, arctan
  (vForce: `vvsinf`/`vvcosf`/`vvtanf`/`vvasinf`/`vvacosf`/`vvatanf`).
- **Hyperbolic** (`Hyperbolic.{h,cpp}`): sinh, cosh, tanh (saves output —
  grad uses `1 - z²`).
- **Activation** (`Activation.{h,cpp}`): relu via `vDSP_vthres` at threshold
  zero. Backward via `positive_mask_storage`.
- **Scalar-parameterized** (`ScalarParam.{h,cpp}`): pow_scalar (x^c),
  rpow_scalar (c^x, saves output), clip (with min/max). Each has its own
  forward signature with extra scalar args; base's apply still works.
- **Discrete / no-grad** (`Discrete.{h,cpp}`): round (banker's via
  `vvnintf`/`vvnint`), floor (`vvfloorf`/`vvfloor`), ceil (`vvceilf`/
  `vvceil`), invert (bitwise NOT for I8/I16/I32/I64/Bool — scalar loop).
- **vForce wrappers extended** with: vasin/vacos/vatan, vsinh/vcosh, vlog2,
  vfabs, vrec, vfloor, vceil, vround, vtan (F32 + F64 each).
- **Storage-level helpers extended** with `mul_scalar_storage`,
  `exp_storage`, `sqrt_storage`, `abs_storage`, `sign_storage`,
  `reciprocal_storage`, `sin/cos/tan/asin/acos/atan_storage`,
  `sinh/cosh/tanh_storage`, `in_range_mask_storage`, `positive_mask_storage`.
- Python bindings expose all 22 via `lucid._C.engine.{neg,abs,sign,…}`.
- OpRegistry now contains **35 ops** (8 binary + 22 unary + 5 no-grad).

### Added — Phase 3.1 (binary op family)
- **6 element-wise binary ops** on `BinaryOp<Derived>` CRTP, each in its own
  file under `autograd/ops/binary/`:
  - `Sub` — `vDSP_vsub`; backward `(g, -g)`, no saved inputs.
  - `Mul` — `vDSP_vmul`; saves inputs; backward `(g*b, g*a)`.
  - `Div` — `vDSP_vdiv`; saves inputs; backward `(g/b, -g*a/b²)`.
  - `Pow` — `vvpowf`; saves inputs; backward `(b·a^(b-1)·g, log(a)·a^b·g)`.
    AMP policy `ForceFP32` (precision-sensitive).
  - `Maximum`, `Minimum` — `vDSP_vmax`/`vmin`; saves inputs; mask-based
    backward (`a >= b` and strict `<`, ties to a — matches PyTorch).
- **`Matmul`** in `autograd/ops/linalg/Matmul.{h,cpp}` — 2-D only, F32/F64,
  forward via `cblas_sgemm`/`dgemm` (Apple AMX coprocessor on M-series).
  Backward: `dA = dC @ B^T`, `dB = A^T @ dC`. Inherits `FuncOp<...,2>`
  directly (skips BinaryOp's equal-shape contract).
- **Backend extensions**:
  - `Vdsp.{h,cpp}` — `vmax`/`vmin` (F32, F64), comparison masks (`vge_mask`,
    `vle_mask`) for min/max backward.
  - `Vforce.{h,cpp}` — `vpow_f32` (vvpowf), `vpow_f64` (vvpow).
- **Storage-level math helpers** in `Helpers.{h,cpp}` — `negate_storage`,
  `multiply_storages`, `divide_storages`, `square_storage`, `clone_storage`,
  `log_storage`, `pow_storage`, `ge_mask_storage`, `lt_mask_storage`,
  `add_scalar_storage`. Used inside `grad_formula` of binary ops without
  creating intermediate TensorImpls.
- Python bindings expose `lucid._C.{add,sub,mul,div,pow,maximum,minimum,matmul}`.
- OpRegistry now contains 8 ops; all have schema_v1, AMP policy, deterministic flag.

### Changed — Repo restructure: C++ tree moved to `lucid/_C/`
- All C++ engine sources moved from `lucid/_backend/_C/engine/` to
  `lucid/_C/`. The `engine/backend/cpu/` redundancy ("backend" appearing
  twice in the path) is gone — paths are now ~3 levels shallower:
  - before: `lucid/_backend/_C/engine/backend/cpu/Blas.h`
  - after:  `lucid/_C/backend/cpu/Blas.h`
- Tokenizers moved alongside: `lucid/_C/tokenizers/`.
- Build target renames:
  - `lucid._backend._C.engine.engine` → `lucid._C.engine`
  - `lucid._backend._C.tokenizers.core` → `lucid._C.tokenizers.core`
- Python import callsites updated:
  - `lucid/data/tokenizers/bpe.py`, `wordpiece.py`
  - `lucid/models/text/bert/_tokenizer.py`
- `lucid/_backend/` is now Python-only (`core.py`, `metal.py`) — no more
  `_C/` subdirectory. Pure Python dispatch / Metal helpers live there.
- Internal `#include` paths within the C++ tree are unchanged (relative
  paths inside `_C/` are the same as they were inside `engine/`).

### Added — Phase 3.0 (CRTP infrastructure + Accelerate backend wrappers)
- `core/AmpPolicy.{h,cpp}` — `AmpPolicy` enum (`Promote`/`KeepInput`/`ForceFP32`),
  `amp::AutocastGuard` RAII, `amp::active_dtype()`, `amp::is_active()`. Each op
  declares its policy in `OpSchema`; Phase 3.5 NN ops will start consulting it.
- `core/OpSchema.h` — versioned op metadata struct (`name`, `version`,
  `amp_policy`, `deterministic`, `determinism_note`).
- `core/OpRegistry.{h,cpp}` — process-wide map populated by `LUCID_REGISTER_OP`
  static initializers; construct-on-first-use storage avoids static-init order
  fiasco. `schema_hash` (FNV-1a) for Phase 5.5 checkpoint validation.
- `core/Profiler.{h,cpp}` — thread-local active-profiler pointer + `OpScope`
  RAII (timing, memory delta, flops). Near-zero overhead when no profiler
  active (single thread-local pointer + null check).
- `backend/cpu/Vdsp.{h,cpp}` — Apple Accelerate vDSP wrappers for element-wise
  binary (vadd/vsub/vmul/vdiv F32+F64), unary (vneg/vabs/vsq), scalar-vector
  (vsadd/vsmul), relu (vthres at 0), plus integer scalar fallbacks.
- `backend/cpu/Vforce.{h,cpp}` — Apple Accelerate vForce transcendentals
  (vexp/vlog/vsqrt/vtanh/vsin/vcos, F32+F64).
- `backend/cpu/Blas.{h,cpp}` — `cblas_sgemm`/`dgemm`/`sgemv`/`dgemv` wrappers
  (row-major, transpose flags). Used by Phase 3.1 `matmul`.
- `autograd/FuncOp.h` — CRTP base template `FuncOp<Derived, N_IN>` carrying
  shared backward-node state (input shapes/out_shape/dtype/device, optionally
  saved input storages).
- `autograd/ops/binary/BinaryOp.h` — CRTP base for all binary ops; handles
  validation, allocation, profiler scope, autograd graph wiring,
  broadcast-undo on backward. Derived implements only `cpu_kernel` and
  `grad_formula`.
- `autograd/ops/binary/Add.{h,cpp}` — first production op rebuilt on
  `BinaryOp<AddBackward>`. Forward via `vDSP_vadd` (F32) / `vDSP_vaddD` (F64) /
  scalar fallback (I32/I64). Registered in OpRegistry with schema v1.
- pybind11 bindings: `bind_amp.cpp`, `bind_profiler.cpp`,
  `bind_op_registry.cpp`. Python module exposes `AmpPolicy`, `AutocastGuard`,
  `OpSchema`, `OpEvent`, `Profiler`, `op_lookup`, `op_registry_all`,
  `schema_hash`, `current_profiler`, `set_current_profiler`,
  `amp_active_dtype`, `amp_is_active`.

### Removed — Phase 3.0
- `lucid/_backend/_C/engine/autograd/ops/Add.{h,cpp}` (Phase 2 hand-written
  test op; replaced by the CRTP-based version at `ops/binary/Add.{h,cpp}`).
  Python `test_add` is kept as an alias of the new `add`.

### Added — Phase 0.6 (Engineering hygiene)
- `.clang-format` (Google base, project-tuned column / include / brace rules).
- `.clang-tidy` (modernize, performance, readability, bugprone, core-guidelines).
- `.editorconfig` (universal editor consistency).
- `lucid/_backend/_C/engine/api.h` — `LUCID_API` / `LUCID_INTERNAL` /
  `LUCID_NOCOPY` / `LUCID_NOMOVE` macros.
- `lucid/_backend/_C/engine/version.{h,cpp}` — semantic version + ABI version
  symbols.
- `lucid/_backend/_C/engine/core/fwd.h` — forward declarations for the public
  type set, used to keep header fan-in low.
- `docs/STYLE.md`, `docs/ARCHITECTURE.md`, `docs/CONTRIBUTING.md`.
- `scripts/build_compile_commands.sh` — generates `compile_commands.json` for
  clangd / clang-tidy / IDE tooling.
- This `CHANGELOG.md`.

### Added — Phase 2.5 (Determinism + concurrency)
- `lucid::Generator` (Philox-4x32-10 counter-based RNG); seedable, has a
  64-bit counter, embeds a `std::mutex` for the rare cross-thread case.
- `lucid::default_generator()` — process-shared singleton for
  `lucid.random.seed()`.
- `lucid::Determinism::set_enabled` / `is_enabled` (process-global atomic).
- `docs/concurrency.md` — engine threading contract.
- Python bindings: `Generator`, `default_generator`, `set_deterministic`,
  `is_deterministic`.

### Added — Phase 1.5 (Production foundations)
- Typed exception hierarchy: `LucidError` base + `OutOfMemory`,
  `ShapeMismatch`, `DtypeMismatch`, `DeviceMismatch`, `VersionMismatch`,
  `GpuNotAvailable`, `IndexError`, `NotImplementedError`. Pybind11 translator
  surfaces them as Python classes with the same names.
- `lucid::MemoryTracker` — per-device atomic counters
  (current / peak / alloc_count / free_count). `Allocator` updates them on
  alloc and via `shared_ptr` deleter on free.
- Python API: `lucid._backend._C.engine.memory_stats(device)`,
  `reset_peak_memory_stats(device)`.
- `LUCID_BUILD_MODE` env var: `release` (default) / `debug` / `debug-asan` /
  `debug-tsan` / `debug-ubsan`. `setup.py` applies sanitizer flags only to
  extensions opted-in via `"sanitizable": true`.
- `scripts/ci_sanitizer.sh` for CI gating.

### Added — Phase 2 (Autograd engine MVP)
- `lucid::Node`, `lucid::Edge` (`weak_ptr<Node>` to break cycles).
- `lucid::AccumulateGrad` — leaf-tensor sentinel that writes grad into
  `TensorImpl::grad_storage_`.
- `lucid::Engine::backward(root, retain_graph)` — DFS topological sort,
  reverse traversal, per-node grad accumulation.
- `lucid::Helpers` — `make_zero_storage`, `make_ones_storage`,
  `reduce_grad_to_shape`, `accumulate_into`.
- Test op `add_op` with `AddBackward` for end-to-end engine verification.

### Added — Phase 1 (TensorImpl + zero-copy bridge)
- `lucid::Dtype` (9 values: F16/F32/F64, I8/I16/I32/I64, Bool, C64).
- `lucid::Allocator` — 64-byte aligned `posix_memalign` with shared_ptr deleter.
- `lucid::Storage = std::variant<CpuStorage, GpuStorage>`.
- `lucid::TensorImpl` — fields per the plan, `data_as_python()` returns a
  zero-copy `numpy.ndarray` (CPU).
- `lucid::GradMode` + `NoGradGuard` (thread-local).
- Python bindings: `TensorImpl`, `Device`, `Dtype`, `NoGradGuard`,
  `grad_enabled`, `set_grad_enabled`.

### Removed — Phase 1
- `lucid/_backend/_C/runtime/` (legacy `_C_func_op_raw` Python orchestrator).
- `USE_CPP_FUNC_OP` flag from `lucid/__init__.py`.
- `_cpp_func_op` decorator from `lucid/_backend/core.py`.

### Changed — Phase 0
- `pyproject.toml`: removed orphan `[project.optional-dependencies]` block
  that newer setuptools rejects without a `[project]` table.
- `setup.py`: test extras moved to `extras_require={"test": [...]}`.
