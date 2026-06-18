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

### Fixed — GPU optimizer step leaked unbounded memory

`optim.step()` wrote each parameter back as an *unevaluated* MLX array, so on the
GPU (Metal) path every step composed its update on top of the prior step's still-
lazy graph — pinning all prior steps' compute → **unbounded active-memory / RSS
growth until OOM** (Adam / SGD / SGD-momentum / RMSprop, ~60-90 B/step, +31 MB
RSS per 400 steps). It only surfaced in loops that *don't* call `.item()` every
step (sparse logging); `mx.synchronize()` drains the queue but does not eval the
never-submitted lazy arrays. `Optimizer::step` now flushes the step's updated
GPU parameter arrays with a single batched `mlx::core::eval` — near-zero cost
(the next forward forces this compute anyway) and no host copy. Active memory is
now flat across 500 steps (regression test added). The CPU path mutates in place
and was unaffected.

### Changed — comparison / bitwise / floor-division now broadcast (NumPy-style)

`<`, `<=`, `>`, `>=`, `==`, `!=`, `&`, `|`, `^`, `<<`, `>>` and `//` previously
required identical operand shapes; they now broadcast like the arithmetic ops
and every reference framework — `x < 0.5`, `mask & 0xF0`, `x // 16` no longer
need a full-shape right operand. Internally this let scalar promotion produce a
0-dim constant instead of materialising the left operand's full shape, which
(a) skips a buffer allocation in eager and (b) is what lets **scalar arithmetic
(`x * 0.5`, `1.0 - x`, `(x - μ) / 2`) ride the symbolic-batch compile path** —
a full-shape scalar pinned the trace-time batch and aborted MPSGraph. With this
plus the view-op fix below, `where`, manual LayerNorm and most hand-written
arithmetic now compile under symbolic batch.

### Changed — `lucid.compile(..., dynamic=True)`: symbolic batch by default, never crashes

`dynamic=True` previously raised `NotImplementedError` (later: required an env
opt-in for the symbolic path). It now **attempts a single symbolic-batch
executable shared across all batch sizes by default**, guarded by a per-model
safety gate decided on the first trace:

- **gate clears the graph** → one symbolic executable, reused for every batch
  size (no recompile when the batch changes). Transformers, CNNs, MLPs and
  hand-written arithmetic (`x*0.5`, `where`, manual LayerNorm) all qualify
  (parity ≤ 7e-7 across batch sizes).
- **gate rejects it** — the graph bakes the batch into a constant MPSGraph can't
  infer (explicit broadcast / `expand` / `repeat`, a `concat`/`stack` on the
  batch axis, or a batch-shaped factory like `zeros_like(x)` / an RNN's zero
  hidden init) — or the symbolic lowering otherwise fails (off-dim-0 view) →
  robust **per-shape static caching**. Correct, never crashes — just recompiles
  per distinct shape.

So `dynamic=True` never crashes or silently mis-shapes a real model: it shares
one executable where provably safe and falls back to per-shape static otherwise.
`LUCID_COMPILE_DYNAMIC=0` forces pure static (no symbolic attempt). The compiled
training step (`make_step`) stays per-shape static.

The same applies to `make_step(..., dynamic=True)`, which is always per-shape
static (the backward graph of common reductions aborts under a symbolic batch).

### Fixed — symbolic-batch view ops (the keystone that unlocks transformers + CNNs)

Under the symbolic-batch path, the compile emitters for the **view ops** (`reshape`
/ `flatten` / `squeeze` / `contiguous` and the reduce-squeeze) copied a target
shape verbatim from the trace, pinning the batch to its concrete trace-time value
— which mismatched the symbolic-batch input and aborted MPSGraph's MLIR pass
(uncatchable). A shared `reshape_dynamic_aware` helper now marks the target's
leading dim `-1` when the input carries a symbolic batch, so MPSGraph infers it.
With this, the earlier "conv aborts" belief is corrected — **conv2d and SDPA ops
are themselves symbolic-batch-safe; the abort was always a surrounding view op**
— and real `TransformerEncoder`s (attention head-split + SDPA + merge) and CNNs
(conv + flatten) now share **one** symbolic-batch executable across batch sizes
(parity ≤ 7e-7 across BS {2,4,8}). The static-compile path is unchanged (the
helper is a no-op without a symbolic input). Remaining known limit: explicit
scalar broadcast (`x * 0.5`) still materialises a batch-shaped constant and
aborts under symbolic — fused-op models (BERT / GPT / ResNet / ViT) avoid it, so
the symbolic path stays an experimental opt-in.

### Added — Grouped-query / multi-query attention

`nn.MultiheadAttention` gained `num_kv_heads` (keyword, must divide `num_heads`;
`None` → standard MHA, fully backward-compatible). With fewer K/V heads the module
projects a smaller key/value space — `k_proj_weight` / `v_proj_weight` become
`(num_kv_heads * head_dim, …)`, shared across `num_heads // num_kv_heads` query
heads — and the **K/V cache stores the smaller set** (the real GQA win for
incremental decode). Each K/V head is repeated to match the queries just before
attention via the new `nn.functional.repeat_kv(hidden_states, n_rep)`. `1` = MQA.
The pattern used by Llama / Mistral / Qwen / Gemma. Parity-verified against the
reference SDPA's `enable_gqa=True` and a hand-rolled GQA (≤1e-5).  Two thin
convenience subclasses are also exported for discoverability:
`nn.GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, …)` (with
`num_kv_heads` required) and `nn.MultiQueryAttention(embed_dim, num_heads, …)`
(fixes `num_kv_heads=1`) — both delegate to `MultiheadAttention`.

### Fixed — device/dtype preserved on empty-input gradient norm + NaN quantile

`clip_grad_norm_` / `get_total_norm` returned their zero norm via `zeros(1)`
(global-default device/dtype) when no parameter carried a gradient, and
`nanquantile`'s all-NaN early returns (both the `dim=None` and per-`dim`
paths) built the NaN result without `device=input.device`. Both silently
produced a CPU / float32 tensor for GPU or mixed-precision inputs, causing
downstream device-mismatch errors. Each path now preserves the input's
device and dtype (falling back to the global default only when there are no
parameters at all). Surfaced by a stability audit.

### Fixed — error handling no longer masks the real failure

`DataLoader` now distinguishes a queue timeout from a worker crash / closed
queue — the latter shuts down and re-raises the *real* error instead of a
bogus "timed out". `linalg` `*_ex` (cholesky / inv / solve) catch only the
engine's `LucidError`, so OOM / unrelated errors propagate rather than being
reported as a singular matrix. `Module.__call__` runs the always-forward
hooks under their own guard so a failing cleanup hook can't mask the original
forward exception. Sharded serialization validates that metadata is
JSON-serializable before attaching it to the index.

### Fixed — exception-safe MPSGraph compile

The five `MpsBuilder::compile_*` paths allocated `CompiledExecutable` with a
raw `new` before running fallible work (set insertion); a throw mid-build
leaked it. They now use `make_unique` + `release()` (ownership unchanged).

### Tooling

Dev-infra hardening, no runtime impact: `ci.yml` now gates main code-pushes
(+ ccache, `pybind11` pinned for reproducible stub generation); an api-data
drift gate (source-hash `--check` + post-processor diff) plus a
full-pipeline pre-commit hook; a reST→markdown converter regression suite,
ESLint, and a Playwright smoke for the docs site (contract audit now 30
rules); compile-db deploy target aligned to 26.0; and dedicated padding /
upsampling / transformer unit tests. The docs `SubsectionHeading` recipe is
now a single component.

### Added — Reference-style KV cache for incremental generation

Autoregressive generation gained a full key/value cache subsystem under
`lucid.utils.cache`, mirroring the established reference-framework layout.
`DynamicCache` grows its per-layer K/V by concatenation and ships the usual
maintenance helpers (`crop` / `reset` / `batch_repeat_interleave` /
`batch_select_indices` / `reorder_cache`); `StaticCache` pre-allocates a fixed
`(B, H, max_cache_len, D)` buffer and writes each step in place at the next
position; and `EncoderDecoderCache` pairs a growing self-attention cache with a
fill-once, read-only cross-attention cache (tracked by an `is_updated` flag).
`generate(cache_implementation="dynamic"|"static", max_cache_len=...)` selects
the cache; sampling was factored into a shared `lucid/models/_sampling.py` reused
across every causal-LM head.

`StaticCache.update` returns a **filled-prefix view** of its buffer (narrowed to
the written width), so the decoder-only attention does q·kᵀ over only the real
keys — O(filled), not O(max_cache_len). This makes `StaticCache` the *faster*
path on compute-bound batches (in-place write + filled-prefix attention beat
`DynamicCache`'s growing concat — ~1.4× at GPT-2-base / B=16), tied at `B=1`, and
makes an over-sized `max_cache_len` free. Both caches turn the prefix re-encode
from O(T²) into O(T).

### Added — `scatter_set` engine primitive

Added a set-mode member to the axis-scatter family (`scatter_add` / `amax` /
`amin` / `prod` / **`set`**) across the CPU (Accelerate) and GPU (MLX
`put_along_axis`) backends, the autograd VJP, and a `MPSGraphScatterModeSet`
compile emitter. `index_copy` now lowers to a **single** `scatter_set` instead of
a `scatter_add`-of-delta, so it is one op on every stream — including inside a
compiled graph.

### Changed — `GenerationMixin` renamed to `CausalLMMixin`; vectorized sampling

The generation mixin was renamed `GenerationMixin` → `CausalLMMixin` to avoid
implying diffusion / other generative families; it is causal-LM specific. The
sampling helpers (repetition penalty, top-k, top-p, multinomial) were rewritten
as fully vectorized on-device tensor ops, removing the per-row `.item()` CPU
round-trips that previously dominated decode.

### Fixed — `TransformerDecoder` forwards key-padding masks

`TransformerDecoder.forward` silently dropped `tgt_key_padding_mask` /
`memory_key_padding_mask` instead of threading them into each layer, so padded
cross-attention attended to pad positions. Both masks are now passed through.

### Performance — `StaticCache` attends over the filled prefix

`StaticCache.update` narrows its returned K/V view to the filled prefix (via
`lucid.narrow`) so attention costs O(filled) not O(max_cache_len); the write and
the stored buffer stay full, so it is bit-identical to the masked-full-buffer
result. Recovers the up-to-~12× wasted q·kᵀ an over-sized buffer used to spend,
making eager `StaticCache` faster than `DynamicCache` on compute-bound batches.

### Removed — `compile_decode` generation path

The opt-in compiled single-token decode (`generate(compile_decode=True)`, the
`StaticCache` MPSGraph decode drivers, `supports_compiled_static_decode`) was
removed: measurement showed it was dominated in every regime — the full-buffer
`copy_`-back and launch-bound dispatch are the ceiling, and the eager
`StaticCache` filled-prefix path above is faster (1.4× at batch, tied at `B=1`)
without compiling. `DynamicCache` (default) and eager `StaticCache` cover all
generation; no public cache class changed.

---

## [3.5.0] — 2026-06-05

### Fixed — Inference no longer leaks the autograd graph (saved-output self-cycle)

Eager forward-only / inference loops *without* `no_grad` grew unbounded in memory
and slowed per-call until OOM (115 → 265 → 1459 → 4699 ms → SIGKILL after a
handful of `m(ids)` calls). The bogus "BERT 4.25× reference" headline was this
blow-up contaminating the timing median. Root cause:
`AutogradNode::saved_impl_output_` held this node's *own* output `TensorImpl` with
a strong `shared_ptr`, and the output already owns the node via `grad_fn` — a
`node ↔ output` self-cycle, the only forward-pointing strong reference in an
otherwise backward-pointing autograd DAG. With no `backward()` to clear `grad_fn`
(inference), every per-call graph was retained forever. Made
`saved_impl_output_` a `weak_ptr`; the create_graph reader re-fetches the live
grad_fn-bearing output via `lock()` (or reconstructs a data-only leaf from the
saved Storage). First-order backward and inference are untouched; tanh/sigmoid/exp
double-backward parity ≤ 7e-9. BERT-base inference: a 40-call eager loop is now
flat at 6.4 GB (was OOM by call ~6); inference under `no_grad` was always fine.

### Performance — Element-wise activations fused via `mx.compile`

GPU activations were multi-op MLX composites (`gelu` ≈ 9 primitives), and MLX
eager does not fuse element-wise chains, so each primitive was a separate Metal
kernel launch + a full DRAM round-trip. A matched primitive bench showed GEMMs /
fused SDPA / LayerNorm already at parity-or-better vs the reference framework, but
GELU was 2.7× slower (1.71 vs 0.63 ms on `(32,128,3072)`) — about 70 % of BERT's
per-layer gap. Added `mlx_unary_fused` / `mlx_binary_fused` helpers that route a
capture-less composite through `mlx::core::compile(fn, shapeless=true)`, tracing
it once into a single fused kernel reused across all shapes and dtypes, and routed
the genuine composites (`silu`, `gelu`, `gelu_exact`, `softplus`, `selu`, `mish` —
forward and backward) through them. `gelu_exact` 1.71 → 0.55 ms (faster than the
reference's 0.63); **BERT-base inference forward 106 → 92 ms = 1.3× → ~1.02× the
reference**. Bit-exact (metal-vs-CPU maxdiff ≤ 6e-7; F16/BF16 exact). Single-op
activations (relu/sigmoid/exp) are left unfused (no gain). Helps every model with
composite activations (ViT / ConvNeXt / EfficientNet …).

### Performance — On-device RNG (dropout mask + general random tensors)

Random tensors were built by a per-element CPU Philox loop then uploaded to the
device. Because `m.train()` enables dropout, this made BERT's *training-mode*
forward 10× its inference forward (956 vs 92 ms) — a single dropout mask on
`(32,128,768)` cost ~19 ms. (Isolation proved the autograd graph adds only ~3 ms
and the backward is a healthy ~1.3× the forward — the cost was entirely the
dropout RNG.) Added `IBackend::bernoulli_mask` and
`random_{uniform,normal,bernoulli,randint}` with GPU overrides that fill in one
kernel via `mlx::core::random`, keyed from the framework `Generator` (one draw per
fill) so results stay reproducible from the global seed; the CPU path is
unchanged. Single dropout 18.9 → 1.57 ms (12×); **BERT-base training step
(fwd+bwd) 1067 → 230 ms = 4.6×, faster than the reference's 284 ms**;
`bert_base()` weight init on a metal device 853 → 5 ms (170×). Exact mask/sample
values differ from the old CPU stream by design (Generator advances one draw per
fill, not `numel`); same-seed reproducibility holds.

### Performance — BERT self-attention routed through fused scaled-dot-product attention

`_BERTSelfAttention` computed attention by hand (`q kᵀ / scale` → (+ additive
mask) → softmax → dropout → `@ v`), materializing the `(B, H, T, T)` scores tensor
across several kernels. It now calls `F.scaled_dot_product_attention` (one fused
kernel, no scores materialization). Q/K/V stay three separate `Linear`s, so
pretrained checkpoints load unchanged, and the result is bit-exact with the manual
path (metal parity maxdiff 0.0 masked, 2e-7 unmasked). `attention.self` 1.29×
unmasked / 1.19× masked; **BERT-base eager forward 92 → 84 ms = ~0.94× the
reference** (compiled forward 81 ms = ~0.90×).

### Performance — ConvNeXt compile: depthwise conv-grad off the MPSGraph slow path

MPSGraph's grouped (depthwise) conv weight-gradient hits a ~14× slower path when
its incoming gradient is a `transposeTensor:` output (ConvNeXt's channels-last
permute VJP), and the generic grouped kernel is itself ~9× slower than the
dedicated depthwise op at 7×7. The cost was re-localized to the channels-last
**permute**, not LayerNorm (an earlier guess). `Conv2dVjp` now (a) launders the
gradient through a non-foldable `× 2.0` / `× 0.5` pair to dodge the transpose slow
path (bit-exact, rel 2.6e-7) and (b) routes the depthwise weight-gradient through
the dedicated `depthwiseConvolution2DWeightsGradient` kernel. **ConvNeXt-tiny
compile 2258 → 658 ms (8.65× → 2.52× reference), then to ~0.85× (a win) once the
dedicated kernel lands**; ResNet / MobileNet show no regression; 131 compile
tests pass.

### Fixed — Numerically stable `F.log_softmax` (and `cross_entropy`)

`F.log_softmax` computed the naive `log(softmax(x))` — even though its own
docstring promised the max-subtracting `x - log(sum(exp(x)))` form and the C++
engine already has a stable `log_softmax` kernel. At large logits the non-max
softmax probabilities round to 0, so the subsequent `log(0) = -inf` produced
`-inf`/`nan` log-probs, loss, and gradients (broke past `|logit| ≈ 90`). The
Python wrapper now routes through the engine's `log_softmax_op`
(`x - m - log(sum(exp(x - m)))`); `cross_entropy` (= `log_softmax` + NLL)
inherits the fix. Forward output is identical to the naive form at normal logits
(matches the numpy reference) but stays finite to `|logit| = 1000+`; verified
finite loss + gradients where it previously NaN'd. `F.softmax` was already stable
(routed straight to the engine kernel). Surfaced while training a from-scratch
ResNet-50 on CIFAR-10 whose logits drifted into the unstable range. New
regression tests in `test_nn_functional.py`.

### Added — Pretrained weights: DDPM (CIFAR-10 + LSUN-Church)

The first **generative** pretrained weights — the official ``google/ddpm-*``
diffusers checkpoints (Apache-2.0), converted into Lucid's ``DDPMUNet``.
``pretrained=True`` is an *inference-ready generator* (the ``_gen`` wrapper
samples images via ancestral DDPM); denoising-step parity vs the reference
``UNet2DModel`` is ≤ 6.6e-4 (CIFAR-10 4.8e-6):

- **ddpm_cifar** / **ddpm_cifar_gen** ← ``google/ddpm-cifar10-32`` (32×32).
  `DDPMCifarWeights.CIFAR10`.
- **ddpm_lsun** / **ddpm_lsun_gen** ← ``google/ddpm-church-256`` (256×256).
  `DDPMChurchWeights.LSUN_CHURCH`.

New converter ``tools/convert_weights/ddpm.py`` maps the diffusers
``UNet2DModel`` to Lucid: stage-grouped ``down_blocks[i].resnets[j]`` →
flat ``down_res[k]``; split ``to_q``/``to_k``/``to_v``/``to_out.0`` (Linear) →
fused ``qkv``/``proj`` (Conv2d 1×1).  ``ddpm_imagenet64`` keeps no weights (its
config is Improved-DDPM, not the original; no matching checkpoint).  NCSN / VAE
are not shipped — no canonical permissively-licensed checkpoint exists.

### Changed — DDPM model: canonical Ho-2020 conventions (parity fixes)

Loading the official checkpoints exposed three deviations from the canonical
DDPM U-Net (all now fixed; random-init behaviour is unchanged in shape):

- **Time embedding** uses the ``[sin, cos]`` ordering (Ho 2020 / diffusers
  ``flip_sin_to_cos=False``); Lucid previously emitted ``[cos, sin]``.
- **Downsample** uses asymmetric ``(0,1,0,1)`` padding + a ``padding=0``
  stride-2 conv (TF ``SAME`` on even inputs), not a symmetric ``padding=1``.
- **GroupNorm** ``eps`` is ``1e-6`` (was ``1e-5``).

### Changed — `nn.TimestepEmbedding` gains `flip_sin_to_cos`

New keyword-only ``flip_sin_to_cos: bool = True`` selects the raw sinusoid
ordering: ``True`` (default, unchanged) → ``[cos, sin]``; ``False`` →
``[sin, cos]`` (original DDPM).  Backward-compatible — existing callers are
unaffected.

### Added — Pretrained weights: BERT fine-tuned task heads (SQuAD + CoNLL NER)

Three *full* fine-tuned BERT checkpoints on canonical NLP benchmarks ship via
:mod:`lucid.weights`, so the QA / NER factories are now inference-ready with
``pretrained=True`` (parity vs the reference framework ≤ 2.3e-5 on logits):

- **bert_base_qa** ← ``csarron/bert-base-uncased-squad-v1`` (MIT, SQuAD v1.1;
  EM ≈ 80.9 / F1 ≈ 88.1).  `BERTBaseQAWeights.SQUAD_V1`.
- **bert_large_qa** *(new factory)* ← the official Google
  ``bert-large-uncased-whole-word-masking-finetuned-squad`` (Apache-2.0;
  EM ≈ 86.9 / F1 ≈ 93.2 — the strongest BERT SQuAD result).
  `BERTLargeQAWeights.SQUAD_V1`.
- **bert_base_token_cls** ← ``dslim/bert-base-NER`` (MIT, CoNLL-2003; F1 ≈ 91.3).
  This is a **cased** BERT-Base (vocab 28 996, 9-way BIO head), so
  ``pretrained=True`` builds the matching cased config — tokenize with the
  cased :class:`BERTTokenizer`.  `BERTBaseNERWeights.CONLL2003`.

Converter (``tools/convert_weights/bert.py``) gains ``qa`` / ``token_cls``
kinds: the head + encoder come from the task model, and the pooler (dropped by
HF's ``add_pooling_layer=False`` but present in the checkpoint file) is
recovered via ``AutoModel`` so Lucid's ``BERTModel`` pooler slot fills with the
checkpoint's own weights.  GLUE/SST-2 is intentionally not shipped — no clean
canonical + permissively-licensed BERT checkpoint exists.

### Changed — Text task-head factories: best-available canonical weights

QA (``bert_base_qa`` / ``bert_large_qa``) and NER (``bert_base_token_cls``)
``pretrained=`` now loads the *full fine-tuned* task model above (was: encoder
+ random head).  Sequence-classification (``*_cls``) keeps the encoder-into-
``.bert`` behaviour below (no canonical GLUE checkpoint).

### Changed — Text task-head factories load the pretrained encoder (head random)

The eight downstream task-head factories now honour `pretrained=` / `weights=`
by loading the matching pretrained **encoder** trunk into their encoder
submodule and leaving the task head randomly initialised — the standard
fine-tuning starting point, mirroring the reference
`AutoModelForX.from_pretrained(<encoder>)` behaviour.  No new checkpoints are
uploaded: each head reuses its family's existing encoder checkpoint.

- **bert** — `bert_base_cls`, `bert_large_cls`, `bert_base_token_cls`,
  `bert_base_qa` load `bert_base` / `bert_large` encoder weights into
  `model.bert` (`strict=True`; classifier / `qa_outputs` head stays random).
- **gpt** / **gpt2** — `gpt_cls`, `gpt2_small_cls` load the `gpt` /
  `gpt2_small` decoder trunk into `model.transformer`.
- **roformer** — `roformer_cls`, `roformer_token_cls` load the `roformer`
  encoder into `model.roformer`.
- Each factory gains a keyword-only `weights=<EncoderWeightsEnum>` argument and
  `pretrained: bool | str`; `pretrained=False` is unchanged (fully random).
  The full task (GLUE / SQuAD / NER) checkpoints are intentionally **not**
  shipped — they are dataset-specific, not architecture-canonical.  The
  `transformer`-family heads keep random init (no pretrained encoder exists).

### Added — Pretrained weights: RoFormer (Chinese, CLUECorpusSmall)

- **roformer** (Su et al. 2021) — `roformer` + `roformer_mlm` ←
  `junnyu/roformer_chinese_base` (Chinese WordPiece, 50 000 vocab).
  Forward parity vs the reference framework ≤6.4e-6.

### Changed — RoFormer interleaved RoPE + opt-in `apply_rotary_emb(interleaved=)`

RoFormer is the *original* rotary-position-embedding paper and uses the
**interleaved** pairing `(x_{2i}, x_{2i+1})`, whereas Lucid's shared RoPE
uses the half-split LLaMA pairing `(x_i, x_{i+d/2})`.  Loading the upstream
checkpoint into the half-split path produced garbage (last_hidden_state
diverged by 1.62).  Fix:

- `nn.functional.apply_rotary_emb` gains an opt-in keyword-only
  `interleaved: bool = False`; the default path is byte-identical to the
  prior half-split behaviour (`_rotate_half`, `cat([freqs, freqs])`) — a
  regression guard confirms `apply_rotary_emb(...)` ≡ `_rotate_half` at
  0.0 diff, and the negative control (half-split on RoFormer weights) gives
  1.82.  RoFormer uses a family-local interleaved cos/sin table
  (each frequency repeated twice), byte-identical to the upstream table.
- The converter drops HF's fixed sinusoidal RoPE buffer
  (`*embed_positions.weight`) + the tied duplicate `decoder.bias`, and
  injects the untrained pooler from init (it is absent upstream and affects
  only `pooler_output`).

### Added — Pretrained weights: GPT-1 + GPT-2 text families

Decoder-only pretrained weights, all loadable + runnable out of the box.
Conversion is an identity key map plus the **Conv1D→Linear weight
transpose** (HF stores the attention/MLP projection weights as `(in, out)`;
Lucid `nn.Linear` wants `(out, in)` — applied to every `c_attn` / `c_proj`
/ `c_fc` weight; `c_proj` is square so only forward parity catches a missing
transpose).  Forward parity vs the reference framework: GPT-1 1.3e-5,
GPT-2 1.4e-4.

- **gpt** (GPT-1, Radford et al. 2018) — `gpt` + `gpt_lm` ←
  `openai-community/openai-gpt` (BookCorpus, 40 478 vocab).
  - **Fix:** `GPTConfig.layer_norm_eps` corrected 1e-12 → **1e-5** (the
    reference GPT-1 / GPT-2 value); the inherited 1e-12 broke parity at
    5.3e-2 over the 12 post-LN blocks.
- **gpt2** (Radford et al. 2019) — `gpt2_{small,medium,large,xlarge}` +
  their `_lm` heads (8 factories) ← `gpt2` / `gpt2-medium` / `gpt2-large` /
  `gpt2-xl` (WebText, byte-level BPE 50 257 vocab; `lm_head` tied to `wte`).

### Changed — Acronym capitalization in the BERT family

Renamed every `Bert*` identifier to `BERT*` (class names, config, tokenizer,
weights enums, internal `_BERT*` helpers) to match the codebase's
acronym-caps convention (`GPT2Model`, `ViTBase16Weights`, `CvT13Weights`).
Lowercase forms are unchanged: factory names (`bert_base`), module paths,
`model_type="bert"`, and Hub slugs (`lucid-dl/bert-base`).

### Added — Pretrained weights: BERT text family (Wikipedia + BookCorpus)

First text-domain pretrained weights.  `pretrained=True` downloads the
upstream checkpoint from the Hub, verifies its SHA-256, and strict-loads
it — `bert_base(pretrained=True)` runs an encoder forward pass out of the
box.  Conversion is a pure identity parameter map (Lucid mirrors the HF
`BertModel` naming one-for-one), validated by forward parity vs the
reference framework (`last_hidden_state` 8.0e-6 base; masked-LM logits
4.1e-5 base, top-prediction exact).

- **bert** — eight checkpoints on the Hub under `lucid-dl/bert-*`:
  - Base encoders `bert_{tiny,mini,small,medium}` ← Turc et al. 2019
    "Well-Read Students" pre-distilled sizes
    (`google/bert_uncased_L-*_H-*_A-*`); `bert_base` / `bert_large` ←
    Devlin et al. 2018 `google-bert/bert-{base,large}-uncased`.
  - Masked-LM heads `bert_base_mlm` / `bert_large_mlm` ← the same
    checkpoints' `cls.predictions` head (sourced from
    `BertForPreTraining` so the pooler weights are real; the NSP head and
    HF's duplicate tied `decoder.bias` are dropped).
  - All `WIKIPEDIA_BOOKSCORPUS` tag, uncased, 30 522-token vocab; tokenize
    with `lucid.models.text.bert.BertTokenizer`.
- **Infrastructure** — text models consume token ids, so the conversion
  card renderer (`tools/convert_weights/_templates.py`) gained text task
  → HF `pipeline_tag` mappings (`base`→feature-extraction,
  `masked-lm`→fill-mask, `causal-lm`→text-generation, …) and a token-id
  usage snippet.  Weight entries carry a no-op preprocessing transform
  (tokenization is a `lucid.utils.tokenizer` concern, not a tensor op).

### Added — Pretrained weights: Mask2Former Swin (ADE20k semantic segmentation)

The most complex segmenter — deformable pixel decoder + masked-attention
transformer decoder. `pretrained=True` loads the official ADE20k
checkpoint and infers out of the box.

- **mask2former** — `mask2former_swin_{tiny,small,base,large}` ← HF
  `facebook/mask2former-swin-{tiny,small,base,large}-ade-semantic`
  (Cheng et al., 2022; ADE20k, 150 classes; 47.7 / 51.3 / 53.9 / 56.1
  mIoU; 47.4–216M params).  Full rebuild to HF's tree (semantic parity
  ≤8.4e-6): HF-style **Swin** backbone, **MSDeformAttn pixel decoder**
  (6-layer deformable encoder on the new
  `multi_scale_deformable_attention` op), **9-layer masked-attention
  transformer decoder** (cross-attn over 3 cycling feature levels,
  attn-mask from the previous layer's `sigmoid(mask) < 0.5`) + learnable
  queries/level embeddings, class + mask MLP heads.  base/large use the
  384-pretrained Swin (window 12; tiny/small window 7) — and the
  Mask2Former `SwinBackbone` runs with `always_partition=True`, so
  shifted-window attention is **kept** even when the feature map equals
  the window (the opposite of the classification Swin).  Enums on
  `lucid.models.weights` (tag `ADE20K`, `Segmentation` transforms).
  (The simplified ResNet/FPN Mask2Former stub was replaced; HF ships
  only Swin ADE-semantic checkpoints.)

- **detection ops** — `multi_scale_deformable_attention` added to
  `_detection` (a `grid_sample` composite; reference parity 9.5e-7),
  reusable for any deformable-attention model.

### Added — Pretrained weights: MaskFormer (ADE20k semantic segmentation)

Mask-classification transformer segmenter — extends the segmentation
sweep beyond per-pixel FCN.

- **maskformer** — `maskformer_resnet50` / `maskformer_resnet101` ← HF
  `facebook/maskformer-resnet{50,101}-ade` (Cheng et al., 2021; ADE20k,
  150 classes; 44.5 / 45.5 mIoU).  Full `_model.py` rebuild to HF's
  module tree (parity 1.7e-5 on the semantic logits): HF-style ResNet
  encoder + FPN pixel decoder (conv+GroupNorm, `mask_projection`),
  DETR-style 6-layer transformer decoder with per-layer query/spatial
  positional injection (cross-attention over the projected C5 feature),
  a 3-layer mask MLP (`mask_embedder`), and reference-exact sine
  position embedding.  Semantic output drops the no-object slot
  (`softmax(class)[…,:-1] ⊗ sigmoid(mask)`, einsum at FPN resolution
  then bilinear upsample) → `(B, 150, H, W)`.  Enums
  `MaskFormerResNet50Weights` / `MaskFormerResNet101Weights` on
  `lucid.models.weights` (tag `ADE20K`, `Segmentation` transforms).

### Added — Pretrained weights: FCN (semantic-segmentation pattern)

First semantic-segmentation model with pretrained weights — extends the
detection sweep to dense prediction.

- **fcn** — `fcn_resnet50` / `fcn_resnet101` ← torchvision
  `FCN_ResNet{50,101}_Weights.COCO_WITH_VOC_LABELS_V1` (Long et al.,
  2015; 21 classes = 20 Pascal-VOC + background; 60.5 / 63.7 mIoU).
  Near-identity converter (the 334-key state dict matches torchvision
  exactly bar the stem naming — `backbone.conv1`/`bn1` →
  `backbone.stem.0`/`stem.1`).  One reference-faithfulness fix to
  `_make_layer`: a dilated stage's **first block keeps the previous
  stage's dilation** (layer3 → 1 then 2, layer4 → 2 then 4) instead of
  applying the new dilation uniformly — this closed a 2.1 → 1.0e-5
  per-pixel logit gap.  Enums `FCNResNet50Weights` / `FCNResNet101Weights`
  on `lucid.models.weights` (tag `COCO_WITH_VOC_LABELS_V1`, `Segmentation`
  transforms preset).

### Added — Pretrained weights: Mask R-CNN ResNet-50-FPN (COCO instance segmentation)

Instance segmentation — `pretrained=True` loads the official COCO
checkpoint and returns detections + per-instance masks out of the box.

- **mask_rcnn** — `mask_rcnn_resnet50_fpn` ← torchvision
  `MaskRCNN_ResNet50_FPN_Weights.COCO_V1` (He et al.; 91 classes, box AP
  37.9 / mask AP 34.6, 44.4M params).  Built on the shared Faster R-CNN
  ResNet-50-FPN infra (backbone / FPN / RPN / box head byte-identical →
  295 shared keys) plus the mask branch (`roi_heads.mask_head` 4×Conv3×3
  + `roi_heads.mask_predictor` ConvTranspose→Conv1×1, 12 keys, 307 total,
  strict-load).  Mask RoI-align at 14×14 → 28×28 per-instance class
  masks.  Mask-branch parity 1.4e-6.  Enum `MaskRCNNResNet50FPNWeights`
  on `lucid.models.weights` (tag `COCO_V1`, `Detection` transforms).

### Added — Pretrained weights: Faster R-CNN ResNet-50-FPN (COCO, two-stage)

First two-stage detector with pretrained weights — `pretrained=True`
loads the official COCO checkpoint and runs inference out of the box.

- **faster_rcnn** — `faster_rcnn_resnet50_fpn` ← torchvision
  `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` (Ren et al.; 91 classes, box
  AP 37.0, 41.8M params).  Full architecture rebuild from the legacy
  VGG16 single-scale design to the reference ResNet-50-FPN two-stage
  pipeline: `_FrozenBatchNorm2d` (eps 0) ResNet backbone +
  `_FeaturePyramidNetwork` (inner/layer blocks + LastLevelMaxPool) + RPN
  (3×3 conv + cls/bbox heads over 5-level anchors) + RoIHeads
  (`multiscale_roi_align` with the canonical FPN level assignment →
  TwoMLPHead → FastRCNNPredictor) + per-class decode / NMS post-process.
  Verified by a staged parity gate (backbone+FPN 5.7e-6, RPN 1.6e-5, RoI
  head 2.3e-5, final detections 1.8e-4).  Enum
  `FasterRCNNResNet50FPNWeights` on `lucid.models.weights` (tag `COCO_V1`,
  `Detection` transforms).

- **detection ops** — `roi_align` rewritten with a vectorized 4-corner
  bilinear gather reproducing the reference RoIAlign exactly across all
  `sampling_ratio` / `aligned` settings (the reference's
  `MultiScaleRoIAlign` uses `aligned=False`); new shared `_detection`
  modules (`_FrozenBatchNorm2d`, `_ResNetBody`, `_FeaturePyramidNetwork`,
  `_ReferenceAnchorGenerator`, `multiscale_roi_align`) reused across the
  R-CNN family.

### Added — Pretrained weights: DETR (COCO detection sweep — pilot)

First object-detection model with full COCO-pretrained weights, opening
the detection/segmentation weight sweep (classification sweep complete).
`pretrained=True` now yields an inference-ready detector reproducing the
paper's box AP, not just an ImageNet backbone.

- **detr** — `detr_resnet50` / `detr_resnet101` ← Facebook DETR
  (`facebookresearch/detr`, torch hub; Apache-2.0; COCO 2017, 91 classes,
  box AP 42.0 / 43.5).  Reference-faithful rebuild of `_model.py` so the
  official checkpoint loads strict + reproduces inference (parity:
  logits 2.1e-5, boxes 8.3e-6):
  - custom DETR transformer replacing `nn.Transformer` — reference-exact
    submodule names, **no** final encoder norm, a final decoder norm, and
    per-layer positional re-injection (encoder Q/K; decoder self Q/K =
    `tgt + query_pos`, cross Q = `tgt + query_pos` / K = `memory + pos`);
  - `_PositionEmbeddingSine` (normalize, scale 2π, temperature 1e4)
    matching the reference to 8e-7;
  - `_FrozenBatchNorm2d` backbone (eval-only, no `num_batches_tracked`);
  - COCO config `num_classes=91`.
  - Enums `DETRResNet50Weights` / `DETRResNet101Weights` on
    `lucid.models.weights` (tag `COCO_2017`, `Detection` transforms preset).

The detection conversion infrastructure (task-aware model cards, the
`Detection` preset wiring, `ObjectDetectionOutput`) is reused unchanged.


### Added — Pretrained weights: GoogLeNet + EfficientFormer (reference-faithful rebuilds)

Two families needed a topology rebuild to match their upstream
checkpoints; both now ship pretrained weights (enums on
`lucid.models.weights`, parity ~6e-6):

- **googlenet** — `googlenet_cls` ← torchvision
  `GoogLeNet_Weights.IMAGENET1K_V1` (Szegedy et al., 2014).  Rebuilt to
  the batch-normalised reference: every conv is now a `Conv→BN(eps
  1e-3)→ReLU` block, the "5×5" branch uses the reference's 3×3 conv,
  the stem/inter-stage pools use ceil-mode sizing (a stateless
  `_CeilMaxPool2d` shim emulates `ceil_mode`, which the engine pool
  ignores), `maxpool4` is kernel-2, and the classifier owns the two
  auxiliary heads so the checkpoint key-set matches.
- **efficientformer** — `efficientformer_l{1,3,7}_cls` ← timm
  `efficientformer_l*.snap_dist_in1k` (Li et al., 2022; SNAP_DIST_IN1K).
  Full rebuild to timm's module tree: `_Stem`, per-stage
  `downsample`+`blocks`, 4D `_MetaBlock2d` (pooling token-mixer +
  conv-MLP) with a parameter-free `_Flat` marker at the 4D→3D
  transition, 3D `_MetaBlock1d` (LeViT-style attention with a learned
  `attention_biases` table), and a distilled dual head averaged at
  inference.

### Fixed — Reference-parity model corrections (ViT / MaxViT / MobileNetV1 / ResNeSt / Swin)

Five families carried subtle forward-path divergences from their
reference implementations that blocked pretrained-weight parity.  Each
fix is a faithfulness correction (verified to bring the smallest variant
to ~1e-5 logit parity), not a weight-fitting hack:

- **vit** — `ViTConfig.layer_norm_eps` added (default `1e-6`, the
  original ViT value) and threaded through both per-block norms and the
  final pre-head norm; Lucid previously used the generic `1e-5`.
  (`vit_base_32` parity 0.017 → 1.9e-5.)
- **maxvit** — three corrections in `_model.py`: BatchNorm `eps=1e-3`
  (was `1e-5`), GELU `approximate="tanh"`, and the relative-position
  bias offset sign (`cj - ci`, key-minus-query).  (`maxvit_tiny` 5.43 →
  3.8e-6.)
- **mobilenet** (v1) — stem + depthwise/pointwise activations switched
  from `ReLU` to `ReLU6` (all public checkpoints train with ReLU6).
  (`mobilenet_v1` 28.0 → 1.0e-5.)
- **resnest** — stage 1 no longer force-enables AVD pooling; AVD now
  fires only on downsampling blocks (`stride > 1`), matching the
  reference.  (`resnest_50` 4.39 → 1.8e-6.)
- **swin** — `_SwinBlock` now disables the cyclic shift and its
  attention mask when the window covers the whole feature map
  (`window_size >= min(H, W)`, e.g. the 7×7 stage 3), matching the
  reference.  (`swin_tiny` 0.53 → 3.3e-6.)

### Added — Pretrained weights: ViT / MaxViT / MobileNetV1 / ResNeSt / Swin

Enabled by the corrections above; all enums on `lucid.models.weights`:

- **vit** — `vit_{base_16,base_32,large_16,large_32}_cls` ← torchvision
  `ViT_*_Weights.IMAGENET1K_V1` (Dosovitskiy et al., 2020).
- **maxvit** — `maxvit_{tiny,small,base,large}_cls` ← timm
  `maxvit_*_tf_224.in1k` (Tu et al., 2022; IN1K).
- **mobilenet** — `mobilenet_v1_cls` ← timm
  `mobilenetv1_100.ra4_e3600_r224_in1k` (Howard et al., 2017).
- **resnest** — `resnest_{50,101,200,269}_cls` ← timm
  `resnest{50d,101e,200e,269e}.in1k` (Zhang et al., 2020; per-variant
  eval resolution 224/256/320/416).
- **swin** — `swin_{tiny,small,base}_cls` ← torchvision
  `Swin_*_Weights.IMAGENET1K_V1`; `swin_large_cls` ← timm
  `swin_large_patch4_window7_224.ms_in22k_ft_in1k` (Liu et al., 2021).

### Added — Pretrained weights: PVTv2 (all 6 variants)

- **pvt** — `pvt_v2_b{0,1,2,3,4,5}_cls` ← timm `pvt_v2_b*.in1k`
  (Wang et al., 2022; IN1K).  Clean identity-rename map; b0/b1 parity
  ~7e-4; E2E verified b0/b1/b5.  Presets 224 crop / 249 resize /
  bicubic.  Enums on `lucid.models.weights`.  `pvt_v2_b1` required a
  config correction (`_CFG_B1` depths `(2,2,4,2)` → `(2,2,2,2)`,
  17.3M → 14.0M params) to match the paper / timm before its
  checkpoint could load.

### Added — Pretrained weights: InceptionNeXt

- **inception_next** — `inception_next_{tiny,small,base}_cls` ← timm
  `inception_next_*.sail_in1k` (Yu et al., 2023; SAIL_IN1K).  Parity
  1.2e-5 (tiny).  Enums on `lucid.models.weights`.

### Added — Pretrained weights: EfficientNet (B0–B7)

- **efficientnet** — `efficientnet_b{0..7}_cls` ← torchvision
  `EfficientNet_B{0..7}_Weights.IMAGENET1K_V1` (Tan & Le, 2019).  Pure
  key-rename (torchvision's nested `features[1..7]` MBConv stages
  flattened onto Lucid's `features[3..N]`; the head's flat index is
  derived from the per-variant block count).  Per-variant presets
  (B0 224/256 … B7 600/600, all bicubic).  b0 parity 2.6e-6; E2E
  verified b0 + b4.  Enums on `lucid.models.weights`.

### Added — Pretrained weights: VGG (full set, 8 variants)

- **vgg** — `vgg_{11,13,16,19}_cls` + `vgg_{11,13,16,19}_bn_cls` ←
  torchvision `VGG*_Weights.IMAGENET1K_V1` (Simonyan & Zisserman, 2014).
  Rename key map; E2E verified (incl. 528 MB vgg_16_bn).  acc@1 from
  torchvision meta (69.0–74.2).  Enums on `lucid.models.weights`.

### Added — Pretrained weights: SE-ResNet

- **senet** — `se_resnet_{18,34,101,152}_cls` ← timm `legacy_seresnet*.in1k`
  (IN1K); `se_resnet_50_cls` ← timm `seresnet50.ra2_in1k` (RA2_IN1K).
  Clean topology match; enums on `lucid.models.weights`.

### Added — Pretrained weights: ResNet (full set) + ResNeXt

- **resnet** — ImageNet-1k weights for the 4 remaining canonical ResNets
  + both Wide ResNets (resnet_18 already shipped): `resnet_34_cls`,
  `resnet_50_cls`, `resnet_101_cls`, `resnet_152_cls`,
  `wide_resnet_50_cls`, `wide_resnet_101_cls` ← torchvision
  `*_Weights.IMAGENET1K_V1`.  Identity-ish key map (stem/head rename);
  E2E verified incl. wide_resnet_101 (127M).
- **resnext** — `resnext_50_32x4d_cls`, `resnext_101_32x8d_cls` ←
  torchvision `*_Weights.IMAGENET1K_V2`; `resnext_101_32x4d_cls` ← timm
  `resnext101_32x4d.gluon_in1k` (GLUON_IN1K).  Enums on
  `lucid.models.weights`.

### Added — Pretrained weights (multi-agent batch: 6 families)

- Pretrained ImageNet-1k weights for 8 variants across 6 vision families,
  recipe-authored + parity-verified by a multi-agent workflow and uploaded
  to the `lucid-dl` HF org.  Each loads via `<factory>(pretrained=True)` /
  `weights=<Enum>` (enums re-exported from `lucid.models.weights`):
  - **inception** `inception_v3_cls` ← torchvision `Inception_V3_Weights.IMAGENET1K_V1`
  - **inception_resnet** `inception_resnet_v2_cls` ← timm `inception_resnet_v2.tf_in1k` (TF_IN1K)
  - **xception** `xception_cls` ← timm `legacy_xception.tf_in1k` (TF_IN1K; 299 crop / bicubic / 0.5 mean-std)
  - **mobilenet_v2** `mobilenet_v2_cls` ← torchvision `MobileNet_V2_Weights.IMAGENET1K_V1`
  - **mobilenet_v3** `mobilenet_v3_large_cls` / `mobilenet_v3_small_cls` ← torchvision `MobileNet_V3_*_Weights.IMAGENET1K_V1`
  - **sknet** `sk_resnet_18_cls` / `sk_resnet_34_cls` ← timm `skresnet{18,34}.ra_in1k` (RA_IN1K)
  
  All verified end-to-end (download → SHA → load → forward); the workflow's
  parity checks landed in the 1e-6 range vs the source models.  Minor
  per-family model fixes were applied to reach exact parity (eps/bias/head
  details).

**Strong-augment suite — RandomErasing + AutoAugment family + Mixup/CutMix + RA-Sampler.**
Closes the torchvision `ClassificationPresetTrain` parity gap in one phased PR:
8 new public classes, 5 new functional ops, integrated into the G0
`TransformsPreset` framework, with reference-framework numerical parity
verified across 180 parity tests.

### Added — DenseNet pretrained weights + DenseNet-161

- **`lucid.models.vision.densenet`** — pretrained ImageNet-1k weights for
  all four canonical DenseNet variants (Huang et al., CVPR 2017),
  converted from torchvision's ``DenseNet*_Weights.IMAGENET1K_V1`` and
  hosted on ``lucid-dl/densenet-{121,161,169,201}``:
  - `densenet_121_cls`  8.0M   74.43% top-1
  - `densenet_161_cls`  28.7M  77.14%  ← **new variant** (k=48, 96-ch stem)
  - `densenet_169_cls`  14.1M  75.60%
  - `densenet_201_cls`  20.0M  76.90%
  
  ``densenet_161`` / ``densenet_161_cls`` are newly added — the wide
  k=48 / 96-channel-stem variant from paper Table 1 that Lucid did not
  previously register (the existing line was 121/169/201/264).
  ``densenet_264`` keeps random-init only (no public checkpoint).
  
  Lucid's DenseNet mirrors torchvision's module layout exactly, so the
  converter is a pure **identity key map** — no rewrites.  Numerical
  parity vs torchvision: max abs logit diff ``3-7e-6``.  Load via:
  ```python
  from lucid.models import densenet_161_cls
  from lucid.models.weights import DenseNet161Weights
  m = densenet_161_cls(pretrained=True)
  m = densenet_161_cls(weights=DenseNet161Weights.IMAGENET1K_V1)
  ```

### Changed — AlexNet single-stream re-derivation

- **`lucid.models.vision.alexnet`** — backbone re-implemented as the
  Krizhevsky 2014 single-stream "One Weird Trick" variant
  (arXiv:1404.5997) with channel widths `(64, 192, 384, 256, 256)` and
  no `LocalResponseNorm`, replacing the previous NIPS 2012 paper-faithful
  channel widths `(96, 256, 384, 384, 256)` with LRN.  This is the
  topology every published reference ImageNet checkpoint targets, so
  pretrained weights now load directly.  R-CNN's internal AlexNet-style
  backbone (`lucid.models.vision.rcnn._AlexNetBackbone`) is kept on the
  original 96-channel topology — R-CNN is faithful to Girshick CVPR 2014
  and is unaffected by this change.
  - `alexnet()` param count: 3.7M → 2.5M (conv trunk only)
  - `alexnet_cls()` param count: 62.4M → 61.1M
  - `feature_info` channels: `[96, 256, 384, 384, 256]` →
    `[64, 192, 384, 256, 256]`

### Added — Pretrained weights

- **`lucid.models.vision.alexnet`** — pretrained ImageNet-1k weights for
  `alexnet_cls`.  Hosted on
  [`lucid-dl/alexnet`](https://huggingface.co/lucid-dl/alexnet).
  Converted from `torchvision.AlexNet_Weights.IMAGENET1K_V1`
  (acc@1 = 56.522 / acc@5 = 79.066).  Numerical parity vs reference:
  max abs logit diff `9.5e-7` on a random 224×224 input.  Load via:
  ```python
  from lucid.models.vision.alexnet import alexnet_cls, AlexNetWeights
  m = alexnet_cls(pretrained=True)                       # DEFAULT tag
  m = alexnet_cls(pretrained="IMAGENET1K_V1")            # by tag name
  m = alexnet_cls(weights=AlexNetWeights.IMAGENET1K_V1)  # by enum
  ```

### Added — CoAtNet paper-faithful variants (no pretrained weights yet)

- **`lucid.models.vision.coatnet`** — full Table 5 lineup from Dai et al.,
  NeurIPS 2021 (arXiv:2106.04803).  Previously only `coatnet_0` was
  registered; this PR fills in the 5 remaining paper-cited variants
  (CoAtNet-1 through CoAtNet-5).  Existing 4-stage builder accepts the
  new `(stem_width, dims, blocks_per_stage, attn_heads)` tuples
  unchanged — no architectural surgery required.  No pretrained
  weights ship: the only public CoAtNet checkpoints (timm's `_rw_*`
  series) are a Ross Wightman re-implementation with a different
  topology (different MBConv expand ratio + `se_early` block + head),
  so they are not paper-cited and therefore skipped per H11.  New
  factories:
  - `coatnet_1` / `coatnet_1_cls` — 42M params, 83.3% top-1
  - `coatnet_2` / `coatnet_2_cls` — 75M params, 84.1% top-1
  - `coatnet_3` / `coatnet_3_cls` — 168M params, 84.5% top-1
  - `coatnet_4` / `coatnet_4_cls` — 275M params, 85.0% top-1
  - `coatnet_5` / `coatnet_5_cls` — 688M params, 85.8% top-1
  
  CoAtNet-6 and CoAtNet-7 land in the immediately-following commit
  (mixed-S3 stage support via a new `mixed_s3` config field).

- **`lucid.models.vision.coatnet`** — CoAtNet-6 and CoAtNet-7 from
  paper §A.2 / Table 12.  Adds a new `mixed_s3: tuple[int, int, int] | None`
  field to `CoAtNetConfig` and a corresponding branch in `_build_body`
  so stage 3 can host an MBConv sub-stage followed by a 1×1 channel
  expansion and a transformer sub-stage at the wider width (paper:
  *"we move 2/3 of the MBConv blocks of S2 into S3 and double its
  hidden dimension"*).  ~1.5B / ~2.4B params respectively — only
  meaningful with very-large-scale pretraining; the paper's headline
  88.4-89.0% ImageNet numbers come from JFT-3B + 512×512 finetune.
  Not buildable on a 16 GB Mac host; unit tests therefore exercise
  the mixed-S3 builder via a tiny proportional config that fits in
  test memory.

- **`lucid.models.vision.convnext`** — pretrained ImageNet-1k weights
  for `convnext_{tiny,small,base,large}_cls` (Liu et al., 2022, Table 9).
  Hosted on
  [`lucid-dl/convnext-{tiny,small,base,large}`](https://huggingface.co/lucid-dl).
  Converted from torchvision's `ConvNeXt_*_Weights.IMAGENET1K_V1` tag
  (which redistributes Facebook AI Research's official checkpoints).
  Numerical parity vs torchvision: max abs logit diff `4-5e-6` across
  all three sizes verified (large skipped on 16 GB host).  Acc@1:
  82.520 / 83.616 / 84.062 / 84.414.
  - `ConvNeXtConfig` gains a `layer_norm_eps: float = 1e-6` field; every
    LayerNorm in the trunk (stem / block / downsample / head) now uses
    this eps.  Was the root cause of a `5.6e-2` initial parity gap —
    Lucid's default LayerNorm eps `1e-5` diverges numerically from the
    reference ConvNeXt recipe (paper / FAIR official / torchvision all
    use `1e-6`).
  - New `Architecture.transform_value` optional hook in
    `tools/convert_weights/_base.py` to massage individual tensors
    before they land in Lucid; ConvNeXt's `layer_scale` ships as
    `(C, 1, 1)` for NCHW broadcasting and the converter squeezes it
    down to `(C,)` for Lucid's explicit elementwise multiply.
  - Load via:
    ```python
    from lucid.models import convnext_tiny_cls
    from lucid.models.weights import ConvNeXtTinyWeights
    m = convnext_tiny_cls(pretrained=True)
    m = convnext_tiny_cls(weights=ConvNeXtTinyWeights.IMAGENET1K_V1)
    ```
  - `convnext_xlarge_cls` weights (350M params, ImageNet-22k → 1k) land
    in a follow-up commit sourced from timm — torchvision does not
    publish a 1k-class xlarge head.

- **`lucid.models.vision.cvt`** — pretrained weights for all three
  paper-cited variants (Wu et al., ICCV 2021), sourced from Microsoft's
  HuggingFace ``transformers`` CvT checkpoints.  Required three model
  fixes to reach exact parity with the reference:
  - **Stem padding** — the 7×7 stride-4 stem uses ``padding=2`` (not
    ``kernel//2 = 3``); the wrong padding shifted the entire stage-0
    feature map.
  - **Attention scale** — CvT scales attention logits by the *full*
    embedding dim (``dim ** -0.5``), not the per-head dim.  Invisible
    on stage 0 (single head) but a large divergence on stages 1-2.
  - **CLS token** — the last stage carries a learnable CLS token that
    participates in attention (split off before the conv projection,
    re-attached before the linear projection) and is what the
    classifier reads (``layernorm(cls).mean(1)``).  Added a
    ``cls_token`` per-stage flag to ``CvTConfig`` + the full split /
    re-attach path through ``_ConvProj`` / ``_CvTAttention`` /
    ``_CvTStage``.  Also dropped the spurious per-stage LayerNorm and
    biased the conv-projection's linear layer to match the reference.
  
  Hosted on [`lucid-dl/cvt-{13,21,w24}`](https://huggingface.co/lucid-dl).
  Verified parity vs ``transformers``: cvt_13 backbone CLS-token max abs
  diff ``9.2e-5``, full-logit head max abs diff ``4.5e-6`` with top-5
  predictions matching exactly.  (The reference torch CvT forward
  SIGBUS-crashes in this M1 / py3.14 env — a torch CPU-BLAS GEMM bug
  unrelated to Lucid — so parity was measured on the backbone CLS token
  at a reduced 64×64 input plus a numpy-computed head reference.)
  cvt_13/cvt_21 are ImageNet-1k @ 224; cvt_w24 is ImageNet-22k →
  ImageNet-1k @ 384 (277M params).  Load via:
  ```python
  from lucid.models import cvt_13_cls
  from lucid.models.weights import CvT13Weights
  m = cvt_13_cls(pretrained=True)
  m = cvt_13_cls(weights=CvT13Weights.IN1K)
  ```

- **`lucid.models.vision.cspnet`** — paper-faithful rebuild + pretrained
  ImageNet-1k weights for the three paper-cited variants (Wang et al.,
  CVPRW 2020).  Pre-3.5 CSPNet shipped only ``cspresnet_50`` and at
  the wrong channel widths (7.5M vs paper 21.6M).  Replaced with a
  generic ``CSPNet`` trunk parameterised by ``CSPNetConfig`` (per-stage
  ``depths`` / ``out_chs`` / ``strides`` / ``groups`` / ``expand_ratio`` /
  ``block_ratio`` / ``bottle_ratio`` / ``cross_linear`` / ``down_growth``
  / ``block_type`` tuples) covering all three paper architectures.
  All three variants match the timm-reported param count to 0.00%:
  ``cspresnet_50`` 21.62M, ``cspresnext_50`` 20.57M, ``cspdarknet_53``
  27.64M.
  
  Weights sourced from ``timm/csp{resnet50,resnext50,darknet53}.ra_in1k``,
  hosted on ``lucid-dl/csp{resnet-50,resnext-50,darknet-53}``.
  Numerical parity vs timm: max abs logit diff ``3-4e-6`` across all
  three variants.
  
  Module + state-dict naming mirrors ``timm.models.cspnet`` so the
  converter is a *single* trivial rename (``head.fc → classifier``).
  ``_ConvBnAct`` uses ``LeakyReLU(0.01)`` everywhere (paper / timm
  recipe) and the stem is 7×7 stride-2 + 3×3 max-pool stride-2 — total
  stem stride 4, matching ResNet.
  
  Load via:
  ```python
  from lucid.models import cspresnet_50_cls
  from lucid.models.weights import CSPResNet50Weights
  m = cspresnet_50_cls(pretrained=True)
  m = cspresnet_50_cls(weights=CSPResNet50Weights.RA_IN1K)
  ```

- **`lucid.models.vision.crossvit`** — paper-faithful rebuild + pretrained
  ImageNet-1k weights for all six paper-cited variants (Chen et al.,
  ICCV 2021, Table 2).  Pre-3.5 CrossViT was a single-stage toy at
  20-50% of paper-cited params; replaced with the full paper
  architecture (dual-input 240/224, K=3 ``MultiScaleBlock``s, CLS-token
  cross-attention per stage).  All six variants land within 1% of the
  paper-published param count (7.0 / 8.6 / 26.7 / 27.4 / 43.3 /
  105.0 M).  Sourced from ``timm/crossvit_<variant>_240.in1k``, hosted
  on [`lucid-dl/crossvit-<variant>`](https://huggingface.co/lucid-dl).
  Numerical parity vs timm: max abs logit diff ``7.7e-6`` on crossvit_tiny.
  
  Side-effect: added a Python-level **bicubic 2D resampler** to the
  CrossViT model (the large-branch input is bicubic-rescaled, and
  Lucid's engine-level ``F.interpolate`` does not yet ship a bicubic
  kernel).  Implemented as a separable row+col pass with the
  Mitchell-Netravali ``a=-0.75`` weights on top of Lucid native ops
  (``arange`` + ``clamp`` + advanced indexing); matches PyTorch
  ``F.interpolate(mode='bicubic', align_corners=False)`` bit-for-bit
  on the CrossViT input distribution.  Promotion to a proper engine
  kernel is filed for a separate PR.
  
  Load via:
  ```python
  from lucid.models import crossvit_tiny_cls
  from lucid.models.weights import CrossViTTinyWeights
  m = crossvit_tiny_cls(pretrained=True)
  m = crossvit_tiny_cls(weights=CrossViTTinyWeights.IN1K)
  ```

- **`lucid.models.vision.convnext`** — pretrained weights for
  `convnext_xlarge_cls` (Liu et al., 2022, Table 11).  Hosted on
  [`lucid-dl/convnext-xlarge`](https://huggingface.co/lucid-dl/convnext-xlarge).
  Sourced from `timm/convnext_xlarge.fb_in22k_ft_in1k` (Facebook AI
  Research's ImageNet-22k pretraining → ImageNet-1k finetune; ~350M
  params, acc@1 ≈ 87.0% at 224×224).  Numeric parity vs timm: max
  abs logit diff `2.4e-5`.  Preset uses **bicubic** interpolation +
  resize_size=256 (timm default; differs from the four
  IMAGENET1K_V1 torchvision-sourced variants which use bilinear +
  variant-specific resize).  Adds a new `ConvNeXtTimmArch`
  converter class in `tools/convert_weights/convnext.py` for sourcing
  weights from timm rather than torchvision (key layout differs:
  `stem.0/1`, `stages.S.blocks.N.{conv_dw, mlp.fc1/fc2}`,
  `stages.S.downsample.0/1`, `head.norm/fc`).  Load via:
  ```python
  from lucid.models import convnext_xlarge_cls
  from lucid.models.weights import ConvNeXtXLargeWeights
  m = convnext_xlarge_cls(pretrained=True)
  m = convnext_xlarge_cls(weights=ConvNeXtXLargeWeights.FB_IN22K_FT_IN1K)
  ```

### Added

- **`lucid.utils.transforms`** — 4 new policy classes + 1 new transform:
  - `RandomErasing(p, scale, ratio, value)` (Zhong et al., 2017,
    arXiv:1708.04896) — single rectangular region erase, with
    constant / per-channel-tuple / `"random"` (i.i.d. normal) fill.
  - `TrivialAugmentWide(num_magnitude_bins, interpolation, fill, p)`
    (Müller & Hutter, 2021 — arXiv:2103.10158) — uniform sample of
    1 op + 1 magnitude per call.
  - `RandAugment(num_ops, magnitude, num_magnitude_bins, ...)` (Cubuk
    et al., 2020 — arXiv:1909.13719) — `num_ops` ops uniform-sampled
    with replacement; shared magnitude.
  - `AutoAugment(policy, num_magnitude_bins, ...)` (Cubuk et al.,
    2019 — arXiv:1805.09501) — 3 paper-faithful policy tables
    (`"imagenet"` / `"cifar10"` / `"svhn"`), 25 sub-policies each
    (verbatim from Tables 2/6/7).
- **`lucid.utils.transforms.functional`** — 5 new ops:
  - `adjust_sharpness(img, factor)` — PIL `ImageFilter.SMOOTH`
    convention with 1-pixel border preserved (parity with reference
    framework).
  - `autocontrast(img)` — per-channel min-max stretch to `[0, 1]`,
    flat channel passthrough.
  - `posterize(img, num_bits)` — bit-mask quantisation via
    uint8 round-trip, bit-exact PIL parity.
  - `solarize(img, threshold)` — invert pixels at or above threshold.
  - `invert(img)` — `1 - img`.
- **`lucid.utils.data`** — 4 new public exports:
  - `MixupCollator(alpha, *, num_classes, p)` (Zhang et al., 2018 —
    arXiv:1710.09412) — Beta(α,α) lambda mix; produces soft labels.
  - `CutMixCollator(alpha, *, num_classes, p)` (Yun et al., 2019 —
    arXiv:1905.04899) — random patch paste; effective λ recomputed
    after border clamping per paper Eq. 1.
  - `RandomMixupCutMixCollator(...)` — uniform random choice between
    the two per batch.
  - `RASampler(dataset, num_replicas, rank, shuffle, seed, num_repeats)`
    (Hoffer et al., 2020 — arXiv:1901.09335) — emits each unique
    index `num_repeats` times consecutively for in-batch
    augmentation diversity.
- **`lucid.utils.transforms.ImageClassificationAugment`** — 2 new
  `__init__` kwargs:
  - `auto_augment: str | None = None` — timm-style spec parser
    accepts `"ta_wide"`, `"ta"`, `"ra[-mM][-nN]"`,
    `"aa_imagenet"` / `"aa_cifar10"` / `"aa_svhn"`.
  - `random_erasing: float = 0.0` — applied **after** `Normalize`
    per reference recipe ordering.
- **AutoAugment shared infrastructure** — internal module
  `lucid.utils.transforms._autoaugment` exposes a 15-op vocabulary
  (`_OP_NAMES`), magnitude lookup (`_magnitudes_for`), and dispatch
  (`apply_op`) shared by all three policy classes; new op vocabulary
  is `Identity / ShearX / ShearY / TranslateX / TranslateY / Rotate /
  Brightness / Color / Contrast / Sharpness / Posterize / Solarize /
  AutoContrast / Equalize / Invert`.

### Verified — Numerical reference-framework parity

180 parity tests in `lucid/test/parity/transforms/` and
`lucid/test/parity/utils/` opt into reference-framework comparison
(`pytest -m parity`).  All pass against torch 2.12 / torchvision 0.27.

Reference-parity conventions adopted in `apply_op`:

- **ShearX / ShearY** anchor at `center=[0, 0]` (top-left, legacy
  AutoAugment paper convention) and sign-flip the angle to match the
  reference framework's matrix convention (Lucid's `affine_matrix`
  uses forward-warp ``y_out += tan·x``; reference uses inverse-warp
  ``y_out -= tan·x``).
- **Rotate** sign-flips the angle to match the reference framework's
  image-convention (positive degrees → clockwise) instead of Lucid's
  default math-convention (positive degrees → counter-clockwise).
- **TranslateX / TranslateY** — no flip needed; Lucid accepts the
  magnitude as a fraction of image size (per ``ImageClassificationAugment``
  contract), reference framework accepts integer pixels, converted at
  the parity-test boundary.

Known structural difference (not a bug): Lucid's `warp_affine` uses
`align_corners=True` while reference framework's `F.affine` uses
`align_corners=False`, so bilinear-interpolated geometric outputs
differ by up to ~1 source-sampling pixel.  Parity tests use NEAREST
interpolation with marker-position checks (semantic correctness) plus
a bounded-mean drift check on BILINEAR random images.

### Fixed

- `F.adjust_sharpness` — 1-pixel border is now preserved (matches
  PIL `ImageFilter.SMOOTH` convention) instead of bleeding through
  zero-padded `conv2d`.  Previously border pixels could clip to 1.0
  on a uniform input.

### Tested

- **615 new unit tests** across 5 files (`test_erasing.py`,
  `test_autoaugment_ops.py`, `test_mix.py`, `test_ra_sampler_spec.py`,
  plus extensions to `test_presets.py`).
- **180 new parity tests** across 5 files (`test_strong_aug_functional_parity.py`,
  `test_autoaugment_family_parity.py`, `test_random_erasing_parity.py`,
  `test_mix_collators_parity.py`).
- Full repo regression: 856/856 model tests, 770/770 utils unit
  tests, 30/30 weights tests — zero regression.

### Documentation

- Added retrospective `obsidian/retro/retro-strong-augment-suite.md`.
- Added op-contract `obsidian/op-contracts/op-autoaugment-magnitude.md`.
- Added engine quirk `obsidian/engine/engine-multiplicative-mask-broadcast.md`.
- Updated `obsidian/api/api-python-utils-data.md` for the 4 new
  exports (`__all__` count 21 → 25).
- Updated `obsidian/INDEX.md` with 3 new note references.

---

## [3.4.1] — 2026-05-22

**No-compile speed sweep across the norm family + Adam + ReLU.**  Pure
algorithmic restructures (CSE, `mean = sum × 1/N`, saved intermediates,
bias-correction folding) closing the remaining ref-framework gap on
ResNet-18 by 12.7 % wall, plus a critical AMP correctness fix that
enables transformer F16 training on Lucid for the first time since
3.3.0.  No new public API surface, no graph capture, no MPSGraph
dispatch changes — same MLX path, fewer kernel dispatches and fewer
recomputations.

**Headline (Mac Studio M4 Max, 5-run median):**

  ResNet-18 / CIFAR-10 / BS=32 F32:  36.76 → 32.10 ms  (-12.7 %)
                                     1.79× → ~1.53× ref-framework
  GPT-2-base step F32 (BS=32):                 ~193 ms   (~1.085× ref)
  GPT-2-base step AMP F16 (BS=32):  FAILED → 188 ms     (~1.091× ref)

**4-axis ref-framework gap after this release:**

| Workload              | Lucid vs ref |
|---                    |---           |
| Conv F32 (ResNet-18)  | 1.53×        |
| Conv AMP F16          | 1.13×        |
| Transformer F32       | 1.085×       |
| Transformer AMP F16   | 1.091×       |

Three of four are within 10 % — production parity.  The Conv F32 gap
remains the residual (multi-output backward structurally blocks MLX
kernel fusion); AMP F16 closes it.

### Performance

- **BN forward + running-stats EMA fusion** — `BatchNormNdBackward<N>::forward`
  now accepts optional `running_mean` / `running_var` / `momentum` parameters
  and performs the EMA update **in-place** inside the C++ forward kernel,
  reusing the same `saved_mean_` / `saved_rstd_` it already computes for
  autograd.  Eliminates the duplicate mean+var reduction over `x` that the
  Python `_update_running_stats` path required.  ResNet-18 step
  36.76 → 34.25 ms (−2.5 ms / −7 %).
- **Conv backward contig sweep** — drop trailing `mlx::core::contiguous` on
  returned `dW` in `conv_nd_backward` / `conv_transpose_nd_backward` and
  the input contigs feeding `conv_general` in the `compute_dx` / `compute_dW`
  lambdas.  Mirrors the 3.1.0 forward sweep + 3.4.0 norm-backward sweep.
  Conv2d backward microbench (ResNet-18 shapes): 21.36 → 20.76 ms.
- **Fused ReLU backward** — new `IBackend::relu_backward` virtual; GPU lowers
  to a single `mlx::core::where(greater(x, 0), g, 0)` kernel, CPU to a tight
  loop.  Replaces the prior 3-op chain (greater + astype + multiply) which
  MLX did not fuse across the Bool→F32 typecast boundary.  ReLU bwd
  microbench: 0.307 → 0.268 ms / call (−12.7 %), per-op parity with ref.
- **BN backward algorithmic restructure** — CSE `multiply(grad, xnorm)`
  (was computed twice) and replace the 4-reduction structure (sum + sum +
  mean + mean) with 2 keep-dims sums and 2 scalar multiplies by 1/N.
  ResNet-18 wall 33.74 → 32.78 ms (−1 ms).
- **BN forward polish** — same `mean → sum × 1/N` pattern in the forward
  path, plus drop trailing `contiguous(...)` on y / mean / rstd outputs.
  Framework polish (no measurable wall change as forward already fuses
  1.9×; fewer kernel dispatches per BN forward).
- **BN saved-xnorm** — `IBackend::batch_norm_forward` now returns 4 elements
  `[y, mean, rstd, xnorm]` (xnorm is the existing lazy MLX intermediate,
  zero extra forward cost); `IBackend::batch_norm_backward` gains a
  `saved_xnorm` parameter and the MLX path consumes it directly, skipping
  the `centered = x - mean` + `xnorm = centered * rstd` recomputation
  (2 element-wise ops on the full input tensor per BN bwd call).
  ResNet-18 wall 32.78 → 32.36 ms; BN bwd microbench shapes all 10–20 %
  faster.
- **Adam bias-correction algebraic fold** — fold `1/bc1` and `√bc2` /
  `1/√bc2` factors into pre-computed scalars `lr_eff = lr · √bc2 / bc1`
  and `eps_eff = ε · √bc2`, dropping the `m_hat = m_new / bc1` and
  `v_hat = v_new / bc2` materialisations per parameter (2 full-tensor
  broadcast multiplies × 60+ params per Adam.step).  ResNet-18 wall
  32.36 → 32.10 ms; pure Adam GPU 3.71 → 3.56 ms.
- **Norm family-wide `mean → sum × 1/N`** — extend the BN backward
  restructure to `layer_norm_backward`, `rms_norm_backward`,
  `group_norm_backward`.  ResNet wall unchanged (no LN/RMS/GN there);
  framework-wide benefit for every transformer / Llama-family LLM /
  segmentation workload using these norms.

### Fixed

- **AMP plumbing for LayerNorm / RMSNorm / GroupNorm forwards** — the three
  op schemas declared `AmpPolicy::ForceFP32` but their forwards never used
  `SchemaGuard` / `astype_op` to enforce it.  Under `with amp.autocast(F16):`
  the upstream op (typically `Embedding` with `AmpPolicy::Promote`) emits
  F16 while `gamma` / `beta` Parameter leaves stay F32, so the norm op's
  pre-cast `x->dtype() == gamma->dtype()` check threw `DtypeMismatch` and
  broke every transformer AMP F16 training run.  Fix mirrors
  `BatchNormNdBackward::forward`'s AMP path: `SchemaGuard` +
  `astype_op(x / gamma / beta, eff_dt)` so all three operands share the
  policy-resolved dtype before the kernel.  GPT-2-base AMP F16:
  FAILED → 188 ms (1.091× ref).

### Investigated, reverted

- **Conv-BN-ReLU fused autograd node (Option β)** — implemented a
  standalone `ConvBnRelu2dBackward` (`FuncOp<,5>`) + `conv_bn_relu_2d_op`
  fused forward that replaces 3 separate apply() calls with 1.  Bit-perfect
  parity (max|Δ|=0 on output + 5 gradients) but wall step +0.14 ms vs the
  unfused path (32.36 → 32.36 ms, within noise).  The autograd Engine
  cycle savings (~18 cycles × ~20 µs ≈ 0.36 ms expected) are dominated by
  MLX kernel execution time — which is unchanged because the fused C++
  apply() builds the same lazy MLX chain as the three separate apply()
  calls.  Implication: filling the
  `FusionPass::try_fuse_conv_bn_relu` placeholder properly would not
  deliver wall improvement on MLX 0.31.  ResNet wall is at the no-compile
  MLX floor.  Code reverted to keep the working tree clean; finding
  documented in
  `obsidian/perf/perf-bn-running-stats-fusion-2026-05-22.md`.

### Tests

- 221/221 nn parity + unit tests pass (`pytest lucid/test/parity/nn/
  lucid/test/unit/nn/`).
- 27/27 optim parity + unit tests pass.

### Vault notes

- `obsidian/perf/perf-bn-running-stats-fusion-2026-05-22.md` — full
  arc of this release, including the negative finding on Option β
  and the no-compile MLX floor analysis.
- `obsidian/api/api-cpp-tree.md` — Last verified bumped through every
  C++ surface change (BN forward 4-element return, BN backward
  `saved_xnorm` param, IBackend `relu_backward` virtual).

---

## [3.4.0] — 2026-05-21

**Per-op MPSGraph dispatch arrives.**  Apple's MPSGraph (the graph compiler
PyTorch MPS uses internally) now coexists with MLX inside the same
`GpuBackend`.  Per-op `should_dispatch_*` policies route the slowest
seven op families to fused MPSGraph kernels with an executable cache;
everything else still runs through MLX.  `device="metal"` semantics
are unchanged — the dispatch is invisible to user code, toggleable via
`LUCID_MPS_DISABLE=1` and observable via `LUCID_MPS_DEBUG=1`.

Two op families also got **MLX-side algorithm fixes** instead of MPSGraph
dispatch — embedding backward replaces a catastrophic onehot-matmul
(820 MB intermediate at GPT-2 vocab) with `mlx::core::scatter_add_axis`,
and LayerNorm / RMSNorm forward replace the 7-op composition with MLX's
own fused `fast::layer_norm` / `fast::rms_norm` primitives.

Phase 0 microbenchmarks (see `lucid_smoke/bench_op_microbench.py`,
measured on Mac Studio M4 Max) drove every dispatch decision.  Where
MPSGraph turned out to be net-negative (softmax backward, smaller silu
shapes), it was demoted from the gate.

### Performance — Mac Studio M4 Max (F32, vs PyTorch MPS 2.9.1)

| Op (worst-shape) | 3.3.0 (Lucid) | 3.4.0 (Lucid) | Torch | ratio change |
|---|---:|---:|---:|---|
| `gelu` fwd ffn-big (32×128×3072) | 13.90 ms | **1.18 ms** | 0.44 ms | 31× → **2.67×** torch |
| `gelu` bwd ffn-big | 14.37 ms | **1.66 ms** | 0.47 ms | 31× → **3.55×** torch |
| `embedding` bwd gpt2 (B×L=4096, V=50257) | 34.73 ms | **0.76 ms** | 1.26 ms | 28× → **0.61×** torch ← Lucid faster |
| `embedding` bwd bert (V=30522) | 21.13 ms | **0.57 ms** | 0.92 ms | 24× → **0.61×** torch ← Lucid faster |
| `layer_norm` fwd llama (16×256×4096) | 2.43 ms | **0.55 ms** | 0.55 ms | 4.4× → **1.00×** torch (parity) |
| `layer_norm` fwd gpt2 (32×128×768) | 0.46 ms | **0.21 ms** | 0.23 ms | 2.2× → **0.94×** torch ← Lucid faster |
| `layer_norm` bwd llama (gated dispatch) | 4.95 ms | 4.96 ms* | 2.57 ms | 1.92× → 1.93× torch |
| `rms_norm` fwd llama | 1.51 ms | **0.53 ms** | 0.53 ms | 2.8× → **1.01×** torch (parity) |
| `batch_norm` train bwd large_acts (32×64×112²) | 7.04 ms | **4.03 ms** | 1.31 ms | 5.47× → **3.08×** torch |
| `silu` bwd ffn-big (gated dispatch) | 2.63 ms | **1.72 ms** | 0.46 ms | 5.75× → **3.68×** torch |

*Within noise on this run; dispatch threshold tuned by gate.

End-to-end on a representative GPT-2-base step (B=16, L=128, V=50257,
d_model=768, d_ff=3072): **~205 ms (estimated 3.3.0) → 112 ms (3.4-dev)**
on Mac Studio M4 Max — roughly **2.2× → 1.21× PyTorch MPS** for transformer
training.  ResNet-18 / CIFAR-10 5-epoch unchanged from 3.3.0
(302.9 s / 1.79× PyTorch) — Wave A targets transformer hot paths, not
CNN ops.

### Added

- **New C++ engine sub-directory** `lucid/_C/backend/gpu/mps/` housing
  `MpsBridge.{h,mm}` (process-wide `MTLDevice` + `MTLCommandQueue`,
  `array_to_buffer` / `buffer_to_array` round-trip primitives — see
  `obsidian/engine/engine-mps-bridge-2026-05.md` for the bridge design),
  `MpsDispatch.{h,cpp}` (per-op heuristics + `LUCID_MPS_DISABLE` /
  `LUCID_MPS_DEBUG` env vars), and `MpsKernels.{h,mm}` (per-op fused
  MPSGraph kernels with a process-wide `MPSGraphExecutable` cache —
  graph compile is one-time per (shape, dtype, eps) signature; warm
  calls reuse the executable).
- **New `_C_engine.gelu_exact(a)`** Python binding wired into Python
  `F.gelu(x, approximate="none")` (the default exact erf-based GELU).
  Replaces a 10-op Python `_erf_approx` polynomial composition with a
  single autograd-aware engine call that dispatches the forward and
  backward to a fused MPSGraph kernel (or falls back to an MLX
  composition using `mlx::core::erf` natively).  Eliminates ~9 `_C_engine`
  ops from every `F.gelu(x)` Python call.
- **`IBackend::gelu_exact` + `IBackend::gelu_exact_backward`** virtuals,
  with implementations in both CPU and GPU backends; new
  `lucid::GeluExactBackward` autograd node (in `Activation.h`) and
  matching schema entry `"gelu_exact"`.
- **`IBackend::silu_backward`** virtual.  Previously the SiLU backward
  was composed at the autograd-node level from seven storage primitives
  (`sigmoid_storage` / `mul_scalar_storage` / …); it now delegates to the
  backend so the GPU path can dispatch a fused MPSGraph kernel for
  FFN-scale activations and the CPU path can use a single scalar loop.
- **`lucid_smoke/bench_op_microbench.py`** — 15 op families × ~60 shape
  variants × {F32, F16} microbench harness with paired Lucid (MLX) and
  PyTorch (MPS) measurements; outputs JSON, drove every dispatch
  decision in this release.

### Changed

- **`F.gelu(x, approximate="none")`** internally uses
  `_C_engine.gelu_exact` (a single autograd-aware op) instead of the
  10-op Python `_erf_approx` composition.  Numerical output is closer to
  PyTorch's exact GELU (uses `mlx::core::erf` / MPSGraph `erfWithTensor:`
  directly, vs the Abramowitz-Stegun polynomial approximation that the
  Python path used previously).  Bit-for-bit parity with PyTorch MPS
  (`max|lucid - torch| = 0` on every shape tested).  Private helper
  `_erf_approx` removed from `lucid/nn/functional/activations.py`.
- **`GpuBackend::embedding_backward`** rewritten from an onehot-matmul
  composition (`(M_total × N) × (N × D)` matmul of an
  `(M_total, N)`-shaped float-mask onehot tensor — 820 MB intermediate
  for GPT-2 `(B*L=4096, V=50257)`) to a direct
  `mlx::core::scatter_add_axis` call with index broadcast.  Same MLX
  primitive that the engine's existing `scatter_add_axis` op binds —
  see `obsidian/engine/engine-mlx-scatter-axis-vs-multiaxis` for the
  convention.  46× faster than 3.3.0 on GPT-2-input scale and now beats
  PyTorch MPS by 1.7×.  No MPSGraph dispatch needed.
- **`GpuBackend::layer_norm_forward`** and `rms_norm_forward` now use
  `mlx::core::fast::layer_norm` / `fast::rms_norm` (MLX's fused
  primitives, single Metal kernel) for the main output; saved-tensor
  `mean` and `rstd` for the backward are computed in parallel via
  `mlx::core::mean` + `var` reductions.  4-5× forward speedup on
  llama-scale, matches PyTorch MPS parity.
- **`IBackend::batch_norm_backward`** signature extended with
  `double eps` parameter; `BatchNormNdBackward<N>` stores `eps_` on
  the autograd node so the GPU MPSGraph kernel can reconstruct
  variance from `saved_rstd` (`var = 1/rstd² - eps`) for the canonical
  `normalizationGammaGradient*` / `normalizationBetaGradient*` /
  `normalizationGradient*` ops.  CPU backend ignores the new parameter
  (uses `saved_rstd` directly via the chain-rule formula).  Closes 1.75×
  of the BN large_acts backward gap (5.47× → 3.08× PyTorch).
- **`GpuBackend::softmax_backward`** wired with an MPSGraph dispatch
  hook gated at `axis_size >= 1024`, but the gate is currently hardcoded
  to return false — Phase 4 measurement showed both the MPSGraph
  canonical `softMaxGradientWithIncomingGradient:` and a hand-rolled
  chain were ~30 % slower than the MLX chain on `(4096, 50257)`
  (MPSGraph framework overhead exceeded the kernel saving).  Kernel
  code retained for reference / future re-evaluation.
- **`GpuBackend::gelu` / `gelu_backward`** (tanh-approximation variant)
  now route through MPSGraph when policy allows (universal — no shape
  gate).  Bit-for-bit parity with the MLX fallback path.

### Fixed

- **Depthwise / grouped Conv2d backward on Metal** no longer raises
  `[conv] If groups > 1, the output channels must be divisible by the
  number of groups. Got 1 output channels and 128 groups.`  Both the
  dx-via-flipped-conv and dW-via-channel-permute tricks in
  `GpuBackend::conv_nd_backward` were passing `opts.groups` to MLX's
  `conv_general`, but the channel rearrangement those tricks perform
  doesn't compose with MLX's grouped conv layout — they only worked
  for `groups == 1` despite the production code calling MLX with
  `opts.groups` regardless.  Fix: when `opts.groups > 1`, slice `x` /
  `W` / `grad_out` per group, run the ungrouped conv-tricks with
  `groups=1` on each slice, then concatenate.  Mirrors the per-group
  loop pattern already in `CpuBackend::conv_nd_backward`.  Six new
  parametrized parity tests at
  `lucid/test/parity/nn/test_conv_parity.py::TestConvGroupedParity`
  cover `{groups=2, groups=4, depthwise=16}` × `{forward, backward}`.
  ResNet-18 (which uses `groups=1` only) unaffected; **MobileNet /
  EfficientNet / depthwise paths now trainable on Metal**.  See
  `obsidian/debug/debug-conv-grouped-backward-2026-05.md`.
- **`Tensor.numpy()` (and other GPU→CPU bridge sites)** on a
  lazily-transposed MLX array no longer returns bytes in the
  underlying buffer's layout instead of the logical shape order.
  `conv_nd_forward`'s output transpose was deliberately left lazy
  (perf, see 3.0.3) so downstream MLX ops could fuse the stride;
  but `MlxBridge::download_gpu_to_cpu` was reading
  `arr.data<uint8_t>()` directly, which per MLX docs is contiguous
  bytes regardless of stride.  Fix: force
  `mlx::core::contiguous(arr)` before the memcpy.  Only affected
  direct `.numpy()` / `.tolist()` / `.to(device='cpu')` on raw
  unmaterialised conv outputs; downstream training paths were fine
  because the next op (BN, ReLU, …) materialised through MLX's
  stride-aware kernels.  Regression covered by new
  `lucid/test/unit/metal/test_metal.py::TestMetalLazyTransposeBridge`
  (permute-view + grouped Conv2d forward + grouped Conv2d backward
  grad).  See `obsidian/engine/engine-mlx-data-ignores-strides.md`.

### Performance (additional measurement notes)

- **`LUCID_MPS_DEBUG=1`** env var prints each dispatch decision plus
  per-call phase timing (sync / buffer alloc / kernel run) to stderr.
- **`LUCID_MPS_DISABLE=1`** env var disables the dispatch entirely
  (every op falls through to the MLX path) — useful for A/B comparing
  the dispatch vs the underlying MLX path, or for narrowing down
  which op a regression came from.
- Shape gates per op (Phase 4 tuned, M4 Max measurements):
  - `gelu` / `gelu_exact`: universal (every shape wins)
  - `batch_norm_train` fwd + bwd: `numel >= 8 M` (ImageNet-scale only;
    ResNet stays on MLX)
  - `layer_norm_backward`: `normalized_size >= 512 AND outer >= 256`
    (real transformer layers; skip Q/K/V small projections)
  - `silu_backward`: `numel >= 6 M` (FFN scale; CNN activation shapes
    stay on MLX)
  - `softmax_backward`: disabled (MPSGraph was net-negative)

### Vault notes

- `obsidian/architecture/arch-mps-dispatch-2026-05.md` — design rationale
- `obsidian/engine/engine-mps-bridge-2026-05.md` — bridge primitives + spike
- `obsidian/engine/engine-mlx-data-ignores-strides.md` — strided-bridge fix
- `obsidian/debug/debug-conv-grouped-backward-2026-05.md` — grouped conv fix
- `obsidian/perf/perf-mlx-op-baseline-2026-05.md` — Phase 0 baseline
- `obsidian/perf/perf-mpsgraph-shortlist-2026-05.md` — shortlist decisions

---

## [3.3.0] — 2026-05-20

AMP / mixed-precision activation **end-to-end**, with five engine-level
fixes that turn the surface plumbing into a feature users can actually
train with.  The autocast infrastructure (`AutocastGuard`,
`SchemaGuard`, `AmpPolicy`) has been wired into the codebase since the
3.0 series, but the **four hot ops that would benefit most —
`LinearBackward`, `ConvNdBackward<N>`, `MatmulBackward`,
`BatchNormNdBackward<N>`** — never called `SchemaGuard` to honor the
autocast scope.  Result: `with lucid.amp.autocast(F16):` had **no
effect** on the entire backbone of every CNN / MLP / transformer —
F32 was used regardless.  This release plumbs SchemaGuard into all
four; on top of that, five subtle gradient-flow bugs that would have
caused the plumbed AMP path to silently produce **random-accuracy
gradients** (10 % top-1 on ResNet-18 / CIFAR-10 — the model can't
learn even though wall-clock looks 2.5× faster) are fixed below.

### Fixed

- **AMP autocast scope now actually changes dtype** for `Conv1d` /
  `Conv2d` / `Conv3d` (`AmpPolicy::Promote`), `nn.Linear`
  (`AmpPolicy::Promote`), `lucid.matmul` (`AmpPolicy::Promote`), and
  `nn.BatchNorm{1,2,3}d` (`AmpPolicy::ForceFP32`).  Each forward
  invokes `SchemaGuard{schema_v1, x->dtype(), x->device()}` to
  resolve `effective_dtype()`, then casts its operands to that
  dtype.  Outside an autocast scope this is a no-op; inside
  `with amp.autocast(F16):` Conv / Linear / Matmul dispatch native
  F16 to MLX (Apple GPU runs F16 conv/matmul at roughly 1.4–1.8×
  FP32 throughput).
- **AMP cast now preserves the autograd chain (`AstypeBackward`).**
  The first attempt at the four ops above routed the cast through
  `detail::maybe_cast_for_kernel`, which produced a TensorImpl with
  `requires_grad=false`.  Downstream, `NaryKernel::wire_autograd`
  saw `any_grad=false` on the cast inputs and silently skipped the
  whole backward wiring — every parameter inside an autocast scope
  had `p.grad is None` after `loss.backward()`, the optimizer step
  ran on zero updates, and the model trained at random accuracy
  even with `GradScaler` enabled.  Replaced with the autograd-aware
  `astype_op`, which now wires a new `AstypeBackward` node when the
  input has `requires_grad`; the backward casts the incoming
  gradient back to the source dtype, keeping the chain intact and
  ensuring F32 parameter slots accumulate F32 gradients.  Same-dtype
  `astype_op(t, t->dtype())` is a no-op fast path that returns the
  input verbatim — the F32 baseline pays zero overhead.
- **`Tensor.to(dtype=...)` no longer strips the autograd grad_fn.**
  The same-device branch in `_to.py` called `clone_with_grad`
  unconditionally, which creates a fresh leaf `TensorImpl` and drops
  the `grad_fn` that `astype_op` had just wired.  Now the call is
  guarded behind `impl.requires_grad != self._impl.requires_grad` —
  the cast's freshly-installed backward node survives, so
  `logits.float()` mid-graph (the canonical AMP wrap-up that brings
  loss inputs back to F32) propagates gradients correctly.
- **`GradScaler.unscale_` no longer flushes F16 gradients to zero.**
  At the default `init_scale=2**16`, the unscale coefficient
  `1/65536 ≈ 1.526e-5` is **subnormal** in F16 (min-normal is
  `6.1e-5`); Apple Silicon's Metal backend flushes subnormals to
  zero, so a naive `full(shape, inv_scale, g.dtype=F16, ...)` coef
  becomes the zero tensor and every unscaled gradient collapses to
  0.  Additionally, mixed-dtype multiply (F16 grad × F32 coef) is
  rejected by `BinaryKernel`'s same-dtype validator (would
  segfault).  Now `unscale_` casts the gradient to F32 first, then
  multiplies by an F32 coef — matching the reference framework's
  AMP path.  The optimizer receives clean F32 gradients to update
  the F32 parameter slots with.
- **`AccumulateGrad` and `accumulate_into` now coerce mixed-dtype
  gradients.**  In ResNet-style residual blocks two paths can
  converge on the same parameter at different effective dtypes —
  e.g. the `c1` Conv emits F16 grad while a sibling BN
  (`ForceFP32`) emits F32.  The leaf accumulator now casts incoming
  grads to the parameter's own dtype before storing; the
  intermediate-node accumulator (`accumulate_into` in
  `autograd/Helpers.cpp`) casts the source to the destination's
  dtype before adding.  Without this, GPU `mlx::core::add` and CPU
  `cpu_add_inplace` would throw `DtypeMismatch` and the backward
  pass would fail to even complete on every multi-path AMP model.
- **`gpu_backend::full(dtype=F16)`** was `NotImplementedError`.
  Triggered by Python scalar promotions touching a cast F16 tensor
  inside autocast (e.g. BatchNorm's running-stats update builds an
  F16 scalar tensor).  Now creates an F32 scalar then `astype` to
  F16 — same path the rest of the F16 dispatch uses.  Also wired
  I8 / I16 cases that had been left as `NotImplementedError`.
- **`BatchNormEvalBackward` (the `model.eval()` BN path) now honors
  AMP.**  The eval-mode op declared `AmpPolicy::ForceFP32` in its
  schema but `forward()` never called `SchemaGuard` — under
  `model.eval()` + `with amp.autocast(F16):` the F16 activations
  were passed straight into the F32 running-mean / running-var
  buffers.  The backend's `batch_norm_eval_forward` doesn't validate
  dtype consistency and silently produced NaN outputs, so on
  Mac Studio ResNet-18 / CIFAR-10 the **training loop converged
  normally (80 % train acc) but `evaluate()` reported
  `test_loss=nan, test_acc=10 %`**.  Now `forward()` resolves
  `eff_dt = F32` via SchemaGuard and `astype_op`-casts all five
  inputs (x / mean / var / γ / β) up to F32 before the kernel call,
  matching the training-mode path.
- **`BatchNorm` / `InstanceNorm` running stats no longer drift to
  F16 under autocast.**  ``_update_running_stats`` mixes the F32
  buffer with the batch reduction in a sequence of ``mul`` / ``add``
  ops; ``mul`` is registered ``AmpPolicy::Promote`` and ``var`` is
  ``AmpPolicy::KeepInput``, so inside ``with amp.autocast(F16):``
  every iteration silently demoted the running buffers from F32 to
  F16.  After thousands of training steps the F16 precision loss
  poisoned the stats and ``model.eval()`` produced 10–18 % test
  accuracy on CIFAR-10 even though training itself converged.
  Wrapped the update in ``AutocastGuard(F32)`` (RAII via CPython
  refcount drop on exit) and ``.to(buffer_dtype)``-cast ``x`` *before*
  computing batch_mean / batch_var so the F32 buffers stay F32 and
  the variance is computed at full precision.  Mac Studio
  ResNet-18 / CIFAR-10 after the fix: ``test_acc=77.75 %`` vs F32
  baseline ``79.95 %`` — the residual 2.2 pp gap is the standard
  F16-precision tax, not a correctness bug.
- **Strict dtype-match checks** in Conv / Linear / Matmul / BN
  fired before `SchemaGuard` could unify dtypes under autocast
  (upstream Conv had cast x to F16 while `gamma` / `beta` were
  still F32 on the Parameter slots).  Moved each check to *after*
  the AMP cast, where all operands share the policy-resolved
  dtype.

### Optimised

- **`eval_tensors_async` — new C++ binding around
  `mlx::core::async_eval`.**  Schedules batched GPU evaluation
  *without* blocking the CPU thread, but still finalises the MLX
  lazy expression graph at schedule time so parent activation
  references are released and downstream allocators can reclaim
  memory.  Exported as `lucid._C.engine.eval_tensors_async(list)`.
  Drop-in replacement for `eval_tensors` whenever the caller only
  needs the lazy-chain break, not the GPU completion barrier.
- **`_eval_running_stats_metal` switched to `eval_tensors_async`.**
  BatchNorm / InstanceNorm running-stats update no longer issues a
  GPU sync per layer.  3.2.1 had introduced the sync as a memory
  leak fix (running stats live outside the loss graph, so they were
  never evaluated and accumulated one full forward graph per step);
  switching to `async_eval` keeps that fix while dropping the
  per-BN barrier.  **Single biggest win in this release** — see the
  Performance section.
- **`Adam` / `AdamW` per-step scalar cache (`AdamScalarCache`).**
  The 8 broadcast scalars (`β1`, `1-β1`, `β2`, `1-β2`, `eps`, `lr`,
  `1/bc1`, `1/bc2`) are now built **once per step** at the first
  parameter and reused across the remaining ~64 parameters of a
  typical ResNet-18 step — instead of constructing
  `params.size() * 8` MLX scalar arrays per call (520 for
  ResNet-18).  Cache invalidates when `step_count_` advances or the
  param dtype changes (mixed-precision case) or `set_lr` is called.
  Effect on the timing-loop is currently in the measurement-noise
  band (~0.2 ms / step on M4 Max), but the infrastructure is sound
  and pays off more once the MLX schedule gets cheaper or the
  pattern is reused by other optimizers.
- **Backward-path `contiguous(...)` sweep — extension of 3.1.0's
  forward sweep.**  Removed redundant `wrap_mlx_array(contiguous(...))`
  on the return path of `batch_norm_backward`, `layer_norm_backward`,
  `rms_norm_backward`, and `group_norm_backward` — MLX `multiply` /
  `sum` produce contiguous tensors and `reshape` preserves
  contiguity, so the trailing materialisation was breaking lazy
  fusion at the autograd boundary.  Forward kernels of these ops
  were already cleaned in 3.1.0; the 3.4-perf-sweep applies the
  same treatment to their backward halves.  Per-call savings are
  small (~0.1 ms) but consistent with the 3.1.0 sweep philosophy.
- **`_update_running_stats` micro-clean.**  Dropped the redundant
  `.detach()` calls on `batch_mean` / `batch_var` / `new_rm` /
  `new_rv` inside the `with no_grad():` context — every tensor
  computed there already has `requires_grad=False`, so detach was a
  Python+pybind round-trip with no observable effect.  Same change
  applied to InstanceNorm's equivalent path.

### Performance

- **M1 Pro TinyResNet train step** (BS=32, 4 residual blocks):
  - F32 baseline:    median 93.3 ms
  - AMP F16:         median 86.5 ms (**−7 %**)
- **Mac Studio M4 Max ResNet-18 / CIFAR-10** (BS=32, 1 epoch,
  measured *after* the 3.4 perf sweep above):

  | Mode             | 3.3.0 baseline | **3.3.0 final** | Δ            |
  |------------------|----------------|-----------------|--------------|
  | F32 (1 epoch)    | 68.23 s        | **58.62 s**     | **−14.1 %**  |
  | F32 (step median)| 43.24 ms       | **36.27 ms**    | **−16.1 %**  |
  | F32 forward      | 16.69 ms       | **9.91 ms**     | **−40.6 %**  |
  | AMP F16 (1 epoch)| 88.86 s        | **77.58 s**     | **−12.7 %**  |
  | AMP F16 (step)   | 56.66 ms       | **49.27 ms**    | **−13.0 %**  |

  vs **PyTorch MPS 2.9.1** on the same workload (`torch_resnet_cifar`
  with `torch.from_numpy` pre-tensorised data, identical optimizer /
  hyperparams):

  | Mode    | PyTorch | Lucid 3.3.0 baseline | **Lucid 3.3.0 final** |
  |---------|---------|----------------------|------------------------|
  | F32     | 32.46 s | 68.23 s (2.10×)      | **58.62 s (1.80×)**    |
  | AMP F16 | 68.58 s | 88.86 s (1.30×)      | **77.58 s (1.13×)**    |

  Memory remains better than the reference framework — peak GPU
  398 MiB (F32) / 444 MiB (AMP) vs reference 430 / 510 MiB
  respectively.  Numerical parity preserved (Lucid F32 test acc
  55.10 % vs reference 55.41 % on 1-epoch CIFAR-10, within noise).
  AMP correctness verified end-to-end at 5 epochs: 77.75 % vs F32
  79.95 %, the standard F16-precision tax.

  The single biggest contributor is the `async_eval` switch — it
  removes 16 GPU sync barriers per forward (one per BatchNorm
  layer in ResNet-18), saving ~5.9 ms / forward and ~9.2 s / epoch.

  The residual 1.80× gap on F32 is *not* Python-overhead bound:
  step time scales **linearly with batch size** (BS=32 → 36 ms,
  BS=128 → 143 ms; ratio 3.93×) — Python dispatch overhead is
  fixed-cost so the time would grow *sub-linearly* if Python were
  the bottleneck.  The gap is dominated by **MLX op-level lazy
  graph vs MPSGraph fused-kernel** scheduling:  Apple's
  MPSGraph fuses `conv + bias + activation + BN` into a single
  kernel launch, while MLX schedules each op as a separate node.
  ResNet-18 backward (where the gap is largest, ~6.4 ms vs
  PyTorch 11.55 ms — yes, Lucid backward kernels are actually
  *faster* in isolation, but full-step is slower because Adam
  and forward Python overhead are larger).  Closing this requires
  either `lucid.compile()` (graph capture + MLX `compile`) or
  upstream MLX adopting MPSGraph kernels — both out of scope here.

### Investigated, no action

- **Adam foreach-fusion** — projected 8–15 % speedup based on the
  per-param Python-dispatch model.  Actual measurement: Adam.step()
  lazy-build cost is **0.48 ms (0.5 %)** of training step on M1 Pro.
  The 27 ms figure that motivated the projection was eval-time GPU
  compute (deferred to next `loss.item()`), not Python dispatch.
  Lucid's C++ `Optimizer::step()` already loops in a single pybind
  dispatch and MLX's lazy graph already batches the per-param
  updates.  No real opportunity; the projection assumed PyTorch's
  per-param Python-dispatch model.
- **Other optimizers (SGD / RMSProp / NAdam / RAdam / Adamax /
  Adagrad / Adadelta) per-step scalar cache.**  Same pattern as
  `AdamScalarCache` would mechanically apply, but the Adam version
  produced noise-level savings on M4 Max (~0.2 ms / step) and the
  other optimizers have *fewer* per-step scalars to hoist.  Skipped
  in this release; the infrastructure landed in `Adam.h` /
  `Adam.cpp` so a follow-up can roll it out symmetrically when the
  underlying MLX dispatch cost rises (or when a benchmark surfaces
  a real win).
- **`conv_nd_backward` redundant-`contiguous(...)` removal.**  Each
  of the trailing `contiguous` wraps there carries an explicit
  "PERF" comment justifying its presence (MLX `conv_general`
  kernel selection requires contiguous inputs and the
  transpose-then-contig sequence is wired into the
  forward-symmetric pattern).  Touching them would be a
  correctness risk for marginal gain.
- **`DataLoader` per-step overhead** — measured at ~1.55 ms / step
  on the current workload, which translates to ~2.4 s / epoch for
  1562 iterations.  Decomposition: 0.34 ms for the
  `next(iter)` + `.to('metal')` + sync of one batch, the rest is
  Python-side loop bookkeeping (collate, list construction).  The
  inherent batch-slicing + collate API cost cannot be made smaller
  without breaking the `__getitem__`-per-sample contract that
  user-defined `Dataset` subclasses depend on; switching to
  `TensorDataset` (vectorised `__getitems__` fast path that landed
  in 3.2.0) did not change the median step time on this workload
  because the timing was already dominated by the GPU step, not
  the data prep.

### API

- **`lucid.amp.autocast(dtype=lucid.float16)`** is functional
  end-to-end for the four ops above.  Usage matches the reference
  framework's `cuda.amp.autocast`:
  ```python
  import lucid.amp as amp
  scaler = amp.GradScaler()
  for x, y in loader:
      opt.zero_grad()
      with amp.autocast(dtype=lucid.float16):
          out = model(x)              # Conv / Linear / Matmul → F16
          out_f32 = out.float()       # cast back for loss
          loss = loss_fn(out_f32, y)  # F32 loss (BN forces F32 inside)
      scaler.scale(loss).backward()    # grads land at parameter dtype
      scaler.step(opt)                 # unscale in F32
      scaler.update()
  ```
  On CPU the F16 path is silently demoted to F32 (Accelerate has no
  F16 arithmetic).  No public Python API surface change — the
  autocast context and `GradScaler` already existed.  New C++
  export: `lucid::AstypeBackward` (autograd node for `astype_op`,
  exposed only through the existing `astype_op` free function).

### Tests

All 50 AMP / autograd / integration tests pass; 596 nn + ops unit
tests pass; existing F32 training paths are byte-identical to 3.2.2
(SchemaGuard fast-path returns input dtype when no autocast active).
`test_amp_train.py` updated to cast the model output back to F32
before computing the loss — the canonical AMP pattern that the
plumbing now actually exercises.  Pre-existing 5 `flatten` failures
in `test_parity_func.py::TestJVPParity` / `TestJacFwdParity` /
`TestVmapJacfwdParity` predate this release (verified on 3.2.2
baseline) and are unrelated; see func module backlog.

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

[Unreleased]: https://github.com/ChanLumerico/lucid/compare/v3.5.0...HEAD
[3.5.0]: https://github.com/ChanLumerico/lucid/releases/tag/v3.5.0
[3.0.0]: https://github.com/ChanLumerico/lucid/releases/tag/v3.0.0
