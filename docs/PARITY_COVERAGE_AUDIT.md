# Parity Coverage Audit

Snapshot of `tests/parity/` coverage against the C++ engine surface.

**Generated:** 2026-04-28
**Engine commit:** see `git rev-parse HEAD`
**Total parity assertions at audit time:** 1237 passed, 377 skipped, 0 failed

---

## Top-line numbers

| Metric | Count | % of surface |
| --- | --- | --- |
| **Total ops in engine surface** (excluding infra) | 226 | 100% |
| **Spec'd in some form** (any axis) | 209 | 92% |
| &nbsp;&nbsp;&nbsp;&nbsp;CPU + GPU forward + backward verified | **99** | **44%** |
| &nbsp;&nbsp;&nbsp;&nbsp;Forward-only (`skip_grad=True`) | 76 | 34% |
| &nbsp;&nbsp;&nbsp;&nbsp;Spec'd but with no `skip_grad` flag determined | 34 | 15% |
| **No spec at all** | 17 | 8% |

> **The headline number — "100% legacy parity" — refers to API surface (173 legacy ops → 173 implemented, 1:1).**
> This audit is a *different* dimension: how thoroughly each op's CPU/GPU forward and backward paths are verified by the parity harness against PyTorch.

---

## A. Fully verified — CPU + GPU forward + backward (99 ops)

These pass all 6 harness axes (`test_forward_CPU`, `test_forward_GPU`, `test_cross_device_forward`, `test_backward_CPU`, `test_backward_GPU`, `test_cross_device_backward`).

This is the gold standard. Every op in this set is known to:
- compute identical (within tolerance) results to PyTorch on CPU and GPU
- produce identical gradients to torch.autograd on CPU and GPU
- not silently diverge between the two device paths

Includes the bulk of bfunc / ufunc / einops / utils / linalg-forward / nn-typical-paths.

---

## B. Forward-only verified — `skip_grad=True` (76 ops)

These pass forward axes but explicitly skip backward. The reasons fall into clear categories:

### B.1 Inherently non-differentiable (~25 ops)
Index / categorical / boolean output — no gradient defined.
```
argmax  argmin  argsort  sort  topk  gather  where  diag (extract)
equal  not_equal  greater  greater_equal  less  less_equal
bitwise_and  bitwise_or  invert  floordiv
```

### B.2 Tie-break ambiguity (~5 ops)
Differentiable in principle but engine and torch break ties differently — comparison would be flaky.
```
max_pool1d  max_pool2d  max_pool3d  adaptive_max_pool1d/2d/3d
maximum  minimum  hard_sigmoid  hard_swish
```

### B.3 Constructors / random (~14 ops)
No input tensor → no gradient flow possible.
```
zeros  ones  eye  arange  linspace  full  *_like
rand  randn  uniform  normal  randint  bernoulli
```

### B.4 Inference-mode / non-trainable inputs (~10 ops)
The op runs in inference mode or accepts inputs that torch refuses requires_grad on.
```
batch_norm_eval        # running stats are non-trainable
alpha_dropout          # masks
embedding              # indices are integer
dropout / dropoutnd / drop_block / drop_path  # masks; only output flows
bce_loss / bce_with_logits  # torch rejects requires_grad on weight/pos_weight
one_hot                # integer label input
sinusoidal_pos_embedding  # constructor
clip_                  # in-place; autograd graph version-mismatch by design
pow_                   # ditto
```

### B.5 Linalg forward-only (10 ops)
Engine has no backward implementation for linalg yet.
```
linalg.inv  linalg.det  linalg.solve  linalg.cholesky  linalg.norm
linalg.qr  linalg.svd  linalg.pinv  linalg.matrix_power  linalg.eig
```

### B.6 Specials / discrete rounding (~7 ops)
```
ceil  floor  round  sign      # piecewise-constant; engine returns 0 grad which torch matches by passing through identity
relu6                          # boundary-jump
masked_fill                    # mask is non-differentiable
linspace                       # constructor
```

**Action item:** B.5 (linalg backward) is the only category with genuine "could be implemented" gradient work. The rest are correct-by-design `skip_grad`.

---

## C. No spec at all (17 ops)

### C.1 Carve-outs / data-dependent output (5)
Already documented in [`memory/feedback_cpu_gpu_backends.md`](../memory/feedback_cpu_gpu_backends.md). Output device is CPU regardless of input device.
```
nonzero  unique  histogram  histogram2d  histogramdd
```
Behavior is verified through smoke tests in `test_determinism.py` and ad-hoc checks but not via the OpSpec harness.

### C.2 Spec writeable, not yet written (4)
Just an oversight — straight to-do.
```
bitwise_xor                      # trivial — twin of bitwise_and/or
nn.grid_sample                   # forward + backward both natively MLX, ready to verify
nn.global_response_norm          # standard norm op
nn.rotate                        # we already migrated GPU path; verified ad-hoc, no spec
```

### C.3 OpSpec pattern doesn't fit (5)
Need harness extension or special handling.
```
pow_scalar  rpow_scalar              # Tensor⊕scalar pattern, no second TensorImpl input
nn.affine_grid                       # output is a grid, comparison with torch needs tweak
nn.rotary_pos_embedding              # param-rich constructor
nn.scaled_dot_product_attention_with_weights  # returns 2 outputs (attn + weights)
```

### C.4 Composite layout-only ops (3)
Trivial wrappers around existing ops.
```
split        # superseded by split_at (already spec'd)
squeeze_all  # superseded by squeeze + squeeze_all-via-default
unbind       # like chunk; spec'able
```

---

## D. What "100% verified" would require

To get every CPP op fully validated under the parity harness:

| Step | Effort | Coverage gain |
| --- | --- | --- |
| Add specs for **C.2** (4 ops) | 30 min | 99 → 103 |
| Add specs for **C.4** (3 ops) | 30 min | 103 → 106 |
| Mark **C.1** carve-outs explicitly in `conftest.py` (5 ops) | 15 min | clarity |
| Extend harness for **C.3** (5 ops, scalar-bind / multi-output) | 2-3 h | 106 → 111 |
| Implement linalg backward (B.5) — non-trivial math (10 ops) | days | 111 → 121 |
| Wire `skip_grad=False` where torch path is actually OK (e.g., re-examine `dropout` p=0 backward, etc.) | 1-2 h | maybe +5 |

**Practical ceiling without major engine work:** ~125/226 (55%) full fwd+bwd verified. The rest are correctly forward-only or carve-outs.

**Realistic ceiling including linalg backward:** ~130/226 (57%).

The remaining ~95 will always be `skip_grad=True` because they are genuinely non-differentiable (index ops, constructors, comparisons, tie-break ambiguous reductions). That's not a coverage failure — it's the math.

---

## E. Honest summary

- **"All legacy ops implemented"** ✅ — 173/173, 1:1 mapping (some renamed: `multiply→mul`, `cross_entropy→cross_entropy_loss`, etc.)
- **"All ops compile and bind"** ✅ — 226/226 callable from Python
- **"All ops have at least a forward parity test"** ⚠️ — 209/226 (92%), 17 no-spec
- **"All differentiable ops have backward parity"** ⚠️ — 99 backward-verified out of ~125 differentiable; remaining 26 either skip_grad-by-circumstance (B.4-style) or no-spec (C.2/C.3/C.4)
- **"Engine is bug-free wrt PyTorch"** ⚠️ — UBSan clean, 1237 specs pass, but recent fixes (`bce_with_logits` broadcast, `floordiv` GPU) prove silent bugs slip past until specifically tested. Parity coverage gap = bug-discovery gap.

---

## F. Action items (prioritized)

If we want to drive this to a "really done" state:

1. **C.2 + C.4 (~7 ops, 1h)** — easy spec additions
2. **C.3 harness extension (5 ops, 2-3h)** — pow_scalar / multi-output / constructor patterns
3. **Re-examine B.4 ops where backward is actually safe (~5 ops, 1h)** — embedding (grad on weights), dropout (grad through mask), etc.
4. **Linalg backward (B.5, 10 ops, days)** — defer until model training requires it
5. **Mark carve-outs explicitly** — done in memory file, surface in code comments too

After 1-3 we'd be at ~115/226 with backward parity and 100% forward parity, which is the practical maximum for a non-trainable-linalg engine.
