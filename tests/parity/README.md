# Lucid C++ engine — parity harness

A spec-driven pytest suite that verifies every C++ engine op on six axes:

| Axis | What it checks |
| --- | --- |
| `test_forward_CPU` | engine CPU forward == torch |
| `test_forward_GPU` | engine GPU forward == torch |
| `test_cross_device_forward` | engine CPU forward == engine GPU forward |
| `test_backward_CPU` | engine CPU backward == torch.autograd |
| `test_backward_GPU` | engine GPU backward == torch.autograd |
| `test_cross_device_backward` | engine CPU backward == engine GPU backward |

PyTorch is the reference; the DUT is `lucid._C.engine` (no Python wrapper —
the harness calls the engine directly).

## Run

```bash
pytest tests/parity/                      # full suite
pytest tests/parity/ -k matmul            # only matmul-related specs
pytest tests/parity/ -k forward           # only forward axes
pytest tests/parity/ -k "GPU and not backward"
```

When MLX/Metal isn't usable on the host, all `*_GPU` and `*_cross_device*`
tests are skipped automatically.

## Add a new op

1. Find or create the right `specs_*.py` file (one per op family).
2. Append an `OpSpec` to its `SPECS` list:

```python
OpSpec(
    name="my_op_4x5",
    engine_fn=lambda ts: E.my_op(ts[0], ts[1]),
    torch_fn=lambda ts: torch.my_op(ts[0], ts[1]),
    input_shapes=[(4, 5), (4, 5)],
    atol=1e-4, rtol=1e-4,            # tighten/loosen as needed
    skip_grad=False,                 # True for non-differentiable
    skip_gpu=False,                  # True only if op truly can't go GPU
)
```

3. If random inputs from a single seed don't suit (e.g. SPD matrices for
   `cholesky`, positive values for `log`), pass `input_gen=lambda rng: [...]`
   instead of `input_shapes`.

4. If the default sum-to-scalar reduction for backward isn't meaningful,
   pass `post_fn=lambda out, engine: ...`. The harness calls it once with
   `engine=lucid._C.engine` (engine path) and once with `engine=torch`
   (reference path). Use the engine arg to dispatch the right `sum`/`mean`.

That's it — all six axes are derived automatically.

## Tolerances

Defaults are `atol=rtol=1e-4`, which is tight for f32 element-wise ops.
Loosen to `1e-3` for ops that go through MLX matmul or Apple Accelerate
sgemm (those use fast-math and accumulate slightly differently than torch).
Loosen to `1e-2` only for genuinely lossy paths (f32 inv, qr, det).

## Why this exists

Before this harness, op-by-op verification was scattered across one-shot
scripts in `scripts/verify_*.py`. That made it easy to add an op without
adding its tests, and easy to claim "verified" when only one device path
had been touched. The harness forces a single declarative entry per op
and refuses to declare success until every axis passes.
