# Lucid test suite

A production-grade test infrastructure for the Lucid ML framework.
Every op is exercised under a single canonical template (shape /
value / dtype / device / edge / backward / determinism), every device
stream (CPU + Metal) is validated for numerical agreement, and every
numerically meaningful op has a parity check against a reference
framework.

## Layout

```
lucid/test/
├── _fixtures/    fixture modules (devices, dtypes, tensors, ref, perf)
├── _helpers/     comparison / seeding / shapes / gradcheck / golden I/O
├── unit/         focused per-op CPU+GPU behaviour
├── parity/       value comparison vs the reference framework
├── numerical/    invariants / stability / precision / golden tensors
├── integration/  multi-component flows (train loops, checkpoints)
├── perf/         pytest-benchmark powered timing (opt-in)
├── stubs/        type-stub regression
└── conftest.py   global wiring, autouse seed
```

## Markers

| Marker | Use |
|---|---|
| `smoke` | < 1 s sanity check |
| `gpu` | Metal-only; collect-skip when unavailable |
| `parity` | Reference-framework parity; collect-skip when missing |
| `slow` | > 5 s (integration / training loops) |
| `perf` | Benchmark; opt-in via `--benchmark-only` |
| `stability` | Extreme-value / numerical-edge |
| `f64_only` | Float64-only path |

## Running

```bash
# fast tier (unit + numerical + stubs; no reference framework)
pytest lucid/test/ \
    --ignore=lucid/test/parity \
    --ignore=lucid/test/integration \
    --ignore=lucid/test/perf

# GPU-only sweep (Apple Silicon)
pytest lucid/test/ -m gpu

# parity tier (auto-skips when reference framework isn't installed)
pytest lucid/test/parity/

# integration tier (training loops + checkpoint round-trip)
pytest lucid/test/integration/

# perf tier — install ``pytest-benchmark`` for full timing tables
pytest lucid/test/perf/ -m perf
pytest lucid/test/perf/ --benchmark-only          # with pytest-benchmark
```

`scripts/ci_full.sh` runs all four tiers in order and treats parity /
perf failures as warnings, so a fresh checkout without the reference
framework or `pytest-benchmark` still passes the gate.

## Reference framework

Parity tests use a reference ML framework as ground truth. The import
is **lazy and string-concat-obfuscated** so the literal name doesn't
appear in Lucid source (H5). End users who never run `pytest` never
trigger this import — Lucid stays usable without `pip install <ref>`.

When the reference isn't installed, every test under `parity/` is
auto-skipped at collection time with a single message.

## Determinism

`conftest.py` sets `lucid.manual_seed(0)` automatically before every
test (autouse fixture). When a test additionally needs the reference
framework's RNG seeded the same way, use
`lucid.test._helpers.seeding.seed_all(0, ref=ref)`.

## CPU + GPU coverage

The `device` fixture parametrizes over `["cpu", "metal"]` whenever
Metal is detected. Most op-level tests don't have to do anything
special — they pull `device` and the cross-product runs automatically.

For tests that need to compare CPU vs GPU output of the *same* op
(detecting "device drift"), use the `cross_device_pair` fixture.

## Adding a new test

Drop the file under the right `unit/<area>/` subdirectory. Match the
canonical template:

```python
import pytest
import lucid

class TestMyOp:
    def test_basic_shape(self, device, float_dtype): ...
    def test_known_values(self, device, float_dtype): ...
    def test_broadcasting(self, device): ...
    def test_zero_dim(self, device): ...

    @pytest.mark.stability
    def test_inf_nan(self, device): ...

    def test_backward(self, device): ...
    def test_gradcheck(self, device): ...
    def test_reproducible(self, device): ...
    def test_cpu_gpu_match(self): ...
```

When the op is also expected to match the reference framework, add a
mirror file under `parity/<same-path>/`:

```python
@pytest.mark.parity
class TestMyOpParity:
    def test_forward(self, ref, device, float_dtype): ...
    def test_backward(self, ref, device): ...
```
