# Contributing to Lucid

Lucid is a production-grade ML framework for Apple Silicon, with a PyTorch-compatible Python surface backed by a custom C++ engine running on MLX (GPU) and Apple Accelerate (CPU). This guide covers everything you need to contribute correctly.

**Read this document fully before opening a PR.** Most rejections come from violating the hard rules in §3, which are non-negotiable.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Development environment](#2-development-environment)
3. [Hard rules — non-negotiable](#3-hard-rules--non-negotiable)
4. [Coding conventions](#4-coding-conventions)
5. [Architecture invariants](#5-architecture-invariants)
6. [Adding a new op](#6-adding-a-new-op)
7. [Testing](#7-testing)
8. [Changelog and commits](#8-changelog-and-commits)
9. [Pull request checklist](#9-pull-request-checklist)
10. [Out-of-scope](#10-out-of-scope)

---

## 1. Project overview

Lucid targets Apple Silicon exclusively (M1–M4, macOS arm64). Linux, Windows, and x86 are out of scope and will not be supported in this major version.

### Layer stack

```
Python public API       lucid.* / lucid.nn.* / lucid.optim.*
Python composite/dispatch  lucid/_ops/composite/, _dispatch.py
pybind11 boundary       lucid/_C/engine.cpython-*.so
C++ engine
  tensor/    TensorImpl, AutogradMeta, view semantics
  core/      Dtype, Shape, Device, Error, Allocator
  backend/
    cpu/     Apple Accelerate (BLAS / LAPACK / vDSP / vForce / BNNS)
    gpu/     MLX (mlx::core::*) + MetalAllocator
  kernel/    IKernel, UnaryKernel, BinaryKernel, primitives
  autograd/  Node, Engine, AutogradNode<Derived,N>
  ops/       bfunc / ufunc / gfunc / utils / linalg / einops / nn
  optim/     SGD / Adam / AdamW / RMSprop / …
  registry/  OpSchema, OpRegistry, BindingGen
  bindings/  bind_*.cpp (auto-generated where possible)
```

### Layer dependency DAG (top → bottom; no reverse imports allowed)

```
bindings → ops → kernel → autograd → backend → tensor → core
                       ↑
                  registry  (orthogonal; used by bindings + kernel)
                       ↑
                  primitives (kernel/primitives/; may use backend)
```

`tools/check_layers.py` enforces this in CI. A violation fails the build.

### Backend rule

| Stream | Backend | Notes |
|--------|---------|-------|
| CPU | Apple Accelerate only | vDSP / vForce / BLAS / LAPACK / BNNS |
| GPU | MLX only | mlx::core::* |
| `lucid.linalg` on CPU | MLX (exception) | MLX itself is CPU-backed here; wrap result back as GPU |
| Data-dependent output shapes | CPU round-trip | Unavoidable; document the reason in a comment |

---

## 2. Development environment

### Requirements

- macOS 26 Tahoe (arm64) or later, M1 or later
- Python 3.14 only (PEP 649 lazy annotations — H1/H7 require it)
- MLX ≥ 0.31 (matches the `macosx_26_0_arm64` wheel + `mlx-metal` split package)
- CMake ≥ 3.24
- Ninja ≥ 1.11
- Xcode Command Line Tools

### Install for development

```bash
git clone https://github.com/ChanLumerico/lucid.git
cd lucid
pip install -e ".[dev]"
```

For parity tests against the reference framework, install it separately and run:

```bash
pip install -e ".[test]"
# install reference framework separately
pytest lucid/test/parity/ -m parity
```

For documentation:

```bash
pip install -e ".[docs]"
```

### Build the C++ engine

```bash
pip install -e . --no-build-isolation
# or build only the extension:
cmake --build build/temp.macosx-*/lucid__C_engine/ -j$(sysctl -n hw.ncpu)
```

Sanitizer builds for memory/UB checking:

```bash
LUCID_BUILD_MODE=debug-asan  pip install -e . --no-build-isolation
LUCID_BUILD_MODE=debug-ubsan pip install -e . --no-build-isolation
```

### Static analysis tools

```bash
ruff check lucid/           # Python linting
mypy --strict lucid/        # Python type checking (see mypy.ini for rationale)
bash tools/check_format.sh  # clang-format + clang-tidy for C++
```

---

## 3. Hard rules — non-negotiable

Every one of these rules is enforced in CI. A PR that violates any of them will be rejected without review.

### H1 — No `from __future__ import annotations`

This import is forbidden in every file under `lucid/`. Python 3.14's lazy annotations make it redundant and it interferes with runtime type inspection.

Detection: `grep -r "from __future__ import annotations" lucid/`

### H2 — `lucid._C` imports must use `_C_{name}` aliases

```python
# Correct
from lucid._C import engine as _C_engine

# Wrong — all of these are banned
import lucid._C.engine as engine
import lucid._C.engine as _eng
import lucid._C.engine as _e
import lucid._C.engine as _ce
```

### H3 — CPU = Accelerate only; GPU = MLX only

Never mix backends. The two permitted exceptions are `lucid.linalg` (MLX on CPU stream) and ops with data-dependent output shapes (CPU round-trip). Every other case must use the canonical backend for its stream.

### H4 — No external library imports inside Lucid internals

Lucid is a standalone framework. The only dependencies permitted inside `lucid/` are the C++ engine and the Python standard library. **`numpy`, `scipy`, and any other third-party package are forbidden** in all compute paths:

- `lucid/_ops/composite/`
- `lucid/_tensor/` (except the 6 bridge points below)
- `lucid/nn/`
- `lucid/optim/`
- `lucid/autograd/`
- `lucid/linalg/`
- `lucid/fft/`
- `lucid/signal/`
- `lucid/special/`
- `lucid/distributions/`
- `lucid/einops/`
- `lucid/amp/`
- `lucid/profiler/`

**The 6 permitted bridge boundaries** (and only these):

| # | Location | What it may import |
|---|----------|--------------------|
| 1 | `lucid/_factories/converters.py` | numpy — external tensor → Lucid conversion |
| 2 | `lucid/_tensor/tensor.py` — `.numpy()`, `__dlpack__`, `_to_impl` only | numpy / dlpack |
| 3 | `lucid/_tensor/_repr.py` | numpy — display only |
| 4 | `lucid/_types.py` | typing protocols — no runtime compute |
| 5 | `lucid/serialization/` + `lucid/optim/{optimizer,lbfgs}.py` state_dict paths | numpy — checkpoint serialization only |
| 6 | `lucid/utils/data/dataloader.py` | numpy — external data ingest |

If you see `import numpy` anywhere outside these six locations, it is a violation.

### H5 — No "torch" or "PyTorch" in source

The words "torch" and "PyTorch" are banned from source code, comments, docstrings, and error messages. Use "reference framework" or similar neutral phrasing instead.

**Only exception:** `lucid/test/_fixtures/ref_framework.py` — the test infrastructure is opt-in (`pip install lucid[test]` + `pytest -m parity`) and may use the literal name. All other test files receive the reference framework only through the `ref` fixture.

Detection: `grep -ri "torch\|pytorch" lucid/ --include="*.py" --exclude-dir=test`

### H6 — No "cuda"

Lucid is Apple Silicon only. Use `metal` everywhere. "cuda" is banned.

Detection: `grep -ri "cuda" lucid/`

### H7 — No string type hints

Forward references as string literals are banned. Use a `TYPE_CHECKING` block instead:

```python
# Wrong
def forward(self, x: "Tensor") -> "Tensor": ...

# Correct
from __future__ import annotations  # NO — see H1
# Instead:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

def forward(self, x: Tensor) -> Tensor: ...
```

Python 3.14 lazy annotations make `TYPE_CHECKING`-style imports sufficient for type checkers without string literals.

### H8 — One canonical path per op; no sub-package shortcuts

- `lucid.linalg` ops are accessed only as `lucid.linalg.<name>`.
- `lucid.einops` ops are accessed only as `lucid.einops.<name>`.
- Do not add top-level aliases like `lucid.norm`, `lucid.cross`, `lucid.einsum`, `lucid.vander`.
- Do not add Tensor method shortcuts like `tensor.norm()`, `tensor.cross()`.

Each op has exactly one canonical path. Duplicating it creates maintenance debt.

### H9 — No `*args` / `**kwargs` in `.pyi` stubs

Every function and method in a `.pyi` file must have an explicit signature that matches the implementation 1:1 — parameter names, types, defaults, positional-only `/`, and keyword-only `*` separators.

```python
# Wrong
def add(self, *args: Any, **kwargs: Any) -> Tensor: ...

# Correct
def add(self, other: Tensor | float, /, *, alpha: float = 1.0) -> Tensor: ...
```

The only exception is a genuinely variadic API such as `def cat(*tensors: Tensor) -> Tensor`.

Detection: `grep -rn "\*args\|\*\*kwargs" lucid/ --include="*.pyi"`

---

## 4. Coding conventions

### Python

| Rule | Detail |
|------|--------|
| **S1** | 4-space indent, 100-column lines |
| **S2** | `ruff check lucid/` and `mypy --strict lucid/` must pass |
| **S6** | Only `_wrap` / `_unwrap` in `lucid/_dispatch.py` may cross the `Tensor ↔ TensorImpl` boundary |
| **S7** | Tier 1 (`lucid.*`) exposes only ops, factories, dtypes, grad-control, and sub-packages. `Module`, `Parameter`, `Linear`, `Adam`, `DataLoader`, etc. live in Tier 2 (`lucid.nn.*`, `lucid.optim.*`, etc.) and must not appear in `lucid.__all__` |
| **S8** | Tier 3 (`lucid._*`) is private — never import it from public API |
| **S9** | dtype aliases are module attributes: `lucid.float = lucid.float32` — never shadow Python builtins |
| **S18** | Write no comments by default. Add one only when the **why** is non-obvious: a hidden constraint, a subtle invariant, or a workaround for a specific bug |
| **S19** | Implement only what the task requires. No speculative abstractions. Three similar lines is better than a premature helper |

### C++

| Rule | Detail |
|------|--------|
| **S1** | 4-space indent, brace-attached (Google base), 100-column — enforced by `.clang-format` |
| **S3** | `clang-format --dry-run --Werror` + `clang-tidy` must pass (`bash tools/check_format.sh`) |
| **S10** | Never `throw std::runtime_error` or `std::invalid_argument` directly — use the `LucidError` hierarchy + `ErrorBuilder` chain from `lucid/_C/core/Error.h` |
| **S11** | Every public C++ type must be tagged with the `LUCID_API` macro (default visibility is hidden) |
| **S12** | Every op must register an `OpSchema` (name, version, AMP policy, determinism flag, complexity class) via `lucid/_C/registry/` |
| **S20** | Every in-place op must call `bump_version()` on the tensor; every forward must call `set_saved_versions(...)` — both are required for backward version checks to work |

### Strict type hints everywhere

Every function in `lucid/` (public, private, helper) must annotate every parameter and return type. No bare `def f(x):`. This is verified by `tools/check_stubs.py`.

---

## 5. Architecture invariants

These must hold at all times. A PR that breaks any of them cannot merge.

| # | Invariant |
|---|-----------|
| **A1** | `Tensor._impl` is always a `TensorImpl`. Nothing else may be stored there. |
| **A2** | Direct access to `._impl` is allowed only inside `lucid/_tensor/tensor.py` and `lucid/_dispatch.py`. Everywhere else must go through `_wrap` / `_unwrap`. |
| **A3** | Scalar operands are coerced via `_to_tensor` — this is how `a + 3` works. |
| **A4** | `Parameter` is a `Tensor` subclass — `isinstance(p, Tensor)` is always `True`. |
| **A5** | `Optimizer` extracts `_impl` at construction time. Post-construction parameter changes go through `add_param_group`. |
| **A6** | `Module.training` propagates to all children via `train()`. |
| **A7** | `nn.functional` is stateless — Tensor in, Tensor out. No module state. |
| **A8** | Import order: `_dispatch → _tensor → _ops → _factories → autograd → nn → optim → lucid`. No cycles. |
| **A9** | `state_dict` returns `OrderedDict` with a `_metadata` attribute (PyTorch-compatible). `lucid.save` separates and re-attaches it. |
| **A10** | `_load_from_state_dict` signature: `(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)`. |

---

## 6. Adding a new op

### Composite op (pure Python, no C++ changes)

A composite op is implemented entirely in Python on top of existing engine primitives. Use this path whenever possible.

1. Add the implementation to the appropriate file in `lucid/_ops/composite/`:
   - `elementwise.py`, `reductions.py`, `shape.py`, `blas.py`, `statistics.py`,
     `indexing.py`, `predicates.py`, `dtype.py`, `constants.py`
2. Add the name to that file's `__all__`.
3. The name is automatically included in `COMPOSITE_NAMES` (via `lucid/_ops/composite/__init__.py`) and exposed through the top-level lazy loader.
4. Add an `OpEntry` in `lucid/_ops/_registry.py` with `method_name` and `free_fn_name`.
5. Write a unit test in `lucid/test/unit/` and a parity test in `lucid/test/parity/`.

### Engine op (new C++ op)

Use the scaffolding CLI to generate the boilerplate:

```bash
python tools/new_op.py <OpName> --kind <unary|binary|composite>
# preview without writing:
python tools/new_op.py <OpName> --kind unary --dry-run
```

This generates 9 files automatically. Then:

1. Implement the op in `lucid/_C/ops/<family>/MyOp.{h,cpp}`.
2. Register an `OpSchema` in `lucid/_C/registry/`.
3. Add a pybind11 binding in `lucid/_C/bindings/bind_<family>.cpp`.
4. Update `lucid/_ops/_registry.py` with an `OpEntry`.
5. Update the relevant `.pyi` stub (or run `python tools/gen_pyi.py`).
6. Rebuild the engine: `cmake --build build/temp.macosx-*/lucid__C_engine/ -j$(sysctl -n hw.ncpu)`
7. Write unit + parity tests.

### Parity test requirement

Every new public API **must** ship both:
- A unit test in `lucid/test/unit/` (no reference framework dependency)
- A parity test in `lucid/test/parity/` using the `ref` fixture

```python
# Example parity test
@pytest.mark.parity
def test_my_op_parity(ref):
    x = lucid.tensor([1.0, 2.0, 3.0])
    x_ref = ref.tensor([1.0, 2.0, 3.0])
    assert_close(lucid.my_op(x), ref.my_op(x_ref))
```

---

## 7. Testing

### Test suite layout

```
lucid/test/
├── unit/        Pure Lucid tests — no reference framework
├── nn/          nn.Module / nn.functional
├── autograd/    backward / gradcheck / higher-order
├── linalg/      decomposition correctness
├── parity/      @pytest.mark.parity — numerical parity vs reference framework
├── integration/ @pytest.mark.slow — end-to-end (model training)
└── helpers/     Numerics / parity utilities
```

### Running tests

```bash
# Fast unit tests only (no reference framework needed)
pytest lucid/test/unit/ -q

# Full suite (excluding parity)
pytest lucid/test/ --ignore=lucid/test/parity -q

# Full suite including parity (requires reference framework)
pytest lucid/test/ -q

# C++ Google Test suite
ctest --test-dir build/temp.macosx-*/lucid__C_engine/ --output-on-failure
```

### Test markers

| Marker | Meaning |
|--------|---------|
| `parity` | Numerical parity vs reference framework; auto-skips if not installed |
| `smoke` | Quick sanity check (< 1 s) |
| `slow` | End-to-end, > 5 s (training loops, full model forward) |
| `gpu` | Requires Apple Silicon GPU (Metal); auto-skips otherwise |
| `perf` | `pytest-benchmark` timing test; opt-in via `--benchmark-only` |
| `stability` | Numerical edge cases (inf / nan / subnormal) |
| `f64_only` | Float64-only path; skipped when only float32 is available |

### Quick smoke test (MLP XOR)

```python
import lucid
import lucid.nn as nn
import lucid.optim as optim

m = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())
opt = optim.Adam(m.parameters(), lr=0.01)
X = lucid.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=lucid.float32)
Y = lucid.tensor([[0],[1],[1],[0]], dtype=lucid.float32)
for _ in range(500):
    loss = lucid.nn.functional.mse_loss(m(X), Y)
    opt.zero_grad()
    loss.backward()
    opt.step()
assert loss.item() < 0.05
```

### Phase exit gate

Before any PR can merge, **all of the following must pass**:

```bash
ruff check lucid/
mypy --strict lucid/
pytest lucid/test/ -q
ctest --test-dir build/temp.macosx-*/lucid__C_engine/ --output-on-failure

python tools/check_layers.py      # layer dependency DAG
python tools/check_stubs.py       # .pyi stub freshness
python tools/check_op_api.py      # OpEntry → engine_fn consistency
python tools/check_doxygen.py     # C++ doc coverage
bash   tools/check_format.sh      # clang-format + clang-tidy
bash   scripts/ci_full.sh         # full CI gate
```

---

## 8. Changelog and commits

Lucid uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Adding a changelog entry

Use the helper tool:

```bash
python tools/changelog.py add --section Added "Brief description of the change"
```

Or edit `CHANGELOG.md` manually under the `[Unreleased]` section. Omitting the changelog entry from a PR will block merge.

### Changelog sections

Standard sections from Keep a Changelog (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`) plus two project-specific ones:

- **Performance** — measured speed or memory wins (include numbers)
- **Tooling** — dev-only changes that don't affect runtime (CI, lints, scaffolding)

### Commit style

There is no enforced commit message format, but each commit should be scoped to a single logical change. Phase-level work lands as a series of focused commits, not one giant squash.

---

## 9. Pull request checklist

Before opening a PR, verify every item:

- [ ] Hard rules H1–H9 are not violated (run the detection commands from §3)
- [ ] `ruff check lucid/` passes
- [ ] `mypy --strict lucid/` passes (error count must not increase)
- [ ] `pytest lucid/test/ -q` passes (1,500+ tests)
- [ ] C++ Google Test suite passes (if C++ was modified)
- [ ] `tools/check_layers.py` passes (no new layer dependency violations)
- [ ] `tools/check_stubs.py` passes (stubs are up to date)
- [ ] `tools/check_doxygen.py` passes at 100% if C++ public API was added
- [ ] `bash tools/check_format.sh` passes (if C++ was modified)
- [ ] Unit test added for new public API
- [ ] Parity test added for new public API (via `ref` fixture)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No `torch` / `pytorch` / `cuda` in any new or modified source
- [ ] No `from __future__ import annotations` introduced
- [ ] No `*args` / `**kwargs` in any new `.pyi` signatures
- [ ] No external library imports outside the 6 permitted bridge boundaries

---

## 10. Out-of-scope

The following will not be accepted in this major version:

- Linux / Windows / x86_64 cross-platform support
- CUDA or NCCL distributed training
- Quantization (int8 / int4) — float-only for now
- ONNX export (Lucid's own `.lucid` format is in scope; ONNX is not)
- TorchScript / FX graph
- `torch.compile`-style JIT (deferred to a future phase)
- `__torch_function__` protocol
- Multi-process `DataLoader` (`num_workers > 0`)

PRs adding any of these will be closed.

---

## Questions?

If something in this guide is unclear or contradicts the code, the code is the ground truth. Open an issue and ask — do not guess.
