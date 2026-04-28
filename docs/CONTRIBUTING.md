# Contributing to Lucid (C++ engine rebuild branch)

This document is the operational guide for contributing while the rebuild is
in flight. It supplements the user-facing README.

## Pre-flight checklist for every change

```
# 1. Format + static analysis
tools/check_format.sh --tidy

# 2. Layer dependency check
python tools/check_layers.py
python tools/check_op_api.py
python tools/check_phase1.py
python tools/audit_tensorimpl_access.py

# 3. Build (release + UBSan)
python3 -m pip install cmake ninja pybind11
python3 -m pip install -e . --no-build-isolation
LUCID_BUILD_MODE=debug-ubsan python3 -m pip install -e . --no-build-isolation

# 4. Test
pytest tests/parity/ --tb=short -q
./scripts/ci_sanitizer.sh ubsan

# 5. CHANGELOG
# Add an entry under "## Unreleased" — Added / Changed / Fixed / Removed.

# Or run the full gate:
./scripts/ci_full.sh
```

## Phase 0.5 build infrastructure policy

Phase 0.5 installs the production build infrastructure: CMake-backed extension
builds, Apple Silicon-only CI, wheel artifact generation, fresh-venv wheel
import verification, coverage report generation, and placeholder perf jobs.

Coverage thresholds and performance regression blocking are intentionally
report-only at this point. `tools/coverage.sh` prints the planned gates
(`lucid/_C/ops/**` and `lucid/_C/kernel/**` >= 80%, `backend/**` >= 70%,
`core/**` >= 90%) but does not fail by default. Set
`LUCID_COVERAGE_ENFORCE=1` to exercise the hard gate locally. CI should turn
that flag on after Phase 7 expands parity coverage and Phase 8 replaces the
perf placeholder with real benchmarks.

## Phase order

The rebuild proceeds in strict phase order. Don't start a phase before its
predecessor's exit gate is green. Current state is in
`/Users/chanlee/.claude/plans/lucid-c-linked-wigderson.md`.

| Phase | Status | Owner |
|---|---|---|
| 0  Baseline freeze | ✅ | this branch |
| 1  TensorImpl + zero-copy | ✅ | this branch |
| 1.5 Production foundations | ✅ | this branch |
| 2  Autograd engine MVP | ✅ | this branch |
| 2.5 Determinism + concurrency | ✅ | this branch |
| 0.6 Engineering hygiene | ✅ | this branch (this doc) |
| 3  Op families | ⏳ next | this branch |
| 4  Optimizers | ⏳ | future |
| 5  Python migration | ⏳ | future |
| 6  JIT rebuild | ⏳ | future |
| 7  Inference build + C ABI | ⏳ | future |

## Adding a new op (Phase 3+)

See `docs/ARCHITECTURE.md` § "Adding a new op". The summary:

1. Declare in `autograd/ops/<family>/<name>.h` with full contract.
2. Implement forward in `backend/cpu/...` (and `backend/gpu/...` if applicable).
3. Implement backward as a CRTP subclass.
4. Register `OpSchema` (auto via static initializer in the header).
5. Bind to Python in `bindings/bind_ops_<family>.cpp`.
6. Add parity test against current Python Lucid (atol=1e-3).
7. Add `CHANGELOG.md` entry.

## Code review expectations

For C++ files:
- Style is enforced by `.clang-format` — reviewers don't comment on it.
- Layering rule (`docs/ARCHITECTURE.md`) is non-negotiable. PRs that break
  upward includes get rejected.
- Every public function has a Doxygen contract block.
- Every typed exception throw site has the right subclass.
- No new `std::runtime_error`, `std::invalid_argument`, etc. in non-test code.
- Allocations go through `allocate_aligned_bytes`.

For Python:
- Type hints required on all new public APIs.
- No `param.data = ...` writes (Phase 5 will tear these out; new code
  shouldn't add more).

## Branching

`claude/beautiful-cannon-440988` (this branch) tracks the rebuild from
`origin/lucid-3.0`. PRs against this branch until Phase 5.12 ships, then
merge into `lucid-3.0` and tag `v3.0.0-rc1`.

## Filing bugs / questions

For now, this is a single-developer project. Issues / discussion go through
the project devlog (`velog.io/@lumerico284`). When the framework is ready
for external contributors (post-Phase 7), this section will expand to GitHub
issue templates and a contributor licensing agreement.

## Definition of done (per phase)

A phase is "done" when:
1. All exit-gate criteria in the plan file pass.
2. `pytest tests/parity/` is green.
3. UBSan build is green on the parity surface.
4. `CHANGELOG.md` is updated.
5. The plan file's todos for that phase are all `[completed]`.

A phase is **not** "done" merely because the code compiles.
