# Lucid C++ Engine — Style Guide

Enforced by `.clang-format` (formatting), `.clang-tidy` (semantics), and
`tools/check_layers.py` (layer dependencies). Run `tools/check_format.sh` and
`tools/check_layers.py` before pushing. CI will reject diffs.

## Principles

1. **Read the contract before the implementation.** Every public function /
   class has a Doxygen-style block describing intent, contract, and failure
   modes. The body should be the smallest correct expansion of that contract.
2. **Errors are typed, not stringly.** Throw `lucid::ShapeMismatch`, not
   `std::runtime_error("shape mismatch")`. See `core/Exceptions.h`.
3. **No public `.data` mutation contracts.** Internal storage is owned by
   `TensorImpl`; mutation goes through named methods (`copy_from`,
   `_apply_step`, `zero_grad`, …). Never expose raw pointers from public APIs.
4. **Single direction of dependency.** Higher layers may include lower layers,
   but lower layers may not include higher layers. The enforced order is:
   `bindings -> optim/random/ops -> kernel -> autograd -> backend -> tensor -> core`.
   `registry` is orthogonal: it depends only on `core`, and is consumed by
   `kernel`/`bindings`.

## Naming

| Construct | Convention | Example |
|---|---|---|
| Type / class / struct / enum | `CamelCase` | `TensorImpl`, `Generator` |
| Function / method | `lower_case` | `data_as_python`, `next_uniform_float` |
| Member variable | `lower_case_` (trailing `_`) | `shape_`, `version_` |
| Local variable, parameter | `lower_case` | `nbytes`, `device` |
| Constant / constexpr | `kCamelCase` | `kCpuAlignment` |
| Macro | `LUCID_UPPER_SNAKE` | `LUCID_API` |
| Namespace | `lower_case` | `lucid::bindings` |
| File | match the primary type | `TensorImpl.{h,cpp}` |

## Headers

- `#pragma once` at top, no include guards.
- Order of includes (clang-format enforces):
  1. matching header (`#include "Foo.h"` from `Foo.cpp`)
  2. pybind11 (`<pybind11/...>`)
  3. C++ stdlib (`<vector>`, `<memory>`, …)
  4. C / system (`<stdio.h>`, `<Accelerate/Accelerate.h>`)
  5. project (relative path: `"../core/Dtype.h"`)
- Forward-declare in `core/fwd.h` whenever you only need a pointer/reference;
  keeps compile times sane as the op count grows.

## Documentation

Every public class and free function gets a Doxygen-style block:

```cpp
/// Brief sentence describing the operation's intent.
///
/// @param a   First input. Must have shape matching `b` after broadcasting.
/// @param b   Second input.
/// @returns   The element-wise sum, on the device of `a`.
///
/// @throws lucid::ShapeMismatch   If `a` and `b` cannot be broadcast together.
/// @throws lucid::DtypeMismatch   If `a.dtype != b.dtype`.
///
/// @amp_policy   Promote
/// @determinism  deterministic
/// @complexity   O(numel(out))
TensorImplPtr add_op(const TensorImplPtr& a, const TensorImplPtr& b);
```

Internal helpers (`anonymous namespace`) get a single-line comment.

## Op contract template

For every op landed in Phase 3+:

```cpp
/// @op           name
/// @schema_v     1
/// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64, ...}
/// @outputs      (c: Tensor<T,*>)
/// @amp_policy   Promote | KeepInput | ForceFP32
/// @determinism  deterministic | nondeterministic-fast
/// @complexity   O(numel(out))
/// @throws       (list)
///
/// Forward:  c[i] = a[i] OP b[i]
/// Backward: dx = ...,  dy = ...
```

## Errors

- Throw the most specific `LucidError` subclass.
- The exception message includes the *context* (op name, parameter name).
  No bare `"shape mismatch"` — always `"add_op: shape mismatch"`.
- Never throw across the C ABI boundary (Phase 7); convert to `LUCID_ERR_*`
  return codes there.

## Memory

- All allocations of tensor data go through `lucid::allocate_aligned_bytes()`,
  not raw `new` / `malloc`. The Allocator updates `MemoryTracker` automatically.
- Use `std::shared_ptr` for ownership of `TensorImpl` and `Node`.
- Use `std::weak_ptr` for autograd `Edge` and `AccumulateGrad::leaf_` to break
  cycles.
- Never store raw `T*` from a `shared_ptr` past the call where it's borrowed.

## Threading

See `docs/concurrency.md`. In short:

- One `TensorImpl` per thread for writes; reads can be shared.
- One `Generator` per thread, or hold its `mutex()`.
- `GradMode` is thread-local; don't read across threads.
- `Determinism` is process-global atomic.

## C++ language features

- Target C++20. Use:
  - `std::span` over `T*+size`, where appropriate.
  - `std::variant` for sum types (e.g. `Storage`).
  - `[[nodiscard]]` on factory functions and observers.
  - Concepts where they make a contract checkable.
- Avoid:
  - Raw `new` / `delete` (use smart pointers or RAII wrappers).
  - `using namespace std;` at file scope.
  - Cyclic includes (use `core/fwd.h`).
  - Hidden allocations in hot loops (every op should declare its allocations
    explicitly in its docblock).

## Testing

- C++-level: pytest verification scripts double as integration tests, hitting
  the engine via pybind11. Phase 3+ adds C++ unit tests under
  `lucid/_C/test/` for kernel-only logic.
- Sanitizer CI: `./scripts/ci_sanitizer.sh` (UBSan default; ASan with brew
  Python).
- Full local gate: `./scripts/ci_full.sh` runs release build, parity, UBSan,
  layer lint, compile command generation, clang-format, and clang-tidy.
- C++ builds use `-Wall -Wextra -Wpedantic -Werror` in every build mode, with
  `-Wno-unused-parameter` for explicitly-unused CRTP/backend parameters.

## Forbidden in production code

| Pattern | Why | Use instead |
|---|---|---|
| `throw std::runtime_error(...)` | not catchable per-category | `lucid::LucidError` subclasses |
| `printf` / `std::cout` | not capturable in profiling | (Phase 5+) Lucid logger |
| `std::rand()` | non-deterministic global state | `lucid::Generator` |
| `posix_memalign` direct | bypasses `MemoryTracker` | `allocate_aligned_bytes()` |
| Magic numerics inline | hides intent, hides re-tuning | named `constexpr kFoo` |
