# Phase 2 TensorImpl Audit

Status: Phase 2.0 baseline.

## Scope

This audit records the current direct-field surface before TensorImpl
encapsulation. The intent is to avoid guessing which accessors and mutators are
actually needed.

## Current Usage

The candidate direct-field scan currently finds:

- Phase 2.0 baseline direct internal field references: 3478
- After Phase 2.1 core/autograd/binding migration: 3415

This is a conservative text scan, so it includes some similarly named Node
fields. It is still useful as a monotonic migration counter while direct
TensorImpl access is being removed.

The dominant read-only TensorImpl fields are:

- `shape_`: shape planning, output construction, scope metadata, backward saved
  shapes.
- `dtype_`: dispatch, allocation, scope metadata, dtype checks.
- `device_`: CPU/GPU dispatch, scope metadata, device checks.
- `stride_`: numpy exposure and contiguity/view logic.
- `storage_`: kernel input, saved inputs for backward, optimizer parameter
  mutation.

The mutation fields are narrower:

- `requires_grad_`: set on outputs when autograd wiring is installed.
- `is_leaf_`: set false on non-leaf outputs.
- `grad_fn_`: installed on outputs; lazily installed as `AccumulateGrad` for
  leaves; cleared by backward when the graph is released.
- `grad_storage_`: accumulated by autograd and reset by `zero_grad()`.
- `version_`: bumped by `copy_from()` and in-place ops; saved by backward nodes
  for mutation checks.

## Accessor Plan

Phase 2 starts with accessors while fields remain public:

- Read accessors: `storage()`, `shape()`, `stride()`, `dtype()`, `device()`,
  `requires_grad()`, `is_leaf()`, `version()`, `grad_fn()`, `grad_storage()`.
- Narrow mutators: `mutable_storage()`, `mutable_grad_storage()`,
  `set_requires_grad()`, `set_leaf()`, `set_grad_fn()`, `clear_grad_fn()`,
  `set_grad_storage()`, `bump_version()`.

The final private-field gate should reject direct access to TensorImpl internals
outside explicitly allowed implementation files.

## Migration Order

1. Replace binding and helper reads with accessors. Completed for
   `bindings/bind_tensor.cpp`, `core/Validate.cpp`, autograd accumulation and
   version checks, and the `Optimizer` base class.
2. Replace op read-only metadata usage (`shape_`, `dtype_`, `device_`) in broad
   mechanical batches.
3. Replace autograd wiring writes with the narrow mutators.
4. Make fields private.
5. Add `tools/check_phase2.py` to CI to keep the boundary closed.
