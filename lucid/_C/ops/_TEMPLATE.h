// =====================================================================
// Lucid op header template — copy this when adding a new op.
// =====================================================================
//
// Replace `MyOp` / `MyOpBackward` / `my_op` / etc. throughout. Delete this
// file from the build (`cpp_extensions.json`) — the template is for reference,
// not compilation.
//
// Required for every op (see docs/CONTRIBUTING.md and docs/ARCHITECTURE.md):
//   1. Doxygen contract block on the public free function.
//   2. CRTP backward node.
//   3. OpSchema registration (Phase 3.0 will add the macro; for now do it by
//      hand via a static-init helper).
//   4. AmpPolicy (Phase 3.5 — for now leave the field, default Promote).
//   5. Determinism declaration (Phase 3.8 enforces).
//   6. Parity test in `lucid/test/parity/cpp_engine/`.
//   7. CHANGELOG.md entry under "Unreleased / Added".

#pragma once

#include <memory>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

/// @op           my_op
/// @schema_v     1
/// @inputs       (a: Tensor<T,*>, b: Tensor<T,*>)  T in {F32, F64}
/// @outputs      (c: Tensor<T,*>)
/// @amp_policy   Promote
/// @determinism  deterministic
/// @complexity   O(numel(out))
///
/// @throws lucid::ShapeMismatch  If `a` and `b` cannot be broadcast together.
/// @throws lucid::DtypeMismatch  If dtypes differ.
/// @throws lucid::DeviceMismatch If devices differ.
///
/// Forward:  c[i] = a[i] OP b[i]
/// Backward: dx = ...,  dy = ...
LUCID_API TensorImplPtr my_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
