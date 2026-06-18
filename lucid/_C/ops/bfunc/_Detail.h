// lucid/_C/ops/bfunc/_Detail.h
//
// Internal utilities shared across every binary-op translation unit in
// ``ops/bfunc/``.  Nothing in this header is part of the public API; it
// is included only by the ``*.cpp`` files inside ``ops/bfunc/``.
//
// The header has two responsibilities:
//
//   1. Provide the :func:`validate_pair` input-validation helper that every
//      binary forward op calls at entry to enforce dtype / device consistency
//      with uniform error messages.  (Shape consistency is handled by NumPy
//      broadcasting — see :func:`bfunc_detail::broadcast_pair` in _Broadcast.h —
//      not enforced here.)
//   2. Re-export the :func:`helpers::allocate_cpu` and :func:`helpers::fresh`
//      convenience functions under the :namespace:`bfunc_detail`
//      namespace so each translation unit can write
//      ``bfunc_detail::allocate_cpu(...)`` instead of repeating a
//      ``using ::lucid::helpers::allocate_cpu;`` declaration locally.
//
// Notes
// -----
// All symbols live in :namespace:`lucid::bfunc_detail` and are
// ``inline``, so multiple translation units may include this header
// without ODR violations.

#pragma once

#include <cstring>
#include <stdexcept>
#include <variant>

#include "../../api.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::bfunc_detail {

using ::lucid::helpers::allocate_cpu;
using ::lucid::helpers::fresh;

// Validate that two binary-op operands have matching dtype and device.
//
// Asserts non-null pointers, identical :enum:`Dtype`, and identical
// :enum:`Device` for the inputs.  This is the lightweight validator
// used by ops that *do* support broadcasting (e.g. :func:`add_op`,
// :func:`mul_op`) — shape compatibility is then checked later by the
// broadcast logic itself.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Left operand.  Must be non-null.
// b : TensorImplPtr
//     Right operand.  Must be non-null.
// op : const char*
//     Caller's op name, embedded verbatim in every error message for
//     easier triage (e.g. ``"add"``, ``"mul"``).
//
// Raises
// ------
// RuntimeError
//     If either ``a`` or ``b`` is null — raised through :class:`ErrorBuilder`.
// DtypeMismatch
//     If ``a->dtype() != b->dtype()`` (no implicit dtype promotion).
// DeviceMismatch
//     If ``a->device() != b->device()`` (no implicit cross-device move).
//
// See Also
inline void validate_pair(const TensorImplPtr& a, const TensorImplPtr& b, const char* op) {
    if (!a || !b)
        ErrorBuilder(op).fail("null input");
    if (a->dtype() != b->dtype())
        throw DtypeMismatch(std::string(dtype_name(a->dtype())),
                            std::string(dtype_name(b->dtype())), std::string(op));
    if (a->device() != b->device())
        throw DeviceMismatch(std::string(device_name(a->device())),
                             std::string(device_name(b->device())), std::string(op));
}

}  // namespace lucid::bfunc_detail
