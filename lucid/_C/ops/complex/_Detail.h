// lucid/_C/ops/complex/_Detail.h
//
// Internal helpers shared across the complex-viewing ops (``real`` / ``imag``
// / ``complex`` / ``conj``).  Header-only inline helpers — pure dispatch
// glue, no kernels live here.
//
// Unlike the FFT ops these go through ``IBackend`` rather than a dedicated
// kernel registry: each backend provides its own native implementation
// (CPU = Apple Accelerate vDSP / interleaved element walks, GPU =
// ``mlx::core::real`` / ``imag`` / ``conjugate``).  The helpers here
// validate dtypes at op boundaries and re-export the canonical ``fresh``
// TensorImpl constructor that wraps a freshly-allocated ``Storage`` into a
// new tensor with the requested ``shape`` / ``dtype`` / ``device`` metadata.
//
// Dtype mapping used by these ops
// -------------------------------
// Complex view → real projection :  C64 → F32, C128 → F64.
// Real combine → complex result  :  F32 → C64, F64 → C128.
//
// Notes
// -----
// Only ``C64`` is currently dispatched at the op layer; ``C128`` is reserved
// for future support and gated at the validator.

#pragma once

#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"

namespace lucid::complex_detail {

using ::lucid::helpers::fresh;

// Assert that ``dt`` is a complex dtype, raising otherwise.
//
// Used as a precondition guard by ``real_op`` and ``imag_op`` which require a
// complex-valued input.  Calling on a real dtype raises ``NotImplemented``
// from the named op via ``ErrorBuilder``.
//
// Parameters
// ----------
// dt : Dtype
//     Dtype to check.  Only ``Dtype::C64`` is currently accepted.
// op : const char*
//     Name of the calling op, used in the error message.
//
// Raises
// ------
// NotImplemented
//     If ``dt`` is not a complex dtype.
inline void require_complex(Dtype dt, const char* op) {
    if (dt != Dtype::C64)
        ErrorBuilder(op).not_implemented("expected C64 input, got " + std::string(dtype_name(dt)));
}

// Assert that ``dt`` is a real-floating dtype, raising otherwise.
//
// Used by ``complex_combine`` to gate the ``re`` / ``im`` arguments to
// ``complex_op``: only ``F16`` / ``F32`` / ``F64`` are accepted.  Any
// integer, boolean, or complex dtype triggers a ``NotImplemented`` error
// from the named op via ``ErrorBuilder``.
//
// Parameters
// ----------
// dt : Dtype
//     Dtype to check.
// op : const char*
//     Name of the calling op, used in the error message.
//
// Raises
// ------
// NotImplemented
//     If ``dt`` is not one of ``F16`` / ``F32`` / ``F64``.
inline void require_real_float(Dtype dt, const char* op) {
    if (dt != Dtype::F32 && dt != Dtype::F16 && dt != Dtype::F64)
        ErrorBuilder(op).not_implemented("expected real floating dtype, got " +
                                         std::string(dtype_name(dt)));
}

}  // namespace lucid::complex_detail
