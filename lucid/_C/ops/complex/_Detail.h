// lucid/_C/ops/complex/_Detail.h
//
// Internal helpers shared across the complex-viewing ops (real / imag /
// complex / conj).  Header-only inline helpers — pure dispatch glue.
//
// Unlike the FFT ops these go through ``IBackend`` (each backend provides
// its own native implementation: CPU = Apple Accelerate vDSP / interleaved
// element walks, GPU = ``mlx::core::real`` / ``imag`` / ``conjugate``).

#pragma once

#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"

namespace lucid::complex_detail {

using ::lucid::helpers::fresh;

// Reject any dtype that isn't C64 (complex64) for ops that require a
// complex input (``real``, ``imag``).
inline void require_complex(Dtype dt, const char* op) {
    if (dt != Dtype::C64)
        ErrorBuilder(op).not_implemented("expected C64 input, got " + std::string(dtype_name(dt)));
}

// Reject any dtype that isn't real (F32 / F16 / F64) — used by
// ``complex_combine`` which builds C64 from two real tensors.
inline void require_real_float(Dtype dt, const char* op) {
    if (dt != Dtype::F32 && dt != Dtype::F16 && dt != Dtype::F64)
        ErrorBuilder(op).not_implemented("expected real floating dtype, got " +
                                         std::string(dtype_name(dt)));
}

}  // namespace lucid::complex_detail
