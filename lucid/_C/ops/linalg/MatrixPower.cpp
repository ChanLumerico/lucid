// lucid/_C/ops/linalg/MatrixPower.cpp
//
// Implementation of the integer matrix power forward op.
//
// This file is intentionally thin: all algorithmic logic (repeated squaring,
// identity short-circuit, inversion for negative p) lives in the backend
// implementation (IBackend::linalg_matrix_power).
//
// CPU path: backend/cpu/ handles repeated squaring using CBLAS matrix
//           multiply and, for p < 0, LAPACK dgetrf/dgetri for the inverse.
// GPU path: mlx::core matrix multiply with repeated squaring.
//
// Autograd: not wired.  The output is a leaf node in the gradient graph.

#include "MatrixPower.h"

#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

// Compute A^p for square matrix a.
//
// The output has the same shape as a (matrix powers preserve shape).
// The local variable is named p to match the mathematical convention A^p;
// the public API parameter is named n for numpy.linalg.matrix_power parity.
// Both name the same integer exponent; they are identical at the call site.
TensorImplPtr matrix_power_op(const TensorImplPtr& a, int p) {
    using namespace linalg_detail;
    Validator::input(a, "matrix_power.a").non_null();
    require_float(a->dtype(), "matrix_power");
    require_square_2d(a->shape(), "matrix_power");
    OpScopeFull scope{"matrix_power", a->device(), a->dtype(), a->shape()};

    Storage out = backend::Dispatcher::for_device(a->device())
                      .linalg_matrix_power(a->storage(), a->shape(), p, a->dtype());
    // Output shape equals input shape: A^p is always the same size as A.
    return fresh(std::move(out), a->shape(), a->dtype(), a->device());
}

}  // namespace lucid
