// lucid/_C/ops/linalg/SolveTriangular.cpp
//
// Implements solve_triangular_op via IBackend::linalg_solve_triangular()
// → LAPACK strtrs_/dtrtrs_.

#include "SolveTriangular.h"

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

TensorImplPtr solve_triangular_op(const TensorImplPtr& a,
                                  const TensorImplPtr& b,
                                  bool upper,
                                  bool unitriangular) {
    using namespace linalg_detail;
    Validator::input(a, "solve_triangular.a").float_only().square_2d();
    Validator::input(b, "solve_triangular.b").float_only();
    if (a->dtype() != b->dtype())
        ErrorBuilder("solve_triangular").fail("A and b must have the same dtype");
    if (a->device() != b->device())
        ErrorBuilder("solve_triangular").fail("A and b must be on the same device");

    auto out_storage = backend::Dispatcher::for_device(a->device())
                           .linalg_solve_triangular(a->storage(), b->storage(), a->shape(),
                                                    b->shape(), upper, unitriangular, a->dtype());

    return fresh(std::move(out_storage), b->shape(), b->dtype(), b->device());
}

}  // namespace lucid
