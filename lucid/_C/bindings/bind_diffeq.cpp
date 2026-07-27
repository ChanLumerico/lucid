// lucid/_C/bindings/bind_diffeq.cpp
//
// Registers differential-equation primitives on the `lucid._C.engine.diffeq`
// sub-module (created in bind.cpp).  Exactly one op lives here:
//   - rk_combine: y0 + dt * sum_i coeffs[i] * ks[i], the affine form every
//                 explicit Runge-Kutta stage and update reduces to.
//
// The integration loop itself stays in Python (lucid/diffeq/_solvers.py) — the
// right-hand side f(t, y) is a Python callable, and driving the loop from C++
// would make the ops layer call back up into Python, inverting the layer DAG.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/diffeq/RkCombine.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers rk_combine on the diffeq sub-module.
void register_diffeq(py::module_& m) {
    // Coefficients are host doubles rather than tensors on purpose: a Butcher
    // tableau lives on the host, so passing it as tensors would force a
    // device synchronisation on every stage of every step.
    m.def("rk_combine", &rk_combine_op, py::arg("y0"), py::arg("ks"), py::arg("coeffs"),
          py::arg("dt"), "y0 + dt * sum_i coeffs[i] * ks[i] (elementwise, all inputs same shape).");
}

}  // namespace lucid::bindings
