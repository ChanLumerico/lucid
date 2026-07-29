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
#include "../ops/diffeq/BroydenProbe.h"
#include "../ops/diffeq/RkCombine.h"
#include "../ops/diffeq/RkErrorNorm.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers rk_combine on the diffeq sub-module.
void register_diffeq(py::module_& m) {
    // Coefficients are host doubles rather than tensors on purpose: a Butcher
    // tableau lives on the host, so passing it as tensors would force a
    // device synchronisation on every stage of every step.
    m.def("rk_combine", &rk_combine_op, py::arg("y0"), py::arg("ks"), py::arg("coeffs"),
          py::arg("dt"), "y0 + dt * sum_i coeffs[i] * ks[i] (elementwise, all inputs same shape).");

    // Returns a host float, not a tensor: the step controller branches on
    // this value, so it has to cross to the host either way.  Producing it
    // in one kernel makes that exactly one sync per step.
    m.def("rk_error_norm", &rk_error_norm_op, py::arg("y0"), py::arg("y1"), py::arg("ks"),
          py::arg("coeffs"), py::arg("dt"), py::arg("rtol"), py::arg("atol"),
          "RMS norm of (dt * sum_i coeffs[i] * ks[i]) / (atol + rtol * max(|y0|, |y1|)).");

    // Returned as a tuple of host floats for the same reason: an implicit
    // method's quasi-Newton iteration branches on all three every pass, and
    // computing them together makes that one device synchronisation instead
    // of three.
    m.def(
        "broyden_probe",
        [](const TensorImplPtr& residual, const TensorImplPtr& step, const TensorImplPtr& info) {
            const BroydenProbeResult r = broyden_probe_op(residual, step, info);
            return py::make_tuple(r.residual_sq, r.step_sq, r.info);
        },
        py::arg("residual"), py::arg("step"), py::arg("info") = py::none(),
        "(sum(residual^2), sum(step^2), info) for one Broyden iteration, in one pass.");
}

}  // namespace lucid::bindings
