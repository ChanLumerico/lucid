// lucid/_C/bindings/bind_einops.cpp
//
// Registers ``einsum`` on the `lucid._C.engine.einops` sub-module (created in
// bind.cpp).  ``rearrange``, ``reduce``, and ``repeat`` lower in Python
// (``lucid/einops/``) directly to primitive view/reduction ops.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/einops/Einops.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers einsum on the einops sub-module.
void register_einops(py::module_& m) {
    // einsum accepts a variadic operand list as a Python list of tensors.
    m.def(
        "einsum",
        [](const std::string& pattern, const std::vector<TensorImplPtr>& operands) {
            return einsum_op(pattern, operands);
        },
        py::arg("pattern"), py::arg("operands"),
        "Einstein summation. C++ pairwise reduce; ellipsis '...' not supported.");
}

}  // namespace lucid::bindings
