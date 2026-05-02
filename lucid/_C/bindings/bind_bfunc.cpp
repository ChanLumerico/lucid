#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/bfunc/Add.h"
#include "../ops/bfunc/Bitwise.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Div.h"
#include "../ops/bfunc/Dot.h"
#include "../ops/bfunc/Floordiv.h"
#include "../ops/bfunc/Inner.h"
#include "../ops/bfunc/Inplace.h"
#include "../ops/bfunc/Matmul.h"
#include "../ops/bfunc/Maximum.h"
#include "../ops/bfunc/Minimum.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Outer.h"
#include "../ops/bfunc/Pow.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/bfunc/Tensordot.h"
#include "BindingGen.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_bfunc(py::module_& m) {
    // ----- Element-wise arithmetic (BindingGen binary) -----
    bind_binary<AddBackward>(m, &add_op, "Element-wise a + b (vDSP_vadd).");
    bind_binary<SubBackward>(m, &sub_op, "Element-wise a - b (vDSP_vsub).");
    bind_binary<MulBackward>(m, &mul_op, "Element-wise a * b (vDSP_vmul).");
    bind_binary<DivBackward>(m, &div_op, "Element-wise a / b (vDSP_vdiv).");
    bind_binary<PowBackward>(m, &pow_op, "Element-wise a ** b (vForce vvpowf).");
    bind_binary<MaximumBackward>(m, &maximum_op, "Element-wise max(a, b) (vDSP_vmax).");
    bind_binary<MinimumBackward>(m, &minimum_op, "Element-wise min(a, b) (vDSP_vmin).");

    // ----- Matmul (custom signature) -----
    m.def("matmul", &matmul_op, py::arg("a"), py::arg("b"),
          "2-D matrix multiply a @ b (cblas_sgemm/dgemm via Apple AMX).");

    // ----- Comparisons -----
    m.def("equal", &equal_op, py::arg("a"), py::arg("b"));
    m.def("not_equal", &not_equal_op, py::arg("a"), py::arg("b"));
    m.def("greater", &greater_op, py::arg("a"), py::arg("b"));
    m.def("greater_equal", &greater_equal_op, py::arg("a"), py::arg("b"));
    m.def("less", &less_op, py::arg("a"), py::arg("b"));
    m.def("less_equal", &less_equal_op, py::arg("a"), py::arg("b"));

    // ----- Bitwise -----
    m.def("bitwise_and", &bitwise_and_op, py::arg("a"), py::arg("b"));
    m.def("bitwise_or", &bitwise_or_op, py::arg("a"), py::arg("b"));
    m.def("bitwise_xor", &bitwise_xor_op, py::arg("a"), py::arg("b"));

    // ----- Linear-algebra contractions -----
    m.def("dot", &dot_op, py::arg("a"), py::arg("b"));
    m.def("inner", &inner_op, py::arg("a"), py::arg("b"));
    m.def("outer", &outer_op, py::arg("a"), py::arg("b"));
    m.def("tensordot", &tensordot_op, py::arg("a"), py::arg("b"), py::arg("axes_a"),
          py::arg("axes_b"));

    // ----- Floor division -----
    m.def("floordiv", &floordiv_op, py::arg("a"), py::arg("b"),
          "Element-wise floor(a / b). Output dtype is Int64.");

    // ----- In-place variants -----
    m.def("add_", &add_inplace_op, py::arg("a"), py::arg("b"));
    m.def("sub_", &sub_inplace_op, py::arg("a"), py::arg("b"));
    m.def("mul_", &mul_inplace_op, py::arg("a"), py::arg("b"));
    m.def("div_", &div_inplace_op, py::arg("a"), py::arg("b"));
    m.def("pow_", &pow_inplace_op, py::arg("a"), py::arg("b"));
    m.def("maximum_", &maximum_inplace_op, py::arg("a"), py::arg("b"));
    m.def("minimum_", &minimum_inplace_op, py::arg("a"), py::arg("b"));

    m.def("test_add", &add_op, py::arg("a"), py::arg("b"),
          "Alias for `add`; kept for Phase 2 test scripts.");
}

}  // namespace lucid::bindings
