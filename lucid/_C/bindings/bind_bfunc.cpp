// lucid/_C/bindings/bind_bfunc.cpp
//
// Registers all binary tensor operations on the top-level engine module.
// Binary ops fall into four groups:
//   1. Differentiable element-wise arithmetic (add, sub, mul, div, pow,
//      maximum, minimum) — registered via the bind_binary<> helper which reads
//      the Python name from BackwardNode::schema_v1.
//   2. matmul and contraction ops (dot, inner, outer, tensordot) — registered
//      individually because their signatures differ.
//   3. Comparison ops (equal, not_equal, greater, etc.) — no backward node;
//      output is always Bool dtype.
//   4. Bitwise ops (bitwise_and, bitwise_or, bitwise_xor) — integer tensors only.
//   5. In-place variants (add_, sub_, ...) — mutate `a` in-place, returning it;
//      the trailing underscore follows reference convention.

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

// Registers all binary ops on the module.
void register_bfunc(py::module_& m) {
    // Differentiable element-wise arithmetic ops.  bind_binary reads the Python
    // name from each BackwardNode's schema so op name and registry entry stay
    // in sync without manual string literals.
    bind_binary<AddBackward>(m, &add_op, "Element-wise a + b (vDSP_vadd).");
    bind_binary<SubBackward>(m, &sub_op, "Element-wise a - b (vDSP_vsub).");
    bind_binary<MulBackward>(m, &mul_op, "Element-wise a * b (vDSP_vmul).");
    bind_binary<DivBackward>(m, &div_op, "Element-wise a / b (vDSP_vdiv).");
    bind_binary<PowBackward>(m, &pow_op, "Element-wise a ** b (vForce vvpowf).");
    bind_binary<MaximumBackward>(m, &maximum_op, "Element-wise max(a, b) (vDSP_vmax).");
    bind_binary<MinimumBackward>(m, &minimum_op, "Element-wise min(a, b) (vDSP_vmin).");

    // matmul is registered manually because its C++ op does not follow the
    // standard binary signature (it handles batched dims internally).
    m.def("matmul", &matmul_op, py::arg("a"), py::arg("b"),
          "2-D matrix multiply a @ b (cblas_sgemm/dgemm via Apple AMX).");

    // Comparison ops return Bool tensors and have no autograd support.
    m.def("equal", &equal_op, py::arg("a"), py::arg("b"));
    m.def("not_equal", &not_equal_op, py::arg("a"), py::arg("b"));
    m.def("greater", &greater_op, py::arg("a"), py::arg("b"));
    m.def("greater_equal", &greater_equal_op, py::arg("a"), py::arg("b"));
    m.def("less", &less_op, py::arg("a"), py::arg("b"));
    m.def("less_equal", &less_equal_op, py::arg("a"), py::arg("b"));

    // Bitwise ops operate on integer-dtype tensors only.
    m.def("bitwise_and", &bitwise_and_op, py::arg("a"), py::arg("b"));
    m.def("bitwise_or", &bitwise_or_op, py::arg("a"), py::arg("b"));
    m.def("bitwise_xor", &bitwise_xor_op, py::arg("a"), py::arg("b"));

    // Contraction ops with non-standard signatures.
    m.def("dot", &dot_op, py::arg("a"), py::arg("b"));
    m.def("inner", &inner_op, py::arg("a"), py::arg("b"));
    m.def("outer", &outer_op, py::arg("a"), py::arg("b"));
    m.def("tensordot", &tensordot_op, py::arg("a"), py::arg("b"), py::arg("axes_a"),
          py::arg("axes_b"));

    m.def("floordiv", &floordiv_op, py::arg("a"), py::arg("b"),
          "Element-wise floor(a / b). Output dtype is Int64.");

    // In-place variants mutate `a` and return it.  They bypass the autograd
    // graph (version counter is bumped to invalidate saved references).
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
