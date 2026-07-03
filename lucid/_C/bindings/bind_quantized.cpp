// lucid/_C/bindings/bind_quantized.cpp
//
// Registers the low-precision GEMM primitives on the
// `lucid._C.engine.quantized` sub-module (created in bind.cpp).  These wrap
// MLX's group-wise affine quantization kernels and run on the GPU stream only.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/quantized/QuantizedMatmul.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_quantized(py::module_& m) {
    m.def("quantize", &quantize_op, py::arg("w"), py::arg("group_size") = 64,
          py::arg("bits") = 4,
          "Group-wise quantize a float weight → [packed_weight, scales, biases].");
    m.def("dequantize", &dequantize_op, py::arg("w"), py::arg("scales"),
          py::arg("biases") = TensorImplPtr{}, py::arg("group_size") = 64,
          py::arg("bits") = 4, "Reconstruct a float weight from its packed form.");
    m.def("quantized_matmul", &quantized_matmul_op, py::arg("x"), py::arg("w"),
          py::arg("scales"), py::arg("biases") = TensorImplPtr{},
          py::arg("transpose") = true, py::arg("group_size") = 64, py::arg("bits") = 4,
          "x @ packed_w using MLX's low-precision GEMM (GPU only).");
}

}  // namespace lucid::bindings
