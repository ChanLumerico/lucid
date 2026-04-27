#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../ops/gfunc/Gfunc.h"
#include "../core/Shape.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

Shape vec_to_shape(const std::vector<std::int64_t>& v) {
    return Shape(v.begin(), v.end());
}

}  // namespace

void register_gfunc(py::module_& m) {
    m.def("zeros",
          [](std::vector<std::int64_t> shape, Dtype dt, Device device,
             bool requires_grad) {
              return zeros_op(vec_to_shape(shape), dt, device, requires_grad);
          },
          py::arg("shape"), py::arg("dtype") = Dtype::F32,
          py::arg("device") = Device::CPU, py::arg("requires_grad") = false);

    m.def("ones",
          [](std::vector<std::int64_t> shape, Dtype dt, Device device,
             bool requires_grad) {
              return ones_op(vec_to_shape(shape), dt, device, requires_grad);
          },
          py::arg("shape"), py::arg("dtype") = Dtype::F32,
          py::arg("device") = Device::CPU, py::arg("requires_grad") = false);

    m.def("full",
          [](std::vector<std::int64_t> shape, double value, Dtype dt,
             Device device, bool requires_grad) {
              return full_op(vec_to_shape(shape), value, dt, device,
                             requires_grad);
          },
          py::arg("shape"), py::arg("fill_value"),
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("empty",
          [](std::vector<std::int64_t> shape, Dtype dt, Device device,
             bool requires_grad) {
              return empty_op(vec_to_shape(shape), dt, device, requires_grad);
          },
          py::arg("shape"), py::arg("dtype") = Dtype::F32,
          py::arg("device") = Device::CPU, py::arg("requires_grad") = false);

    m.def("eye", &eye_op,
          py::arg("N"), py::arg("M") = -1, py::arg("k") = 0,
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("arange", &arange_op,
          py::arg("start"), py::arg("stop"), py::arg("step") = 1.0,
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("linspace", &linspace_op,
          py::arg("start"), py::arg("stop"), py::arg("num") = 50,
          py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
          py::arg("requires_grad") = false);

    m.def("diag", &diag_op, py::arg("v"), py::arg("k") = 0);

    // ----- _like family -----
    m.def("zeros_like", &zeros_like_op,
          py::arg("a"), py::arg("requires_grad") = false);
    m.def("ones_like",  &ones_like_op,
          py::arg("a"), py::arg("requires_grad") = false);
    m.def("empty_like", &empty_like_op,
          py::arg("a"), py::arg("requires_grad") = false);
    m.def("full_like",  &full_like_op,
          py::arg("a"), py::arg("fill_value"),
          py::arg("requires_grad") = false);
}

}  // namespace lucid::bindings
