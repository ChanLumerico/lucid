#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "op.hpp"

#include <optional>
#include <string>

namespace py = pybind11;

using lucid::backend::runtime::FuncOpExecutor;
using lucid::backend::runtime::FuncOpSpec;

PYBIND11_MODULE(core, m) {
    py::class_<FuncOpSpec>(m, "_C_FuncOpSpec")
        .def(py::init<>())
        .def_readwrite("n_in", &FuncOpSpec::n_in)
        .def_readwrite("n_ret", &FuncOpSpec::n_ret)
        .def_readwrite("has_gradient", &FuncOpSpec::has_gradient)
        .def_readwrite("device", &FuncOpSpec::device);

    m.def(
        "_C_func_op",
        &FuncOpExecutor::execute,
        py::arg("op_self"),
        py::arg("forward_func"),
        py::arg("args"),
        py::arg("kwargs"),
        py::arg("spec")
    );

    m.def(
        "_C_func_op_raw",
        [](
            const py::object& op_self,
            const py::object& forward_func,
            const py::tuple& args,
            const py::dict& kwargs,
            std::optional<std::size_t> n_in,
            std::optional<std::size_t> n_ret,
            bool has_gradient,
            const std::string& device
        ) -> py::object {
            FuncOpSpec spec;
            spec.n_in = n_in;
            spec.n_ret = n_ret;
            spec.has_gradient = has_gradient;
            spec.device = device;

            return FuncOpExecutor::execute(
                op_self,
                forward_func,
                args,
                kwargs,
                spec
            );
        },
        py::arg("op_self"),
        py::arg("forward_func"),
        py::arg("args"),
        py::arg("kwargs"),
        py::arg("n_in") = std::nullopt,
        py::arg("n_ret") = std::nullopt,
        py::arg("has_gradient") = true,
        py::arg("device") = "cpu"
    );
}

