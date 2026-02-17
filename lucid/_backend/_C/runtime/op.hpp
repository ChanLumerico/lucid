#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <optional>
#include <string>

namespace py = pybind11;

namespace lucid::backend::runtime {

    struct FuncOpSpec {
        std::optional<std::size_t> n_in = std::nullopt;
        std::optional<std::size_t> n_ret = std::nullopt;

        bool has_gradient = true;
        std::string device = "cpu";
    };

    class FuncOpExecutor {
        public:
            static py::object execute(
                const py::object& op_self,
                const py::object& forward_func,
                const py::tuple& args,
                const py::dict& kwargs,
                const FuncOpSpec& spec
            );
    };
}
