#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "../ops/einops/Einops.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

int parse_einops_reduction(const std::string& reduction) {
    if (reduction == "mean")
        return 1;
    if (reduction == "sum")
        return 2;
    if (reduction == "max")
        return 3;
    if (reduction == "min")
        return 4;
    if (reduction == "prod")
        return 5;
    ErrorBuilder("einops.reduce").fail("unknown reduction '" + reduction + "'");
}

}  // namespace

void register_einops(py::module_& m) {
    m.def(
        "rearrange",
        [](const TensorImplPtr& a, const std::string& pattern,
           const std::map<std::string, std::int64_t>& axes_lengths) {
            return einops_rearrange_op(a, pattern, axes_lengths);
        },
        py::arg("a"), py::arg("pattern"),
        py::arg("axes_lengths") = std::map<std::string, std::int64_t>{},
        "einops-style rearrange — pattern e.g. 'b (h w) c -> b h w c'.");

    m.def(
        "reduce",
        [](const TensorImplPtr& a, const std::string& pattern, const std::string& reduction,
           const std::map<std::string, std::int64_t>& axes_lengths) {
            return einops_reduce_op(a, pattern, parse_einops_reduction(reduction), axes_lengths);
        },
        py::arg("a"), py::arg("pattern"), py::arg("reduction") = std::string("mean"),
        py::arg("axes_lengths") = std::map<std::string, std::int64_t>{},
        "einops-style reduce — collapses axes that disappear from rhs.");

    m.def(
        "repeat",
        [](const TensorImplPtr& a, const std::string& pattern,
           const std::map<std::string, std::int64_t>& axes_lengths) {
            return einops_repeat_op(a, pattern, axes_lengths);
        },
        py::arg("a"), py::arg("pattern"),
        py::arg("axes_lengths") = std::map<std::string, std::int64_t>{},
        "einops-style repeat — inserts and tiles new axes.");

    m.def(
        "einsum",
        [](const std::string& pattern, const std::vector<TensorImplPtr>& operands) {
            return einsum_op(pattern, operands);
        },
        py::arg("pattern"), py::arg("operands"),
        "Einstein summation. C++ pairwise reduce; ellipsis '...' not supported.");
}

}  // namespace lucid::bindings
