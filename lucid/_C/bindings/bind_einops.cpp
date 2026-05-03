// lucid/_C/bindings/bind_einops.cpp
//
// Registers einops-style tensor manipulation ops on the `lucid._C.engine.einops`
// sub-module (created in bind.cpp).  The four ops are:
//   - rearrange: reshape / transpose via a string pattern like 'b (h w) c -> b h w c'.
//   - reduce:    rearrange + reduction; axes absent from the rhs are contracted.
//   - repeat:    rearrange with new axes inserted; new axis sizes in axes_lengths.
//   - einsum:    general Einstein summation over an explicit operand list.
//
// The `reduction` argument to `reduce` is accepted as a plain Python string
// ("mean", "sum", "max", "min", "prod") and converted to an integer tag by the
// local parse_einops_reduction() helper before being forwarded to the C++ op.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"
#include "../ops/einops/Einops.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

// Converts a reduction name string to the integer tag expected by
// einops_reduce_op.  Throws a LucidError (via ErrorBuilder) for unknown names
// so that Python callers receive a typed exception rather than a raw crash.
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

// Registers rearrange, reduce, repeat, and einsum on the einops sub-module.
void register_einops(py::module_& m) {
    // rearrange delegates to einops_rearrange_op; axes_lengths is optional and
    // defaults to an empty map for patterns that do not decompose/compose axes.
    m.def(
        "rearrange",
        [](const TensorImplPtr& a, const std::string& pattern,
           const std::map<std::string, std::int64_t>& axes_lengths) {
            return einops_rearrange_op(a, pattern, axes_lengths);
        },
        py::arg("a"), py::arg("pattern"),
        py::arg("axes_lengths") = std::map<std::string, std::int64_t>{},
        "einops-style rearrange — pattern e.g. 'b (h w) c -> b h w c'.");

    // reduce accepts a string reduction name; parse_einops_reduction converts
    // it to the integer tag before calling einops_reduce_op.
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
