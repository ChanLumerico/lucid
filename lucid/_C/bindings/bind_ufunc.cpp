#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Element-wise unary
#include "../ops/ufunc/Activation.h"
#include "../ops/ufunc/Arith.h"
#include "../ops/ufunc/Discrete.h"
#include "../ops/ufunc/Exponential.h"
#include "../ops/ufunc/Hyperbolic.h"
#include "../ops/ufunc/ScalarParam.h"
#include "../ops/ufunc/Softmax.h"
#include "../ops/ufunc/Trig.h"
// Reductions
#include "../ops/ufunc/Reductions.h"
// Transpose / swapaxes (axis-permutation forms in ufunc.py)
#include "../ops/ufunc/Transpose.h"
// Variance / trace / scans
#include "../ops/ufunc/Var.h"
#include "../ops/ufunc/Trace.h"
#include "../ops/ufunc/Scan.h"
// In-place variants
#include "../ops/ufunc/Inplace.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_ufunc(py::module_& m) {
    // ----- Arithmetic -----
    m.def("neg",        &neg_op,        py::arg("a"));
    m.def("abs",        &abs_op,        py::arg("a"));
    m.def("sign",       &sign_op,       py::arg("a"));
    m.def("reciprocal", &reciprocal_op, py::arg("a"));
    m.def("square",     &square_op,     py::arg("a"));
    m.def("cube",       &cube_op,       py::arg("a"));

    // ----- Exponential / log -----
    m.def("exp",        &exp_op,        py::arg("a"));
    m.def("log",        &log_op,        py::arg("a"));
    m.def("log2",       &log2_op,       py::arg("a"));
    m.def("sqrt",       &sqrt_op,       py::arg("a"));

    // ----- Trig -----
    m.def("sin",        &sin_op,        py::arg("a"));
    m.def("cos",        &cos_op,        py::arg("a"));
    m.def("tan",        &tan_op,        py::arg("a"));
    m.def("arcsin",     &arcsin_op,     py::arg("a"));
    m.def("arccos",     &arccos_op,     py::arg("a"));
    m.def("arctan",     &arctan_op,     py::arg("a"));

    // ----- Hyperbolic -----
    m.def("sinh",       &sinh_op,       py::arg("a"));
    m.def("cosh",       &cosh_op,       py::arg("a"));
    m.def("tanh",       &tanh_op,       py::arg("a"));

    // ----- Activation -----
    m.def("relu",         &relu_op,         py::arg("a"));
    m.def("sigmoid",      &sigmoid_op,      py::arg("a"));
    m.def("silu",         &silu_op,         py::arg("a"));
    m.def("gelu",         &gelu_op,         py::arg("a"));
    m.def("leaky_relu",   &leaky_relu_op,   py::arg("a"), py::arg("slope") = 0.01);
    m.def("softplus",     &softplus_op,     py::arg("a"));
    m.def("softmax",      &softmax_op,      py::arg("a"), py::arg("axis") = -1);
    m.def("elu",          &elu_op,          py::arg("a"), py::arg("alpha") = 1.0);
    m.def("selu",         &selu_op,         py::arg("a"));
    m.def("mish",         &mish_op,         py::arg("a"));
    m.def("hard_sigmoid", &hard_sigmoid_op, py::arg("a"));
    m.def("hard_swish",   &hard_swish_op,   py::arg("a"));
    m.def("relu6",        &relu6_op,        py::arg("a"));

    // ----- Scalar-parameterized -----
    m.def("pow_scalar",  &pow_scalar_op,  py::arg("a"), py::arg("exp"));
    m.def("rpow_scalar", &rpow_scalar_op, py::arg("base"), py::arg("a"));
    m.def("clip",        &clip_op,        py::arg("a"), py::arg("min"), py::arg("max"));

    // ----- Discrete (no grad) -----
    m.def("round",      &round_op,      py::arg("a"));
    m.def("floor",      &floor_op,      py::arg("a"));
    m.def("ceil",       &ceil_op,       py::arg("a"));
    m.def("invert",     &invert_op,     py::arg("a"));

    // ----- Reductions (collapse axes) -----
    m.def("sum",  &sum_op,  py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false,
          "Reduce-sum along given axes (empty = all).");
    m.def("mean", &mean_op, py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false,
          "Reduce-mean.");
    m.def("prod", &prod_op, py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false,
          "Reduce-product.");
    m.def("max",  &max_op,  py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false,
          "Reduce-max.");
    m.def("min",  &min_op,  py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false,
          "Reduce-min.");

    // ----- Axis-permutation forms (transpose / T / mT / swapaxes) -----
    m.def("permute",   &permute_op,   py::arg("a"), py::arg("perm"),
          "General axis permutation.");
    m.def("transpose", &transpose_op, py::arg("a"),
          "Reverse all axes (alias for `T`).");
    m.def("T",         &T_op,         py::arg("a"),
          "Reverse all axes.");
    m.def("mT",        &mT_op,        py::arg("a"),
          "Swap last two axes (matrix transpose).");
    m.def("swapaxes",  &swapaxes_op,
          py::arg("a"), py::arg("axis1"), py::arg("axis2"),
          "Swap two axes.");

    // ----- Phase 4c: var / trace / cumsum / cumprod -----
    m.def("var", &var_op, py::arg("a"),
          py::arg("axes") = std::vector<int>{}, py::arg("keepdims") = false,
          "Variance reduction (sample variance, ddof=0).");
    m.def("trace", &trace_op, py::arg("a"),
          "Sum of the main diagonal (last 2 axes).");
    m.def("cumsum",  &cumsum_op,  py::arg("a"), py::arg("axis") = -1);
    m.def("cumprod", &cumprod_op, py::arg("a"), py::arg("axis") = -1);

    // ----- In-place unary variants -----
    m.def("neg_",        &neg_inplace_op,        py::arg("a"));
    m.def("abs_",        &abs_inplace_op,        py::arg("a"));
    m.def("sign_",       &sign_inplace_op,       py::arg("a"));
    m.def("reciprocal_", &reciprocal_inplace_op, py::arg("a"));
    m.def("square_",     &square_inplace_op,     py::arg("a"));
    m.def("cube_",       &cube_inplace_op,       py::arg("a"));
    m.def("exp_",        &exp_inplace_op,        py::arg("a"));
    m.def("log_",        &log_inplace_op,        py::arg("a"));
    m.def("log2_",       &log2_inplace_op,       py::arg("a"));
    m.def("sqrt_",       &sqrt_inplace_op,       py::arg("a"));
    m.def("sin_",        &sin_inplace_op,        py::arg("a"));
    m.def("cos_",        &cos_inplace_op,        py::arg("a"));
    m.def("tan_",        &tan_inplace_op,        py::arg("a"));
    m.def("arcsin_",     &arcsin_inplace_op,     py::arg("a"));
    m.def("arccos_",     &arccos_inplace_op,     py::arg("a"));
    m.def("arctan_",     &arctan_inplace_op,     py::arg("a"));
    m.def("sinh_",       &sinh_inplace_op,       py::arg("a"));
    m.def("cosh_",       &cosh_inplace_op,       py::arg("a"));
    m.def("tanh_",       &tanh_inplace_op,       py::arg("a"));
    m.def("round_",      &round_inplace_op,      py::arg("a"));
    m.def("floor_",      &floor_inplace_op,      py::arg("a"));
    m.def("ceil_",       &ceil_inplace_op,       py::arg("a"));
    m.def("clip_",       &clip_inplace_op,
          py::arg("a"), py::arg("min"), py::arg("max"));
}

}  // namespace lucid::bindings
