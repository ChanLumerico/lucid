#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BindingGen.h"

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
// Transpose / swapaxes
#include "../ops/ufunc/Transpose.h"
// Variance / trace / scans
#include "../ops/ufunc/Scan.h"
#include "../ops/ufunc/Trace.h"
#include "../ops/ufunc/Var.h"
// In-place variants
#include "../core/TensorImpl.h"
#include "../ops/ufunc/Inplace.h"

namespace py = pybind11;

namespace lucid::bindings {

void register_ufunc(py::module_& m) {
    // ----- Arithmetic (BindingGen unary) -----
    bind_unary<NegBackward>(m, &neg_op);
    bind_unary<AbsBackward>(m, &abs_op);
    bind_unary<SignBackward>(m, &sign_op);
    bind_unary<ReciprocalBackward>(m, &reciprocal_op);
    bind_unary<SquareBackward>(m, &square_op);
    bind_unary<CubeBackward>(m, &cube_op);

    // ----- Exponential / log -----
    bind_unary<ExpBackward>(m, &exp_op);
    bind_unary<LogBackward>(m, &log_op);
    bind_unary<Log2Backward>(m, &log2_op);
    bind_unary<SqrtBackward>(m, &sqrt_op);

    // ----- Trig -----
    bind_unary<SinBackward>(m, &sin_op);
    bind_unary<CosBackward>(m, &cos_op);
    bind_unary<TanBackward>(m, &tan_op);
    bind_unary<AsinBackward>(m, &arcsin_op);
    bind_unary<AcosBackward>(m, &arccos_op);
    bind_unary<AtanBackward>(m, &arctan_op);

    // ----- Hyperbolic -----
    bind_unary<SinhBackward>(m, &sinh_op);
    bind_unary<CoshBackward>(m, &cosh_op);
    bind_unary<TanhBackward>(m, &tanh_op);

    // ----- Activation -----
    bind_unary<ReluBackward>(m, &relu_op);
    bind_unary<SigmoidBackward>(m, &sigmoid_op);
    bind_unary<SiluBackward>(m, &silu_op);
    bind_unary<GeluBackward>(m, &gelu_op);
    bind_unary_extra<LeakyReluBackward>(m, &leaky_relu_op, py::arg("slope") = 0.01);
    bind_unary<SoftplusBackward>(m, &softplus_op);
    bind_unary_extra<EluBackward>(m, &elu_op, py::arg("alpha") = 1.0);
    bind_unary<SeluBackward>(m, &selu_op);
    bind_unary<MishBackward>(m, &mish_op);
    bind_unary<HardSigmoidBackward>(m, &hard_sigmoid_op);
    bind_unary<HardSwishBackward>(m, &hard_swish_op);
    bind_unary<Relu6Backward>(m, &relu6_op);

    // ----- Softmax (has axis arg) -----
    m.def("softmax", &softmax_op, py::arg("a"), py::arg("axis") = -1);

    // ----- Scalar-parameterized (non-standard signatures) -----
    m.def("pow_scalar", &pow_scalar_op, py::arg("a"), py::arg("exp"));
    m.def("rpow_scalar", &rpow_scalar_op, py::arg("base"), py::arg("a"));
    m.def("clip", &clip_op, py::arg("a"), py::arg("min"), py::arg("max"));

    // ----- Discrete (BindingGen unary) -----
    bind_unary<RoundBackward>(m, &round_op);
    bind_unary<FloorBackward>(m, &floor_op);
    bind_unary<CeilBackward>(m, &ceil_op);
    bind_unary<InvertBackward>(m, &invert_op);

    // ----- Reductions (BindingGen reduce) -----
    bind_reduce<SumBackward>(m, &sum_op, "Reduce-sum along given axes (empty = all).");
    bind_reduce<MeanBackward>(m, &mean_op, "Reduce-mean.");
    bind_reduce<ProdBackward>(m, &prod_op, "Reduce-product.");
    bind_reduce<MaxBackward>(m, &max_op, "Reduce-max.");
    bind_reduce<MinBackward>(m, &min_op, "Reduce-min.");

    // ----- Axis-permutation forms -----
    m.def("permute", &permute_op, py::arg("a"), py::arg("perm"), "General axis permutation.");
    m.def("transpose", &transpose_op, py::arg("a"), "Reverse all axes (alias for `T`).");
    m.def("T", &T_op, py::arg("a"), "Reverse all axes.");
    m.def("mT", &mT_op, py::arg("a"), "Swap last two axes (matrix transpose).");
    m.def("swapaxes", &swapaxes_op, py::arg("a"), py::arg("axis1"), py::arg("axis2"),
          "Swap two axes.");

    // ----- Var / trace / cumsum / cumprod -----
    m.def("var", &var_op, py::arg("a"), py::arg("axes") = std::vector<int>{},
          py::arg("keepdims") = false, "Variance reduction (sample variance, ddof=0).");
    m.def("trace", &trace_op, py::arg("a"), "Sum of the main diagonal (last 2 axes).");
    m.def("cumsum", &cumsum_op, py::arg("a"), py::arg("axis") = -1);
    m.def("cumprod", &cumprod_op, py::arg("a"), py::arg("axis") = -1);

    // ----- In-place unary variants -----
    m.def("neg_", &neg_inplace_op, py::arg("a"));
    m.def("abs_", &abs_inplace_op, py::arg("a"));
    m.def("sign_", &sign_inplace_op, py::arg("a"));
    m.def("reciprocal_", &reciprocal_inplace_op, py::arg("a"));
    m.def("square_", &square_inplace_op, py::arg("a"));
    m.def("cube_", &cube_inplace_op, py::arg("a"));
    m.def("exp_", &exp_inplace_op, py::arg("a"));
    m.def("log_", &log_inplace_op, py::arg("a"));
    m.def("log2_", &log2_inplace_op, py::arg("a"));
    m.def("sqrt_", &sqrt_inplace_op, py::arg("a"));
    m.def("sin_", &sin_inplace_op, py::arg("a"));
    m.def("cos_", &cos_inplace_op, py::arg("a"));
    m.def("tan_", &tan_inplace_op, py::arg("a"));
    m.def("arcsin_", &arcsin_inplace_op, py::arg("a"));
    m.def("arccos_", &arccos_inplace_op, py::arg("a"));
    m.def("arctan_", &arctan_inplace_op, py::arg("a"));
    m.def("sinh_", &sinh_inplace_op, py::arg("a"));
    m.def("cosh_", &cosh_inplace_op, py::arg("a"));
    m.def("tanh_", &tanh_inplace_op, py::arg("a"));
    m.def("round_", &round_inplace_op, py::arg("a"));
    m.def("floor_", &floor_inplace_op, py::arg("a"));
    m.def("ceil_", &ceil_inplace_op, py::arg("a"));
    m.def("clip_", &clip_inplace_op, py::arg("a"), py::arg("min"), py::arg("max"));
}

}  // namespace lucid::bindings
