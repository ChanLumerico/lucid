// lucid/_C/test/ops/test_diffeq.cpp
// Tests for the fused Runge-Kutta stage combination (rk_combine_op).

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../autograd/Engine.h"
#include "../../core/Error.h"
#include "../../ops/diffeq/RkCombine.h"
#include "../../ops/ufunc/Reductions.h"

using namespace lucid;
using namespace lucid::test;

namespace {

/// Create a leaf tensor that requires gradient.
TensorImplPtr leaf(const Shape& shape, double val) {
    return full_op(shape, val, Dtype::F32, Device::CPU, /*requires_grad=*/true);
}

}  // namespace

TEST(RkCombine, ForwardWeightedSum) {
    // 1 + 2.0 * (0.5 * 2 + 0.25 * 3) = 4.5
    auto y0 = cpu_full({4}, 1.0);
    auto k1 = cpu_full({4}, 2.0);
    auto k2 = cpu_full({4}, 3.0);
    auto y = rk_combine_op(y0, {k1, k2}, {0.5, 0.25}, 2.0);

    EXPECT_EQ(y->shape(), Shape({4}));
    EXPECT_TENSOR_NEAR(y, 4.5f, 1e-6f);
}

TEST(RkCombine, ForwardEulerStep) {
    // A single Euler step is coeffs={1}: y1 = y0 + dt * k1.
    auto y0 = cpu_full({2, 3}, 5.0);
    auto k1 = cpu_full({2, 3}, -1.0);
    auto y = rk_combine_op(y0, {k1}, {1.0}, 0.25);

    EXPECT_EQ(y->shape(), Shape({2, 3}));
    EXPECT_TENSOR_NEAR(y, 4.75f, 1e-6f);
}

TEST(RkCombine, ZeroCoefficientIsSkipped) {
    // The zero-coefficient term must not contribute; RK4's third stage row
    // is (0, 1/2), so this path runs on every classical RK4 step.
    auto y0 = cpu_full({3}, 1.0);
    auto k1 = cpu_full({3}, 100.0);
    auto k2 = cpu_full({3}, 2.0);
    auto y = rk_combine_op(y0, {k1, k2}, {0.0, 0.5}, 1.0);

    EXPECT_TENSOR_NEAR(y, 2.0f, 1e-6f);
}

TEST(RkCombine, NoStagesCopiesBase) {
    auto y0 = cpu_full({3}, 7.0);
    auto y = rk_combine_op(y0, {}, {}, 0.5);

    EXPECT_EQ(y->shape(), Shape({3}));
    EXPECT_TENSOR_NEAR(y, 7.0f, 1e-6f);
}

TEST(RkCombine, NegativeStepIntegratesBackwards) {
    auto y0 = cpu_full({2}, 1.0);
    auto k1 = cpu_full({2}, 4.0);
    auto y = rk_combine_op(y0, {k1}, {1.0}, -0.5);

    EXPECT_TENSOR_NEAR(y, -1.0f, 1e-6f);
}

TEST(RkCombine, BackwardScalesEachStage) {
    // d(sum y)/dy0 = 1 and d(sum y)/dk_i = dt * coeffs[i].
    const double dt = 2.0;
    auto y0 = leaf({4}, 1.0);
    auto k1 = leaf({4}, 2.0);
    auto k2 = leaf({4}, 3.0);
    auto z = sum_op(rk_combine_op(y0, {k1, k2}, {0.5, 0.25}, dt), {}, false);
    Engine::backward(z);

    ASSERT_TRUE(has_grad(y0));
    ASSERT_TRUE(has_grad(k1));
    ASSERT_TRUE(has_grad(k2));
    for (float v : grad_to_float_vec(y0)) EXPECT_NEAR(v, 1.0f, 1e-6f);
    for (float v : grad_to_float_vec(k1)) EXPECT_NEAR(v, 1.0f, 1e-6f);
    for (float v : grad_to_float_vec(k2)) EXPECT_NEAR(v, 0.5f, 1e-6f);
}

TEST(RkCombine, BackwardZeroCoefficientGivesZeroGrad) {
    auto y0 = leaf({3}, 1.0);
    auto k1 = leaf({3}, 9.0);
    auto z = sum_op(rk_combine_op(y0, {k1}, {0.0}, 1.0), {}, false);
    Engine::backward(z);

    ASSERT_TRUE(has_grad(k1));
    for (float v : grad_to_float_vec(k1)) EXPECT_NEAR(v, 0.0f, 1e-6f);
}

TEST(RkCombine, BackwardAccumulatesRepeatedStage) {
    // The same tensor appearing in two slots must accumulate both scales:
    // dt * (c_0 + c_1) = 2.0 * (0.5 + 0.25).
    auto y0 = leaf({4}, 1.0);
    auto k = leaf({4}, 2.0);
    auto z = sum_op(rk_combine_op(y0, {k, k}, {0.5, 0.25}, 2.0), {}, false);
    Engine::backward(z);

    ASSERT_TRUE(has_grad(k));
    for (float v : grad_to_float_vec(k)) EXPECT_NEAR(v, 1.5f, 1e-6f);
}

TEST(RkCombine, RejectsCoeffLengthMismatch) {
    auto y0 = cpu_ones({2});
    auto k1 = cpu_ones({2});
    EXPECT_THROW(rk_combine_op(y0, {k1}, {0.5, 0.5}, 1.0), LucidError);
}

TEST(RkCombine, RejectsShapeMismatch) {
    auto y0 = cpu_ones({2, 3});
    auto k1 = cpu_ones({3, 2});
    EXPECT_THROW(rk_combine_op(y0, {k1}, {1.0}, 1.0), ShapeMismatch);
}

TEST(RkCombine, RejectsDtypeMismatch) {
    auto y0 = cpu_ones({2}, Dtype::F32);
    auto k1 = cpu_ones({2}, Dtype::F64);
    EXPECT_THROW(rk_combine_op(y0, {k1}, {1.0}, 1.0), DtypeMismatch);
}
