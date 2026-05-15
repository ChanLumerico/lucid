// lucid/_C/test/nn/test_loss.cpp
// Tests for mse_loss_op and cross_entropy_op.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../nn/Loss.h"

using namespace lucid;
using namespace lucid::test;

TEST(MSELoss, PerfectPredictionZeroLoss) {
    auto pred   = cpu_full({4}, 2.0);
    auto target = cpu_full({4}, 2.0);
    // reduction=1 (mean)
    auto loss = mse_loss_op(pred, target, 1);
    EXPECT_TENSOR_NEAR(loss, 0.0f, 1e-6f);
}

TEST(MSELoss, KnownValue) {
    // pred=[1], target=[3] → MSE = (1-3)^2 = 4
    auto pred   = cpu_full({1}, 1.0);
    auto target = cpu_full({1}, 3.0);
    auto loss = mse_loss_op(pred, target, 1);
    EXPECT_TENSOR_NEAR(loss, 4.0f, 1e-5f);
}

TEST(MSELoss, ScalarOutput) {
    auto pred   = cpu_ones({4, 6});
    auto target = cpu_zeros({4, 6});
    auto loss = mse_loss_op(pred, target, 1);
    // Shape should be scalar
    EXPECT_TENSOR_SHAPE(loss, (Shape{}));
}

TEST(MSELoss, NonNegative) {
    auto pred   = cpu_full({8}, 0.5);
    auto target = cpu_full({8}, 2.5);
    auto loss = mse_loss_op(pred, target, 1);
    auto val = to_float_vec(loss);
    ASSERT_EQ(val.size(), 1u);
    EXPECT_GE(val[0], 0.0f);
}
