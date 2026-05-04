// lucid/_C/test/nn/test_linear.cpp
// Tests for the linear_op C++ function.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../nn/Linear.h"

using namespace lucid;
using namespace lucid::test;

TEST(LinearOp, OutputShape) {
    // x: (4, 8), W: (6, 8), b: (6,) → out: (4, 6)
    auto x = cpu_ones({4, 8});
    auto W = cpu_ones({6, 8});
    auto b = cpu_zeros({6});
    auto out = linear_op(x, W, b);
    EXPECT_TENSOR_SHAPE(out, (Shape{4, 6}));
}

TEST(LinearOp, ZeroWeightZeroOutput) {
    auto x = cpu_full({3, 4}, 5.0);
    auto W = cpu_zeros({2, 4});
    auto b = cpu_zeros({2});
    auto out = linear_op(x, W, b);
    EXPECT_TENSOR_NEAR(out, 0.0f, 1e-6f);
}

TEST(LinearOp, BiasAdded) {
    auto x = cpu_zeros({3, 4});
    auto W = cpu_zeros({2, 4});
    auto b = cpu_full({2}, 3.0);
    auto out = linear_op(x, W, b);
    EXPECT_TENSOR_NEAR(out, 3.0f, 1e-6f);
}

TEST(LinearOp, IdentityWeightProducesInput) {
    auto x = cpu_full({3, 4}, 2.0);
    // W = I_4 (4×4 identity)
    auto W = cpu_eye(4);
    auto b = cpu_zeros({4});
    auto out = linear_op(x, W, b);
    EXPECT_TENSORS_NEAR(out, x, 1e-5f);
}

TEST(LinearOp, FiniteOutput) {
    auto x = cpu_full({4, 8}, 0.1);
    auto W = cpu_full({6, 8}, 0.1);
    auto b = cpu_zeros({6});
    auto out = linear_op(x, W, b);
    ASSERT_TENSOR_FINITE(out);
}
