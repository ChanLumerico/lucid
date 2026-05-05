// lucid/_C/test/autograd/test_gradients.cpp
// Tests for backward pass: gradient shapes and basic correctness.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../autograd/Engine.h"
#include "../../core/GradMode.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Exponential.h"
#include "../../ops/ufunc/Reductions.h"
#include "../../ops/bfunc/Add.h"
#include "../../ops/bfunc/Mul.h"

using namespace lucid;
using namespace lucid::test;

namespace {

/// Create a leaf tensor that requires gradient.
TensorImplPtr leaf(const Shape& shape, double val = 1.0, Dtype dtype = Dtype::F32) {
    return full_op(shape, val, dtype, Device::CPU, /*requires_grad=*/true);
}

}  // namespace

TEST(GradAdd, GradWrtBothInputs) {
    // z = x + y, dz/dx = 1, dz/dy = 1
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(add_op(x, y), {}, false);
    Engine::backward(z);

    ASSERT_TRUE(has_grad(x));
    ASSERT_TRUE(has_grad(y));
    auto gx = grad_to_float_vec(x);
    auto gy = grad_to_float_vec(y);
    for (float v : gx) EXPECT_NEAR(v, 1.0f, 1e-6f);
    for (float v : gy) EXPECT_NEAR(v, 1.0f, 1e-6f);
}

TEST(GradMul, GradWrtX) {
    // z = x * y, dz/dx = y = 3.0
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);
    Engine::backward(z);

    ASSERT_TRUE(has_grad(x));
    for (float v : grad_to_float_vec(x)) EXPECT_NEAR(v, 3.0f, 1e-5f);
}

TEST(GradMul, GradWrtY) {
    // z = x * y, dz/dy = x = 2.0
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);
    Engine::backward(z);

    ASSERT_TRUE(has_grad(y));
    for (float v : grad_to_float_vec(y)) EXPECT_NEAR(v, 2.0f, 1e-5f);
}

TEST(GradExp, GradWrtX) {
    // dexp(x)/dx = exp(x); at x=0, exp(0)=1
    auto x = leaf({4}, 0.0);
    auto y = sum_op(exp_op(x), {}, false);
    Engine::backward(y);

    ASSERT_TRUE(has_grad(x));
    for (float v : grad_to_float_vec(x)) EXPECT_NEAR(v, 1.0f, 1e-5f);
}

TEST(GradShape, GradSameNumelAsInput) {
    // Gradient of sum(x^2) wrt {3,4} tensor should have 12 elements.
    auto x = leaf({3, 4}, 1.0);
    auto y = sum_op(mul_op(x, x), {}, false);
    Engine::backward(y);

    ASSERT_TRUE(has_grad(x));
    EXPECT_EQ(grad_numel(x), x->numel());
}

TEST(NoGrad, NoGradContextDisablesGrad) {
    auto x = leaf({4}, 1.0);
    {
        NoGradGuard guard;
        auto y = add_op(x, x);
        EXPECT_FALSE(y->requires_grad());
    }
}
