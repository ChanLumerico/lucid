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
    auto t = cpu_full(shape, val, dtype);
    return t->clone_with_grad(true);
}

/// Run backward on a scalar output tensor.
void backward(const TensorImplPtr& loss) {
    Engine::backward(loss, false);
}

}  // namespace

TEST(GradAdd, GradWrtBothInputs) {
    // z = x + y, dz/dx = 1, dz/dy = 1
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(add_op(x, y), {}, false);
    backward(z);

    ASSERT_NE(x->grad(), nullptr);
    ASSERT_NE(y->grad(), nullptr);
    EXPECT_TENSOR_NEAR(x->grad(), 1.0f, 1e-6f);
    EXPECT_TENSOR_NEAR(y->grad(), 1.0f, 1e-6f);
}

TEST(GradMul, GradWrtX) {
    // z = x * y, dz/dx = y
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);
    backward(z);

    ASSERT_NE(x->grad(), nullptr);
    EXPECT_TENSOR_NEAR(x->grad(), 3.0f, 1e-5f);  // grad = y = 3.0
}

TEST(GradMul, GradWrtY) {
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);
    backward(z);

    ASSERT_NE(y->grad(), nullptr);
    EXPECT_TENSOR_NEAR(y->grad(), 2.0f, 1e-5f);  // grad = x = 2.0
}

TEST(GradExp, GradWrtX) {
    // dexp(x)/dx = exp(x)
    auto x = leaf({4}, 0.0);  // x = 0, exp(0) = 1
    auto y = sum_op(exp_op(x), {}, false);
    backward(y);

    ASSERT_NE(x->grad(), nullptr);
    EXPECT_TENSOR_NEAR(x->grad(), 1.0f, 1e-5f);
}

TEST(GradShape, GradSameShapeAsInput) {
    auto x = leaf({3, 4}, 1.0);
    auto y = sum_op(mul_op(x, x), {}, false);
    backward(y);

    ASSERT_NE(x->grad(), nullptr);
    EXPECT_TENSOR_SHAPE(x->grad(), (Shape{3, 4}));
}

TEST(NoGrad, NoGradContextDisablesGrad) {
    auto x = leaf({4}, 1.0);
    {
        NoGradGuard guard;
        auto y = add_op(x, x);
        EXPECT_FALSE(y->requires_grad());
    }
}
