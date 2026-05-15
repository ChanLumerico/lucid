// lucid/_C/test/ops/test_ufunc.cpp
// Tests for unary operations (neg, abs, exp, log, relu, etc.)
// Uses typed test suites to run each test for float32 and float64.

#include <gtest/gtest.h>
#include <cmath>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Exponential.h"
#include "../../ops/ufunc/Trig.h"
#include "../../ops/ufunc/Activation.h"
#include "../../ops/ufunc/Reductions.h"

using namespace lucid;
using namespace lucid::test;

template <typename T>
struct DtypeOf;
template <> struct DtypeOf<float>  { static constexpr Dtype value = Dtype::F32; };
template <> struct DtypeOf<double> { static constexpr Dtype value = Dtype::F64; };

// ── Typed test fixture ────────────────────────────────────────────────────────

template <typename T>
struct UnaryTest : ::testing::Test {
    static constexpr Dtype kDtype = DtypeOf<T>::value;
};

using FloatTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(UnaryTest, FloatTypes);

// ── neg ───────────────────────────────────────────────────────────────────────

TYPED_TEST(UnaryTest, NegShape) {
    auto x = cpu_ones({3, 4}, this->kDtype);
    auto y = neg_op(x);
    EXPECT_TENSOR_SHAPE(y, (Shape{3, 4}));
}

TYPED_TEST(UnaryTest, NegValues) {
    auto x = cpu_full({4}, 2.0, this->kDtype);
    auto y = neg_op(x);
    EXPECT_TENSOR_NEAR(y, -2.0f, 1e-6f);
}

TYPED_TEST(UnaryTest, NegNegIsIdentity) {
    auto x = cpu_full({4}, 3.0, this->kDtype);
    auto y = neg_op(neg_op(x));
    EXPECT_TENSORS_NEAR(y, x, 1e-6f);
}

// ── abs ───────────────────────────────────────────────────────────────────────

TYPED_TEST(UnaryTest, AbsPositive) {
    auto x = cpu_full({4}, 2.0, this->kDtype);
    auto y = abs_op(x);
    EXPECT_TENSOR_NEAR(y, 2.0f, 1e-6f);
}

TYPED_TEST(UnaryTest, AbsNegative) {
    auto x = cpu_full({4}, -3.0, this->kDtype);
    auto y = abs_op(x);
    EXPECT_TENSOR_NEAR(y, 3.0f, 1e-6f);
}

// ── exp ───────────────────────────────────────────────────────────────────────

TYPED_TEST(UnaryTest, ExpAtZeroIsOne) {
    auto x = cpu_zeros({4}, this->kDtype);
    auto y = exp_op(x);
    EXPECT_TENSOR_NEAR(y, 1.0f, 1e-6f);
}

TYPED_TEST(UnaryTest, ExpPreservesShape) {
    auto x = cpu_zeros({3, 5}, this->kDtype);
    EXPECT_TENSOR_SHAPE(exp_op(x), (Shape{3, 5}));
}

TYPED_TEST(UnaryTest, ExpAndLogInverse) {
    auto x = cpu_full({4}, 2.0, this->kDtype);
    auto y = log_op(exp_op(x));
    EXPECT_TENSORS_NEAR(y, x, 1e-4f);
}

// ── relu ──────────────────────────────────────────────────────────────────────

TYPED_TEST(UnaryTest, ReluZerosNegatives) {
    auto x = cpu_full({4}, -1.0, this->kDtype);
    auto y = relu_op(x);
    EXPECT_TENSOR_NEAR(y, 0.0f, 1e-7f);
}

TYPED_TEST(UnaryTest, ReluKeepsPositives) {
    auto x = cpu_full({4}, 2.0, this->kDtype);
    auto y = relu_op(x);
    EXPECT_TENSOR_NEAR(y, 2.0f, 1e-6f);
}

// ── sigmoid ───────────────────────────────────────────────────────────────────

TYPED_TEST(UnaryTest, SigmoidAtZeroIsHalf) {
    auto x = cpu_zeros({4}, this->kDtype);
    auto y = sigmoid_op(x);
    EXPECT_TENSOR_NEAR(y, 0.5f, 1e-6f);
}

// ── sqrt ──────────────────────────────────────────────────────────────────────

TYPED_TEST(UnaryTest, SqrtAtFour) {
    auto x = cpu_full({4}, 4.0, this->kDtype);
    auto y = sqrt_op(x);
    EXPECT_TENSOR_NEAR(y, 2.0f, 1e-5f);
}

// ── sum (reduction) ───────────────────────────────────────────────────────────

TEST(ReductionF32, SumAllElements) {
    // ones of shape (3, 4) → sum = 12
    auto x = cpu_ones({3, 4});
    auto s = sum_op(x, {}, false);
    EXPECT_TENSOR_SHAPE(s, (Shape{}));  // scalar
    EXPECT_TENSOR_NEAR(s, 12.0f, 1e-5f);
}

TEST(ReductionF32, SumAlongAxis0) {
    // ones (3, 4), sum along axis 0 → shape (4,), all 3.0
    auto x = cpu_ones({3, 4});
    auto s = sum_op(x, {0}, false);
    EXPECT_TENSOR_SHAPE(s, (Shape{4}));
    EXPECT_TENSOR_NEAR(s, 3.0f, 1e-5f);
}

TEST(ReductionF32, MeanAllElements) {
    auto x = cpu_ones({3, 4});
    auto m = mean_op(x, {}, false);
    EXPECT_TENSOR_NEAR(m, 1.0f, 1e-5f);
}
