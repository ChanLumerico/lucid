// lucid/_C/test/ops/test_reductions.cpp
// Tests for reduction operations: sum, mean, max, min with axes.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../ops/ufunc/Reductions.h"

using namespace lucid;
using namespace lucid::test;

TEST(Sum, AllDims) {
    auto x = cpu_ones({2, 3, 4});  // 24 elements
    auto s = sum_op(x, {}, false);
    EXPECT_TENSOR_SHAPE(s, (Shape{}));
    EXPECT_TENSOR_NEAR(s, 24.0f, 1e-5f);
}

TEST(Sum, KeepDims) {
    auto x = cpu_ones({3, 4});
    auto s = sum_op(x, {0}, true);
    EXPECT_TENSOR_SHAPE(s, (Shape{1, 4}));
    EXPECT_TENSOR_NEAR(s, 3.0f, 1e-5f);
}

TEST(Sum, Axis1) {
    // ones (3, 4), sum over axis 1 → shape (3,), values all 4.0
    auto x = cpu_ones({3, 4});
    auto s = sum_op(x, {1}, false);
    EXPECT_TENSOR_SHAPE(s, (Shape{3}));
    EXPECT_TENSOR_NEAR(s, 4.0f, 1e-5f);
}

TEST(Mean, AllDims) {
    auto x = cpu_full({4, 4}, 3.0);
    auto m = mean_op(x, {}, false);
    EXPECT_TENSOR_NEAR(m, 3.0f, 1e-5f);
}

TEST(Max, AllDims) {
    auto x = cpu_full({2, 3}, 5.0);
    auto m = max_op(x, {}, false);
    EXPECT_TENSOR_NEAR(m, 5.0f, 1e-5f);
}

TEST(Min, AllDims) {
    auto x = cpu_full({2, 3}, -2.0);
    auto m = min_op(x, {}, false);
    EXPECT_TENSOR_NEAR(m, -2.0f, 1e-5f);
}
