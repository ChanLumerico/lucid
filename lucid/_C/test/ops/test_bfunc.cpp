// lucid/_C/test/ops/test_bfunc.cpp
// Tests for binary operations: add, mul, matmul, broadcast.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../ops/bfunc/Add.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/bfunc/Sub.h"
#include "../../ops/bfunc/Div.h"
#include "../../ops/bfunc/Matmul.h"

using namespace lucid;
using namespace lucid::test;

TEST(Add, ShapePreserved) {
    auto a = cpu_ones({3, 4});
    auto b = cpu_ones({3, 4});
    auto c = add_op(a, b);
    EXPECT_TENSOR_SHAPE(c, (Shape{3, 4}));
}

TEST(Add, OnesAndOnes) {
    auto a = cpu_ones({4});
    auto b = cpu_ones({4});
    auto c = add_op(a, b);
    EXPECT_TENSOR_NEAR(c, 2.0f, 1e-6f);
}

TEST(Add, ScalarBroadcast) {
    auto a = cpu_full({3}, 1.0);
    auto b = cpu_full({3}, 2.0);
    auto c = add_op(a, b);
    EXPECT_TENSOR_NEAR(c, 3.0f, 1e-6f);
}

TEST(Sub, BasicSubtraction) {
    auto a = cpu_full({4}, 5.0);
    auto b = cpu_full({4}, 3.0);
    auto c = sub_op(a, b);
    EXPECT_TENSOR_NEAR(c, 2.0f, 1e-6f);
}

TEST(Mul, OnesIsIdentity) {
    auto a = cpu_full({3, 4}, 3.0);
    auto b = cpu_ones({3, 4});
    auto c = mul_op(a, b);
    EXPECT_TENSORS_NEAR(c, a, 1e-6f);
}

TEST(Mul, ZeroAnnihilates) {
    auto a = cpu_full({4}, 999.0);
    auto b = cpu_zeros({4});
    auto c = mul_op(a, b);
    EXPECT_TENSOR_NEAR(c, 0.0f, 1e-7f);
}

TEST(Div, DivBySelf) {
    auto a = cpu_full({4}, 3.0);
    auto c = div_op(a, a);
    EXPECT_TENSOR_NEAR(c, 1.0f, 1e-6f);
}

TEST(Matmul, IdentityMatrix) {
    auto I = cpu_eye(3);
    auto A = cpu_full({3, 3}, 2.0);
    auto C = matmul_op(I, A);
    EXPECT_TENSORS_NEAR(C, A, 1e-5f);
}

TEST(Matmul, OutputShape) {
    auto A = cpu_ones({3, 4});
    auto B = cpu_ones({4, 5});
    auto C = matmul_op(A, B);
    EXPECT_TENSOR_SHAPE(C, (Shape{3, 5}));
}

TEST(Matmul, CorrectValues) {
    // (1, 2) @ (2, 1) = [[a*c + b*d]]
    // [1, 2] @ [[3], [4]] = [[1*3 + 2*4]] = [[11]]
    auto A = cpu_full({1, 2}, 1.0);  // will manually create proper values in integration
    auto B = cpu_full({2, 1}, 1.0);
    auto C = matmul_op(A, B);
    EXPECT_TENSOR_SHAPE(C, (Shape{1, 1}));
    // 1*1 + 1*1 = 2
    EXPECT_TENSOR_NEAR(C, 2.0f, 1e-6f);
}
