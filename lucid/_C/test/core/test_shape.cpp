// lucid/_C/test/core/test_shape.cpp
// Tests for Shape arithmetic: numel, strides, broadcasting.

#include <gtest/gtest.h>
#include "../../core/Shape.h"

using namespace lucid;

TEST(ShapeNumel, ScalarIsOne) {
    EXPECT_EQ(shape_numel({}), 1u);
}

TEST(ShapeNumel, VectorCorrect) {
    EXPECT_EQ(shape_numel({5}), 5u);
}

TEST(ShapeNumel, MatrixCorrect) {
    EXPECT_EQ(shape_numel({3, 4}), 12u);
}

TEST(ShapeNumel, HigherDimCorrect) {
    EXPECT_EQ(shape_numel({2, 3, 4, 5}), 120u);
}

TEST(ContiguousStride, ScalarEmpty) {
    auto s = contiguous_stride({}, 4);
    EXPECT_TRUE(s.empty());
}

TEST(ContiguousStride, VectorStride) {
    auto s = contiguous_stride({5}, 4);
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(s[0], 4);  // stride in bytes = element_size * next_dim_product
}

TEST(ContiguousStride, MatrixRowMajor) {
    // (3, 4) float32: row stride = 4*4=16, col stride = 4
    auto s = contiguous_stride({3, 4}, 4);
    ASSERT_EQ(s.size(), 2u);
    EXPECT_EQ(s[0], 4 * 4);  // 16 bytes per row
    EXPECT_EQ(s[1], 4);      // 4 bytes per element
}

TEST(ShapeEquality, SameShapeEqual) {
    Shape a{2, 3, 4};
    Shape b{2, 3, 4};
    EXPECT_EQ(a, b);
}

TEST(ShapeEquality, DifferentShapeNotEqual) {
    Shape a{2, 3};
    Shape b{3, 2};
    EXPECT_NE(a, b);
}
