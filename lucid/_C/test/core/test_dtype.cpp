// lucid/_C/test/core/test_dtype.cpp
// Tests for Dtype enum, size computation, predicates, and name mapping.

#include <gtest/gtest.h>
#include "../../core/Dtype.h"

using namespace lucid;

TEST(DtypeSize, Float32IsFourBytes) {
    EXPECT_EQ(dtype_size(Dtype::F32), 4u);
}

TEST(DtypeSize, Float16IsTwoBytes) {
    EXPECT_EQ(dtype_size(Dtype::F16), 2u);
}

TEST(DtypeSize, Float64IsEightBytes) {
    EXPECT_EQ(dtype_size(Dtype::F64), 8u);
}

TEST(DtypeSize, Int32IsFourBytes) {
    EXPECT_EQ(dtype_size(Dtype::I32), 4u);
}

TEST(DtypeSize, Int8IsOneByte) {
    EXPECT_EQ(dtype_size(Dtype::I8), 1u);
}

TEST(DtypeSize, BoolIsOneByte) {
    EXPECT_EQ(dtype_size(Dtype::Bool), 1u);
}

TEST(DtypeName, F32Name) {
    auto name = dtype_name(Dtype::F32);  // string_view
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name.find("32"), std::string_view::npos);
}

TEST(DtypeName, I64Name) {
    auto name = dtype_name(Dtype::I64);
    EXPECT_FALSE(name.empty());
}

TEST(DtypeIsFloat, F32IsFloat) {
    EXPECT_TRUE(is_floating_point(Dtype::F32));
}

TEST(DtypeIsFloat, I32IsNotFloat) {
    EXPECT_FALSE(is_floating_point(Dtype::I32));
}

TEST(DtypeIsFloat, BoolIsNotFloat) {
    EXPECT_FALSE(is_floating_point(Dtype::Bool));
}

TEST(DtypeIsInt, I32IsInt) {
    EXPECT_TRUE(is_integral(Dtype::I32));
}

TEST(DtypeIsInt, F32IsNotInt) {
    EXPECT_FALSE(is_integral(Dtype::F32));
}
