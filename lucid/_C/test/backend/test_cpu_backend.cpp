// lucid/_C/test/backend/test_cpu_backend.cpp
// Tests for CPU tensor creation and basic properties.
// Uses the public gfunc API (zeros_op/ones_op/full_op) which routes through
// the CPU backend; no direct IBackend calls needed.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"

using namespace lucid;
using namespace lucid::test;

TEST(CpuBackend, ZerosAreZero) {
    auto t = cpu_zeros({4, 4});
    EXPECT_TENSOR_NEAR(t, 0.0f, 1e-7f);
}

TEST(CpuBackend, OnesAreOne) {
    auto t = cpu_ones({4});
    EXPECT_TENSOR_NEAR(t, 1.0f, 1e-7f);
}

TEST(CpuBackend, FullFillsValue) {
    auto t = cpu_full({3, 3}, 7.5);
    EXPECT_TENSOR_NEAR(t, 7.5f, 1e-6f);
}

TEST(CpuBackend, TwoSeparateAllocsAreIndependent) {
    // Two tensors with identical values but different allocations.
    auto a = cpu_full({4}, 3.0);
    auto b = cpu_full({4}, 3.0);
    EXPECT_TENSORS_NEAR(a, b, 1e-7f);
    EXPECT_NE(a.get(), b.get());
}

TEST(CpuBackend, DeviceIsCPU) {
    auto t = cpu_zeros({2, 3});
    EXPECT_EQ(t->device(), Device::CPU);
}

TEST(CpuBackend, ShapeCorrect) {
    auto t = cpu_zeros({5, 6, 7});
    EXPECT_TENSOR_SHAPE(t, (Shape{5, 6, 7}));
}

TEST(CpuBackend, NumelCorrect) {
    auto t = cpu_zeros({3, 4, 5});
    EXPECT_EQ(t->numel(), 60u);
}

TEST(CpuBackend, FiniteValues) {
    auto t = cpu_full({4, 4}, 1.23);
    ASSERT_TENSOR_FINITE(t);
}

TEST(CpuBackend, DtypeF32) {
    auto t = cpu_zeros({4}, Dtype::F32);
    EXPECT_TENSOR_DTYPE(t, Dtype::F32);
}

TEST(CpuBackend, DtypeF64) {
    auto t = cpu_zeros({4}, Dtype::F64);
    EXPECT_TENSOR_DTYPE(t, Dtype::F64);
}
