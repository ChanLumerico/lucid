// lucid/_C/test/core/test_tensor_impl.cpp
// Tests for TensorImpl construction, properties, views, and version tracking.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"

using namespace lucid;
using namespace lucid::test;

// ── Construction ──────────────────────────────────────────────────────────────

TEST(TensorImplConstruct, ZerosShape) {
    auto t = cpu_zeros({3, 4});
    EXPECT_TENSOR_SHAPE(t, (Shape{3, 4}));
}

TEST(TensorImplConstruct, ZerosDtype) {
    auto t = cpu_zeros({3, 4}, Dtype::F32);
    EXPECT_TENSOR_DTYPE(t, Dtype::F32);
}

TEST(TensorImplConstruct, ZerosDevice) {
    auto t = cpu_zeros({3, 4});
    EXPECT_EQ(t->device(), Device::CPU);
}

TEST(TensorImplConstruct, NumelCorrect) {
    auto t = cpu_zeros({3, 4});
    EXPECT_EQ(t->numel(), 12u);
}

TEST(TensorImplConstruct, OnesValues) {
    auto t = cpu_ones({4});
    EXPECT_TENSOR_NEAR(t, 1.0f, 1e-7f);
}

TEST(TensorImplConstruct, FullValues) {
    auto t = cpu_full({5}, 3.14);
    EXPECT_TENSOR_NEAR(t, 3.14f, 1e-5f);
}

// ── Properties ────────────────────────────────────────────────────────────────

TEST(TensorImplProps, RequiresGradDefault) {
    auto t = cpu_zeros({3});
    EXPECT_FALSE(t->requires_grad());
}

TEST(TensorImplProps, RequiresGradTrue) {
    auto& b = Dispatcher::for_device(Device::CPU);
    auto base = b.zeros({3}, Dtype::F32, Device::CPU);
    auto t = base->clone_with_grad(true);
    EXPECT_TRUE(t->requires_grad());
}

TEST(TensorImplProps, IsContiguousForNewTensor) {
    auto t = cpu_zeros({3, 4});
    EXPECT_TRUE(t->is_contiguous());
}

TEST(TensorImplProps, NdimCorrect) {
    auto t = cpu_zeros({2, 3, 4});
    EXPECT_EQ(t->shape().size(), 3u);
}

// ── Version tracking ──────────────────────────────────────────────────────────

TEST(TensorImplVersion, InitialVersionZero) {
    auto t = cpu_zeros({4});
    EXPECT_EQ(t->version(), 0u);
}

TEST(TensorImplVersion, BumpVersionIncrements) {
    auto t = cpu_zeros({4});
    t->bump_version();
    EXPECT_EQ(t->version(), 1u);
    t->bump_version();
    EXPECT_EQ(t->version(), 2u);
}

// ── Identity/zero ─────────────────────────────────────────────────────────────

TEST(TensorImplValues, ZerosAreZero) {
    auto t = cpu_zeros({6});
    EXPECT_TENSOR_NEAR(t, 0.0f, 1e-7f);
}

TEST(TensorImplValues, FiniteValues) {
    auto t = cpu_full({4, 4}, 1.5);
    ASSERT_TENSOR_FINITE(t);
}
