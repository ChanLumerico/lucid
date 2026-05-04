// lucid/_C/test/backend/test_cpu_backend.cpp
// Tests for the CPU backend: zeros, ones, full, clone, matmul.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../backend/Dispatcher.h"
#include "../../backend/IBackend.h"

using namespace lucid;
using namespace lucid::test;

namespace {
IBackend& cpu() { return Dispatcher::for_device(Device::CPU); }
}

TEST(CpuBackend, ZerosAreZero) {
    auto t = cpu().zeros({4, 4}, Dtype::F32, Device::CPU);
    EXPECT_TENSOR_NEAR(t, 0.0f, 1e-7f);
}

TEST(CpuBackend, OnesAreOne) {
    auto t = cpu().full({4}, 1.0, Dtype::F32, Device::CPU);
    EXPECT_TENSOR_NEAR(t, 1.0f, 1e-7f);
}

TEST(CpuBackend, FullFillsValue) {
    auto t = cpu().full({3, 3}, 7.5, Dtype::F32, Device::CPU);
    EXPECT_TENSOR_NEAR(t, 7.5f, 1e-6f);
}

TEST(CpuBackend, CloneIsIndependent) {
    auto t = cpu().full({4}, 3.0, Dtype::F32, Device::CPU);
    auto c = t->clone_with_grad(false);
    EXPECT_TENSORS_NEAR(t, c, 1e-7f);
    EXPECT_NE(t.get(), c.get());
}

TEST(CpuBackend, DeviceIsCPU) {
    auto t = cpu().zeros({2, 3}, Dtype::F32, Device::CPU);
    EXPECT_EQ(t->device(), Device::CPU);
}

TEST(CpuBackend, ShapeCorrect) {
    auto t = cpu().zeros({5, 6, 7}, Dtype::F32, Device::CPU);
    EXPECT_TENSOR_SHAPE(t, (Shape{5, 6, 7}));
}

TEST(CpuBackend, NumelCorrect) {
    auto t = cpu().zeros({3, 4, 5}, Dtype::F32, Device::CPU);
    EXPECT_EQ(t->numel(), 60u);
}

TEST(CpuBackend, FiniteValues) {
    auto t = cpu().full({4, 4}, 1.23, Dtype::F32, Device::CPU);
    ASSERT_TENSOR_FINITE(t);
}
