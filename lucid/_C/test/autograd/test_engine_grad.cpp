// lucid/_C/test/autograd/test_engine_grad.cpp
// Tests for Engine::grad — functional gradients that never write .grad.

#include <gtest/gtest.h>

#include "../../autograd/Engine.h"
#include "../../core/Error.h"
#include "../../ops/bfunc/Add.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/ufunc/Reductions.h"
#include "numeric_assert.h"
#include "tensor_factory.h"

using namespace lucid;
using namespace lucid::test;

namespace {

TensorImplPtr leaf(const Shape& shape, double val) {
    return full_op(shape, val, Dtype::F32, Device::CPU, /*requires_grad=*/true);
}

}  // namespace

TEST(EngineGrad, ReturnsGradientForRequestedInput) {
    // z = x * y, dz/dx = y = 3
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);

    auto grads = Engine::grad(z, Storage{CpuStorage{}}, {x});
    ASSERT_EQ(grads.size(), 1u);
    ASSERT_NE(grads[0], nullptr);
    EXPECT_TENSOR_NEAR(grads[0], 3.0f, 1e-5f);
}

TEST(EngineGrad, LeavesRequestedInputGradUntouched) {
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);

    Engine::grad(z, Storage{CpuStorage{}}, {x});
    EXPECT_FALSE(has_grad(x));
}

TEST(EngineGrad, LeavesUnrequestedLeafGradUntouched) {
    // The defect this exists to prevent: y is in the graph but was not asked
    // for, so its .grad must stay empty.  A full backward would fill it.
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);

    Engine::grad(z, Storage{CpuStorage{}}, {x});
    EXPECT_FALSE(has_grad(y));
    EXPECT_FALSE(has_grad(x));
}

TEST(EngineGrad, PreservesAPreExistingGrad) {
    // A .grad left over from an earlier backward must survive the call
    // unchanged, for every leaf.
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    Engine::backward(sum_op(mul_op(x, y), {}, false));
    ASSERT_TRUE(has_grad(x));
    ASSERT_TRUE(has_grad(y));
    const auto x_before = grad_to_float_vec(x);
    const auto y_before = grad_to_float_vec(y);

    auto z2 = sum_op(mul_op(x, y), {}, false);
    Engine::grad(z2, Storage{CpuStorage{}}, {x});

    EXPECT_EQ(grad_to_float_vec(x), x_before);
    EXPECT_EQ(grad_to_float_vec(y), y_before);
}

TEST(EngineGrad, ReturnsOnePerInputInOrder) {
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);

    auto grads = Engine::grad(z, Storage{CpuStorage{}}, {y, x});
    ASSERT_EQ(grads.size(), 2u);
    EXPECT_TENSOR_NEAR(grads[0], 2.0f, 1e-5f);  // dz/dy = x
    EXPECT_TENSOR_NEAR(grads[1], 3.0f, 1e-5f);  // dz/dx = y
}

TEST(EngineGrad, ReturnsNullForAnUnreachableInput) {
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto outsider = leaf({4}, 9.0);
    auto z = sum_op(mul_op(x, y), {}, false);

    auto grads = Engine::grad(z, Storage{CpuStorage{}}, {x, outsider});
    ASSERT_EQ(grads.size(), 2u);
    EXPECT_NE(grads[0], nullptr);
    EXPECT_EQ(grads[1], nullptr);
}

TEST(EngineGrad, AccumulatesAcrossBranches) {
    // z = x * x, dz/dx = 2x = 4.  Both edges land on the same leaf.
    auto x = leaf({4}, 2.0);
    auto z = sum_op(mul_op(x, x), {}, false);

    auto grads = Engine::grad(z, Storage{CpuStorage{}}, {x});
    ASSERT_NE(grads[0], nullptr);
    EXPECT_TENSOR_NEAR(grads[0], 4.0f, 1e-5f);
    EXPECT_FALSE(has_grad(x));
}

TEST(EngineGrad, HonoursAnExplicitSeed) {
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = mul_op(x, y);  // not reduced: seed shape must match

    auto seed = full_op({4}, 2.0, Dtype::F32, Device::CPU, false);
    auto grads = Engine::grad(z, seed->storage(), {x});
    ASSERT_NE(grads[0], nullptr);
    EXPECT_TENSOR_NEAR(grads[0], 6.0f, 1e-5f);  // 2 * y
}

TEST(EngineGrad, DifferentiatesAnInteriorTensor) {
    // Asking for a non-leaf must not stop gradients reaching what is below it.
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto mid = mul_op(x, y);
    auto z = sum_op(mul_op(mid, mid), {}, false);

    auto grads = Engine::grad(z, Storage{CpuStorage{}}, {mid, x});
    ASSERT_EQ(grads.size(), 2u);
    ASSERT_NE(grads[0], nullptr);
    ASSERT_NE(grads[1], nullptr);
    EXPECT_TENSOR_NEAR(grads[0], 12.0f, 1e-4f);  // d/dmid of mid^2 = 2*6
    EXPECT_TENSOR_NEAR(grads[1], 36.0f, 1e-4f);  // chain through y = 12*3
}

TEST(EngineGrad, LeafRootDifferentiatesToTheSeed) {
    auto x = leaf({4}, 2.0);
    auto grads = Engine::grad(x, Storage{CpuStorage{}}, {x});
    ASSERT_NE(grads[0], nullptr);
    EXPECT_TENSOR_NEAR(grads[0], 1.0f, 1e-6f);
    EXPECT_FALSE(has_grad(x));
}

TEST(EngineGrad, RetainGraphAllowsASecondCall) {
    auto x = leaf({4}, 2.0);
    auto y = leaf({4}, 3.0);
    auto z = sum_op(mul_op(x, y), {}, false);

    auto first = Engine::grad(z, Storage{CpuStorage{}}, {x}, /*retain_graph=*/true);
    auto second = Engine::grad(z, Storage{CpuStorage{}}, {y}, /*retain_graph=*/true);
    EXPECT_TENSOR_NEAR(first[0], 3.0f, 1e-5f);
    EXPECT_TENSOR_NEAR(second[0], 2.0f, 1e-5f);
    EXPECT_FALSE(has_grad(x));
    EXPECT_FALSE(has_grad(y));
}

TEST(EngineGrad, RejectsNullRootAndInputs) {
    auto x = leaf({4}, 2.0);
    auto z = sum_op(mul_op(x, x), {}, false);
    EXPECT_THROW(Engine::grad(nullptr, Storage{CpuStorage{}}, {x}), LucidError);
    EXPECT_THROW(Engine::grad(z, Storage{CpuStorage{}}, {nullptr}), LucidError);
}
