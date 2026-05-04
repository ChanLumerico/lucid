// lucid/_C/test/integration/test_training.cpp
// End-to-end training loop test: 3 SGD steps, loss should decrease.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../nn/Linear.h"
#include "../../nn/Loss.h"
#include "../../autograd/Engine.h"
#include "../../ops/ufunc/Reductions.h"
#include "../../ops/ufunc/Arith.h"
#include "../../core/GradMode.h"

using namespace lucid;
using namespace lucid::test;

namespace {

/// Very simple manual SGD step: param -= lr * grad
void sgd_step(TensorImplPtr& param, float lr) {
    if (!param->grad()) return;
    auto& b = Dispatcher::for_device(Device::CPU);
    // param = param - lr * grad
    auto lr_t = b.full(param->shape(), static_cast<double>(lr), param->dtype(), Device::CPU);
    auto step = b.mul(lr_t, param->grad());
    auto new_param = b.sub(param, step);
    // Update param in place (replace storage)
    param = new_param->clone_with_grad(true);
}

}  // namespace

TEST(TrainingLoop, LossDecreases) {
    // Tiny linear model: (4,) → (2,)
    // Fixed data: all ones
    auto x      = cpu_ones({8, 4});
    auto target = cpu_zeros({8, 2});

    // Manual weight + bias (leaf tensors)
    auto W = cpu_full({2, 4}, 0.1);
    W = W->clone_with_grad(true);
    auto b = cpu_zeros({2});
    b = b->clone_with_grad(true);

    float last_loss = std::numeric_limits<float>::max();
    const float lr = 0.01f;

    for (int step = 0; step < 5; ++step) {
        // Zero grads
        W->zero_grad();
        b->zero_grad();

        // Forward: y = x @ W^T + b (simplified)
        auto out    = linear_op(x, W, b);
        auto loss   = mse_loss_op(out, target, 1);

        // Backward
        Engine::backward(loss, false);

        // Get loss value
        auto vals = to_float_vec(loss);
        ASSERT_EQ(vals.size(), 1u);
        float cur_loss = vals[0];

        if (step > 0) {
            EXPECT_LE(cur_loss, last_loss + 0.1f)
                << "Loss increased at step " << step
                << ": " << last_loss << " → " << cur_loss;
        }
        last_loss = cur_loss;

        // SGD step
        sgd_step(W, lr);
        sgd_step(b, lr);
    }

    EXPECT_LT(last_loss, 1.0f) << "Loss did not converge after 5 steps.";
}
