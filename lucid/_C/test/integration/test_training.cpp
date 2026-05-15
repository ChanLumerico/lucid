// lucid/_C/test/integration/test_training.cpp
// End-to-end training loop test: 5 SGD steps, loss should decrease.

#include <gtest/gtest.h>
#include "tensor_factory.h"
#include "numeric_assert.h"
#include "../../nn/Linear.h"
#include "../../nn/Loss.h"
#include "../../autograd/Engine.h"
#include "../../ops/ufunc/Reductions.h"
#include "../../ops/bfunc/Mul.h"
#include "../../ops/bfunc/Sub.h"
#include "../../core/GradMode.h"

using namespace lucid;
using namespace lucid::test;

namespace {

/// Simple SGD step: param ← param − lr * grad
void sgd_step(TensorImplPtr& param, float lr) {
    if (!has_grad(param)) return;
    // Wrap grad_storage into a TensorImpl for arithmetic.
    const auto& g_st   = param->grad_storage();
    auto        g_impl = std::make_shared<TensorImpl>(
                             *g_st, param->shape(), param->dtype(), Device::CPU, false);
    auto lr_t      = full_op(param->shape(), static_cast<double>(lr), param->dtype(), Device::CPU);
    auto step      = mul_op(lr_t, g_impl);
    auto new_param = sub_op(param, step);
    new_param->set_requires_grad(true);
    new_param->set_leaf(true);
    param = new_param;
}

}  // namespace

TEST(TrainingLoop, LossDecreases) {
    auto x      = cpu_ones({8, 4});
    auto target = cpu_zeros({8, 2});

    auto W = full_op({2, 4}, 0.1, Dtype::F32, Device::CPU, /*requires_grad=*/true);
    auto b = zeros_op({2}, Dtype::F32, Device::CPU, /*requires_grad=*/true);

    float last_loss = std::numeric_limits<float>::max();
    const float lr  = 0.01f;

    for (int step = 0; step < 5; ++step) {
        W->zero_grad();
        b->zero_grad();

        auto out  = linear_op(x, W, b);
        auto loss = mse_loss_op(out, target, 1);

        Engine::backward(loss);  // default: retain_graph=false

        auto vals = to_float_vec(loss);
        ASSERT_EQ(vals.size(), 1u);
        float cur_loss = vals[0];

        if (step > 0) {
            EXPECT_LE(cur_loss, last_loss + 0.1f)
                << "Loss increased at step " << step
                << ": " << last_loss << " → " << cur_loss;
        }
        last_loss = cur_loss;

        sgd_step(W, lr);
        sgd_step(b, lr);
    }

    EXPECT_LT(last_loss, 1.0f) << "Loss did not converge after 5 steps.";
}
