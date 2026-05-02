#include "Dropout.h"

#include <cmath>
#include <string>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/Generator.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"

namespace lucid {

const OpSchema DropoutBackward::schema_v1{
    "dropout", 1, AmpPolicy::KeepInput, false,
    "uses Generator-driven Bernoulli mask; reproducible only with explicit seed"};

TensorImplPtr
DropoutBackward::forward(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    Validator::input(a, "dropout.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("dropout").fail("p must be in [0, 1)");

    if (training && p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    const std::size_t numel = a->numel();

    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        auto bwd = std::make_shared<DropoutBackward>();
        bwd->p_ = 0.0;

        kernel::NaryKernel<DropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    }

    Generator& g = gen ? *gen : default_generator();
    Storage mask = bernoulli_mask_storage_shape(1.0 - p, a->shape(), a->dtype(), a->device(), g);
    const double scale = 1.0 / (1.0 - p);

    Storage scaled_mask = mul_scalar_storage(mask, scale, numel, a->dtype(), a->device());
    Storage y = multiply_storages(a->storage(), scaled_mask, numel, a->dtype(), a->device());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropoutBackward>();
        bwd->p_ = p;
        bwd->mask_ = std::move(scaled_mask);
        kernel::NaryKernel<DropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropoutBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);

    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return DropoutBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(DropoutBackward)

namespace {

Shape channel_mask_shape(const Shape& in) {
    if (in.size() < 2) {
        ErrorBuilder("dropoutnd").fail("expected ndim >= 2, got" + std::to_string(in.size()));
    }
    Shape m(in.size(), 1);
    m[0] = in[0];
    m[1] = in[1];
    return m;
}

Shape sample_mask_shape(const Shape& in) {
    if (in.empty())
        ErrorBuilder("drop_path").fail("input must have ≥1 dim");
    Shape m(in.size(), 1);
    m[0] = in[0];
    return m;
}

}  // namespace

const OpSchema DropoutNdBackward::schema_v1{
    "dropoutnd", 1, AmpPolicy::KeepInput, false,
    "Bernoulli channel mask; reproducible only with explicit seed"};

TensorImplPtr
DropoutNdBackward::forward(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    Validator::input(a, "dropoutnd.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("dropoutnd").fail("p must be in [0, 1)");

    if (training && p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    const std::size_t numel = a->numel();

    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        auto bwd = std::make_shared<DropoutNdBackward>();
        bwd->p_ = 0.0;
        kernel::NaryKernel<DropoutNdBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    }

    Shape mask_shape = channel_mask_shape(a->shape());
    const std::size_t mask_numel =
        static_cast<std::size_t>(mask_shape[0]) * static_cast<std::size_t>(mask_shape[1]);
    Generator& g = gen ? *gen : default_generator();
    Storage small_mask =
        bernoulli_mask_storage_shape(1.0 - p, mask_shape, a->dtype(), a->device(), g);
    const double scale = 1.0 / (1.0 - p);
    Storage scaled_small =
        mul_scalar_storage(small_mask, scale, mask_numel, a->dtype(), a->device());
    auto& be = backend::Dispatcher::for_device(a->device());
    auto [full_mask, y] =
        be.expand_and_multiply(scaled_small, a->storage(), mask_shape, a->shape(), a->dtype());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropoutNdBackward>();
        bwd->p_ = p;
        bwd->mask_ = std::move(full_mask);
        kernel::NaryKernel<DropoutNdBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropoutNdBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr dropoutnd_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return DropoutNdBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(DropoutNdBackward)

namespace {
constexpr double kSeluAlpha = 1.6732632423543772;
constexpr double kSeluLambda = 1.0507009873554805;
constexpr double kAlphaPrime = -kSeluLambda * kSeluAlpha;
}  // namespace

const OpSchema AlphaDropoutBackward::schema_v1{
    "alpha_dropout", 1, AmpPolicy::KeepInput, false,
    "Bernoulli mask + SELU rescaling; reproducible only with explicit seed"};

TensorImplPtr
AlphaDropoutBackward::forward(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    Validator::input(a, "alpha_dropout.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("alpha_dropout").fail("p must be in [0, 1)");

    if (training && p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    const std::size_t numel = a->numel();

    if (!training || p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        auto bwd = std::make_shared<AlphaDropoutBackward>();
        bwd->p_ = 0.0;
        bwd->a_coef_ = 1.0;
        kernel::NaryKernel<AlphaDropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    }

    const double keep = 1.0 - p;
    const double a_coef = std::pow(keep * (1.0 + p * kAlphaPrime * kAlphaPrime), -0.5);
    const double b_coef = -a_coef * p * kAlphaPrime;

    Generator& g = gen ? *gen : default_generator();
    Storage mask = bernoulli_mask_storage_shape(keep, a->shape(), a->dtype(), a->device(), g);

    Storage x_mask = multiply_storages(a->storage(), mask, numel, a->dtype(), a->device());

    Storage alpha_mask = mul_scalar_storage(mask, kAlphaPrime, numel, a->dtype(), a->device());
    Storage neg_alpha_mask = mul_scalar_storage(alpha_mask, -1.0, numel, a->dtype(), a->device());
    Storage one_m_mask_ap =
        add_scalar_storage(neg_alpha_mask, kAlphaPrime, numel, a->dtype(), a->device());

    Storage inner = add_storages(x_mask, one_m_mask_ap, numel, a->dtype(), a->device());
    Storage scaled = mul_scalar_storage(inner, a_coef, numel, a->dtype(), a->device());
    Storage y = add_scalar_storage(scaled, b_coef, numel, a->dtype(), a->device());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<AlphaDropoutBackward>();
        bwd->p_ = p;
        bwd->a_coef_ = a_coef;
        bwd->mask_ = std::move(mask);
        kernel::NaryKernel<AlphaDropoutBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> AlphaDropoutBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }

    Storage scaled_mask = mul_scalar_storage(mask_, a_coef_, numel, dtype_, device_);
    return {multiply_storages(grad_out, scaled_mask, numel, dtype_, device_)};
}

TensorImplPtr alpha_dropout_op(const TensorImplPtr& a, double p, bool training, Generator* gen) {
    return AlphaDropoutBackward::forward(a, p, training, gen);
}

LUCID_REGISTER_OP(AlphaDropoutBackward)

const OpSchema DropBlockBackward::schema_v1{
    "drop_block", 1, AmpPolicy::KeepInput, false,
    "Bernoulli + spatial dilation; reproducible only with explicit seed"};

TensorImplPtr DropBlockBackward::forward(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen) {
    Validator::input(a, "drop_block.a").non_null();
    if (a->shape().size() != 4)
        ErrorBuilder("drop_block").fail("input must be 4-D (N, C, H, W)");
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("drop_block").fail("p must be in [0, 1)");
    if (block_size <= 0)
        ErrorBuilder("drop_block").fail("block_size must be > 0");

    if (p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    const std::size_t numel = a->numel();
    const std::int64_t H = a->shape()[2];
    const std::int64_t W = a->shape()[3];

    const double feat = static_cast<double>(H * W);
    const double valid = static_cast<double>((H - block_size + 1) * (W - block_size + 1));
    const double gamma = p * feat / (block_size * block_size) / (valid + eps);

    Generator& g = gen ? *gen : default_generator();
    Storage seed = bernoulli_mask_storage(gamma, numel, a->dtype(), a->device(), g);
    auto& be = backend::Dispatcher::for_device(a->device());
    Storage keep_storage = be.drop_block_mask(seed, p, block_size, a->shape(), a->dtype());
    Storage y = multiply_storages(a->storage(), keep_storage, numel, a->dtype(), a->device());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropBlockBackward>();
        bwd->p_ = p;
        bwd->mask_ = std::move(keep_storage);
        kernel::NaryKernel<DropBlockBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropBlockBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr drop_block_op(
    const TensorImplPtr& a, std::int64_t block_size, double p, double eps, Generator* gen) {
    return DropBlockBackward::forward(a, block_size, p, eps, gen);
}

LUCID_REGISTER_OP(DropBlockBackward)

const OpSchema DropPathBackward::schema_v1{
    "drop_path", 1, AmpPolicy::KeepInput, false,
    "Per-sample Bernoulli; reproducible only with explicit seed"};

TensorImplPtr
DropPathBackward::forward(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen) {
    Validator::input(a, "drop_path.a").non_null();
    if (p < 0.0 || p >= 1.0)
        ErrorBuilder("drop_path").fail("p must be in [0, 1)");

    if (p > 0.0 && gen == nullptr)
        check_schema_determinism(schema_v1);

    OpScopeFull scope{schema_v1.name, a->device(), a->dtype(), a->shape()};
    const std::size_t numel = a->numel();

    if (p == 0.0) {
        Storage clone = clone_storage(a->storage(), numel, a->dtype(), a->device());
        auto out = std::make_shared<TensorImpl>(std::move(clone), a->shape(), a->dtype(),
                                                a->device(), false);
        kernel::NaryKernel<DropPathBackward, 1>::wire_autograd({a}, out, false);
        return out;
    }

    const double keep = 1.0 - p;
    const std::size_t B = static_cast<std::size_t>(a->shape()[0]);
    Shape sshape = sample_mask_shape(a->shape());
    Generator& g = gen ? *gen : default_generator();
    Storage small = bernoulli_mask_storage_shape(keep, sshape, a->dtype(), a->device(), g);
    if (scale_by_keep && keep > 0.0) {
        small = mul_scalar_storage(small, 1.0 / keep, B, a->dtype(), a->device());
    }
    auto& be = backend::Dispatcher::for_device(a->device());
    auto [full_mask, y] =
        be.expand_and_multiply(small, a->storage(), sshape, a->shape(), a->dtype());

    auto out =
        std::make_shared<TensorImpl>(std::move(y), a->shape(), a->dtype(), a->device(), false);
    scope.set_flops(static_cast<std::int64_t>(numel));

    {
        auto bwd = std::make_shared<DropPathBackward>();
        bwd->mask_ = std::move(full_mask);
        kernel::NaryKernel<DropPathBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    }
    return out;
}

std::vector<Storage> DropPathBackward::apply(Storage grad_out) {
    const std::size_t numel = shape_numel(input_shapes_[0]);
    if (device_ == Device::CPU) {
        auto* mask_cpu = std::get_if<CpuStorage>(&mask_);
        if (!mask_cpu || mask_cpu->nbytes == 0) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    } else {
        auto* mask_gpu = std::get_if<GpuStorage>(&mask_);
        if (!mask_gpu || !mask_gpu->arr) {
            return {clone_storage(grad_out, numel, dtype_, device_)};
        }
    }
    return {multiply_storages(grad_out, mask_, numel, dtype_, device_)};
}

TensorImplPtr drop_path_op(const TensorImplPtr& a, double p, bool scale_by_keep, Generator* gen) {
    return DropPathBackward::forward(a, p, scale_by_keep, gen);
}

LUCID_REGISTER_OP(DropPathBackward)

}  // namespace lucid
