#include "Tri.h"

#include <variant>

#include <mlx/ops.h>

#include "../../autograd/FuncOp.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;

template <typename T>
CpuStorage tri_cpu(
    const CpuStorage& input, const Shape& shape, Dtype dt, int k, bool upper, const char* name) {
    if (shape.size() < 2)
        ErrorBuilder(name).fail("input must have ndim >= 2");
    auto out = allocate_cpu(shape, dt);
    const auto* src = reinterpret_cast<const T*>(input.ptr.get());
    auto* dst = reinterpret_cast<T*>(out.ptr.get());
    const std::int64_t M = shape[shape.size() - 2];
    const std::int64_t N = shape[shape.size() - 1];
    std::size_t batch = 1;
    for (std::size_t d = 0; d < shape.size() - 2; ++d)
        batch *= static_cast<std::size_t>(shape[d]);
    const std::size_t plane = static_cast<std::size_t>(M * N);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::int64_t i = 0; i < M; ++i) {
            for (std::int64_t j = 0; j < N; ++j) {
                const std::size_t f = b * plane + static_cast<std::size_t>(i * N + j);
                const bool keep = upper ? (j - i >= k) : (j - i <= k);
                dst[f] = keep ? src[f] : T{};
            }
        }
    }
    return out;
}

Storage tri_storage(const Storage& input,
                    const Shape& shape,
                    Dtype dt,
                    Device device,
                    int k,
                    bool upper,
                    const char* name) {
    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(input);
        auto out = upper ? ::mlx::core::triu(*ga.arr, k) : ::mlx::core::tril(*ga.arr, k);
        return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
    }
    const auto& ca = std::get<CpuStorage>(input);
    switch (dt) {
        case Dtype::F32:
            return Storage{tri_cpu<float>(ca, shape, dt, k, upper, name)};
        case Dtype::F64:
            return Storage{tri_cpu<double>(ca, shape, dt, k, upper, name)};
        default:
            ErrorBuilder(name).not_implemented("dtype not supported");
    }
}

class TriBackward : public FuncOp<TriBackward, 1> {
public:
    static const OpSchema schema_v1;

    int k_ = 0;
    bool upper_ = false;
    const char* name_ = "tril";

    std::vector<Storage> apply(Storage grad_out) override {
        return {tri_storage(grad_out, out_shape_, dtype_, device_, k_, upper_, name_)};
    }
};

const OpSchema TriBackward::schema_v1{"tri", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, /*internal=*/true};

TensorImplPtr attach_tri_grad(
    const TensorImplPtr& a, TensorImplPtr out, int k, bool upper, const char* name) {
    auto bwd = std::make_shared<TriBackward>();
    bwd->k_ = k;
    bwd->upper_ = upper;
    bwd->name_ = name;
    kernel::NaryKernel<TriBackward, 1>::wire_autograd(std::move(bwd), {a}, out, /*save_ins=*/false);
    return out;
}

TensorImplPtr tri_dispatch(const TensorImplPtr& a, int k, bool upper, const char* name) {
    Validator::input(a, std::string(name) + ".a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{name, device, dt, a->shape()};
    Shape sh = a->shape();
    auto out_storage = tri_storage(a->storage(), sh, dt, device, k, upper, name);
    auto out = fresh(std::move(out_storage), std::move(sh), dt, device);
    return attach_tri_grad(a, std::move(out), k, upper, name);
}

}  // namespace

TensorImplPtr tril_op(const TensorImplPtr& a, int k) {
    return tri_dispatch(a, k, /*upper=*/false, "tril");
}
TensorImplPtr triu_op(const TensorImplPtr& a, int k) {
    return tri_dispatch(a, k, /*upper=*/true, "triu");
}

LUCID_REGISTER_OP(TriBackward)

}  // namespace lucid
