#include "Inv.h"

#include <cstring>
#include <variant>

#include <mlx/linalg.h>

#include "../../autograd/FuncOp.h"
#include "../../backend/cpu/Lapack.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Error.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/ufunc/Arith.h"
#include "../../ops/ufunc/Transpose.h"
#include "_Detail.h"

namespace lucid {

// ---------- Schema & backward ----------

const OpSchema InvBackward::schema_v1{"inv", 1, AmpPolicy::KeepInput};

std::vector<Storage> InvBackward::apply(Storage grad_out) {
    NoGradGuard ng;
    using ::lucid::helpers::fresh;
    // saved_output_ = B = inv(A)
    auto B  = fresh(Storage{saved_output_},       out_shape_,       dtype_, device_);
    auto dB = fresh(std::move(grad_out),           out_shape_,       dtype_, device_);
    // dA = -(B^T @ dB @ B^T)
    auto Bt = mT_op(B);
    auto dA = neg_op(matmul_op(matmul_op(Bt, dB), Bt));
    return {dA->storage()};
}

LUCID_REGISTER_OP(InvBackward)

// ---------- Forward ----------

TensorImplPtr inv_op(const TensorImplPtr& a) {
    using namespace linalg_detail;
    Validator::input(a, "inv.a").float_only().square_2d();
    OpScopeFull scope{"inv", a->device(), a->dtype(), a->shape()};

    if (a->device() == Device::GPU) {
        auto in = as_mlx_array_gpu(a);
        auto raw = ::mlx::core::linalg::inv(in, kMlxLinalgStream);
        auto out = fresh(wrap_gpu_result(std::move(raw), a->dtype()), a->shape(), a->dtype(),
                         a->device());
        auto bwd = std::make_shared<InvBackward>();
        bwd->saved_output_ = out->storage();
        kernel::NaryKernel<InvBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
        return out;
    }

    // CPU path — Apple Accelerate LAPACK, batched over leading dims.
    const auto& sh = a->shape();
    const int n = static_cast<int>(sh[sh.size() - 1]);
    const std::int64_t batch = leading_batch_count(sh, /*mat_dims=*/2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;

    auto out_cpu = allocate_cpu(sh, a->dtype());
    const auto& in_cpu = std::get<CpuStorage>(a->storage());
    std::memcpy(out_cpu.ptr.get(), in_cpu.ptr.get(), in_cpu.nbytes);

    int info = 0;
    if (a->dtype() == Dtype::F32) {
        auto* p = reinterpret_cast<float*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_inv_f32(p + b * per_mat, n, &info);
            check_lapack_info(info, "inv");
        }
    } else {
        auto* p = reinterpret_cast<double*>(out_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            backend::cpu::lapack_inv_f64(p + b * per_mat, n, &info);
            check_lapack_info(info, "inv");
        }
    }
    auto out = fresh(Storage{std::move(out_cpu)}, sh, a->dtype(), a->device());
    auto bwd = std::make_shared<InvBackward>();
    bwd->saved_output_ = out->storage();
    kernel::NaryKernel<InvBackward, 1>::wire_autograd(std::move(bwd), {a}, out, /*save_ins=*/false);
    return out;
}

}  // namespace lucid
