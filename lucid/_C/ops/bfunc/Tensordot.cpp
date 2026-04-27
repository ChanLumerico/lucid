#include "Tensordot.h"

#include <cstring>
#include <numeric>
#include <variant>

#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Exceptions.h"
#include "../../core/Profiler.h"
#include "../../core/TensorImpl.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

}  // namespace

TensorImplPtr tensordot_op(const TensorImplPtr& a, const TensorImplPtr& b,
                           std::vector<int> axes_a, std::vector<int> axes_b) {
    validate_pair(a, b, "tensordot");
    if (axes_a.size() != axes_b.size())
        throw LucidError("tensordot: axes_a and axes_b must have equal length");
    const Dtype dt = a->dtype_;
    const Device device = a->device_;
    OpScope scope{"tensordot", device, dt, Shape{}};

    if (device == Device::GPU) {
        const auto& ga = std::get<GpuStorage>(a->storage_);
        const auto& gb = std::get<GpuStorage>(b->storage_);
        auto out = ::mlx::core::tensordot(*ga.arr, *gb.arr, axes_a, axes_b);
        Shape out_shape;
        for (auto d : out.shape())
            out_shape.push_back(static_cast<std::int64_t>(d));
        return fresh(Storage{gpu::wrap_mlx_array(std::move(out), dt)},
                     std::move(out_shape), dt, device);
    }

    auto contract = [&](const TensorImplPtr& t,
                        const std::vector<int>& axes_contract,
                        bool put_first) {
        const std::size_t nd = t->shape_.size();
        std::vector<bool> is_c(nd, false);
        for (auto ax : axes_contract) {
            int p = ax < 0 ? ax + (int)nd : ax;
            if (p < 0 || p >= (int)nd)
                throw LucidError("tensordot: axis out of range");
            is_c[p] = true;
        }
        std::vector<int> perm;
        std::vector<std::int64_t> kept;
        std::int64_t contract_size = 1;
        if (put_first) {
            for (auto ax : axes_contract) {
                int p = ax < 0 ? ax + (int)nd : ax;
                perm.push_back(p);
                contract_size *= t->shape_[p];
            }
            for (std::size_t d = 0; d < nd; ++d)
                if (!is_c[d]) {
                    perm.push_back((int)d);
                    kept.push_back(t->shape_[d]);
                }
        } else {
            for (std::size_t d = 0; d < nd; ++d)
                if (!is_c[d]) {
                    perm.push_back((int)d);
                    kept.push_back(t->shape_[d]);
                }
            for (auto ax : axes_contract) {
                int p = ax < 0 ? ax + (int)nd : ax;
                perm.push_back(p);
                contract_size *= t->shape_[p];
            }
        }
        Shape src_shape = t->shape_;
        Shape dst_shape;
        for (auto p : perm) dst_shape.push_back(src_shape[p]);
        const auto& cs = std::get<CpuStorage>(t->storage_);
        auto dst = allocate_cpu(dst_shape, dt);
        const std::size_t elem = dtype_size(dt);
        Stride src_stride(nd);
        if (nd > 0) {
            src_stride.back() = 1;
            for (std::ptrdiff_t d = (std::ptrdiff_t)nd - 2; d >= 0; --d)
                src_stride[d] = src_stride[d + 1] * src_shape[d + 1];
        }
        const std::size_t total = shape_numel(dst_shape);
        std::vector<std::int64_t> coord(nd, 0);
        for (std::size_t f = 0; f < total; ++f) {
            std::size_t src_flat = 0;
            for (std::size_t d = 0; d < nd; ++d)
                src_flat += static_cast<std::size_t>(coord[d]) *
                            static_cast<std::size_t>(src_stride[perm[d]]);
            std::memcpy(dst.ptr.get() + f * elem,
                        cs.ptr.get() + src_flat * elem, elem);
            for (std::ptrdiff_t d = (std::ptrdiff_t)nd - 1; d >= 0; --d) {
                if (++coord[d] < dst_shape[d]) break;
                coord[d] = 0;
            }
        }
        return std::tuple<CpuStorage, std::vector<std::int64_t>, std::int64_t>{
            std::move(dst), std::move(kept), contract_size};
    };

    auto [a_cpu, a_kept, a_contract] = contract(a, axes_a, /*put_first=*/false);
    auto [b_cpu, b_kept, b_contract] = contract(b, axes_b, /*put_first=*/true);
    if (a_contract != b_contract)
        throw LucidError("tensordot: contracted dim sizes don't match");

    const std::int64_t M = std::accumulate(a_kept.begin(), a_kept.end(),
                                           (std::int64_t)1, std::multiplies<>());
    const std::int64_t K = a_contract;
    const std::int64_t N = std::accumulate(b_kept.begin(), b_kept.end(),
                                           (std::int64_t)1, std::multiplies<>());

    Shape out_shape(a_kept.begin(), a_kept.end());
    out_shape.insert(out_shape.end(), b_kept.begin(), b_kept.end());
    auto out_cpu = allocate_cpu(out_shape, dt);

    auto gemm = [&](auto* op, const auto* ap, const auto* bp) {
        using T = std::remove_pointer_t<decltype(op)>;
        for (std::int64_t i = 0; i < M; ++i)
            for (std::int64_t j = 0; j < N; ++j) {
                T s{};
                for (std::int64_t k = 0; k < K; ++k)
                    s = s + ap[i * K + k] * bp[k * N + j];
                op[i * N + j] = s;
            }
    };
    if (dt == Dtype::F32)
        gemm(reinterpret_cast<float*>(out_cpu.ptr.get()),
             reinterpret_cast<const float*>(a_cpu.ptr.get()),
             reinterpret_cast<const float*>(b_cpu.ptr.get()));
    else if (dt == Dtype::F64)
        gemm(reinterpret_cast<double*>(out_cpu.ptr.get()),
             reinterpret_cast<const double*>(a_cpu.ptr.get()),
             reinterpret_cast<const double*>(b_cpu.ptr.get()));
    else
        throw NotImplementedError("tensordot: dtype not supported");

    return fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
}

}  // namespace lucid
