#include "Tensordot.h"

#include <cstring>
#include <numeric>
#include <set>
#include <string>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../einops/Einops.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Build an einsum pattern equivalent to tensordot(a, b, axes_a, axes_b).
// Contracted axes share a label; non-contracted axes get distinct labels.
std::string tensordot_einsum_pattern(std::size_t na,
                                     std::size_t nb,
                                     const std::vector<int>& axes_a,
                                     const std::vector<int>& axes_b) {
    auto norm = [](int ax, std::size_t n) { return ax < 0 ? ax + static_cast<int>(n) : ax; };
    std::set<int> ca_set, cb_set;
    for (auto a : axes_a)
        ca_set.insert(norm(a, na));
    for (auto b : axes_b)
        cb_set.insert(norm(b, nb));

    std::string a_lhs(na, '?'), b_lhs(nb, '?'), rhs;
    char free = 'a';
    char shared = 'A';  // contracted labels go in the upper-case alphabet
    // First assign shared labels to contraction pairs, in axes_{a,b} order.
    for (std::size_t i = 0; i < axes_a.size(); ++i) {
        int pa = norm(axes_a[i], na);
        int pb = norm(axes_b[i], nb);
        a_lhs[pa] = shared;
        b_lhs[pb] = shared;
        ++shared;
    }
    // Then assign free labels to non-contracted axes (in original order).
    for (std::size_t i = 0; i < na; ++i) {
        if (a_lhs[i] == '?') {
            a_lhs[i] = free;
            rhs.push_back(free);
            ++free;
        }
    }
    for (std::size_t i = 0; i < nb; ++i) {
        if (b_lhs[i] == '?') {
            b_lhs[i] = free;
            rhs.push_back(free);
            ++free;
        }
    }
    return a_lhs + "," + b_lhs + "->" + rhs;
}

}  // namespace

TensorImplPtr tensordot_op(const TensorImplPtr& a,
                           const TensorImplPtr& b,
                           std::vector<int> axes_a,
                           std::vector<int> axes_b) {
    validate_pair(a, b, "tensordot");
    if (axes_a.size() != axes_b.size())
        ErrorBuilder("tensordot").fail("axes_a and axes_b must have equal length");

    // Autograd-tracked path: dispatch through einsum so primitive op
    // backwards stitch the gradient chain. Inference path keeps the native
    // gemm-based fast path below.
    if (GradMode::is_enabled() && (a->requires_grad() || b->requires_grad())) {
        return einsum_op(
            tensordot_einsum_pattern(a->shape().size(), b->shape().size(), axes_a, axes_b), {a, b});
    }

    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"tensordot", device, dt, Shape{}};

    if (device == Device::GPU) {
        auto out_storage = backend::Dispatcher::for_device(device).tensordot(
            a->storage(), b->storage(), a->shape(), b->shape(), Shape{}, axes_a, axes_b, dt);
        const auto& gs = std::get<GpuStorage>(out_storage);
        Shape out_shape;
        for (auto d : gs.arr->shape())
            out_shape.push_back(static_cast<std::int64_t>(d));
        return fresh(std::move(out_storage), std::move(out_shape), dt, device);
    }

    auto contract = [&](const TensorImplPtr& t, const std::vector<int>& axes_contract,
                        bool put_first) {
        const std::size_t nd = t->shape().size();
        std::vector<bool> is_c(nd, false);
        for (auto ax : axes_contract) {
            int p = ax < 0 ? ax + (int)nd : ax;
            if (p < 0 || p >= (int)nd)
                ErrorBuilder("tensordot").fail("axis out of range");
            is_c[p] = true;
        }
        std::vector<int> perm;
        std::vector<std::int64_t> kept;
        std::int64_t contract_size = 1;
        if (put_first) {
            for (auto ax : axes_contract) {
                int p = ax < 0 ? ax + (int)nd : ax;
                perm.push_back(p);
                contract_size *= t->shape()[p];
            }
            for (std::size_t d = 0; d < nd; ++d)
                if (!is_c[d]) {
                    perm.push_back((int)d);
                    kept.push_back(t->shape()[d]);
                }
        } else {
            for (std::size_t d = 0; d < nd; ++d)
                if (!is_c[d]) {
                    perm.push_back((int)d);
                    kept.push_back(t->shape()[d]);
                }
            for (auto ax : axes_contract) {
                int p = ax < 0 ? ax + (int)nd : ax;
                perm.push_back(p);
                contract_size *= t->shape()[p];
            }
        }
        Shape src_shape = t->shape();
        Shape dst_shape;
        for (auto p : perm)
            dst_shape.push_back(src_shape[p]);
        const auto& cs = std::get<CpuStorage>(t->storage());
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
            std::memcpy(dst.ptr.get() + f * elem, cs.ptr.get() + src_flat * elem, elem);
            for (std::ptrdiff_t d = (std::ptrdiff_t)nd - 1; d >= 0; --d) {
                if (++coord[d] < dst_shape[d])
                    break;
                coord[d] = 0;
            }
        }
        return std::tuple<CpuStorage, std::vector<std::int64_t>, std::int64_t>{
            std::move(dst), std::move(kept), contract_size};
    };

    auto [a_cpu, a_kept, a_contract] = contract(a, axes_a, /*put_first=*/false);
    auto [b_cpu, b_kept, b_contract] = contract(b, axes_b, /*put_first=*/true);
    if (a_contract != b_contract)
        ErrorBuilder("tensordot").fail("contracted dim sizes don't match");

    const std::int64_t M =
        std::accumulate(a_kept.begin(), a_kept.end(), (std::int64_t)1, std::multiplies<>());
    const std::int64_t K = a_contract;
    const std::int64_t N =
        std::accumulate(b_kept.begin(), b_kept.end(), (std::int64_t)1, std::multiplies<>());

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
        ErrorBuilder("tensordot").not_implemented("dtype not supported");

    return fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
}

}  // namespace lucid
