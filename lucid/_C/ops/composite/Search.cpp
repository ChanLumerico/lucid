// lucid/_C/ops/composite/Search.cpp
//
// ``searchsorted`` and ``bucketize`` via O(k log N) binary search in O(k)
// memory.  The previous broadcast-comparison composite materialised an
// ``S + [N]`` intermediate tensor (O(N * k) time and memory, OOM-killing
// 1M x 1M calls — LCD-218).  GPU inputs are transferred to host memory under
// H3's data-dependent index-lookup carve-out (like histogram_op) and the
// resulting int64 indices are uploaded back to the target device.

#include "Search.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/Helpers.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"

namespace lucid {

namespace {

std::string format_shape(const Shape& sh) {
    std::string out = "[";
    for (std::size_t i = 0; i < sh.size(); ++i) {
        if (i > 0)
            out += ", ";
        out += std::to_string(sh[i]);
    }
    out += "]";
    return out;
}

template <typename T>
std::int64_t bsearch_one(const T* seq, std::int64_t n, T v, bool right) {
    std::int64_t lo = 0;
    std::int64_t hi = n;
    if (right) {
        while (lo < hi) {
            const std::int64_t mid = lo + ((hi - lo) >> 1);
            if (!(v < seq[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
    } else {
        while (lo < hi) {
            const std::int64_t mid = lo + ((hi - lo) >> 1);
            if (!(v <= seq[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
    }
    return lo;
}

template <typename T>
void bsearch_rows(const T* seq,
                  const T* vals,
                  std::int64_t* out,
                  std::size_t outer,
                  std::int64_t n_seq,
                  std::size_t n_val,
                  bool seq_1d,
                  bool right) {
    for (std::size_t o = 0; o < outer; ++o) {
        const T* row_seq = seq_1d ? seq : (seq + o * static_cast<std::size_t>(n_seq));
        const T* row_val = vals + o * n_val;
        std::int64_t* row_out = out + o * n_val;
        for (std::size_t j = 0; j < n_val; ++j)
            row_out[j] = bsearch_one(row_seq, n_seq, row_val[j], right);
    }
}

void dispatch_bsearch(Dtype dt,
                      const CpuStorage& seq_s,
                      const CpuStorage& val_s,
                      std::int64_t* out,
                      std::size_t outer,
                      std::int64_t n_seq,
                      std::size_t n_val,
                      bool seq_1d,
                      bool right) {
    switch (dt) {
    case Dtype::F16:
    case Dtype::BF16: {
        const bool bf = dt == Dtype::BF16;
        const std::size_t seq_total = (seq_1d ? 1 : outer) * static_cast<std::size_t>(n_seq);
        const std::size_t val_total = outer * n_val;
        std::vector<float> wide_seq(seq_total), wide_val(val_total);
        backend::detail::half_or_bfloat16_to_float_bulk(
            reinterpret_cast<const std::uint16_t*>(seq_s.ptr.get()), wide_seq.data(), seq_total,
            bf);
        backend::detail::half_or_bfloat16_to_float_bulk(
            reinterpret_cast<const std::uint16_t*>(val_s.ptr.get()), wide_val.data(), val_total,
            bf);
        bsearch_rows(wide_seq.data(), wide_val.data(), out, outer, n_seq, n_val, seq_1d, right);
        return;
    }
    case Dtype::F32:
        bsearch_rows(reinterpret_cast<const float*>(seq_s.ptr.get()),
                     reinterpret_cast<const float*>(val_s.ptr.get()), out, outer, n_seq, n_val,
                     seq_1d, right);
        return;
    case Dtype::F64:
        bsearch_rows(reinterpret_cast<const double*>(seq_s.ptr.get()),
                     reinterpret_cast<const double*>(val_s.ptr.get()), out, outer, n_seq, n_val,
                     seq_1d, right);
        return;
    case Dtype::Bool:
        bsearch_rows(reinterpret_cast<const std::uint8_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::uint8_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I8:
        bsearch_rows(reinterpret_cast<const std::int8_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int8_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I16:
        bsearch_rows(reinterpret_cast<const std::int16_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int16_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I32:
        bsearch_rows(reinterpret_cast<const std::int32_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int32_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I64:
        bsearch_rows(reinterpret_cast<const std::int64_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int64_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    default:
        ErrorBuilder("searchsorted").not_implemented("complex dtype is not supported");
    }
}

void check_search_shapes(const Shape& ss, const Shape& vs, bool& seq_1d) {
    if (ss.empty())
        ErrorBuilder("searchsorted").invalid_argument("sorted_sequence cannot be 0-D");
    seq_1d = ss.size() == 1;
    if (seq_1d)
        return;
    bool match = vs.size() == ss.size();
    for (std::size_t i = 0; match && i + 1 < ss.size(); ++i)
        match = ss[i] == vs[i];
    if (!match)
        ErrorBuilder("searchsorted")
            .invalid_argument("boundaries tensor should be 1 dimension or the first N-1 "
                              "dimensions of boundaries tensor and input value tensor must match, "
                              "got boundaries " +
                              format_shape(ss) + " and input value " + format_shape(vs));
}

}  // namespace

TensorImplPtr
searchsorted_op(const TensorImplPtr& sorted_1d, const TensorImplPtr& values, bool right) {
    Validator::pair(sorted_1d, values, "searchsorted").both_non_null().same_dtype();
    const auto [seq, val] = detail::align_binary_devices(sorted_1d, values, "searchsorted");
    bool seq_1d = true;
    check_search_shapes(seq->shape(), val->shape(), seq_1d);

    const Device dv = val->device();
    const Dtype dt = seq->dtype();
    const Shape& vs = val->shape();
    OpScopeFull scope{"searchsorted", dv, Dtype::I64, vs};
    scope.set_attr("right", right ? std::int64_t{1} : std::int64_t{0});

    auto& be = backend::Dispatcher::for_device(dv);
    CpuStorage seq_cpu = be.to_cpu(seq->storage(), seq->shape());
    CpuStorage val_cpu = be.to_cpu(val->storage(), vs);
    CpuStorage out_cpu = helpers::allocate_cpu(vs, Dtype::I64);

    const std::int64_t n_seq = seq->shape().back();
    const std::size_t total_val = shape_numel(vs);
    const std::size_t n_val =
        seq_1d ? total_val : (vs.empty() ? 1 : static_cast<std::size_t>(vs.back()));
    const std::size_t outer = seq_1d ? 1 : (n_val == 0 ? 0 : total_val / n_val);

    if (total_val > 0) {
        dispatch_bsearch(dt, seq_cpu, val_cpu, reinterpret_cast<std::int64_t*>(out_cpu.ptr.get()),
                         outer, n_seq, n_val, seq_1d, right);
    }

    Storage out_s =
        dv == Device::GPU ? be.from_cpu(std::move(out_cpu), vs) : Storage{std::move(out_cpu)};
    auto out = helpers::fresh(std::move(out_s), vs, Dtype::I64, dv);
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({sorted_1d, values}, out);
    return out;
}

TensorImplPtr
bucketize_op(const TensorImplPtr& values, const TensorImplPtr& boundaries, bool right) {
    Validator::pair(values, boundaries, "bucketize").both_non_null().same_dtype();
    const auto [val, bnd] = detail::align_binary_devices(values, boundaries, "bucketize");
    if (bnd->shape().size() != 1)
        ErrorBuilder("bucketize")
            .invalid_argument("boundaries tensor must be 1 dimension, but got dim(" +
                              std::to_string(bnd->shape().size()) + ")");
    return searchsorted_op(bnd, val, right);
}

}  // namespace lucid
