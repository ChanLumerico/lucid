// lucid/_C/ops/composite/Indexing.cpp
//
// All ops below decompose into existing primitives.  Index dtype is
// validated up front so error messages name the failing op rather than the
// underlying gather/scatter dispatch.

#include "Indexing.h"

#include <cstring>
#include <variant>
#include <vector>

#include "../../backend/Dispatcher.h"
#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../bfunc/Sub.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Concat.h"
#include "../utils/Layout.h"
#include "../utils/Select.h"
#include "../utils/Sort.h"
#include "../utils/View.h"

namespace lucid {

namespace {

// Resolve a possibly-negative ``dim`` against ``a``'s rank.
int wrap_dim(const TensorImplPtr& a, int dim, const char* op) {
    const int ndim = static_cast<int>(a->shape().size());
    int d = dim < 0 ? dim + ndim : dim;
    if (d < 0 || d >= ndim)
        ErrorBuilder(op).index_error("dim out of range");
    return d;
}

// ``gather_op`` requires int32/int64 indices; we surface a per-op error so
// the user knows which call rejected the dtype.
void require_index_dtype(const TensorImplPtr& idx, const char* op) {
    if (idx->dtype() != Dtype::I32 && idx->dtype() != Dtype::I64)
        ErrorBuilder(op).fail("indices must be int32 or int64");
}

}  // namespace

TensorImplPtr take_op(const TensorImplPtr& a, const TensorImplPtr& indices) {
    if (!a || !indices)
        ErrorBuilder("take").fail("null input");
    require_index_dtype(indices, "take");

    // ``gather`` along a freshly-flattened axis 0 lifts the multi-dim layout
    // into a single contiguous buffer; ReshapeBackward + GatherBackward
    // jointly carry the gradient back to the original shape.
    const std::int64_t total = static_cast<std::int64_t>(a->numel());
    auto flat = reshape_op(a, Shape{total});
    return gather_op(flat, indices, 0);
}

TensorImplPtr index_select_op(const TensorImplPtr& a, int dim, const TensorImplPtr& indices) {
    if (!a || !indices)
        ErrorBuilder("index_select").fail("null input");
    require_index_dtype(indices, "index_select");
    if (indices->shape().size() != 1)
        ErrorBuilder("index_select").fail("indices must be 1-D");

    const int d = wrap_dim(a, dim, "index_select");
    const int ndim = static_cast<int>(a->shape().size());
    const std::int64_t k = indices->shape()[0];

    // Reshape the 1-D index list to rank ``a`` with size ``k`` along ``d``
    // and 1 elsewhere; expand to the source shape so ``gather_op``'s same-
    // rank-as-input contract holds.
    Shape idx_reshaped(static_cast<std::size_t>(ndim), 1);
    idx_reshaped[static_cast<std::size_t>(d)] = k;
    auto idx_r = reshape_op(indices, idx_reshaped);
    Shape idx_target = a->shape();
    idx_target[static_cast<std::size_t>(d)] = k;
    auto idx_full = expand_op(idx_r, idx_target);
    return gather_op(a, idx_full, d);
}

TensorImplPtr
narrow_op(const TensorImplPtr& a, int dim, std::int64_t start, std::int64_t length) {
    if (!a)
        ErrorBuilder("narrow").fail("null input");
    const int d = wrap_dim(a, dim, "narrow");
    const std::int64_t size = a->shape()[static_cast<std::size_t>(d)];
    if (start < 0 || length < 0 || start + length > size)
        ErrorBuilder("narrow").index_error("range out of bounds");

    // Full-axis slice — no split needed; return the input directly.
    if (start == 0 && length == size)
        return a;

    // Cut the axis at the window boundaries and pick the middle (or first /
    // last) piece.  ``split_at_op`` carries autograd via SplitSliceBackward.
    std::vector<std::int64_t> cuts;
    int wanted = 0;
    if (start > 0) {
        cuts.push_back(start);
        wanted = 1;
    }
    if (start + length < size)
        cuts.push_back(start + length);
    auto pieces = split_at_op(a, cuts, d);
    return pieces[static_cast<std::size_t>(wanted)];
}

TensorImplPtr scatter_op(const TensorImplPtr& base,
                         int dim,
                         const TensorImplPtr& indices,
                         const TensorImplPtr& src) {
    if (!base || !indices || !src)
        ErrorBuilder("scatter").fail("null input");
    require_index_dtype(indices, "scatter");

    const int d = wrap_dim(base, dim, "scatter");

    // Overwrite via add: feed ``scatter_add`` the delta ``src - base[idx]``.
    auto existing = gather_op(base, indices, d);
    auto delta = sub_op(src, existing);
    return scatter_add_op(base, indices, delta, d);
}

TensorImplPtr
kthvalue_op(const TensorImplPtr& a, std::int64_t k, int dim, bool keepdim) {
    if (!a)
        ErrorBuilder("kthvalue").fail("null input");
    const int d = wrap_dim(a, dim, "kthvalue");
    const std::int64_t size = a->shape()[static_cast<std::size_t>(d)];
    if (k < 1 || k > size)
        ErrorBuilder("kthvalue").fail("k out of range");

    auto sorted = sort_op(a, d);

    // Build a same-rank index tensor whose entries are all ``k - 1`` along
    // ``d`` and 1 elsewhere — what ``gather_op`` needs to pluck a single
    // slice.  We allocate via the backend directly because ``full_like_op``
    // would carry a redundant graph dependency on ``a``.
    Shape idx_shape = a->shape();
    idx_shape[static_cast<std::size_t>(d)] = 1;
    Storage idx_storage = backend::Dispatcher::for_device(a->device()).zeros(idx_shape, Dtype::I64);

    std::size_t numel = 1;
    for (auto s : idx_shape) numel *= static_cast<std::size_t>(s);
    std::vector<std::int64_t> host(numel, k - 1);
    if (a->device() == Device::CPU) {
        auto& cs = std::get<CpuStorage>(idx_storage);
        std::memcpy(cs.ptr.get(), host.data(), numel * sizeof(std::int64_t));
    } else {
        // GPU path: build a CPU buffer then upload via the backend.
        CpuStorage stage;
        stage.dtype = Dtype::I64;
        stage.nbytes = numel * sizeof(std::int64_t);
        stage.ptr = allocate_aligned_bytes(stage.nbytes);
        std::memcpy(stage.ptr.get(), host.data(), stage.nbytes);
        idx_storage =
            backend::Dispatcher::for_device(Device::GPU).from_cpu(std::move(stage), idx_shape);
    }
    auto idx_tensor = std::make_shared<TensorImpl>(
        std::move(idx_storage), idx_shape, Dtype::I64, a->device(), false);

    auto val = gather_op(sorted, idx_tensor, d);
    if (keepdim)
        return val;
    return squeeze_op(val, d);
}

}  // namespace lucid
