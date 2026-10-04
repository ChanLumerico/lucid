// lucid/_C/ops/composite/Indexing.cpp
//
// The ops below decompose into existing primitives, except ``scatter``, which
// dispatches the backend's overwrite kernel and carries its own backward.
// Index dtype is validated up front so error messages name the failing op
// rather than the underlying gather/scatter dispatch.

#include "Indexing.h"

#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"  // detail::ensure_grad_fn
#include "../gfunc/Gfunc.h"
#include "../ufunc/Astype.h"
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
    Validator::pair(a, indices, "index_select").same_device();
    require_index_dtype(indices, "index_select");
    if (indices->shape().size() != 1)
        ErrorBuilder("index_select").fail("indices must be 1-D");

    const int d = wrap_dim(a, dim, "index_select");
    const int ndim = static_cast<int>(a->shape().size());
    const std::int64_t k = indices->shape()[0];

    // Reshape the 1-D index list to rank ``a`` with size ``k`` along ``d``
    // and 1 elsewhere; broadcast it to the source shape so ``gather_op``'s
    // same-rank-as-input contract holds.  ``broadcast_to``, not ``expand``:
    // on the CPU ``expand`` is a zero-stride view, which the gather then
    // copied element by element to make contiguous — 2.4 ms of a 4 ms
    // select of a (3, 375, 500) image.  ``broadcast_to`` materialises it in
    // runs.  The index carries no gradient, so nothing is lost.
    Shape idx_reshaped(static_cast<std::size_t>(ndim), 1);
    idx_reshaped[static_cast<std::size_t>(d)] = k;
    auto idx_r = reshape_op(indices, idx_reshaped);
    Shape idx_target = a->shape();
    idx_target[static_cast<std::size_t>(d)] = k;
    auto idx_full = broadcast_to_op(idx_r, idx_target);
    return gather_op(a, idx_full, d);
}

TensorImplPtr narrow_op(const TensorImplPtr& a, int dim, std::int64_t start, std::int64_t length) {
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

namespace {

// The overwrite's backward.  ``out`` is ``base`` except where the index
// points, which holds ``src``: so ``base`` takes the gradient with those
// positions zeroed, and ``src`` takes the gradient read back from where each
// of its elements landed.  That is the reference's rule, duplicates
// included — every element aimed at one position gathers its gradient,
// though only one of them survived the write.
struct ScatterSetNode : Node {
    int dim_ = 0;
    TensorImplPtr saved_indices_;
    Shape base_shape_;
    Shape idx_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    std::string node_name() const override { return "scatter_set"; }
    void release_saved() override { saved_indices_.reset(); }

    std::vector<Storage> apply(Storage g) override {
        auto& be = backend::Dispatcher::for_device(device_);
        auto g_impl = std::make_shared<TensorImpl>(g, base_shape_, dtype_, device_, false);
        Storage grad_src = gather_op(g_impl, saved_indices_, dim_)->storage();
        Storage grad_base =
            be.scatter_set(g, saved_indices_->storage(), be.zeros(idx_shape_, dtype_), base_shape_,
                           idx_shape_, dim_, dtype_);
        return {std::move(grad_base), std::move(grad_src)};
    }

    // The same two halves as ops, so the gradient is differentiable again:
    // zeroing positions is itself an overwrite, linear in ``g``.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& g) override {
        auto zeros = zeros_op(idx_shape_, g->dtype(), g->device(), false);
        return {scatter_op(g, dim_, saved_indices_, zeros), gather_op(g, saved_indices_, dim_)};
    }
};

std::string shape_str(const Shape& sh) {
    std::string out = "[";
    for (std::size_t i = 0; i < sh.size(); ++i)
        out += (i ? ", " : "") + std::to_string(sh[i]);
    return out + "]";
}

}  // namespace

TensorImplPtr scatter_op(const TensorImplPtr& base,
                         int dim,
                         const TensorImplPtr& indices,
                         const TensorImplPtr& src) {
    if (!base || !indices || !src)
        ErrorBuilder("scatter").fail("null input");
    Validator::pair(base, indices, "scatter").same_device();
    Validator::pair(base, src, "scatter").same_device();
    require_index_dtype(indices, "scatter");

    // A 0-d operand counts as one element along a single axis, as in the
    // reference: a 0-d tensor scatters, and a 1-d one takes a 0-d index.
    const bool scalar = base->shape().empty() || indices->shape().empty() || src->shape().empty();
    if (scalar && base->shape().size() <= 1) {
        if (dim < -1 || dim > 0)
            ErrorBuilder("scatter").index_error("dim out of range");
        const auto one = [](const TensorImplPtr& t) {
            return t->shape().empty() ? reshape_op(t, Shape{1}) : t;
        };
        auto out = scatter_op(one(base), 0, one(indices), one(src));
        return base->shape().empty() ? reshape_op(out, Shape{}) : out;
    }

    const int d = wrap_dim(base, dim, "scatter");
    const int ndim = static_cast<int>(base->shape().size());
    const Shape& bs = base->shape();
    const Shape& is = indices->shape();
    const Shape& ss = src->shape();

    // The index walks src and lands in base, so it may be no larger than src
    // on any axis, nor than base on any but ``dim`` — the reference's rule,
    // and ``scatter_add``'s.
    if (is.size() != bs.size() || ss.size() != bs.size())
        ErrorBuilder("scatter").fail("index " + shape_str(is) + ", self " + shape_str(bs) +
                                     " and src " + shape_str(ss) +
                                     " must have the same number of dimensions");
    for (int i = 0; i < ndim; ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (is[k] > ss[k] || (i != d && is[k] > bs[k]))
            ErrorBuilder("scatter").fail("Expected index " + shape_str(is) +
                                         " to be no larger than self " + shape_str(bs) +
                                         " apart from dimension " + std::to_string(d) +
                                         " and to be no larger than src " + shape_str(ss));
    }
    // Any index into an empty axis is out of range; saying so needs only the
    // shapes, so Metal — which reads no index back — refuses it too.
    if (bs[static_cast<std::size_t>(d)] == 0 && shape_numel(is) > 0)
        ErrorBuilder("scatter").index_error("index out of range: dimension " + std::to_string(d) +
                                            " has size 0");

    // Only the corner of src the index covers is written; cut src to that
    // corner so every backend reads it in the index's own layout.  The
    // values are moved, not converted, so they take base's dtype first —
    // the cast ``index_copy`` makes in Python (``_like_input``).
    TensorImplPtr values = src;
    for (int i = 0; i < ndim; ++i) {
        const auto k = static_cast<std::size_t>(i);
        if (ss[k] > is[k])
            values = narrow_op(values, i, 0, is[k]);
    }
    const Dtype dt = base->dtype();
    if (values->dtype() != dt)
        values = astype_op(values, dt);

    // A true overwrite.  This was a scatter-add of the delta
    // ``src - base[idx]``, which is not one: NaN or inf in ``base`` stayed
    // NaN (``x[isnan(x)] = 0`` did nothing), a value far smaller than the
    // one it replaced rounded away (``x[0] = 1e-30`` wrote 0), bool could
    // not become False, a half overflowed in the subtraction, and a
    // repeated index summed its deltas (CHA-158).
    const Device dv = base->device();
    OpScopeFull scope{"scatter_set", dv, dt, bs};
    scope.set_attr("dim", static_cast<std::int64_t>(d));
    auto& be = backend::Dispatcher::for_device(dv);
    Storage out_s =
        be.scatter_set(base->storage(), indices->storage(), values->storage(), bs, is, d, dt);
    auto out = std::make_shared<TensorImpl>(std::move(out_s), bs, dt, dv, false);
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({base, indices, values}, out);

    if (!GradMode::is_enabled() || !(base->requires_grad() || values->requires_grad()))
        return out;

    auto bwd = std::make_shared<ScatterSetNode>();
    bwd->dim_ = d;
    bwd->saved_indices_ = indices;
    bwd->base_shape_ = bs;
    bwd->idx_shape_ = is;
    bwd->dtype_ = dt;
    bwd->device_ = dv;
    bwd->set_next_edges({Edge(detail::ensure_grad_fn(base), base->grad_output_nr()),
                         Edge(detail::ensure_grad_fn(values), values->grad_output_nr())});
    bwd->set_saved_versions({base->version(), values->version()});
    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

TensorImplPtr kthvalue_op(const TensorImplPtr& a, std::int64_t k, int dim, bool keepdim) {
    if (!a)
        ErrorBuilder("kthvalue").fail("null input");
    // Refused for bool as the reference does; it sorts underneath, and
    // sorting bool is allowed, so without this the answer depended on it.
    if (a->dtype() == Dtype::Bool)
        ErrorBuilder("kthvalue")
            .not_implemented("kthvalue does not accept bool — cast to an integer dtype");
    const int d = wrap_dim(a, dim, "kthvalue");
    const std::int64_t size = a->shape()[static_cast<std::size_t>(d)];
    if (k < 1 || k > size)
        ErrorBuilder("kthvalue").invalid_argument("k out of range");

    auto sorted = sort_op(a, d);

    // Build a same-rank index tensor whose entries are all ``k - 1`` along
    // ``d`` and 1 elsewhere — what ``gather_op`` needs to pluck a single
    // slice.  We allocate via the backend directly because ``full_like_op``
    // would carry a redundant graph dependency on ``a``.
    Shape idx_shape = a->shape();
    idx_shape[static_cast<std::size_t>(d)] = 1;
    Storage idx_storage = backend::Dispatcher::for_device(a->device()).zeros(idx_shape, Dtype::I64);

    std::size_t numel = 1;
    for (auto s : idx_shape)
        numel *= static_cast<std::size_t>(s);
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
    auto idx_tensor = std::make_shared<TensorImpl>(std::move(idx_storage), idx_shape, Dtype::I64,
                                                   a->device(), false);

    auto val = gather_op(sorted, idx_tensor, d);
    if (keepdim)
        return val;
    return squeeze_op(val, d);
}

}  // namespace lucid
