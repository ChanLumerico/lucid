// lucid/_C/ops/utils/View.cpp
//
// Implements reshape, squeeze, and unsqueeze.  On the CPU, a dense input is
// reshaped by relabelling: the result is a view over the same buffer, in the
// input's ViewFamily.  So is a strided CPU view whose new shape only drops or
// inserts size-1 axes.  Anything else — any other CPU view, any GPU tensor —
// is copied by the backend dispatcher's reshape.  Either way the central
// helper build_view_output wires the ViewBackward autograd node so that
// gradients flow back through the changed shape.

#include "View.h"

#include <variant>
#include <vector>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "Contiguous.h"

namespace lucid {

// OpSchema registration for ViewBackward.  The "true" at position 9 marks
// this op as an in-place-compatible view (no data copy is required for the
// forward pass).
const OpSchema ViewBackward::schema_v1{"view", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, true};

namespace {

// Resolve the target shape for a reshape call, handling the single -1
// wildcard dimension.  Returns the fully-specified shape with all dimensions
// filled in.
//
// Throws via ErrorBuilder when:
//   - more than one -1 appears in new_shape
//   - a dimension value other than -1 is negative
//   - the known product of non-wildcard dims is zero (so inference is impossible)
//   - the inferred element count does not equal in_shape's numel
Shape resolve_reshape_shape(const Shape& in_shape, const std::vector<std::int64_t>& new_shape) {
    const std::size_t in_numel = shape_numel(in_shape);
    Shape resolved;
    resolved.reserve(new_shape.size());
    int wildcard_pos = -1;
    std::int64_t known_product = 1;
    for (std::size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (wildcard_pos != -1) {
                ErrorBuilder("reshape").fail("only one -1 is allowed");
            }
            wildcard_pos = static_cast<int>(i);
            resolved.push_back(0);  // placeholder; filled in below
        } else if (new_shape[i] < 0) {
            ErrorBuilder("reshape").fail("negative dim other than -1 is invalid");
        } else {
            known_product *= new_shape[i];
            resolved.push_back(new_shape[i]);
        }
    }
    if (wildcard_pos != -1) {
        if (known_product == 0) {
            ErrorBuilder("reshape").fail("cannot infer -1 from a product of zero");
        }
        if (in_numel % static_cast<std::size_t>(known_product) != 0) {
            throw ShapeMismatch(in_shape, resolved,
                                "reshape: -1 inference failed (numel mismatch)");
        }
        // Fill in the placeholder value computed from the remaining element count.
        resolved[wildcard_pos] = static_cast<std::int64_t>(in_numel) / known_product;
    }
    if (shape_numel(resolved) != in_numel) {
        throw ShapeMismatch(in_shape, resolved, "reshape: total numel mismatch");
    }
    return resolved;
}

// Whether ``out`` only drops or inserts size-1 axes of ``a``'s shape.  If it
// does, ``stride`` gets strides that leave every element of ``a`` where it
// was: the kept axes keep theirs, and a size-1 axis — which never moves the
// address — gets the one a packed layout would give it.
bool squeezes_only(const TensorImpl& a, const Shape& out, Stride& stride) {
    std::vector<std::int64_t> sizes;
    std::vector<std::int64_t> kept;
    for (std::size_t d = 0; d < a.shape().size(); ++d)
        if (a.shape()[d] != 1) {
            sizes.push_back(a.shape()[d]);
            kept.push_back(a.stride()[d]);
        }
    std::vector<std::int64_t> want;
    for (const auto n : out)
        if (n != 1)
            want.push_back(n);
    if (want != sizes)
        return false;
    stride.assign(out.size(), 0);
    std::size_t k = kept.size();
    std::int64_t next = static_cast<std::int64_t>(dtype_size(a.dtype()));
    for (std::size_t d = out.size(); d-- > 0;) {
        stride[d] = out[d] != 1 ? kept[--k] : next;
        next = stride[d] * out[d];
    }
    return true;
}

// Shared implementation used by reshape_op, squeeze_op, squeeze_all_op, and
// unsqueeze_op.  A dense CPU input, or a strided one whose new shape only drops
// or inserts size-1 axes, gives a view over the same buffer; anything else is
// copied.  Either way ViewBackward is attached so that autograd can reshape
// gradients back to the input's shape.
TensorImplPtr build_view_output(const TensorImplPtr& a, Shape out_shape, const char* op_name) {
    Validator::input(a, std::string(op_name) + ".a").non_null();
    OpScopeFull scope{op_name, a->device(), a->dtype(), out_shape};

    TensorImplPtr out;
    if (a->device() == Device::CPU && a->is_dense()) {
        // A dense CPU tensor holds its elements in row-major order from the
        // first byte of its buffer, which is what any shape with the same
        // element count reads there — so the result is a view of it: no
        // copy, and an in-place write to either reaches the other (see
        // ViewFamily).
        Stride stride = contiguous_stride(out_shape, dtype_size(a->dtype()));
        out = TensorImpl::make_view(a, out_shape, std::move(stride));
    } else if (Stride strided; a->device() == Device::CPU && storage_is_cpu(a->raw_storage()) &&
                               squeezes_only(*a, out_shape, strided)) {
        // A strided CPU view whose new shape only drops or inserts size-1
        // axes keeps every element where it was — ``x[:, 1]`` is a column
        // slice with its axis squeezed out — so the result is a view too.
        out = TensorImpl::make_view(a, out_shape, std::move(strided));
    } else {
        // Any other reshape of a CPU tensor that is not dense (a view at an
        // offset, or not contiguous) copies: ``storage()`` hands the backend
        // its elements packed.  GPU tensors keep copy semantics: an MLX array
        // cannot see a write made through another.
        Storage out_storage = backend::Dispatcher::for_device(a->device())
                                  .reshape(a->storage(), a->shape(), out_shape, a->dtype());
        out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(),
                                           a->device(), false);
    }
    // Autograd wires to ``a`` itself either way.
    kernel::NaryKernel<ViewBackward, 1>::wire_autograd({a}, out, false);
    return out;
}

}  // namespace

// Backward for all view ops: reshape the incoming gradient from out_shape_
// back to the original input shape recorded in input_shapes_[0].  Because
// the forward pass was a pure re-interpretation of a contiguous buffer, the
// backward is likewise a zero-copy reshape of the gradient buffer.
std::vector<Storage> ViewBackward::apply(Storage grad_out) {
    auto out = backend::Dispatcher::for_device(device_).reshape(grad_out, out_shape_,
                                                                input_shapes_[0], dtype_);
    return {std::move(out)};
}

// Graph-mode backward: reshape grad_out back to the input shape using reshape_op
// so the operation itself is recorded in the autograd graph.
std::vector<TensorImplPtr> ViewBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // Convert Shape (vector<size_t>) to vector<int64_t> for reshape_op.
    std::vector<std::int64_t> target_shape;
    target_shape.reserve(input_shapes_[0].size());
    for (auto s : input_shapes_[0])
        target_shape.push_back(static_cast<std::int64_t>(s));
    return {reshape_op(grad_out, target_shape)};
}

// Resolve the target shape (handling -1 wildcard inference), then delegate
// to build_view_output.  The resolved shape will have exactly the same numel
// as `a`; any mismatch throws ShapeMismatch before reaching the backend.
TensorImplPtr reshape_op(const TensorImplPtr& a, const std::vector<std::int64_t>& new_shape) {
    Validator::input(a, "reshape.a").non_null();
    Shape resolved = resolve_reshape_shape(a->shape(), new_shape);
    return build_view_output(a, std::move(resolved), "reshape");
}

// Remove the size-1 dimension at `dim` by building a new shape that omits
// the element at index `wrapped`.  Validates that the specified dimension
// exists and has size 1 before forwarding to build_view_output.
TensorImplPtr squeeze_op(const TensorImplPtr& a, int dim) {
    Validator::input(a, "squeeze.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    if (ndim == 0) {
        ErrorBuilder("squeeze").index_error("axis out of range for 0-d tensor");
    }
    const int wrapped = dim < 0 ? dim + ndim : dim;
    if (wrapped < 0 || wrapped >= ndim) {
        ErrorBuilder("squeeze").index_error("axis out of range");
    }
    if (a->shape()[static_cast<std::size_t>(wrapped)] != 1) {
        ErrorBuilder("squeeze").fail("target dim must be size 1");
    }
    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i != wrapped)
            new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
    }
    return build_view_output(a, std::move(new_shape), "squeeze");
}

// Remove every size-1 dimension from `a`, preserving all dimensions whose
// size is greater than 1.  If no size-1 dimensions exist, the output is a
// view with the unchanged shape.
TensorImplPtr squeeze_all_op(const TensorImplPtr& a) {
    Validator::input(a, "squeeze.a").non_null();
    Shape new_shape;
    for (auto d : a->shape()) {
        if (d != 1)
            new_shape.push_back(d);
    }
    return build_view_output(a, std::move(new_shape), "squeeze");
}

// Insert a new size-1 axis at the position given by `wrapped` (already
// resolved from `dim`).  The loop iterates over the output positions [0,
// ndim], inserts a 1 at `wrapped`, and copies the original dimensions at all
// other positions.  Negative dim is resolved against (ndim + 1) so that -1
// refers to the position just before the last dimension of the output.
TensorImplPtr unsqueeze_op(const TensorImplPtr& a, int dim) {
    Validator::input(a, "unsqueeze.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());

    const int wrapped = dim < 0 ? dim + ndim + 1 : dim;
    if (wrapped < 0 || wrapped > ndim) {
        ErrorBuilder("unsqueeze").index_error("axis out of range");
    }
    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) + 1);
    for (int i = 0; i <= ndim; ++i) {
        if (i == wrapped) {
            new_shape.push_back(1);
        }
        if (i < ndim) {
            new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
        }
    }
    return build_view_output(a, std::move(new_shape), "unsqueeze");
}

LUCID_REGISTER_OP(ViewBackward)

}  // namespace lucid
