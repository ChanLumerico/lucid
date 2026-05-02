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

namespace lucid {

const OpSchema ViewBackward::schema_v1{"view", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, true};

namespace {

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
            resolved.push_back(0);
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
        resolved[wildcard_pos] = static_cast<std::int64_t>(in_numel) / known_product;
    }
    if (shape_numel(resolved) != in_numel) {
        throw ShapeMismatch(in_shape, resolved, "reshape: total numel mismatch");
    }
    return resolved;
}

TensorImplPtr build_view_output(const TensorImplPtr& a, Shape out_shape, const char* op_name) {
    Validator::input(a, std::string(op_name) + ".a").non_null();
    OpScopeFull scope{op_name, a->device(), a->dtype(), out_shape};

    Storage out_storage = backend::Dispatcher::for_device(a->device())
                              .reshape(a->storage(), a->shape(), out_shape, a->dtype());
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(),
                                            a->device(), false);

    kernel::NaryKernel<ViewBackward, 1>::wire_autograd({a}, out, false);
    return out;
}

}  // namespace

std::vector<Storage> ViewBackward::apply(Storage grad_out) {
    auto out = backend::Dispatcher::for_device(device_).reshape(grad_out, out_shape_,
                                                                input_shapes_[0], dtype_);
    return {std::move(out)};
}

TensorImplPtr reshape_op(const TensorImplPtr& a, const std::vector<std::int64_t>& new_shape) {
    Validator::input(a, "reshape.a").non_null();
    Shape resolved = resolve_reshape_shape(a->shape(), new_shape);
    return build_view_output(a, std::move(resolved), "reshape");
}

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

TensorImplPtr squeeze_all_op(const TensorImplPtr& a) {
    Validator::input(a, "squeeze.a").non_null();
    Shape new_shape;
    for (auto d : a->shape()) {
        if (d != 1)
            new_shape.push_back(d);
    }
    return build_view_output(a, std::move(new_shape), "squeeze");
}

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
