#include "Meshgrid.h"

#include "../../autograd/FuncOp.h"
#include "../../autograd/Helpers.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"  // detail::ensure_grad_fn
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::check_dtype_device_match;
using utils_detail::fresh;

Storage meshgrid_backward_storage(const Storage& grad,
                                  const Shape& input_shape,
                                  const Shape& output_shape,
                                  int carry_axis,
                                  Dtype dt,
                                  Device device) {
    std::vector<int> axes;
    axes.reserve(output_shape.size() > 0 ? output_shape.size() - 1 : 0);
    for (int d = 0; d < static_cast<int>(output_shape.size()); ++d) {
        if (d != carry_axis)
            axes.push_back(d);
    }
    auto& backend = backend::Dispatcher::for_device(device);
    if (axes.empty())
        return backend.clone(grad, input_shape, dt);
    return backend.reduce_sum(grad, output_shape, backend::ReduceOpts{axes, false}, dt);
}

class MeshgridBackward : public FuncOp<MeshgridBackward, 1> {
public:
    static const OpSchema schema_v1;

    int carry_axis_ = 0;

    std::vector<Storage> apply(Storage grad_out) override {
        return {meshgrid_backward_storage(grad_out, input_shapes_[0], out_shape_, carry_axis_,
                                          dtype_, device_)};
    }
};

const OpSchema MeshgridBackward::schema_v1{"meshgrid", 1, AmpPolicy::KeepInput, true};

TensorImplPtr attach_meshgrid_grad(const TensorImplPtr& input,
                                   TensorImplPtr output,
                                   int carry_axis) {
    auto bwd = std::make_shared<MeshgridBackward>();
    bwd->carry_axis_ = carry_axis;
    kernel::NaryKernel<MeshgridBackward, 1>::wire_autograd(std::move(bwd), {input}, output,
                                                           /*save_ins=*/false);
    return output;
}

}  // namespace

std::vector<TensorImplPtr> meshgrid_op(const std::vector<TensorImplPtr>& xs, bool indexing_xy) {
    check_dtype_device_match(xs, "meshgrid");
    const Dtype dt = xs[0]->dtype();
    const Device device = xs[0]->device();
    OpScopeFull scope{"meshgrid", device, dt, Shape{}};
    const std::size_t N = xs.size();
    std::vector<std::int64_t> dims(N);
    for (std::size_t i = 0; i < N; ++i) {
        if (xs[i]->shape().size() != 1)
            ErrorBuilder("meshgrid").fail("each input must be 1-D");
        dims[i] = xs[i]->shape()[0];
    }
    std::vector<std::int64_t> out_dims = dims;
    if (indexing_xy && N >= 2)
        std::swap(out_dims[0], out_dims[1]);
    Shape out_shape(out_dims.begin(), out_dims.end());
    std::vector<Storage> in_storage;
    in_storage.reserve(xs.size());
    for (const auto& x : xs)
        in_storage.push_back(x->storage());
    std::vector<Storage> out_storage =
        backend::Dispatcher::for_device(device).meshgrid(in_storage, out_shape, dt, indexing_xy);

    std::vector<TensorImplPtr> result;
    result.reserve(out_storage.size());
    for (std::size_t i = 0; i < out_storage.size(); ++i) {
        std::size_t carry_axis = i;
        if (indexing_xy && N >= 2 && i < 2)
            carry_axis = 1 - i;
        auto out = fresh(std::move(out_storage[i]), out_shape, dt, device);
        result.push_back(attach_meshgrid_grad(xs[i], std::move(out), static_cast<int>(carry_axis)));
    }
    return result;
}

LUCID_REGISTER_OP(MeshgridBackward)

}  // namespace lucid
