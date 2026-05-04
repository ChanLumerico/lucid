// lucid/_C/ops/ufunc/Astype.cpp
#include "Astype.h"
#include "../../backend/Dispatcher.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
namespace lucid {

TensorImplPtr astype_op(const TensorImplPtr& a, Dtype dst_dtype) {
    Validator::input(a, "astype").non_null();
    if (a->dtype() == dst_dtype) {
        // No-op: share the storage, just wrap with confirmed dtype.
        return std::make_shared<TensorImpl>(
            a->storage(), a->shape(), dst_dtype, a->device(), false);
    }
    const auto& shape = a->shape();
    auto& be = backend::Dispatcher::for_device(a->device());
    Storage out = be.astype(a->storage(), shape, a->dtype(), dst_dtype);
    return std::make_shared<TensorImpl>(
        std::move(out), shape, dst_dtype, a->device(), false);
}

}  // namespace lucid
