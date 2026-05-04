// lucid/_C/ops/utils/MaskedSelect.cpp
#include "MaskedSelect.h"
#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"

namespace lucid {

TensorImplPtr masked_select_op(const TensorImplPtr& a, const TensorImplPtr& mask) {
    Validator::input(a,    "masked_select.a").non_null();
    Validator::input(mask, "masked_select.mask").non_null();

    auto& be = backend::Dispatcher::for_device(a->device());
    // Count true elements to determine output size.
    Storage count_s = be.masked_select_count(mask->storage(), mask->shape(), mask->dtype());
    // Retrieve count from count_s (CPU scalar int64).
    std::int64_t n = 0;
    {
        const auto& cs = std::get<CpuStorage>(count_s);
        std::memcpy(&n, cs.ptr.get(), sizeof(std::int64_t));
    }
    Shape out_shape{n};
    Storage out = be.masked_select(a->storage(), mask->storage(),
                                    a->shape(), mask->shape(), n, a->dtype());
    return std::make_shared<TensorImpl>(std::move(out), out_shape, a->dtype(), a->device(), false);
}

}  // namespace lucid
