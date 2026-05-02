#pragma once

#include <memory>

namespace lucid {

class TensorImpl;
using TensorImplPtr = std::shared_ptr<TensorImpl>;

TensorImplPtr contiguous_op(const TensorImplPtr& a);

}  // namespace lucid
