#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API std::pair<TensorImplPtr, TensorImplPtr> eig_op(const TensorImplPtr& a);

}  // namespace lucid
