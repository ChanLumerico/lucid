#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

/// QR decomposition. Returns [Q, R].
LUCID_API std::vector<TensorImplPtr> qr_op(const TensorImplPtr& a);

}  // namespace lucid
