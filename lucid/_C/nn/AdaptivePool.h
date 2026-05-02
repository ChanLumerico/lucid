#pragma once

#include <vector>

#include "../api.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr adaptive_max_pool1d_op(const TensorImplPtr& x, int OL);

LUCID_API TensorImplPtr adaptive_max_pool2d_op(const TensorImplPtr& x, int OH, int OW);

LUCID_API TensorImplPtr adaptive_max_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);

LUCID_API TensorImplPtr adaptive_avg_pool1d_op(const TensorImplPtr& x, int OL);

LUCID_API TensorImplPtr adaptive_avg_pool2d_op(const TensorImplPtr& x, int OH, int OW);

LUCID_API TensorImplPtr adaptive_avg_pool3d_op(const TensorImplPtr& x, int OD, int OH, int OW);

}  // namespace lucid
