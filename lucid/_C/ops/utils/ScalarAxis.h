// lucid/_C/ops/utils/ScalarAxis.h
//
// A one-axis op applied to a 0-d tensor.
//
// The reference gives a 0-d tensor one notional axis wherever an op takes a
// dim: 0 and -1 both name it (``wrap_dim``), and the answer is 0-d again —
// sort and cumsum return the scalar, argsort and topk index 0, softmax 1.
// The kernels need a real axis, so the op runs on the scalar viewed as shape
// (1,) along axis 0 and each result is viewed back to 0-d.  Both views are
// reshapes, so autograd and the compile trace see ordinary ops.

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "../../autograd/Helpers.h"
#include "../../core/TensorImpl.h"
#include "View.h"

namespace lucid {

// ``run(a viewed as (1,), 0)`` for a 0-d ``a`` whose ``dim`` names its one
// notional axis, with every result viewed back to 0-d.  ``run`` returns a
// tensor or a vector of tensors.
template <class Run>
auto on_scalar_axis(const TensorImplPtr& a, int dim, const char* op, Run&& run) {
    wrap_dim(dim, 0, op);
    auto out = std::forward<Run>(run)(reshape_op(a, {1}), 0);
    if constexpr (std::is_same_v<decltype(out), std::vector<TensorImplPtr>>) {
        for (auto& t : out)
            t = reshape_op(t, {});
        return out;
    } else {
        return reshape_op(out, {});
    }
}

}  // namespace lucid
