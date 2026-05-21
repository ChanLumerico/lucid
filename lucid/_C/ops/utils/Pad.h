// lucid/_C/ops/utils/Pad.h
//
// Declares the constant-fill padding operation.  The op supports padding each
// dimension independently with (before, after) widths and fills the added
// elements with a caller-supplied constant value.

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Pad a tensor on each side of each dimension with a constant fill value.
//
// ``pad_width`` is a per-dimension list of ``(before, after)`` widths whose
// length must equal ``a.rank()``.  The output preserves the input's dtype and
// device; newly added elements are filled with ``constant``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// pad_width : vector<pair<int64_t, int64_t>>
//     One ``(before, after)`` pair per dimension of ``a``.  Both widths must
//     be non-negative.
// constant : double
//     Fill value for the padded region.  Cast to the input's dtype before
//     storing.
//
// Returns
// -------
// TensorImplPtr
//     Padded tensor with shape
//     ``out.shape[d] = a.shape[d] + before_d + after_d``.
//
// Shape
// -----
// For each dimension $d$:
// $$ \text{out.shape}[d] = \text{a.shape}[d] + \text{pad\_width}[d].\text{first}
//                          + \text{pad\_width}[d].\text{second}. $$
//
// Math
// ----
// $$ y_{[i_0, \ldots, i_{n-1}]} =
//        \begin{cases}
//            x_{[i_0 - b_0, \ldots, i_{n-1} - b_{n-1}]}
//                & \text{if } b_d \le i_d < b_d + D_d \text{ for all } d, \\
//            c   & \text{otherwise},
//        \end{cases} $$
// where $b_d$ is the leading pad width, $D_d$ is the input size at axis
// $d$, and $c$ is ``constant``.
//
// Raises
// ------
// ShapeMismatch
//     If ``pad_width.size() != a.rank()`` or any width is negative.
//
// Notes
// -----
// Backward is implemented in the .cpp via ``PadBackward`` (schema name
// ``"pad"``, ``AmpPolicy::KeepInput``).  It slices the gradient sequentially
// per dimension, discarding ``pad_width[d].first`` leading and
// ``pad_width[d].second`` trailing elements to recover the original input
// region.  Only the constant-fill mode is supported here; reflect / replicate
// modes live in higher-level Python wrappers.
//
// Examples
// --------
// Padding a ``(2, 2)`` tensor with ``pad_width = {(1, 0), (0, 2)}`` and
// ``constant = 0`` yields a ``(3, 4)`` output whose first row and last two
// columns are zeros.
//
// See Also
// --------
// :func:`concatenate_op` — pad is conceptually concat with constant tensors.
LUCID_API TensorImplPtr pad_op(const TensorImplPtr& a,
                               std::vector<std::pair<std::int64_t, std::int64_t>> pad_width,
                               double constant);

}  // namespace lucid
