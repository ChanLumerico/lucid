// lucid/_C/ops/ufunc/Scan.h
//
// Public entry points for inclusive cumulative scan operations along a
// single axis: ``cumsum``, ``cumprod``, ``cummax``, ``cummin``.  Each
// output has exactly the same shape as the input; only the values are
// accumulated along ``axis``.
//
// The corresponding backward nodes (``CumsumBackward``,
// ``CumprodBackward``, ``CummaxBackward``, ``CumminBackward``) are
// defined inside ``Scan.cpp`` in an anonymous namespace and are not
// exposed through this header — callers should not depend on them.
//
// Backward overview
// -----------------
// - **cumsum**  — gradient at position $i$ is the suffix-sum of the
//   upstream gradient: $dx_i = \sum_{j \ge i} dy_j$.  Computed via the
//   *reverse-cumsum trick* $dx = \mathrm{reverse}(\mathrm{cumsum}
//   (\mathrm{reverse}(dy)))$.
// - **cumprod** — similar, weighted by the saved output and divided by
//   the saved input.
// - **cummax** / **cummin** — gradient flows only to the first
//   position that achieved the running extremum; computed by a
//   right-to-left segmented accumulation against the saved output.
//
// Notes
// -----
// All four ops save the output (``cumprod`` also saves the input) for
// use by their backward nodes.  ``axis`` may be negative; it is
// normalised to a non-negative index before being stored on the
// backward node.

#pragma once

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Inclusive cumulative sum along a single axis,
// $y_k = \sum_{j \le k} x_j$.
//
// The output has the same shape and dtype as the input.  Backward
// uses the reverse-cumsum trick to avoid an explicit $O(n^2)$
// summation: reverse the upstream gradient along the axis, take its
// cumsum, then reverse again.
//
// Math
// ----
// $$
//   y_k = \sum_{j \le k} x_j, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \sum_{j \ge i} \frac{\partial \mathcal{L}}{\partial y_j}.
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any non-scalar shape and real dtype.
// axis : int
//     Axis along which to accumulate.  May be negative; in that case
//     it is interpreted as ``axis + ndim``.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype as ``a``.
//
// Raises
// ------
// Error
//     If ``a`` is a scalar (``ndim == 0``) or ``axis`` is out of range.
//
// Shape
// -----
// Output shape equals ``a.shape``.
//
// Notes
// -----
// Dispatch: Accelerate strided cumsum (CPU) / MLX ``cumsum`` (GPU).
// Backward is `O(n)` per axis-stride.
LUCID_API TensorImplPtr cumsum_op(const TensorImplPtr& a, int axis);

// Inclusive cumulative product along a single axis,
// $y_k = \prod_{j \le k} x_j$.
//
// The output has the same shape and dtype as the input.  Backward
// reuses the reverse-cumsum primitive against the saved cumprod
// output and divides by the saved input:
// $dx = \mathrm{reverse}(\mathrm{cumsum}(\mathrm{reverse}(dy \cdot y)))
// / x$.
//
// Math
// ----
// $$
//   y_k = \prod_{j \le k} x_j, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \sum_{j \ge i} \frac{\partial \mathcal{L}}{\partial y_j}
//   \cdot \frac{y_j}{x_i}.
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any non-scalar shape and real dtype.
// axis : int
//     Axis along which to accumulate (negative wraps).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype as ``a``.
//
// Notes
// -----
// Backward is undefined where $x_i = 0$ because the formula divides
// by $x$.  Forward saves both $x$ and $y$ on the backward node.
LUCID_API TensorImplPtr cumprod_op(const TensorImplPtr& a, int axis);

// Inclusive cumulative maximum along a single axis,
// $y_k = \max_{j \le k} x_j$.
//
// Output is non-decreasing along ``axis``.  Backward credits the
// upstream gradient $dy_k$ to the position that **first** achieved the
// running maximum up to $k$ (i.e. the unique $i \le k$ at which the
// running max changed value); positions that merely tie with the
// running max receive no gradient.
//
// Math
// ----
// $$
//   y_k = \max_{j \le k} x_j, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \sum_{k \ge i} dy_k \cdot \mathbb{1}[\,i = \arg\max_{j \le k} x_j\,].
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any non-scalar shape; only F32 / F64 are
//     supported by the backward pass.
// axis : int
//     Axis along which to accumulate (negative wraps).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype as ``a``.
//
// Notes
// -----
// Backward is computed by a right-to-left segmented accumulation on
// CPU (or evaluated on CPU after a GPU round-trip).  Saves only the
// forward output; the input is not needed.
LUCID_API TensorImplPtr cummax_op(const TensorImplPtr& a, int axis);

// Inclusive cumulative minimum along a single axis,
// $y_k = \min_{j \le k} x_j$.
//
// Symmetric to :func:`cummax_op`: gradient flows only to the position
// that first achieved the running minimum at each $k$.
//
// Math
// ----
// $$
//   y_k = \min_{j \le k} x_j, \qquad
//   \frac{\partial \mathcal{L}}{\partial x_i} =
//   \sum_{k \ge i} dy_k \cdot \mathbb{1}[\,i = \arg\min_{j \le k} x_j\,].
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor of any non-scalar shape; backward supports F32 / F64.
// axis : int
//     Axis along which to accumulate (negative wraps).
//
// Returns
// -------
// TensorImplPtr
//     Output tensor of the same shape and dtype as ``a``.
//
// See Also
// --------
// :func:`cummax_op` — symmetric maximum scan.
LUCID_API TensorImplPtr cummin_op(const TensorImplPtr& a, int axis);

}  // namespace lucid
