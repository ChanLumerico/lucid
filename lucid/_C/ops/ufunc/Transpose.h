// lucid/_C/ops/ufunc/Transpose.h
//
// Backward node and public entry points for axis-permutation operations:
// permute, transpose (full reversal), T (.T property, alias for transpose),
// mT (batch matrix-transpose, swaps the last two axes), and swapaxes.
//
// All five public functions delegate to PermuteBackward::forward after building
// the appropriate permutation vector.  The forward pass is a metadata-only
// operation: it shares the underlying Storage and rewrites shape/stride, paying
// only an O(ndim) cost.
//
// Backward: applying the inverse permutation to grad_out restores it to the
// input's axis order.  inverse_perm(p)[i] = j such that p[j] == i.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for an arbitrary axis permutation
// $y_{i_{p_0}, i_{p_1}, \dots} = x_{i_0, i_1, \dots}$ with permutation
// $p$.
//
// The forward pass is a *view* — it does not copy data.  The output
// shares the input's underlying ``Storage`` and only rewrites its
// ``shape`` and ``stride`` according to ``perm``, paying an $O(\text{ndim})$
// bookkeeping cost.  Materialising the permuted layout into contiguous
// memory is the caller's responsibility (via a subsequent
// ``contiguous()`` call), which keeps simple ``.T`` followed by
// ``matmul`` free of redundant copies.
//
// The backward is the same kind of operation in reverse: apply the
// inverse permutation to ``grad_out`` so the gradient is reshuffled
// back to the input's axis order.
//
// Math
// ----
// $$y = \mathrm{permute}(x, p),
//   \qquad \frac{\partial L}{\partial x}
//   = \mathrm{permute}\!\Bigl(\frac{\partial L}{\partial y}, p^{-1}\Bigr)$$
//
// where $p^{-1}[p[i]] = i$ is the inverse permutation.
//
// Shape
// -----
// Output shape is ``[input.shape[perm[0]], input.shape[perm[1]], …]``.
// ``ndim`` is preserved.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"permute"``, ``AmpPolicy::KeepInput`` — pure metadata operation,
//     no promotion required.
// perm_ : std::vector<int>
//     Normalised (non-negative) forward permutation saved by ``forward``;
//     used to derive the inverse permutation in ``apply``.
//
// Raises
// ------
// ValueError
//     If ``perm`` is not a permutation of ``0..ndim-1`` or its length
//     differs from ``ndim``.
// IndexError
//     If any element of ``perm`` is outside ``[-ndim, ndim)``.
//
// Notes
// -----
// Composability: a ``permute`` followed by another ``permute`` does not
// materialise the intermediate layout; both forward passes rewrite
// metadata only, so chains like ``a.T.contiguous()`` pay exactly one
// data copy regardless of how many transposes precede the
// ``contiguous`` call.
class LUCID_API PermuteBackward : public FuncOp<PermuteBackward, 1> {
public:
    static const OpSchema schema_v1;

    // Normalised forward permutation saved from ``forward``; used to derive
    // the inverse permutation in ``apply``.
    std::vector<int> perm_;

    // Normalise ``perm_user``, build the output shape / stride, dispatch
    // the (view-only) forward permute through the backend, and wire the
    // backward node.
    //
    // Parameters
    // ----------
    // a : TensorImplPtr
    //     Input tensor.
    // perm : std::vector<int>
    //     Permutation of ``0..ndim-1``.  Negative indices are wrapped to
    //     ``axis + ndim`` before validation.
    //
    // Returns
    // -------
    // TensorImplPtr
    //     Output tensor sharing storage with ``a`` but with permuted
    //     shape and stride.
    //
    // Raises
    // ------
    // ValueError
    //     If ``perm`` is not a valid permutation of ``0..ndim-1``.
    // IndexError
    //     If any element of ``perm`` falls outside ``[-ndim, ndim)``.
    static TensorImplPtr forward(const TensorImplPtr& a, const std::vector<int>& perm);

    // Eager-mode backward: $\partial L/\partial x =
    // \mathrm{permute}(\partial L/\partial y, p^{-1})$.  ``out_shape_``
    // is the permuted output shape, which becomes the input shape of
    // the inverse permute call.
    std::vector<Storage> apply(Storage grad_out) override;

    // Graph-mode backward used when ``create_graph=True``.  Applies the
    // inverse permutation through ``permute_op`` so the result remains
    // tracked in the autograd graph for second-order differentiation.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override;

    // Node-name override used by the autograd graph dumper.
    std::string node_name() const override { return "permute"; }
};

// Permute all axes of ``a`` according to ``perm``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// perm : std::vector<int>
//     Permutation of ``0..ndim-1``.  Negative indices count from the
//     end.
//
// Returns
// -------
// TensorImplPtr
//     Output tensor sharing storage with ``a`` but with permuted shape
//     and stride (a view; no data copy).
//
// Raises
// ------
// ValueError
//     If ``perm`` is not a valid permutation of ``0..ndim-1``.
// IndexError
//     If any element of ``perm`` is outside ``[-ndim, ndim)``.
//
// See Also
// --------
// transpose_op : Full-reversal permutation.
// swapaxes_op : Swap two specific axes.
// PermuteBackward : Autograd node implementing the gradient rule.
LUCID_API TensorImplPtr permute_op(const TensorImplPtr& a, const std::vector<int>& perm);

// Reverse all axes of ``a``: equivalent to ``permute(a, [ndim-1, …, 1, 0])``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     View of ``a`` with reversed axis order.
//
// See Also
// --------
// T_op : Alias used to implement the ``.T`` property.
// mT_op : Batch-matrix-transpose variant that only swaps the last two
//     axes.
LUCID_API TensorImplPtr transpose_op(const TensorImplPtr& a);

// Backing function for the ``.T`` tensor property — alias of
// ``transpose_op``.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
//
// Returns
// -------
// TensorImplPtr
//     View of ``a`` with reversed axis order.
//
// See Also
// --------
// transpose_op : Identical functionality; this alias exists for the
//     Python-side ``.T`` property binding.
LUCID_API TensorImplPtr T_op(const TensorImplPtr& a);

// Batch matrix-transpose — swap the last two axes only, leaving all
// leading batch axes intact.  Backing function for the ``.mT`` tensor
// property and standard for batched matmul transposes.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor; must have ``ndim >= 2``.
//
// Returns
// -------
// TensorImplPtr
//     View of ``a`` with the final two axes swapped.
//
// Raises
// ------
// ValueError
//     If ``a.ndim < 2``.
LUCID_API TensorImplPtr mT_op(const TensorImplPtr& a);

// Swap two named axes of ``a``, wrapping negative indices.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// axis1, axis2 : int
//     Axes to swap.  Negative values count from the end.
//
// Returns
// -------
// TensorImplPtr
//     View of ``a`` with the two requested axes swapped and all other
//     axes left in place.
//
// Raises
// ------
// IndexError
//     If either axis falls outside ``[-ndim, ndim)`` after
//     normalisation.
LUCID_API TensorImplPtr swapaxes_op(const TensorImplPtr& a, int axis1, int axis2);

}  // namespace lucid
