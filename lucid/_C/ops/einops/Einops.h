// lucid/_C/ops/einops/Einops.h
//
// Public entry-point declarations for the einops operation family in the
// Lucid engine: ``rearrange``, ``reduce``, ``repeat``, and ``einsum``.
// Together these four ops cover the full einops API surface exposed by the
// Python ``lucid.einops`` module.
//
// Each function follows the ``<name>_op`` convention used throughout the
// Lucid C++ codebase.  They take ``TensorImplPtr`` inputs and return a
// ``TensorImplPtr`` output; the pybind11 bindings call these directly.
//
// Pattern syntax
// --------------
// All ops except ``einsum`` accept a string of the form ``"lhs -> rhs"``
// parsed by ``_Pattern.h``:
//
//   - Axis names are alphanumeric identifiers (``[A-Za-z_][A-Za-z0-9_]*``).
//   - Parentheses group multiple axes into one merged dimension (rhs) or
//     decompose a single input dimension into named components (lhs).
//   - Integer literals assert a fixed-size dimension constraint.
//
// Example: ``"b c (h w) -> b (c h) w"``
//   - ``(h w)`` on the lhs splits one input dim into named axes ``h`` and ``w``.
//   - ``(c h)`` on the rhs merges the ``c`` and ``h`` axes into one output dim.
//
// Reduction codes used by ``einops_reduce_op``
// --------------------------------------------
//   1 = mean, 2 = sum, 3 = max, 4 = min, 5 = prod.
// These must stay in sync with the Python-side enum in the bindings layer.
//
// Autograd
// --------
// Rearrange / Reduce / Repeat compose existing differentiable primitives
// (``reshape_op``, ``permute_op``, ``sum_op``, ``broadcast_to_op``, ...) so
// gradients flow through the chain automatically — no dedicated backward
// node is registered for these ops.  ``einsum`` similarly composes
// ``mul_op`` and ``sum_op``.
//
// References
// ----------
// Rogozhnikov, "einops: Clear and Reliable Tensor Manipulations" (ICLR 2022).
// https://github.com/arogozhnikov/einops

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Rearrange (permute and/or reshape) a tensor according to an einops pattern.
//
// The pattern must have the form ``"lhs -> rhs"``.  Both sides must contain
// exactly the same set of named axes; ``rhs`` may group or split them with
// parentheses.  ``axes_lengths`` provides sizes for any axis that cannot be
// inferred from the input shape (typically the components of a decomposed
// group).  No reduction is performed — element count is preserved.
//
// Math
// ----
// Pure index permutation and reshape:
// $$
//   y_{\sigma(i_1, \dots, i_n)} = x_{i_1, \dots, i_n}
// $$
// where $\sigma$ is determined by aligning ``lhs`` and ``rhs`` flat axis lists.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// pattern : std::string
//     Einops rearrange pattern ``"lhs -> rhs"``.
// axes_lengths : std::map<std::string, std::int64_t>
//     Named sizes for axes whose size cannot be inferred from ``a.shape``.
//
// Returns
// -------
// TensorImplPtr
//     Rearranged tensor.  Same element count and dtype as ``a``; shape is
//     determined by ``rhs`` and ``axes_lengths``.
//
// Raises
// ------
// EinopsPatternError
//     If the pattern is malformed, axes are inconsistent across sides, or a
//     required size is missing from ``axes_lengths``.
//
// Examples
// --------
// Pattern ``"b c h w -> b (h w) c"`` flattens spatial dims into a token
// sequence (NCHW → NLC) suitable for transformer-style backbones.
//
// See Also
// --------
// einops_reduce_op, einops_repeat_op, einsum_op
LUCID_API TensorImplPtr
einops_rearrange_op(const TensorImplPtr& a,
                    const std::string& pattern,
                    const std::map<std::string, std::int64_t>& axes_lengths);

// Reduce a tensor over one or more axes specified by an einops pattern.
//
// Axes present on the ``lhs`` but absent from the ``rhs`` are collapsed using
// the given reduction code.  ``axes_lengths`` is forwarded to the internal
// rearrange step that aligns axes before the reduction.
//
// Math
// ----
// For ``reduction = sum``:
// $$
//   y_{j_1, \dots, j_m} = \sum_{k_1, \dots, k_r} x_{i_1, \dots, i_n}
// $$
// where $\{k_\bullet\}$ are the axes dropped between ``lhs`` and ``rhs``.
// The mean / max / min / prod variants substitute the corresponding monoid.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// pattern : std::string
//     Einops pattern ``"lhs -> rhs"`` with at least one axis dropped on the
//     ``rhs``.
// reduction : int
//     Reduction code: ``1=mean``, ``2=sum``, ``3=max``, ``4=min``, ``5=prod``.
//     Must match the Python-side enum.
// axes_lengths : std::map<std::string, std::int64_t>
//     Named sizes for axes whose size cannot be inferred from ``a.shape``.
//
// Returns
// -------
// TensorImplPtr
//     Reduced tensor of the shape implied by ``rhs``.
//
// Raises
// ------
// EinopsPatternError
//     If the pattern is malformed or no axis is dropped (use ``rearrange``
//     instead).
// std::invalid_argument
//     If ``reduction`` is outside the accepted code range.
//
// See Also
// --------
// einops_rearrange_op, einops_repeat_op, einsum_op
LUCID_API TensorImplPtr einops_reduce_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         int reduction,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

// Tile / broadcast a tensor to introduce new axes specified by an einops
// pattern.
//
// Axes present in ``rhs`` but absent from ``lhs`` are new and their sizes
// must be provided in ``axes_lengths``.  Existing axes may additionally be
// reordered and merged exactly as in ``einops_rearrange_op``.  Internally
// the op uses ``unsqueeze`` + ``broadcast_to`` so it avoids data copies
// whenever possible — repeat is materialised only when subsequent ops force
// a contiguous layout.
//
// Math
// ----
// For each new axis $k$ of size $n_k$:
// $$
//   y_{\dots, k, \dots} = x_{\dots},
//   \quad k = 0, 1, \dots, n_k - 1
// $$
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input tensor.
// pattern : std::string
//     Einops pattern ``"lhs -> rhs"`` with at least one axis introduced on
//     the ``rhs``.
// axes_lengths : std::map<std::string, std::int64_t>
//     Sizes for the new axes introduced on the ``rhs``.  Must contain every
//     axis name that does not appear on the ``lhs``.
//
// Returns
// -------
// TensorImplPtr
//     Tensor with new axes broadcast/repeated to the requested sizes.
//
// Raises
// ------
// EinopsPatternError
//     If a new axis on the ``rhs`` is missing from ``axes_lengths``.
//
// See Also
// --------
// einops_rearrange_op, einops_reduce_op, einsum_op
LUCID_API TensorImplPtr einops_repeat_op(const TensorImplPtr& a,
                                         const std::string& pattern,
                                         const std::map<std::string, std::int64_t>& axes_lengths);

// General tensor contraction via an Einstein summation equation.
//
// The equation uses the same syntax as NumPy's ``einsum``: single-character
// labels separated by commas for each input, optionally followed by ``->``
// and the output labels.  If ``->`` is omitted, the output labels are the
// sorted set of labels that appear exactly once across all inputs (implicit
// form).  Repeated labels in a single operand take the diagonal; labels
// shared across operands trigger contraction; labels dropped from the output
// are summed out.
//
// Math
// ----
// $$
//   y_{\ell_\text{out}} = \sum_{\ell_\text{contract}}
//       \prod_{k} x^{(k)}_{\ell^{(k)}}
// $$
//
// Parameters
// ----------
// pattern : std::string
//     Einsum equation, e.g. ``"ij,jk->ik"`` for matrix multiplication or
//     ``"bij,bjk->bik"`` for batched matmul.
// operands : std::vector<TensorImplPtr>
//     The input tensors, one per comma-separated label group.  All operands
//     must share the same device and dtype.
//
// Returns
// -------
// TensorImplPtr
//     Contracted tensor of the shape implied by the output labels.
//
// Raises
// ------
// EinopsPatternError
//     If the equation is malformed, label counts do not match operand
//     dimensions, or contracted dimensions disagree in size.
// DeviceMismatch
//     If operands live on different devices.
// DtypeMismatch
//     If operands have different dtypes.
//
// Notes
// -----
// Ellipsis (``...``) is not yet supported.  Use explicit labels for batch
// dimensions in the meantime.
//
// Examples
// --------
// ``"ij,jk->ik"``    — matrix multiplication.
// ``"bhij,bhjk->bhik"`` — multi-head batched attention contraction.
//
// See Also
// --------
// einops_rearrange_op, einops_reduce_op, einops_repeat_op
LUCID_API TensorImplPtr einsum_op(const std::string& pattern,
                                  const std::vector<TensorImplPtr>& operands);

}  // namespace lucid
