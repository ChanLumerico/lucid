// lucid/_C/ops/einops/Einops.h
//
// Public entry-point declaration for ``einsum`` in the Lucid engine.
// ``rearrange``, ``reduce``, and ``repeat`` lower in Python
// (``lucid/einops/``) directly to primitive view/reduction ops.

#pragma once

#include <string>
#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

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
LUCID_API TensorImplPtr einsum_op(const std::string& pattern,
                                  const std::vector<TensorImplPtr>& operands);

}  // namespace lucid
