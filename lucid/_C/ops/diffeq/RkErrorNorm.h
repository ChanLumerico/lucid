// lucid/_C/ops/diffeq/RkErrorNorm.h
//
// Public API for the fused embedded-error norm that drives adaptive
// Runge-Kutta step-size control.  Returns a host ``double`` rather than a
// tensor: the step controller has to branch on this value, so it must cross
// to the host no matter how it is computed.  Producing it inside one kernel
// makes that exactly one synchronisation per step and leaves no intermediate
// error tensor behind.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Root-mean-square ratio of an embedded error estimate to its tolerance.
//
// Computes, in a single pass and without materialising the error tensor:
//
//     err_i   = dt * sum_j coeffs[j] * ks[j][i]
//     tol_i   = atol + rtol * max(|y0_i|, |y1_i|)
//     result  = sqrt( mean_i( (err_i / tol_i)^2 ) )
//
// The adaptive controller accepts a step when the result is at most 1 and
// otherwise retries with a smaller one, so this scalar is read on the host
// every single step.  Written as a chain of ordinary tensor ops it would cost
// roughly eight passes over the state plus several temporaries and still end
// in the same host read; fused it costs one pass, no temporary, one read.
//
// Parameters
// ----------
// y0 : TensorImplPtr
//     State at the start of the step.  Defines shape, dtype, and device.
// y1 : TensorImplPtr
//     Proposed state at the end of the step.
// ks : vector<TensorImplPtr>
//     Stage derivatives of the step.  May be empty, in which case the error
//     estimate is identically zero and the result is ``0``.
// coeffs : vector<double>
//     Embedded-error weights (``b - b_hat``), one per entry of ``ks``.
// dt : double
//     Step size.  Negative for backwards integration; only the magnitude of
//     the resulting error matters.
// rtol, atol : double
//     Relative and absolute tolerances.  Both must be non-negative and not
//     both zero, otherwise the tolerance is zero and the ratio is infinite.
//
// Returns
// -------
// double
//     The error ratio, on the host.  Never negative; ``0`` when ``ks`` is
//     empty or every coefficient is zero.
//
// Raises
// ------
// ShapeMismatch
//     If ``y1`` or any ``ks[i]`` disagrees with ``y0`` on shape, or if
//     ``ks`` and ``coeffs`` differ in length.
// DtypeMismatch
//     If the operands disagree on dtype.
// DeviceMismatch
//     If the operands reside on different devices.
// NotImplementedError
//     If the dtype is not ``F32`` or ``F64``.  Step control is a scalar
//     diagnostic, so the Python layer promotes lower-precision states before
//     calling rather than the kernel carrying a half-precision path.
//
// Notes
// -----
// Not differentiable, by construction — step-size control is a control-flow
// decision, not part of the value the caller differentiates.  Returning a
// host ``double`` rather than a tensor makes that structural instead of a
// convention someone could accidentally violate.
//
// The accumulation is performed in ``double`` on the CPU path regardless of
// storage dtype, so a large state does not lose the small contributions that
// decide whether a step is accepted.
//
// See Also
// --------
// :func:`rk_combine_op` — the fused stage combination this pairs with.
LUCID_API double rk_error_norm_op(const TensorImplPtr& y0,
                                  const TensorImplPtr& y1,
                                  const std::vector<TensorImplPtr>& ks,
                                  const std::vector<double>& coeffs,
                                  double dt,
                                  double rtol,
                                  double atol);

}  // namespace lucid
