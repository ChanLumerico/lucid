// lucid/_C/ops/diffeq/BroydenProbe.h
//
// Public API for the fused probe that drives one Broyden iteration inside an
// implicit Runge-Kutta step.  Returns three host ``double``s rather than
// tensors, for the same reason ``rk_error_norm`` does: the iteration branches
// on all three, so they have to cross to the host no matter how they are
// computed.  Producing them together makes that exactly one synchronisation
// per iteration instead of three.

#pragma once

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// The three scalars one Broyden iteration has to know.
//
// Attributes
// ----------
// residual_sq : double
//     ``sum_i residual_i^2``.  Compared against the squared convergence
//     tolerance; the square root is never needed, so it is never taken.
// step_sq : double
//     ``sum_i step_i^2``.  Denominator of the rank-one Jacobian update, and
//     zero exactly when the iteration has stalled.
// info : double
//     The linear solve's status, carried through unchanged.  Non-zero means
//     the approximate Jacobian was singular.
struct BroydenProbeResult {
    double residual_sq = 0.0;
    double step_sq = 0.0;
    double info = 0.0;
};

// Answer all three of a Broyden iteration's questions in one pass.
//
// An implicit method solves its stage equations by quasi-Newton iteration,
// and each iteration has to decide three things on the host: whether the
// residual is small enough to stop, whether the linear solve succeeded, and
// whether the step is long enough for the rank-one update to be defined.
//
// Written separately, those are three reductions, three temporaries, and --
// on the GPU -- three pipeline flushes.  MLX evaluates lazily, so reading any
// one scalar waits for everything queued behind it; measured on an M4 Max,
// that wait is around 120 microseconds while the arithmetic it interrupts is
// closer to 10.  The flushes, not the arithmetic, are what an implicit solve
// spends its time on.  Fusing them costs one flush and no temporaries, and on
// the CPU it also drops the three intermediate tensors the Python spelling
// would allocate.
//
// Parameters
// ----------
// residual : TensorImplPtr
//     Current residual vector.  Defines dtype and device; must be F32 or F64.
// step : TensorImplPtr
//     Proposed Newton step.  Same dtype and device as ``residual`` and the
//     same element count, though not necessarily the same shape -- the linear
//     solve returns a column, the residual is flat.
// info : TensorImplPtr
//     Status scalar from the linear solve.  Any dtype; read as a number.
//     May be null, for the seeding call made before any solve has happened;
//     ``info`` then reports zero and only the two norms are computed.
//
// Returns
// -------
// BroydenProbeResult
//     The three scalars, already on the host.
//
// Raises
// ------
// NotImplementedError
//     If ``residual`` is neither F32 nor F64.
// DtypeMismatch, DeviceMismatch, ShapeMismatch
//     If ``step`` disagrees with ``residual`` on dtype, device, or element
//     count.
//
// See Also
// --------
// rk_error_norm_op : the same trade for adaptive step-size control.
LUCID_API BroydenProbeResult broyden_probe_op(const TensorImplPtr& residual,
                                              const TensorImplPtr& step,
                                              const TensorImplPtr& info);

}  // namespace lucid
