// lucid/_C/ops/diffeq/RkCombine.h
//
// Public API for the fused Runge-Kutta stage combination used by the ODE
// integrators in ``lucid.diffeq``.  Every explicit Runge-Kutta method spends
// its per-step tensor arithmetic on exactly one affine form — a base state
// plus a scaled sum of stage derivatives — so a single fused op covers stage
// inputs, the final update, and (later) embedded error estimates.  The
// backward node is defined in RkCombine.cpp and is not exposed here.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/fwd.h"

namespace lucid {

// Combine Runge-Kutta stage derivatives into a new state in one fused op.
//
// Computes ``y0 + dt * sum_i coeffs[i] * ks[i]`` element-wise.  Every
// explicit Runge-Kutta step is built from this form: a stage input uses the
// Butcher row ``a_i`` as ``coeffs``, the final update uses the weights ``b``,
// and an embedded error estimate uses ``b - b*`` with ``y0`` set to a zero
// tensor.
//
// The coefficients arrive as host ``double`` values rather than tensors,
// which is deliberate — a tableau lives on the host, so materialising it as
// device tensors would force a synchronisation on every stage of every step.
// Terms whose coefficient is exactly zero are skipped entirely; strictly
// lower-triangular Butcher rows are mostly zeros, so this removes real work
// rather than a hypothetical case.
//
// Parameters
// ----------
// y0 : TensorImplPtr
//     Base state.  Defines the output shape, dtype, and device.
// ks : vector<TensorImplPtr>
//     Stage derivatives.  May be empty (the result is then a copy of
//     ``y0``).  Each entry must match ``y0`` in shape, dtype, and device.
// coeffs : vector<double>
//     Per-stage weights; must have the same length as ``ks``.
// dt : double
//     Step size multiplying the whole sum.  A negative ``dt`` integrates
//     backwards in time and is fully supported.
//
// Returns
// -------
// TensorImplPtr
//     Freshly allocated tensor with the same shape, dtype, and device as
//     ``y0``.
//
// Shape
// -----
// All inputs share shape ``(D_0, ..., D_{ndim-1})``; the output has that
// same shape.  No broadcasting is performed.
//
// Math
// ----
// $$ y = y_0 + \Delta t \sum_{i} c_i \, k_i $$
//
// Raises
// ------
// ShapeMismatch
//     If any ``ks[i]`` disagrees with ``y0`` on shape, or if ``ks`` and
//     ``coeffs`` differ in length.
// DtypeMismatch
//     If any ``ks[i]`` has a different dtype from ``y0``.
// DeviceMismatch
//     If any ``ks[i]`` resides on a different device from ``y0``.
//
// Notes
// -----
// Backward is exact and cheap: the incoming gradient passes straight through
// to ``y0`` and is scaled by ``dt * coeffs[i]`` for each ``ks[i]``.  Keeping
// a real backward is what lets a solver be differentiated end-to-end
// (discretise-then-optimise); a fused forward without it would silently cut
// the graph.  Schema name ``"rk_combine"``, ``AmpPolicy::Promote`` (matching
// ``add``), deterministic, variadic input arity.
//
// Non-contiguous inputs are materialised through :func:`contiguous_op`
// before the arithmetic, so views integrate without a caller-side copy.
//
// See Also
// --------
// :func:`add_op`, :func:`mul_scalar_op`.
LUCID_API TensorImplPtr rk_combine_op(const TensorImplPtr& y0,
                                      const std::vector<TensorImplPtr>& ks,
                                      const std::vector<double>& coeffs,
                                      double dt);

}  // namespace lucid
