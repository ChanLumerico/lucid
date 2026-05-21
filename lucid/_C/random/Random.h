// lucid/_C/random/Random.h
//
// Tensor-level random-number generation ops exposed to the Python
// :mod:`lucid` namespace via :file:`bindings/bind_random.cpp`.
//
// Each op allocates a fresh leaf :class:`TensorImpl` whose storage is
// filled with samples drawn from the requested distribution.  The
// optional :class:`Generator` argument selects the Philox-4x32-10 PRNG
// state; passing ``nullptr`` routes through the process-global
// :func:`default_generator` so all Lucid random calls share a single
// reproducible stream once :func:`lucid.manual_seed` has been called.
//
// Notes
// -----
// Dispatch:
//
//   * **CPU stream** — :file:`core/Storage`'s ``random_*_storage``
//     helpers, which call vForce / Accelerate vector routines for
//     bulk math and a scalar loop for the bit-stream itself.
//   * **GPU stream** — :func:`mlx::core::random::*`, keeping the
//     result in a :class:`GpuStorage`.
//
// All returned tensors carry ``requires_grad = false`` because random
// sampling is not differentiable in the conventional sense (no
// reparameterisation trick is applied here — that lives in
// :mod:`lucid.distributions`).
//
// See Also
// --------
// :class:`Generator` — Philox bit-stream state.
// :func:`default_generator` — process-global Generator singleton.
// :mod:`lucid.distributions` — high-level (differentiable when
//     possible) distribution objects layered on top of these ops.

#pragma once

#include <cstdint>

#include "../api.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class Generator;

// Return a tensor filled with values sampled uniformly from [0, 1).

// Allocates a tensor filled with uniform samples on $[0, 1)$.
//
// Convenience wrapper around :func:`uniform_op` with
// ``low = 0`` / ``high = 1``.
//
// Math
// ----
// $$x_i \sim \mathcal{U}(0, 1)$$
//
// Parameters
// ----------
// shape : const Shape&
//     Shape of the output tensor.
// dt : Dtype
//     Element dtype.  Floating-point dtypes only (``F32``, ``F64``,
//     ``BF16``, ``F16``).
// device : Device
//     Memory domain of the output (``CPU`` or ``GPU``).
// gen : Generator*, optional
//     Bit-stream state to draw from.  When ``nullptr`` (the default),
//     uses :func:`default_generator`.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated leaf tensor (``requires_grad = false``) of the
//     requested shape / dtype / device.
//
// See Also
// --------
// :func:`uniform_op` — arbitrary $[low, high)$ interval.
// :func:`randn_op` — Gaussian variant.
LUCID_API TensorImplPtr rand_op(const Shape& shape,
                                Dtype dt,
                                Device device,
                                Generator* gen = nullptr);

// Return a tensor filled with values sampled uniformly from [low, high).
// Raises if high <= low.

// Allocates a tensor filled with uniform samples on $[\mathrm{low},
// \mathrm{high})$.
//
// Math
// ----
// $$x_i \sim \mathcal{U}(\mathrm{low}, \mathrm{high})$$
//
// Parameters
// ----------
// shape : const Shape&
//     Shape of the output tensor.
// low : double
//     Inclusive lower bound of the support.
// high : double
//     Exclusive upper bound of the support.  Must be strictly greater
//     than ``low``.
// dt : Dtype
//     Element dtype.  Floating-point only.
// device : Device
//     Memory domain (``CPU`` or ``GPU``).
// gen : Generator*, optional
//     Bit-stream state; falls back to :func:`default_generator` on
//     ``nullptr``.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated leaf tensor (``requires_grad = false``).
//
// Raises
// ------
// LucidError
//     If ``high <= low`` — surfaced as ``ValueError`` on the Python
//     side.
//
// See Also
// --------
// :func:`rand_op` — convenience shortcut for $[0, 1)$.
LUCID_API TensorImplPtr uniform_op(
    const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen = nullptr);

// Return a tensor filled with samples from N(0, 1) (standard normal).

// Allocates a tensor filled with standard-normal samples
// $\mathcal{N}(0, 1)$.
//
// Convenience wrapper around :func:`normal_op` with ``mean = 0`` /
// ``std = 1``.
//
// Math
// ----
// $$x_i \sim \mathcal{N}(0, 1)$$
//
// Parameters
// ----------
// shape : const Shape&
//     Shape of the output tensor.
// dt : Dtype
//     Element dtype.  Floating-point only.
// device : Device
//     Memory domain.
// gen : Generator*, optional
//     Bit-stream state; falls back to :func:`default_generator`.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated leaf tensor (``requires_grad = false``).
//
// See Also
// --------
// :func:`normal_op` — arbitrary mean / std.
// :func:`rand_op` — uniform variant.
LUCID_API TensorImplPtr randn_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 Generator* gen = nullptr);

// Return a tensor filled with samples from N(mean, std^2).
// Raises if std < 0.

// Allocates a tensor filled with Gaussian samples
// $\mathcal{N}(\mathrm{mean}, \mathrm{std}^2)$.
//
// Math
// ----
// $$x_i \sim \mathcal{N}(\mathrm{mean}, \mathrm{std}^2)$$
//
// Parameters
// ----------
// shape : const Shape&
//     Shape of the output tensor.
// mean : double
//     Location parameter $\mu$.
// std : double
//     Scale parameter $\sigma$.  Must be non-negative; a value of
//     ``0`` produces a constant tensor equal to ``mean``.
// dt : Dtype
//     Element dtype.  Floating-point only.
// device : Device
//     Memory domain.
// gen : Generator*, optional
//     Bit-stream state.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated leaf tensor (``requires_grad = false``).
//
// Raises
// ------
// LucidError
//     If ``std < 0``.
//
// See Also
// --------
// :func:`randn_op` — shortcut for $\mathcal{N}(0, 1)$.
LUCID_API TensorImplPtr normal_op(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen = nullptr);

// Return an integer tensor with samples drawn uniformly from [low, high).

// Allocates an integer tensor filled with samples drawn uniformly
// from the integer interval $[\mathrm{low}, \mathrm{high})$.
//
// Math
// ----
// $$x_i \sim \mathcal{U}\{\mathrm{low}, \mathrm{low}+1, \dots,
//   \mathrm{high}-1\}$$
//
// Parameters
// ----------
// shape : const Shape&
//     Shape of the output tensor.
// low : int64_t
//     Inclusive lower bound.
// high : int64_t
//     Exclusive upper bound; must be strictly greater than ``low``.
// dt : Dtype
//     Integer dtype (``I32``, ``I64``, ``U8`` …).
// device : Device
//     Memory domain.
// gen : Generator*, optional
//     Bit-stream state.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated leaf integer tensor (``requires_grad = false``).
//
// Notes
// -----
// The integer dtype must be wide enough to represent every value in
// $[\mathrm{low}, \mathrm{high})$ losslessly; the underlying Storage
// helper does **not** mask or wrap on overflow.
LUCID_API TensorImplPtr randint_op(const Shape& shape,
                                   std::int64_t low,
                                   std::int64_t high,
                                   Dtype dt,
                                   Device device,
                                   Generator* gen = nullptr);

// Return a tensor of Bernoulli samples where each element is 1 with
// probability p and 0 with probability 1-p.

// Allocates a tensor of independent Bernoulli$(p)$ samples.
//
// Each element is independently drawn as ``1`` with probability $p$
// and ``0`` with probability $1 - p$.
//
// Math
// ----
// $$x_i \sim \mathrm{Bernoulli}(p), \quad
//   \Pr(x_i = 1) = p, \quad \Pr(x_i = 0) = 1 - p$$
//
// Parameters
// ----------
// shape : const Shape&
//     Shape of the output tensor.
// p : double
//     Success probability in $[0, 1]$.
// dt : Dtype
//     Output dtype.  Both floating-point (``F32``/``F64`` storing
//     ``0.0``/``1.0``) and integer (``U8``/``I32`` storing ``0``/``1``)
//     are accepted.
// device : Device
//     Memory domain.
// gen : Generator*, optional
//     Bit-stream state.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated leaf tensor (``requires_grad = false``).
//
// See Also
// --------
// :class:`lucid.distributions.Bernoulli` — higher-level
//     distribution object built on top of this op.
LUCID_API TensorImplPtr
bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen = nullptr);

}  // namespace lucid
