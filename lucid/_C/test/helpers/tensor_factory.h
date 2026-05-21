// lucid/_C/test/helpers/tensor_factory.h
//
// Convenience factories for constructing :class:`TensorImpl` objects in
// C++ unit tests.
//
// Uses the public ops/gfunc API (:func:`zeros_op`, :func:`ones_op`,
// :func:`full_op`, :func:`eye_op`) which return :type:`TensorImplPtr`
// directly — no need to interact with :class:`Storage` or
// :class:`IBackend`.  Prefer these helpers over hand-rolling tensors in
// tests: they centralise the "fresh tensor for a unit test" pattern,
// keep tests dtype-parameterisable, and stay in sync with the
// production allocator path so a test's tensor is byte-identical to
// what a real user would obtain via :func:`lucid.zeros` etc.
//
// Notes
// -----
// All factories produce ``requires_grad = false`` tensors — autograd
// state is irrelevant to most fixture setup.  Tests that need
// ``requires_grad`` should toggle it explicitly after construction or
// call the underlying ``*_op`` directly.
//
// See Also
// --------
// :file:`numeric_assert.h` — companion header with GTest matchers that
//     consume the :type:`TensorImplPtr` values produced here.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../../core/fwd.h"
#include "../../core/Dtype.h"
#include "../../core/Device.h"
#include "../../core/Shape.h"
#include "../../ops/gfunc/Gfunc.h"
#include "../../backend/Dispatcher.h"

namespace lucid::test {

// Create a CPU tensor filled with zeros.
//
// Thin wrapper over :func:`zeros_op` pinned to :attr:`Device::CPU` and
// ``requires_grad = false`` — the canonical "blank tensor" used as
// fixture input for unit tests.
//
// Parameters
// ----------
// shape : Shape
//     Target tensor shape.  Empty shape produces a 0-dim scalar.
// dtype : Dtype, optional
//     Element type.  Defaults to :attr:`Dtype::F32`.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated CPU tensor of the given shape, with all elements
//     set to ``0``.
//
// Examples
// --------
// .. code-block:: cpp
//
//     auto x = cpu_zeros({3, 4});                  // F32 by default
//     auto y = cpu_zeros({2}, Dtype::F64);         // double precision
//
// See Also
// --------
// :func:`cpu_ones`, :func:`cpu_full`.
inline TensorImplPtr cpu_zeros(const Shape& shape, Dtype dtype = Dtype::F32) {
    return zeros_op(shape, dtype, Device::CPU, /*requires_grad=*/false);
}

// Create a CPU tensor filled with ones.
//
// Thin wrapper over :func:`ones_op` pinned to :attr:`Device::CPU` and
// ``requires_grad = false``.  Useful as a multiplicative identity input
// or for tests that need a non-zero baseline.
//
// Parameters
// ----------
// shape : Shape
//     Target tensor shape.
// dtype : Dtype, optional
//     Element type.  Defaults to :attr:`Dtype::F32`.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated CPU tensor with all elements set to ``1``.
//
// See Also
// --------
// :func:`cpu_zeros`, :func:`cpu_full`.
inline TensorImplPtr cpu_ones(const Shape& shape, Dtype dtype = Dtype::F32) {
    return ones_op(shape, dtype, Device::CPU, /*requires_grad=*/false);
}

// Create a CPU tensor filled with a constant scalar value.
//
// Thin wrapper over :func:`full_op` pinned to :attr:`Device::CPU` and
// ``requires_grad = false``.  The fill value is accepted as ``double``
// and cast down to :paramref:`dtype` by the backend kernel.
//
// Parameters
// ----------
// shape : Shape
//     Target tensor shape.
// value : double
//     Scalar value broadcast to every element.  Cast to :paramref:`dtype`
//     during fill.
// dtype : Dtype, optional
//     Element type.  Defaults to :attr:`Dtype::F32`.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated CPU tensor with all elements equal to
//     :paramref:`value` (after dtype cast).
//
// Examples
// --------
// .. code-block:: cpp
//
//     auto x = cpu_full({4}, 2.0);          // [2, 2, 2, 2] F32
//     auto y = cpu_full({3}, -1.0, Dtype::F64);
//
// See Also
// --------
// :func:`cpu_zeros`, :func:`cpu_ones`.
inline TensorImplPtr cpu_full(const Shape& shape, double value, Dtype dtype = Dtype::F32) {
    return full_op(shape, value, dtype, Device::CPU, /*requires_grad=*/false);
}

// Create an :math:`N \times N` identity matrix on CPU.
//
// Thin wrapper over :func:`eye_op` that produces a square identity with
// the main diagonal set to ``1`` and all off-diagonal entries set to
// ``0``.  Used in linear-algebra tests as a multiplicative identity
// (``A @ I == A``) or as a reference matrix.
//
// Parameters
// ----------
// n : std::int64_t
//     Side length of the square identity matrix.  The resulting tensor
//     has shape ``(n, n)``.
// dtype : Dtype, optional
//     Element type.  Defaults to :attr:`Dtype::F32`.
//
// Returns
// -------
// TensorImplPtr
//     Newly allocated ``(n, n)`` CPU tensor with ``1`` on the main
//     diagonal and ``0`` elsewhere.
//
// Math
// ----
// $$ I_{ij} = \begin{cases} 1 & i = j \\ 0 & i \ne j \end{cases} $$
inline TensorImplPtr cpu_eye(std::int64_t n, Dtype dtype = Dtype::F32) {
    return eye_op(n, n, 0, dtype, Device::CPU, /*requires_grad=*/false);
}

// Whether the CPU backend is available — always ``true`` in tests.
//
// Provided as a counterpart to :func:`gpu_available` so dtype-parameterised
// fixtures can branch on device availability uniformly.  The CPU backend
// (Accelerate / vDSP) is always present on the supported Apple Silicon
// platforms targeted by Lucid, so this function unconditionally returns
// ``true``.
//
// Returns
// -------
// bool
//     Always ``true``.
//
// See Also
// --------
// :func:`gpu_available`.
inline bool cpu_available() { return true; }

// Whether the GPU backend (MLX / Metal) is available on the current
// host.
//
// Probes :func:`backend::Dispatcher::for_device` with
// :attr:`Device::GPU` and reports success / failure as a boolean.  A
// runtime probe (rather than a compile-time constant) is required
// because Lucid tests may be cross-compiled or run on machines without
// a functioning Metal device.
//
// Returns
// -------
// bool
//     ``true`` if a GPU dispatcher can be constructed, ``false``
//     otherwise (no Metal device, missing MLX initialisation, etc.).
//
// Notes
// -----
// Catches every exception raised by the dispatcher lookup; treat the
// negative result as advisory and skip GPU-only tests via
// ``GTEST_SKIP()`` when this returns ``false``.
//
// See Also
// --------
// :func:`cpu_available`.
inline bool gpu_available() {
    try {
        backend::Dispatcher::for_device(Device::GPU);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace lucid::test
