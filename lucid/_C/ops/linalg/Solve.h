// lucid/_C/ops/linalg/Solve.h
//
// Differentiable general linear solve $AX = B$ — finds $X$ given a
// square invertible $A$ and a right-hand side $B$ without explicitly
// forming the inverse.
//
// Internally LU-factorises $A$ with partial pivoting (LAPACK
// ``*getrf``) then triangular-solves (``*getrs``), exposed through
// ``IBackend::linalg_solve()``.  Compared to materialising $A^{-1}$
// and multiplying, the fused factor-and-solve is roughly twice as fast
// and considerably more numerically stable because the LU factors are
// shared between the two triangular sweeps.
//
// Math
// ----
// Forward:
// $$
//   A X = B \quad\Longrightarrow\quad X = A^{-1} B
// $$
// Backward — differentiating the linear relation $AX = B$ jointly:
// $$
//   \frac{\partial L}{\partial B} = A^{-\top}\, \frac{\partial L}{\partial X}, \qquad
//   \frac{\partial L}{\partial A} = -\,\frac{\partial L}{\partial B}\, X^\top
// $$
// The first identity follows from $A\, dX = dB$; the second from
// $(dA) X + A\, dX = 0$, eliminating $dX$ via the first.
//
// Shape
// -----
// - $A$: ``(..., N, N)`` — leading batch dims iterate independently.
// - $B$: ``(..., N)`` for a single RHS, or ``(..., N, K)`` for $K$
//   simultaneous right-hand sides.
// - $X$: same shape as $B$.
//
// Notes
// -----
// - LAPACK works in FP32/FP64; the GPU stream round-trips through
//   MLX-on-CPU which is itself Accelerate-backed.  ``AmpPolicy::
//   KeepInput`` prevents autocast from dropping precision before the
//   factorisation.
// - The backward call ``solve(Aᵀ, G)`` re-factorises $A^\top$ rather
//   than reusing the forward LU factors — a future optimisation could
//   save the factors to halve the backward factorisation cost.
// - Singular $A$ surfaces as a zero pivot inside LAPACK and is
//   reported as a ``LinAlgError`` from the backend.
//
// References
// ----------
// Giles, "Collected Matrix Derivative Results for Forward and Reverse
// Mode Algorithmic Differentiation" (Oxford, 2008), §2.4.
// Murray, "Differentiation of the Cholesky Decomposition" (arXiv 2016).
// LAPACK ``sgesv``/``dgesv``; *LAPACK Users' Guide* §2.5.

#pragma once

#include "../../api.h"
#include "../../autograd/FuncOp.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Autograd node for the general linear solve $AX = B$.
//
// The forward saves $A$ as ``saved_inputs_[0]`` (required to build
// $A^\top$ for the adjoint solve) and the solution $X$ as
// ``saved_output_`` (required for the outer-product step that yields
// $\partial L / \partial A$).  Although $B$ is technically saved by
// ``NaryKernel`` when ``save_inputs=true``, the backward never reads
// it.
//
// Math
// ----
// Given upstream gradient $G = \partial L / \partial X$:
// $$
//   \frac{\partial L}{\partial B} = \text{solve}(A^\top, G), \qquad
//   \frac{\partial L}{\partial A} = -\,\frac{\partial L}{\partial B}\, X^\top
// $$
// Both expressions compose existing differentiable ops
// ([[solve_op]] / [[matmul_op]] / [[neg_op]] / [[mT_op]]) so that
// second-order gradients flow automatically.
//
// Attributes
// ----------
// schema_v1 : OpSchema
//     ``"solve"``, one saved-input slot, ``AmpPolicy::KeepInput`` —
//     prevents a lossy autocast promotion before the LAPACK call.
//
// References
// ----------
// Giles, "Collected Matrix Derivative Results for Forward and Reverse
// Mode Algorithmic Differentiation" (Oxford, 2008), §2.4.
class LUCID_API SolveBackward : public FuncOp<SolveBackward, 2> {
public:
    static const OpSchema schema_v1;

    // Compute input gradients for the forward solve $AX = B$.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient $G = \partial L / \partial X$, shaped like
    //     the forward output $X$.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Two-element vector ``{dA, dB}`` matching the input ordering
    //     ``[A=0, B=1]`` that ``NaryKernel`` expects, where
    //     ``dB = solve(Aᵀ, G)`` and ``dA = -dB @ Xᵀ``.
    //
    // Notes
    // -----
    // The transpose solve ``solve(Aᵀ, G)`` performs a fresh LU
    // factorisation of $A^\top$ — the forward factors are not cached.
    std::vector<Storage> apply(Storage grad_out) override;
};

// Solve the linear system $AX = B$ for $X$.
//
// Parameters
// ----------
// a : const TensorImplPtr&
//     Square float tensor of shape ``(..., N, N)``.  Must share dtype
//     and device with ``b``.
// b : const TensorImplPtr&
//     Right-hand side of shape ``(..., N)`` or ``(..., N, K)``.
//
// Returns
// -------
// TensorImplPtr
//     Solution tensor $X = A^{-1} B$ with the same shape as ``b``.
//     When grad mode is active, the result has [[SolveBackward]]
//     wired into the autograd graph with $A$ saved as input and $X$
//     saved as output.
//
// Shape
// -----
// - ``a``: ``(..., N, N)``.
// - ``b``: ``(..., N)`` or ``(..., N, K)``.
// - Output: same shape as ``b``.
//
// Raises
// ------
// LinAlgError
//     When ``a`` is not square, not float-typed, when dtypes/devices
//     mismatch between ``a`` and ``b``, or when $A$ is singular
//     (zero pivot detected by LU).
//
// Examples
// --------
// Single RHS solve::
//
//     auto x = solve_op(a, b);  // a: (n, n), b: (n,) → x: (n,)
//
// Batched multi-RHS solve::
//
//     auto x = solve_op(a, b);  // a: (B, n, n), b: (B, n, k)
//
// See Also
// --------
// [[lu_factor_op]] : Standalone LU factorisation (no autograd).
// [[lu_solve_op]]  : Apply pre-computed LU factors (no autograd).
// [[matmul_op]]    : Used in the backward outer product $-dB\, X^\top$.
//
// References
// ----------
// Giles, "Collected Matrix Derivative Results" (Oxford, 2008), §2.4.
// LAPACK ``sgesv``/``dgesv``; *LAPACK Users' Guide* §2.5.
LUCID_API TensorImplPtr solve_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
