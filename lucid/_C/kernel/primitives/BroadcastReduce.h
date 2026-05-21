// lucid/_C/kernel/primitives/BroadcastReduce.h
//
// Re-export of the autograd broadcast-reduction helpers into the kernel
// primitives namespace.
//
// The primary helper exposed here is ``reduce_grad_to_shape()``, which
// sums gradient components over the dimensions that were broadcast
// during the forward pass, restoring the gradient to the original
// pre-broadcast input shape.  This is the canonical "sum_to_shape"
// pattern needed by every binary op whose forward implicitly broadcasts
// its operands.
//
// Math
// ----
// Given a forward broadcast that expanded an input of shape $S_\text{in}$
// to a common shape $S_\text{out}$, the adjoint of that broadcast is a
// sum over the inserted / size-1-expanded axes:
// $$
//   \frac{\partial \mathcal{L}}{\partial X[\mathbf{i}]}
//   = \sum_{\mathbf{j} \in \mathrm{bcast}^{-1}(\mathbf{i})}
//     \frac{\partial \mathcal{L}}{\partial Y[\mathbf{j}]}
// $$
// where $\mathrm{bcast}^{-1}(\mathbf{i})$ is the set of output indices
// that map back to input index $\mathbf{i}$.
//
// Notes
// -----
// The implementation lives in :file:`autograd/Helpers.h`.  This header
// exists only so that kernel-level callers can ``#include
// "primitives/BroadcastReduce.h"`` and pick up the helper without
// reaching into the autograd subtree directly.
//
// See Also
// --------
// BinaryKernel::apply : Calls ``reduce_grad_to_shape`` to fold each
//     operand's broadcast-expanded gradient back to its original shape.
// UnaryKernel::apply : Same pattern for unary ops that broadcast a
//     scalar.

#pragma once

#include "../../autograd/Helpers.h"
