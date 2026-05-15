// lucid/_C/ops/gfunc/Gfunc.h
//
// Generator ("gfunc") ops: tensor-creation functions that produce new leaf
// tensors from scalar parameters or from the metadata of an existing tensor.
//
// The suite covers:
//   zeros / ones / full / empty  — constant-fill ops.
//   eye                          — identity-matrix construction.
//   arange                       — arithmetic progression.
//   linspace                     — linearly spaced grid.
//   diag                         — diagonal extraction / construction.
//   zeros_like / ones_like / empty_like / full_like — shape-inheriting variants.
//
// Design invariants:
//   - None of these ops participate in autograd.  All outputs are leaf nodes
//     with requires_grad determined solely by the argument of the same name.
//     There are no FuncOp / NaryKernel calls in this file.
//   - CPU storage is allocated via Allocator (CpuStorage with a managed
//     unique_ptr).  GPU storage uses IBackend::full / IBackend::eye /
//     IBackend::from_cpu — the last route uploads a CPU buffer to the GPU
//     without a full backend rewrite.
//   - arange and linspace always materialise the buffer on the CPU first and
//     then call from_cpu, because the element computation is a simple scalar
//     loop that is faster to write in portable C++ than to route through
//     a backend-specific kernel.
//   - All shape, dtype, and device arguments are forwarded verbatim to the
//     resulting TensorImpl; no implicit promotion occurs.
//
// These functions are exposed to Python via the _bindings/ layer.  The
// Python-side wrappers translate Python objects (list shapes, dtype strings)
// into the C++ types used here.

#pragma once

#include <cstdint>

#include "../../api.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Create a tensor of given shape filled with zeros.
LUCID_API TensorImplPtr zeros_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 bool requires_grad = false);

// Create a tensor of given shape filled with ones.
LUCID_API TensorImplPtr ones_op(const Shape& shape,
                                Dtype dt,
                                Device device,
                                bool requires_grad = false);

// Create a tensor of given shape filled with a constant scalar value.
LUCID_API TensorImplPtr
full_op(const Shape& shape, double fill_value, Dtype dt, Device device, bool requires_grad = false);

// Create an uninitialised tensor of given shape.
//
// The current implementation zero-fills (via make_zero_storage) to avoid
// UB reads; the name "empty" signals that callers must not rely on the values.
LUCID_API TensorImplPtr empty_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 bool requires_grad = false);

// Create an N×M identity-like matrix with ones on diagonal k.
//
// If M <= 0, M is set to N.  k=0 is the main diagonal; positive k is above,
// negative k is below.
LUCID_API TensorImplPtr eye_op(std::int64_t N,
                               std::int64_t M,
                               std::int64_t k,
                               Dtype dt,
                               Device device,
                               bool requires_grad = false);

// Create a 1-D tensor with values [start, start+step, ...) up to but not
// including stop.
//
// The number of elements is ceil((stop - start) / step).  Raises an error
// when step is zero.
LUCID_API TensorImplPtr arange_op(
    double start, double stop, double step, Dtype dt, Device device, bool requires_grad = false);

// Create a 1-D tensor with num evenly-spaced values between start and stop
// inclusive.
//
// The last element is pinned to stop to avoid floating-point drift.
// Raises an error when num < 0.
LUCID_API TensorImplPtr linspace_op(double start,
                                    double stop,
                                    std::int64_t num,
                                    Dtype dt,
                                    Device device,
                                    bool requires_grad = false);

// Extract a diagonal from a 2-D matrix, or construct a 2-D matrix from a 1-D
// diagonal vector.
//
// Follows numpy.diag semantics: k=0 is the main diagonal, k>0 above, k<0 below.
// Input must be 1-D or 2-D; an error is raised otherwise.
LUCID_API TensorImplPtr diag_op(const TensorImplPtr& v, std::int64_t k = 0);

// Create a zeros tensor with the same shape, dtype, and device as a.
LUCID_API TensorImplPtr zeros_like_op(const TensorImplPtr& a, bool requires_grad = false);

// Create a ones tensor with the same shape, dtype, and device as a.
LUCID_API TensorImplPtr ones_like_op(const TensorImplPtr& a, bool requires_grad = false);

// Create an uninitialised tensor with the same shape, dtype, and device as a.
LUCID_API TensorImplPtr empty_like_op(const TensorImplPtr& a, bool requires_grad = false);

// Create a constant-filled tensor with the same shape, dtype, and device as a.
LUCID_API TensorImplPtr full_like_op(const TensorImplPtr& a,
                                     double fill_value,
                                     bool requires_grad = false);

// Create a 1-D tensor with num values spaced evenly on a log scale.
// Values are: base^linspace(start, stop, num).
LUCID_API TensorImplPtr logspace_op(double start,
                                    double stop,
                                    std::int64_t num,
                                    double base,
                                    Dtype dt,
                                    Device device,
                                    bool requires_grad = false);

// User-facing scatter-add: out = base with src added at positions given by
// indices along dim. Has full autograd support.
// Gradients: d/d(base) = grad_out; d/d(src) = gather(grad_out, dim, index).
LUCID_API TensorImplPtr scatter_add_op(const TensorImplPtr& base,
                                       const TensorImplPtr& indices,
                                       const TensorImplPtr& src,
                                       int dim);

// Scatter-reduce variants — forward is the specified reduce; backward flows
// only through the "winning" element at each output position.
//
// d/d(src[j])  = grad[idx[j]]  if src[j] == output[idx[j]]  else 0
// d/d(base[i]) = grad[i]       if base[i] == output[i]       else 0
// Ties broken by "all winners receive the gradient equally" (matches the
// reference framework).
LUCID_API TensorImplPtr scatter_amax_op(const TensorImplPtr& base,
                                        const TensorImplPtr& indices,
                                        const TensorImplPtr& src,
                                        int dim);
LUCID_API TensorImplPtr scatter_amin_op(const TensorImplPtr& base,
                                        const TensorImplPtr& indices,
                                        const TensorImplPtr& src,
                                        int dim);

// scatter_prod: out[i] = base[i] * prod_j(src[j]) for j where idx[j]==i.
// Gradient: d/d(src[j]) = grad[idx[j]] * (out[idx[j]] / src[j])  (product rule).
// d/d(base[i]) = grad[i] * out[i] / base[i]  (if base[i] != 0).
LUCID_API TensorImplPtr scatter_prod_op(const TensorImplPtr& base,
                                        const TensorImplPtr& indices,
                                        const TensorImplPtr& src,
                                        int dim);

// Sliding-window view along a single dimension.
// Output shape: (*in_shape[:dim], L, *in_shape[dim+1:], size)
// where L = (dim_size - size) / step + 1.
// No gradient (view-like; backward passes through gather).
LUCID_API TensorImplPtr unfold_dim_op(const TensorImplPtr& a, int dim, int size, int step);

}  // namespace lucid
