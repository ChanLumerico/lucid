#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "../core/Device.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {

class TensorImpl;

// Allocate a new CpuStorage, zero-filled, of the given shape and dtype.
Storage make_zero_storage(const Shape& shape, Dtype dtype, Device device);

// Allocate a new CpuStorage, ones-filled, of the given shape and dtype.
// Used for the implicit `loss.backward()` seed.
Storage make_ones_storage(const Shape& shape, Dtype dtype, Device device);

// Sum-reduce `grad` so that its shape matches `target_shape`. Implements the
// broadcast-back behavior used by every binary op's backward. CPU only in
// Phase 2.
Storage reduce_grad_to_shape(const Storage& grad, const Shape& grad_shape,
                             const Shape& target_shape, Dtype dtype,
                             Device device);

// In-place accumulate: dst += src. Both must have matching shape/dtype/device.
// Used by the Engine to merge multiple grad streams arriving at a node.
void accumulate_into(Storage& dst, const Storage& src);

// ----------------------------------------------------------------------
// Storage-level math primitives — used inside binary ops' `grad_formula`.
// Each allocates a fresh CpuStorage of the same numel/dtype and returns it.
// CPU-only in Phase 3.1; GPU paths arrive in Phase 3.6.
// ----------------------------------------------------------------------

Storage negate_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage multiply_storages(const Storage& a, const Storage& b,
                          std::size_t numel, Dtype dt, Device device);
Storage divide_storages(const Storage& a, const Storage& b,
                        std::size_t numel, Dtype dt, Device device);
Storage add_storages(const Storage& a, const Storage& b,
                     std::size_t numel, Dtype dt, Device device);
Storage subtract_storages(const Storage& a, const Storage& b,
                          std::size_t numel, Dtype dt, Device device);
Storage square_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage clone_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage log_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage pow_storage(const Storage& base, const Storage& expo,
                    std::size_t numel, Dtype dt, Device device);

// Element-wise comparison mask (used by Maximum/Minimum backward):
//   ge_mask: out[i] = (a[i] >= b[i]) ? 1 : 0
//   lt_mask: out[i] = (a[i] <  b[i]) ? 1 : 0   (strict less, so ties go to ge)
Storage ge_mask_storage(const Storage& a, const Storage& b,
                        std::size_t numel, Dtype dt, Device device);
Storage lt_mask_storage(const Storage& a, const Storage& b,
                        std::size_t numel, Dtype dt, Device device);

// Element-wise vector + scalar add (out = in + scalar). Used by Pow backward
// to compute (b - 1) without materializing a tensor of ones.
Storage add_scalar_storage(const Storage& s, double scalar, std::size_t numel,
                           Dtype dt, Device device);

// Element-wise vector * scalar (out = in * scalar). Used by op backwards that
// need a constant multiplier (e.g. cube: 3·a²·g, _pow: exp·a^(exp-1)·g).
Storage mul_scalar_storage(const Storage& s, double scalar, std::size_t numel,
                           Dtype dt, Device device);

// More element-wise unary primitives (Phase 3.2). Each allocates a fresh
// CpuStorage; CPU only.
Storage exp_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage sqrt_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage abs_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage sign_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage reciprocal_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

Storage sin_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage cos_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage tan_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage asin_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage acos_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage atan_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

Storage sinh_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage cosh_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage tanh_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

// out[i] = (lo <= in[i] <= hi) ? 1 : 0  — used by Clip backward
Storage in_range_mask_storage(const Storage& s, double lo, double hi,
                              std::size_t numel, Dtype dt, Device device);

// out[i] = (in[i] > 0) ? 1 : 0   — used by ReLU backward
Storage positive_mask_storage(const Storage& s, std::size_t numel,
                              Dtype dt, Device device);

// out[i] = (in[i] >= 0) ? 1 : slope   — used by LeakyReLU backward
Storage leaky_mask_storage(const Storage& s, double slope, std::size_t numel,
                           Dtype dt, Device device);

// out[i] = sigmoid(in[i]) = 1 / (1 + exp(-in[i]))   — used by sigmoid forward
// + silu/softplus backward. Numerically stable (no overflow at large |x|).
Storage sigmoid_storage(const Storage& s, std::size_t numel, Dtype dt,
                        Device device);

// Bernoulli mask: each output element is 1 with probability `keep_prob`,
// 0 otherwise. Uses `Generator` for reproducibility under
// `lucid.set_deterministic(True)`. dtype must be F32 or F64 (mask is the
// same dtype as the tensor it'll multiply).
class Generator;
Storage bernoulli_mask_storage(double keep_prob, std::size_t numel, Dtype dt,
                               Device device, Generator& gen);

// Same as above but uploads the GPU mask with a specified shape (so the mask
// broadcasts cleanly against the input tensor). The CPU branch ignores
// `shape_for_gpu`. Required when feeding the mask into multiply_storages
// against a non-flat input.
Storage bernoulli_mask_storage_shape(double keep_prob, const Shape& shape,
                                       Dtype dt, Device device, Generator& gen);

// --------------------------- Random ops (Phase 3.8) ----------------------
//
// All Lucid random ops draw uniform u32 from a Philox-4x32-10 `Generator`.
// CPU storage is filled directly from the generator. GPU storage is filled
// CPU-side and then uploaded — this preserves bit-exact determinism between
// CPU and GPU for the same seed, at the cost of one host→device copy per
// random tensor. (MLX's own random ops use a different keyed-Philox layout
// that wouldn't match Lucid's seed semantics, so we route everything through
// the Lucid generator.)

/// Uniform [lo, hi). dtype must be F32 or F64.
Storage random_uniform_storage(const Shape& shape, double lo, double hi,
                               Dtype dt, Device device, Generator& gen);

/// Normal(mean, std) via Box-Muller pairs over uniform draws. F32/F64.
Storage random_normal_storage(const Shape& shape, double mean, double std,
                              Dtype dt, Device device, Generator& gen);

/// Bernoulli(p) — output dtype matches `dt` (each cell is 1.0 or 0.0).
/// Same algorithm as `bernoulli_mask_storage` but expressed at op level.
Storage random_bernoulli_storage(const Shape& shape, double p,
                                 Dtype dt, Device device, Generator& gen);

/// Uniform integer in [low, high). dt must be I32 or I64.
Storage random_randint_storage(const Shape& shape,
                               std::int64_t low, std::int64_t high,
                               Dtype dt, Device device, Generator& gen);

// --------------------------- Reduce-op helpers (Phase 3.3) ---------------

/// Normalize axes from user-facing form to a sorted-ascending vector of
/// non-negative axis indices.
///   - empty input  : reduce all axes (returns 0..ndim-1)
///   - negative idx : Python-style wrap (axis + ndim)
///   - duplicates   : deduplicated
///   - out-of-range : throws IndexError
std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim);

/// Compute the output shape after reducing along `axes`. If `keepdims=true`,
/// reduced axes become size 1 in-place; else they're removed.
Shape reduce_output_shape(const Shape& input_shape,
                          const std::vector<int>& axes,
                          bool keepdims);

/// Check that a saved input tensor's version_ still matches the recorded
/// value. Called from FuncOp::validate_versions on each input before backward.
/// Silently no-ops if the weak_ptr is dead (intermediate tensor freed). Throws
/// lucid::VersionMismatch otherwise. Defined in Helpers.cpp so FuncOp.h doesn't
/// have to pull in TensorImpl.h.
class TensorImpl;
void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         std::string_view op_name, std::size_t input_idx);

/// Broadcast a reduced gradient back to the original input shape. `grad_shape`
/// is the output shape of the forward reduce; `axes` are the (sorted-ascending)
/// reduced axis indices; `keepdims` matches the forward call. Returns a fresh
/// Storage of `input_shape` size, with each input position carrying the value
/// from the reduced position it was collapsed into.
Storage broadcast_back_for_reduce(const Storage& grad,
                                  const Shape& grad_shape,
                                  const Shape& input_shape,
                                  const std::vector<int>& axes,
                                  bool keepdims, Dtype dt, Device device);

}  // namespace lucid
