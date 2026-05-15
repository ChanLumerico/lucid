// lucid/_C/random/Random.h
//
// Tensor-level random-number generation operations. Each function allocates
// a new leaf TensorImpl whose storage is filled with random values drawn from
// the specified distribution. The optional Generator* argument selects the
// Philox-based PRNG state; passing nullptr routes through the process-global
// default generator so all Lucid random calls participate in the same
// reproducible stream when a seed has been set.
//
// CPU dispatch ultimately calls vForce or a scalar loop; GPU dispatch
// delegates to mlx::core::random::* and keeps the result in a GpuStorage.
// All returned tensors have requires_grad = false because random ops are
// not differentiable in the conventional sense.

#pragma once

#include <cstdint>

#include "../api.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class Generator;

// Return a tensor filled with values sampled uniformly from [0, 1).
LUCID_API TensorImplPtr rand_op(const Shape& shape,
                                Dtype dt,
                                Device device,
                                Generator* gen = nullptr);

// Return a tensor filled with values sampled uniformly from [low, high).
// Raises if high <= low.
LUCID_API TensorImplPtr uniform_op(
    const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen = nullptr);

// Return a tensor filled with samples from N(0, 1) (standard normal).
LUCID_API TensorImplPtr randn_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 Generator* gen = nullptr);

// Return a tensor filled with samples from N(mean, std^2).
// Raises if std < 0.
LUCID_API TensorImplPtr normal_op(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen = nullptr);

// Return an integer tensor with samples drawn uniformly from [low, high).
LUCID_API TensorImplPtr randint_op(const Shape& shape,
                                   std::int64_t low,
                                   std::int64_t high,
                                   Dtype dt,
                                   Device device,
                                   Generator* gen = nullptr);

// Return a tensor of Bernoulli samples where each element is 1 with
// probability p and 0 with probability 1-p.
LUCID_API TensorImplPtr
bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen = nullptr);

}  // namespace lucid
