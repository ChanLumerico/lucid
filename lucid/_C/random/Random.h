#pragma once

// =====================================================================
// Lucid C++ engine — random tensor ops (Phase 3.8).
// =====================================================================
//
// User-facing ops that produce a fresh tensor of random values. All draws
// route through Lucid's Philox-4x32-10 `Generator` so a seeded Generator
// gives bit-exact identical output across CPU and GPU. Random ops have no
// gradient (kHasGradient=false) — they're pure samplers, not part of the
// autograd graph.
//
// AMP policy: KeepInput (no precision-sensitive intermediate; output dtype
// is what the caller asked for).

#include <cstdint>

#include "../api.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class Generator;

/// Uniform [0, 1). dtype must be F32 or F64. If `gen` is null, uses the
/// process-default generator (`default_generator()`).
LUCID_API TensorImplPtr rand_op(const Shape& shape,
                                Dtype dt,
                                Device device,
                                Generator* gen = nullptr);

/// Uniform [low, high). F32 or F64.
LUCID_API TensorImplPtr uniform_op(
    const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen = nullptr);

/// Standard normal N(0, 1) via Box-Muller. F32 or F64.
LUCID_API TensorImplPtr randn_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 Generator* gen = nullptr);

/// Normal(mean, std). F32 or F64.
LUCID_API TensorImplPtr normal_op(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen = nullptr);

/// Uniform integer in [low, high). dtype must be I32 or I64.
LUCID_API TensorImplPtr randint_op(const Shape& shape,
                                   std::int64_t low,
                                   std::int64_t high,
                                   Dtype dt,
                                   Device device,
                                   Generator* gen = nullptr);

/// Bernoulli(p) with output cast to the requested float dtype (1.0 or 0.0).
LUCID_API TensorImplPtr
bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen = nullptr);

}  // namespace lucid
