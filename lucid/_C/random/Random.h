#pragma once

#include <cstdint>

#include "../api.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

class Generator;

LUCID_API TensorImplPtr rand_op(const Shape& shape,
                                Dtype dt,
                                Device device,
                                Generator* gen = nullptr);

LUCID_API TensorImplPtr uniform_op(
    const Shape& shape, double low, double high, Dtype dt, Device device, Generator* gen = nullptr);

LUCID_API TensorImplPtr randn_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 Generator* gen = nullptr);

LUCID_API TensorImplPtr normal_op(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator* gen = nullptr);

LUCID_API TensorImplPtr randint_op(const Shape& shape,
                                   std::int64_t low,
                                   std::int64_t high,
                                   Dtype dt,
                                   Device device,
                                   Generator* gen = nullptr);

LUCID_API TensorImplPtr
bernoulli_op(const Shape& shape, double p, Dtype dt, Device device, Generator* gen = nullptr);

}  // namespace lucid
