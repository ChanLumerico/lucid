#pragma once

#include <cstdint>

#include "../../api.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

LUCID_API TensorImplPtr zeros_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 bool requires_grad = false);

LUCID_API TensorImplPtr ones_op(const Shape& shape,
                                Dtype dt,
                                Device device,
                                bool requires_grad = false);

LUCID_API TensorImplPtr
full_op(const Shape& shape, double fill_value, Dtype dt, Device device, bool requires_grad = false);

LUCID_API TensorImplPtr empty_op(const Shape& shape,
                                 Dtype dt,
                                 Device device,
                                 bool requires_grad = false);

LUCID_API TensorImplPtr eye_op(std::int64_t N,
                               std::int64_t M,
                               std::int64_t k,
                               Dtype dt,
                               Device device,
                               bool requires_grad = false);

LUCID_API TensorImplPtr arange_op(
    double start, double stop, double step, Dtype dt, Device device, bool requires_grad = false);

LUCID_API TensorImplPtr linspace_op(double start,
                                    double stop,
                                    std::int64_t num,
                                    Dtype dt,
                                    Device device,
                                    bool requires_grad = false);

LUCID_API TensorImplPtr diag_op(const TensorImplPtr& v, std::int64_t k = 0);

LUCID_API TensorImplPtr zeros_like_op(const TensorImplPtr& a, bool requires_grad = false);

LUCID_API TensorImplPtr ones_like_op(const TensorImplPtr& a, bool requires_grad = false);

LUCID_API TensorImplPtr empty_like_op(const TensorImplPtr& a, bool requires_grad = false);

LUCID_API TensorImplPtr full_like_op(const TensorImplPtr& a,
                                     double fill_value,
                                     bool requires_grad = false);

}  // namespace lucid
