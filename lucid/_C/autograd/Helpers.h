#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "../core/Device.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {

class TensorImpl;

Storage make_zero_storage(const Shape& shape, Dtype dtype, Device device);

Storage make_ones_storage(const Shape& shape, Dtype dtype, Device device);

Storage reduce_grad_to_shape(const Storage& grad,
                             const Shape& grad_shape,
                             const Shape& target_shape,
                             Dtype dtype,
                             Device device);

void accumulate_into(Storage& dst, const Storage& src);

Storage negate_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage
multiply_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);
Storage
divide_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);
Storage
add_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);
Storage
subtract_storages(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);
Storage square_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage clone_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage log_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);
Storage
pow_storage(const Storage& base, const Storage& expo, std::size_t numel, Dtype dt, Device device);

Storage
ge_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);
Storage
lt_mask_storage(const Storage& a, const Storage& b, std::size_t numel, Dtype dt, Device device);

Storage
add_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device);

Storage
mul_scalar_storage(const Storage& s, double scalar, std::size_t numel, Dtype dt, Device device);

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

Storage in_range_mask_storage(
    const Storage& s, double lo, double hi, std::size_t numel, Dtype dt, Device device);

Storage positive_mask_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

Storage
leaky_mask_storage(const Storage& s, double slope, std::size_t numel, Dtype dt, Device device);

Storage sigmoid_storage(const Storage& s, std::size_t numel, Dtype dt, Device device);

class Generator;
Storage bernoulli_mask_storage(
    double keep_prob, std::size_t numel, Dtype dt, Device device, Generator& gen);

Storage bernoulli_mask_storage_shape(
    double keep_prob, const Shape& shape, Dtype dt, Device device, Generator& gen);

Storage random_uniform_storage(
    const Shape& shape, double lo, double hi, Dtype dt, Device device, Generator& gen);

Storage random_normal_storage(
    const Shape& shape, double mean, double std, Dtype dt, Device device, Generator& gen);

Storage
random_bernoulli_storage(const Shape& shape, double p, Dtype dt, Device device, Generator& gen);

Storage random_randint_storage(const Shape& shape,
                               std::int64_t low,
                               std::int64_t high,
                               Dtype dt,
                               Device device,
                               Generator& gen);

std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim);

Shape reduce_output_shape(const Shape& input_shape, const std::vector<int>& axes, bool keepdims);

class TensorImpl;
void check_version_match(const std::weak_ptr<TensorImpl>& live,
                         std::int64_t saved_version,
                         std::string_view op_name,
                         std::size_t input_idx);

Storage broadcast_back_for_reduce(const Storage& grad,
                                  const Shape& grad_shape,
                                  const Shape& input_shape,
                                  const std::vector<int>& axes,
                                  bool keepdims,
                                  Dtype dt,
                                  Device device);

}  // namespace lucid
