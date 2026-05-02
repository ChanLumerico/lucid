#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>

#include "../../api.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

namespace lucid::gpu {

::mlx::core::Dtype to_mlx_dtype(Dtype dt);

Dtype from_mlx_dtype(::mlx::core::Dtype dt);

LUCID_API GpuStorage upload_cpu_to_gpu(const CpuStorage& cpu, const Shape& shape);

LUCID_API GpuStorage shared_storage_to_gpu(const SharedStorage& sh, const Shape& shape);

LUCID_API CpuStorage download_gpu_to_cpu(const GpuStorage& gpu, const Shape& shape);

GpuStorage wrap_mlx_array(::mlx::core::array&& arr, Dtype dtype);

::mlx::core::Shape to_mlx_shape(const Shape& shape);

inline Shape mlx_shape_to_lucid(const ::mlx::core::Shape& shape) {
    Shape out;
    out.reserve(shape.size());
    for (auto dim : shape)
        out.push_back(static_cast<std::int64_t>(dim));
    return out;
}

inline ::mlx::core::array mlx_scalar(double v, ::mlx::core::Dtype dt) {
    return ::mlx::core::astype(::mlx::core::array(static_cast<float>(v)), dt);
}

}  // namespace lucid::gpu
