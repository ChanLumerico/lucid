#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::linalg_detail {

using ::lucid::gpu::mlx_shape_to_lucid;
using ::lucid::helpers::fresh;

inline const ::mlx::core::Device kMlxLinalgStream{::mlx::core::Device::cpu};

inline ::mlx::core::array as_mlx_array_gpu(const TensorImplPtr& t) {
    if (t->device() != Device::GPU)
        ErrorBuilder("as_mlx_array_gpu").fail("not a GPU tensor");
    const auto& g = storage_gpu(t->storage());
    return *g.arr;
}

inline Storage wrap_gpu_result(::mlx::core::array&& out, Dtype dtype) {
    return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
}

using ::lucid::helpers::allocate_cpu;

inline std::int64_t leading_batch_count(const Shape& shape, std::size_t mat_dims) {
    if (shape.size() < mat_dims)
        ErrorBuilder("linalg").fail("input rank too small");
    std::int64_t b = 1;
    for (std::size_t i = 0; i + mat_dims < shape.size(); ++i)
        b *= shape[i];
    return b;
}

inline void require_float(Dtype dt, const char* op) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder(op).not_implemented("only F32/F64 supported (got" +
                                         std::string(dtype_name(dt)) + ")");
}

inline void require_square_2d(const Shape& sh, const char* op) {
    if (sh.size() < 2)
        ErrorBuilder(op).fail("input must be at least 2-D");
    if (sh[sh.size() - 1] != sh[sh.size() - 2])
        ErrorBuilder(op).fail("last two dims must be equal (square)");
}

inline void check_lapack_info(int info, const char* op) {
    if (info < 0)
        ErrorBuilder(op).fail("LAPACK invalid argument index" + std::to_string(-info));
    if (info > 0)
        ErrorBuilder(op).fail("LAPACK numerical failure (info=" + std::to_string(info) + ")");
}

}  // namespace lucid::linalg_detail
