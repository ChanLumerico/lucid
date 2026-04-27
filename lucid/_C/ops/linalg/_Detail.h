#pragma once

// =====================================================================
// linalg internal helpers — shared by Inv/Det/Solve/Cholesky/Norm/QR/SVD/
// MatrixPower/Pinv/Eig. All linalg ops route through MLX's CPU stream.
// =====================================================================

#include <mlx/array.h>
#include <mlx/device.h>

#include "../../core/Exceptions.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::linalg_detail {

inline TensorImplPtr fresh(Storage&& s, Shape shape, Dtype dt, Device device) {
    return std::make_shared<TensorImpl>(std::move(s), std::move(shape), dt,
                                        device, /*requires_grad=*/false);
}

inline Shape mlx_shape_to_lucid(const ::mlx::core::Shape& s) {
    Shape out;
    out.reserve(s.size());
    for (auto d : s) out.push_back(static_cast<std::int64_t>(d));
    return out;
}

inline void require_gpu(const TensorImplPtr& t, const char* op) {
    if (t->device_ != Device::GPU)
        throw NotImplementedError(
            std::string(op) +
            ": CPU path not implemented; move tensor to device=\"gpu\"");
}

// MLX linalg ops are evaluated on the CPU stream.
inline const ::mlx::core::Device kMlxCpu{::mlx::core::Device::cpu};

}  // namespace lucid::linalg_detail
