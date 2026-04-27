#pragma once

// =====================================================================
// linalg internal helpers — shared by Inv/Det/Solve/Cholesky/Norm/QR/SVD/
// MatrixPower/Pinv/Eig. All linalg ops route through MLX's CPU stream;
// for inputs that arrive on CPU, this header provides up/down helpers
// that wrap the MLX-CPU-stream call transparently.
// =====================================================================

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
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

// MLX linalg ops are evaluated on the CPU stream.
inline const ::mlx::core::Device kMlxCpu{::mlx::core::Device::cpu};

// Upload an input tensor to an MLX array regardless of source device.
// CPU tensors are uploaded once via the bridge; GPU tensors return their
// existing MLX array directly.  All linalg ops then call MLX with the
// `kMlxCpu` stream so the actual compute happens on CPU.
inline ::mlx::core::array as_mlx_array(const TensorImplPtr& t) {
    if (t->device_ == Device::GPU) {
        const auto& g = std::get<GpuStorage>(t->storage_);
        return *g.arr;
    }
    auto g = gpu::upload_cpu_to_gpu(std::get<CpuStorage>(t->storage_), t->shape_);
    return *g.arr;
}

// Wrap a result MLX array back into the appropriate Storage for `device`.
// For CPU device we download; for GPU we wrap.
inline Storage wrap_result(::mlx::core::array&& out, Dtype dtype, Device device,
                             const Shape& out_shape) {
    if (device == Device::CPU) {
        // Force MLX to materialize the array on its (CPU) stream before we
        // reach into its buffer for the host download.
        out.eval();
        GpuStorage tmp = gpu::wrap_mlx_array(std::move(out), dtype);
        return Storage{gpu::download_gpu_to_cpu(tmp, out_shape)};
    }
    return Storage{gpu::wrap_mlx_array(std::move(out), dtype)};
}

}  // namespace lucid::linalg_detail
