// lucid/_C/ops/quantized/QuantizedMatmul.cpp
//
// GPU-stream int4/int8 GEMM primitives over MLX group-wise affine kernels.
// Pattern B (see ops/fft, ops/linalg GPU paths): call mlx::core directly,
// guarded to Device::GPU, wrap the result back — no IBackend / CpuBackend /
// autograd wiring (inference only).

#include "QuantizedMatmul.h"

#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../../backend/gpu/MlxBridge.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"

namespace lucid {
namespace {

// Extract the backing ``mlx::core::array`` from a GPU tensor (guarded).
::mlx::core::array gpu_array(const TensorImplPtr& t, const char* who) {
    if (!t)
        ErrorBuilder(who).fail("null input tensor");
    if (t->device() != Device::GPU)
        ErrorBuilder(who).fail("quantized ops require GPU (Metal) tensors");
    return *std::get<GpuStorage>(t->storage()).arr;
}

// Wrap an MLX result into a fresh GPU TensorImpl tagged with ``dt``.
TensorImplPtr wrap(::mlx::core::array&& a, Dtype dt) {
    Shape sh = gpu::mlx_shape_to_lucid(a.shape());
    Storage st{gpu::wrap_mlx_array(std::move(a), dt)};
    return ::lucid::helpers::fresh(std::move(st), sh, dt, Device::GPU);
}

}  // namespace

std::vector<TensorImplPtr> quantize_op(const TensorImplPtr& w, int group_size, int bits) {
    ::mlx::core::array wa = gpu_array(w, "quantize");
    std::vector<::mlx::core::array> res = ::mlx::core::quantize(wa, group_size, bits);
    return {
        wrap(std::move(res[0]), Dtype::I32),  // packed uint32 → I32-tagged opaque
        wrap(std::move(res[1]), Dtype::F32),  // scales
        wrap(std::move(res[2]), Dtype::F32),  // biases
    };
}

// The packed weight is uint32 but Lucid tags it I32 (no uint32 dtype).  After a
// serialize / reload round-trip it comes back as a genuine int32 array, which
// MLX's quantized kernels reject.  A zero-cost bitcast to uint32 makes the op
// robust whether the tensor is still uint32 (in-memory) or int32 (reloaded).
::mlx::core::array as_packed_u32(const ::mlx::core::array& w) {
    return ::mlx::core::view(w, ::mlx::core::uint32);
}

TensorImplPtr dequantize_op(const TensorImplPtr& w, const TensorImplPtr& scales,
                            const TensorImplPtr& biases, int group_size, int bits) {
    ::mlx::core::array wa = as_packed_u32(gpu_array(w, "dequantize"));
    ::mlx::core::array sa = gpu_array(scales, "dequantize");
    std::optional<::mlx::core::array> ba =
        biases ? std::optional<::mlx::core::array>(gpu_array(biases, "dequantize"))
               : std::nullopt;
    ::mlx::core::array out = ::mlx::core::dequantize(wa, sa, ba, group_size, bits);
    return wrap(std::move(out), Dtype::F32);
}

TensorImplPtr quantized_matmul_op(const TensorImplPtr& x, const TensorImplPtr& w,
                                  const TensorImplPtr& scales, const TensorImplPtr& biases,
                                  bool transpose, int group_size, int bits) {
    ::mlx::core::array xa = gpu_array(x, "quantized_matmul");
    ::mlx::core::array wa = as_packed_u32(gpu_array(w, "quantized_matmul"));
    ::mlx::core::array sa = gpu_array(scales, "quantized_matmul");
    std::optional<::mlx::core::array> ba =
        biases ? std::optional<::mlx::core::array>(gpu_array(biases, "quantized_matmul"))
               : std::nullopt;
    ::mlx::core::array out =
        ::mlx::core::quantized_matmul(xa, wa, sa, ba, transpose, group_size, bits);
    return wrap(std::move(out), Dtype::F32);
}

}  // namespace lucid
