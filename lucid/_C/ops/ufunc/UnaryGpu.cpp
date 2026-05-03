// lucid/_C/ops/ufunc/UnaryGpu.cpp
//
// All GPU-specific unary op implementations have been migrated to
// backend/gpu/GpuBackend.h and the corresponding GpuBackend::* methods.
// Each unary op that previously required special MLX handling (mlx::core::eval
// barriers, custom array-wrapping logic) is now a regular virtual method on
// IBackend dispatched through Dispatcher::for_device(Device::GPU).
//
// The only exceptions are LeakyReluBackward::forward and EluBackward::forward
// in Activation.cpp, which still own their GPU paths directly because those
// ops take a scalar parameter (slope / alpha) that cannot pass through the
// standard single-input IBackend dispatch signature.
//
// This file is intentionally empty after the migration; it is kept to avoid
// build-system changes that would otherwise need to remove it from CMakeLists.
