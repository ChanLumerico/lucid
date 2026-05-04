// lucid/_C/test/helpers/tensor_factory.h
//
// Convenience factories for constructing TensorImpl objects in C++ unit tests.
// No Python or pybind11 dependency — tests link only against lucid_core and
// the backend libraries.

#pragma once

#include <cstring>
#include <memory>
#include <vector>

#include "../../core/TensorImpl.h"
#include "../../core/Dtype.h"
#include "../../core/Device.h"
#include "../../backend/Dispatcher.h"

namespace lucid::test {

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Fill a freshly-allocated CPU tensor with `value`.
inline TensorImplPtr make_cpu_filled(const Shape& shape, Dtype dtype, float value) {
    auto& b = Dispatcher::for_device(Device::CPU);
    auto t = b.zeros(shape, dtype, Device::CPU);
    auto filled = b.full(shape, static_cast<double>(value), dtype, Device::CPU);
    return filled;
}

/// Create a CPU zeros tensor.
inline TensorImplPtr cpu_zeros(const Shape& shape, Dtype dtype = Dtype::F32) {
    return Dispatcher::for_device(Device::CPU).zeros(shape, dtype, Device::CPU);
}

/// Create a CPU ones tensor.
inline TensorImplPtr cpu_ones(const Shape& shape, Dtype dtype = Dtype::F32) {
    return Dispatcher::for_device(Device::CPU).full(shape, 1.0, dtype, Device::CPU);
}

/// Create a CPU tensor filled with `value`.
inline TensorImplPtr cpu_full(const Shape& shape, double value, Dtype dtype = Dtype::F32) {
    return Dispatcher::for_device(Device::CPU).full(shape, value, dtype, Device::CPU);
}

/// Create an identity matrix (n×n) on CPU.
inline TensorImplPtr cpu_eye(int64_t n, Dtype dtype = Dtype::F32) {
    // Create zeros then set diagonal
    auto& b = Dispatcher::for_device(Device::CPU);
    return b.eye(n, n, 0, dtype, Device::CPU);
}

/// True if the CPU backend is available (always true for tests).
inline bool cpu_available() { return true; }

/// True if the GPU backend (MLX) is available.
inline bool gpu_available() {
    try {
        Dispatcher::for_device(Device::GPU);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace lucid::test
