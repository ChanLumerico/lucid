// lucid/_C/test/helpers/tensor_factory.h
//
// Convenience factories for constructing TensorImpl objects in C++ unit tests.
// Uses the public ops/gfunc API (zeros_op, ones_op, full_op, eye_op) which
// return TensorImplPtr directly — no need to interact with Storage or IBackend.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../../core/fwd.h"
#include "../../core/Dtype.h"
#include "../../core/Device.h"
#include "../../core/Shape.h"
#include "../../ops/gfunc/Gfunc.h"
#include "../../backend/Dispatcher.h"

namespace lucid::test {

/// Create a CPU zeros tensor.
inline TensorImplPtr cpu_zeros(const Shape& shape, Dtype dtype = Dtype::F32) {
    return zeros_op(shape, dtype, Device::CPU, /*requires_grad=*/false);
}

/// Create a CPU ones tensor.
inline TensorImplPtr cpu_ones(const Shape& shape, Dtype dtype = Dtype::F32) {
    return ones_op(shape, dtype, Device::CPU, /*requires_grad=*/false);
}

/// Create a CPU tensor filled with a constant value.
inline TensorImplPtr cpu_full(const Shape& shape, double value, Dtype dtype = Dtype::F32) {
    return full_op(shape, value, dtype, Device::CPU, /*requires_grad=*/false);
}

/// Create an identity matrix (N×N) on CPU.
inline TensorImplPtr cpu_eye(std::int64_t n, Dtype dtype = Dtype::F32) {
    return eye_op(n, n, 0, dtype, Device::CPU, /*requires_grad=*/false);
}

/// Always true — CPU backend is always available in tests.
inline bool cpu_available() { return true; }

/// True if the GPU backend (MLX / Metal) is available.
inline bool gpu_available() {
    try {
        backend::Dispatcher::for_device(Device::GPU);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace lucid::test
