// lucid/_C/test/helpers/numeric_assert.h
//
// Custom GTest matchers and assertion macros for TensorImpl objects.
//
// Usage:
//   EXPECT_TENSOR_SHAPE(t, (Shape{3, 4}));
//   EXPECT_TENSOR_DTYPE(t, Dtype::F32);
//   EXPECT_TENSOR_NEAR(t, 1.0f, 1e-5f);   // all elements near value
//   ASSERT_TENSOR_FINITE(t);               // no NaN / Inf
//   EXPECT_TENSORS_NEAR(a, b, tol);        // element-wise comparison

#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "../../core/fwd.h"
#include "../../core/TensorImpl.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../backend/Dispatcher.h"

namespace lucid::test {

// ── Extract CPU data as float vector ─────────────────────────────────────────
//
// For CPU tensors: directly reinterpret the CpuStorage byte buffer.
// For GPU tensors: transfer to CPU first via IBackend::to_cpu(), then read.

inline std::vector<float> to_float_vec(const TensorImplPtr& t) {
    std::size_t n = t->numel();

    CpuStorage cpu_st;
    if (t->device() == Device::CPU) {
        cpu_st = std::get<CpuStorage>(t->storage());
    } else {
        auto& b = backend::Dispatcher::for_device(Device::CPU);
        cpu_st = b.to_cpu(t->storage(), t->shape());
    }

    if (t->dtype() == Dtype::F64) {
        const auto* raw = reinterpret_cast<const double*>(cpu_st.ptr.get());
        std::vector<float> result(n);
        for (std::size_t i = 0; i < n; ++i) result[i] = static_cast<float>(raw[i]);
        return result;
    }
    const auto* raw = reinterpret_cast<const float*>(cpu_st.ptr.get());
    return std::vector<float>(raw, raw + n);
}

// ── Gradient helpers ──────────────────────────────────────────────────────────
//
// Normal Engine::backward stores the gradient in grad_storage().
// Access helpers that avoid manually reconstructing a TensorImpl.

inline bool has_grad(const TensorImplPtr& t) {
    return t->grad_storage().has_value();
}

/// Extract gradient data as float vector.  Works for F32 and F64.
inline std::vector<float> grad_to_float_vec(const TensorImplPtr& t) {
    const auto& g = t->grad_storage();
    if (!g.has_value()) return {};
    const auto& cpu_st = std::get<CpuStorage>(*g);
    std::size_t n = cpu_st.nbytes / dtype_size(t->dtype());
    if (t->dtype() == Dtype::F64) {
        const auto* raw = reinterpret_cast<const double*>(cpu_st.ptr.get());
        std::vector<float> result(n);
        for (std::size_t i = 0; i < n; ++i) result[i] = static_cast<float>(raw[i]);
        return result;
    }
    const auto* raw = reinterpret_cast<const float*>(cpu_st.ptr.get());
    return std::vector<float>(raw, raw + n);
}

/// Number of elements in the gradient storage.
inline std::size_t grad_numel(const TensorImplPtr& t) {
    const auto& g = t->grad_storage();
    if (!g.has_value()) return 0;
    const auto& cpu_st = std::get<CpuStorage>(*g);
    return cpu_st.nbytes / dtype_size(t->dtype());
}

// ── Shape assertion ───────────────────────────────────────────────────────────
// NOTE: Store the TensorImplPtr in a local so its lifetime covers the
//       shape() reference — avoids UB with temporaries like exp_op(x).

#define EXPECT_TENSOR_SHAPE(tensor_impl_ptr, expected_shape)                   \
    do {                                                                        \
        auto _t = (tensor_impl_ptr);                                           \
        const auto& _s = _t->shape();                                          \
        const Shape _e = (expected_shape);                                     \
        EXPECT_EQ(_s, _e) << "Tensor shape mismatch.";                         \
    } while (0)

// ── Dtype assertion ───────────────────────────────────────────────────────────

#define EXPECT_TENSOR_DTYPE(tensor_impl_ptr, expected_dtype)                   \
    EXPECT_EQ((tensor_impl_ptr)->dtype(), (expected_dtype))                    \
        << "Tensor dtype mismatch."

// ── All elements near a scalar ────────────────────────────────────────────────

#define EXPECT_TENSOR_NEAR(tensor_impl_ptr, scalar_val, tol)                   \
    do {                                                                        \
        auto _vals = lucid::test::to_float_vec(tensor_impl_ptr);               \
        float _expected = static_cast<float>(scalar_val);                      \
        float _tol = static_cast<float>(tol);                                  \
        for (std::size_t _i = 0; _i < _vals.size(); ++_i) {                   \
            EXPECT_NEAR(_vals[_i], _expected, _tol)                            \
                << "Element [" << _i << "] out of tolerance.";                 \
        }                                                                       \
    } while (0)

// ── No NaN or Inf ─────────────────────────────────────────────────────────────

#define ASSERT_TENSOR_FINITE(tensor_impl_ptr)                                  \
    do {                                                                        \
        auto _vals = lucid::test::to_float_vec(tensor_impl_ptr);               \
        for (std::size_t _i = 0; _i < _vals.size(); ++_i) {                   \
            ASSERT_TRUE(std::isfinite(_vals[_i]))                              \
                << "Element [" << _i << "] is NaN or Inf.";                   \
        }                                                                       \
    } while (0)

// ── Element-wise comparison of two tensors ────────────────────────────────────

#define EXPECT_TENSORS_NEAR(lhs, rhs, tol)                                     \
    do {                                                                        \
        auto _a = lucid::test::to_float_vec(lhs);                              \
        auto _b = lucid::test::to_float_vec(rhs);                              \
        ASSERT_EQ(_a.size(), _b.size()) << "Tensor size mismatch.";            \
        float _tol = static_cast<float>(tol);                                  \
        for (std::size_t _i = 0; _i < _a.size(); ++_i) {                      \
            EXPECT_NEAR(_a[_i], _b[_i], _tol)                                 \
                << "Element [" << _i << "] mismatch.";                         \
        }                                                                       \
    } while (0)

}  // namespace lucid::test
