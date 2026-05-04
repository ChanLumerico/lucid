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
#include <vector>

#include <gtest/gtest.h>

#include "../../core/TensorImpl.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"

namespace lucid::test {

// ── Extract CPU data as float vector ─────────────────────────────────────────

inline std::vector<float> to_float_vec(const TensorImplPtr& t) {
    auto& b = Dispatcher::for_device(Device::CPU);
    TensorImplPtr cpu_t = (t->device() == Device::CPU) ? t : b.from_gpu(t);
    const auto* raw = static_cast<const float*>(cpu_t->data_ptr());
    return std::vector<float>(raw, raw + cpu_t->numel());
}

// ── Shape assertion ───────────────────────────────────────────────────────────

#define EXPECT_TENSOR_SHAPE(tensor_impl_ptr, expected_shape)                   \
    do {                                                                        \
        const auto& _s = (tensor_impl_ptr)->shape();                           \
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

// ── Element-wise comparison of two tensors ───────────────────────────────────

#define EXPECT_TENSORS_NEAR(a, b, tol)                                         \
    do {                                                                        \
        auto _a = lucid::test::to_float_vec(a);                                \
        auto _b = lucid::test::to_float_vec(b);                                \
        ASSERT_EQ(_a.size(), _b.size()) << "Tensor size mismatch.";            \
        float _tol = static_cast<float>(tol);                                  \
        for (std::size_t _i = 0; _i < _a.size(); ++_i) {                      \
            EXPECT_NEAR(_a[_i], _b[_i], _tol)                                 \
                << "Element [" << _i << "] mismatch.";                         \
        }                                                                       \
    } while (0)

}  // namespace lucid::test
