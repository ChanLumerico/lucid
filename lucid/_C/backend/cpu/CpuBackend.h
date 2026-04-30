#pragma once

// =====================================================================
// Lucid C++ engine — CpuBackend: IBackend for CPU via Accelerate.
// =====================================================================
//
// Phase 4: implements IBackend using Apple Accelerate (vDSP/vForce/BLAS).
// Registered with Dispatcher at static-init time via g_cpu_registrar.
//
// Layer: backend/cpu/. Depends on backend/ and core/ only.

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../Dispatcher.h"
#include "../IBackend.h"
#include "Blas.h"
#include "Reduce.h"
#include "Shape.h"
#include "Vdsp.h"
#include "Vforce.h"

namespace lucid {
namespace backend {

class CpuBackend final : public IBackend {
public:
    static void register_self() {
        Dispatcher::register_backend(Device::CPU, std::make_unique<CpuBackend>());
    }

    Device device() const noexcept override { return Device::CPU; }

    // ---- Memory -------------------------------------------------------

    Storage zeros(const Shape& shape, Dtype dt) override {
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        std::memset(ptr.get(), 0, nb);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage ones(const Shape& shape, Dtype dt) override {
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        fill_ones(ptr.get(), n, dt);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage clone(const Storage& src, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(src);
        std::size_t nb = cs.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        std::memcpy(ptr.get(), cs.ptr.get(), nb);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    // ---- Elementwise binary -------------------------------------------

    Storage add(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return binary_op(
            a, b, shape, dt,
            [](const float* ap, const float* bp, float* op, std::size_t n) {
                cpu::vadd_f32(ap, bp, op, n);
            },
            [](const double* ap, const double* bp, double* op, std::size_t n) {
                cpu::vadd_f64(ap, bp, op, n);
            },
            [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
                cpu::vadd_i32(ap, bp, op, n);
            },
            [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
                cpu::vadd_i64(ap, bp, op, n);
            });
    }

    Storage sub(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return binary_op(
            a, b, shape, dt,
            [](const float* ap, const float* bp, float* op, std::size_t n) {
                cpu::vsub_f32(ap, bp, op, n);
            },
            [](const double* ap, const double* bp, double* op, std::size_t n) {
                cpu::vsub_f64(ap, bp, op, n);
            },
            [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = ap[i] - bp[i];
            },
            [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = ap[i] - bp[i];
            });
    }

    Storage mul(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return binary_op(
            a, b, shape, dt,
            [](const float* ap, const float* bp, float* op, std::size_t n) {
                cpu::vmul_f32(ap, bp, op, n);
            },
            [](const double* ap, const double* bp, double* op, std::size_t n) {
                cpu::vmul_f64(ap, bp, op, n);
            },
            [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = ap[i] * bp[i];
            },
            [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = ap[i] * bp[i];
            });
    }

    Storage div(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return binary_op(
            a, b, shape, dt,
            [](const float* ap, const float* bp, float* op, std::size_t n) {
                cpu::vdiv_f32(ap, bp, op, n);
            },
            [](const double* ap, const double* bp, double* op, std::size_t n) {
                cpu::vdiv_f64(ap, bp, op, n);
            },
            [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = ap[i] / bp[i];
            },
            [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = ap[i] / bp[i];
            });
    }

    Storage pow(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return binary_op(
            a, b, shape, dt,
            [](const float* ap, const float* bp, float* op, std::size_t n) {
                cpu::vpow_f32(ap, bp, op, n);
            },
            [](const double* ap, const double* bp, double* op, std::size_t n) {
                cpu::vpow_f64(ap, bp, op, n);
            },
            [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = static_cast<std::int32_t>(std::pow(ap[i], bp[i]));
            },
            [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = static_cast<std::int64_t>(std::pow(ap[i], bp[i]));
            });
    }

    Storage maximum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return binary_op(
            a, b, shape, dt,
            [](const float* ap, const float* bp, float* op, std::size_t n) {
                cpu::vmax_f32(ap, bp, op, n);
            },
            [](const double* ap, const double* bp, double* op, std::size_t n) {
                cpu::vmax_f64(ap, bp, op, n);
            },
            [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = std::max(ap[i], bp[i]);
            },
            [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = std::max(ap[i], bp[i]);
            });
    }

    Storage minimum(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        return binary_op(
            a, b, shape, dt,
            [](const float* ap, const float* bp, float* op, std::size_t n) {
                cpu::vmin_f32(ap, bp, op, n);
            },
            [](const double* ap, const double* bp, double* op, std::size_t n) {
                cpu::vmin_f64(ap, bp, op, n);
            },
            [](const std::int32_t* ap, const std::int32_t* bp, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = std::min(ap[i], bp[i]);
            },
            [](const std::int64_t* ap, const std::int64_t* bp, std::int64_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = std::min(ap[i], bp[i]);
            });
    }

    // ---- Elementwise unary --------------------------------------------

    Storage exp(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vexp_f32, cpu::vexp_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::exp(static_cast<float>(ip[i])));
                        });
    }

    Storage log(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vlog_f32, cpu::vlog_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::log(static_cast<float>(ip[i])));
                        });
    }

    Storage sqrt(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vsqrt_f32, cpu::vsqrt_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::sqrt(static_cast<float>(ip[i])));
                        });
    }

    Storage rsqrt(const Storage& a, const Shape& shape, Dtype dt) override {
        // rsqrt: 1/sqrt(x). No direct vForce call — compute via vsqrt then vrec.
        auto sq = sqrt(a, shape, dt);
        const auto& cs = std::get<CpuStorage>(sq);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32)
            cpu::vrec_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                          reinterpret_cast<float*>(ptr.get()), n);
        else if (dt == Dtype::F64)
            cpu::vrec_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                          reinterpret_cast<double*>(ptr.get()), n);
        else
            ErrorBuilder("cpu_backend::rsqrt").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage abs(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(
            a, shape, dt,
            [](const float* ip, float* op, std::size_t n) { cpu::vabs_f32(ip, op, n); },
            [](const double* ip, double* op, std::size_t n) { cpu::vabs_f64(ip, op, n); },
            [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = std::abs(ip[i]);
            });
    }

    Storage neg(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(
            a, shape, dt,
            [](const float* ip, float* op, std::size_t n) { cpu::vneg_f32(ip, op, n); },
            [](const double* ip, double* op, std::size_t n) { cpu::vneg_f64(ip, op, n); },
            [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = -ip[i];
            });
    }

    Storage sign(const Storage& a, const Shape& shape, Dtype dt) override {
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const auto& cs = std::get<CpuStorage>(a);
        if (dt == Dtype::F32) {
            const float* ip = reinterpret_cast<const float*>(cs.ptr.get());
            float* op = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = (ip[i] > 0.f) ? 1.f : ((ip[i] < 0.f) ? -1.f : 0.f);
        } else if (dt == Dtype::F64) {
            const double* ip = reinterpret_cast<const double*>(cs.ptr.get());
            double* op = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = (ip[i] > 0.0) ? 1.0 : ((ip[i] < 0.0) ? -1.0 : 0.0);
        } else if (dt == Dtype::I32) {
            const std::int32_t* ip = reinterpret_cast<const std::int32_t*>(cs.ptr.get());
            std::int32_t* op = reinterpret_cast<std::int32_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = (ip[i] > 0) ? 1 : ((ip[i] < 0) ? -1 : 0);
        } else if (dt == Dtype::I64) {
            const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
            std::int64_t* op = reinterpret_cast<std::int64_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = (ip[i] > 0) ? 1 : ((ip[i] < 0) ? -1 : 0);
        } else {
            ErrorBuilder("cpu_backend::sign").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage floor(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vfloor_f32, cpu::vfloor_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            std::copy(ip, ip + n, op);
                        });
    }

    Storage ceil(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vceil_f32, cpu::vceil_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            std::copy(ip, ip + n, op);
                        });
    }

    Storage round(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vround_f32, cpu::vround_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            std::copy(ip, ip + n, op);
                        });
    }

    Storage sin(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vsin_f32, cpu::vsin_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::sin(static_cast<float>(ip[i])));
                        });
    }

    Storage cos(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vcos_f32, cpu::vcos_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::cos(static_cast<float>(ip[i])));
                        });
    }

    Storage tanh(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vtanh_f32, cpu::vtanh_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::tanh(static_cast<float>(ip[i])));
                        });
    }

    Storage sigmoid(const Storage& a, const Shape& shape, Dtype dt) override {
        // sigmoid(x) = 1 / (1 + exp(-x))
        auto neg_a = neg(a, shape, dt);
        auto exp_neg = exp(neg_a, shape, dt);
        auto one = ones(shape, dt);
        auto denom = add(one, exp_neg, shape, dt);  // 1 + exp(-x)
        // reciprocal: use vrec via rsqrt workaround — but rsqrt is 1/sqrt(x).
        // Implement 1/denom directly with a scalar loop.
        const auto& cs = std::get<CpuStorage>(denom);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            cpu::vrec_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                          reinterpret_cast<float*>(ptr.get()), n);
        } else if (dt == Dtype::F64) {
            cpu::vrec_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                          reinterpret_cast<double*>(ptr.get()), n);
        } else {
            ErrorBuilder("cpu_backend::sigmoid").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage relu(const Storage& a, const Shape& shape, Dtype dt) override {
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const auto& cs = std::get<CpuStorage>(a);
        if (dt == Dtype::F32)
            cpu::vrelu_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                           reinterpret_cast<float*>(ptr.get()), n);
        else if (dt == Dtype::F64)
            cpu::vrelu_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                           reinterpret_cast<double*>(ptr.get()), n);
        else
            ErrorBuilder("cpu_backend::relu").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    // ---- Additional unary (Phase 4.5) ---------------------------------

    Storage log2(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vlog2_f32, cpu::vlog2_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = static_cast<std::int32_t>(
                                    std::log2(static_cast<float>(ip[i])));
                        });
    }

    Storage reciprocal(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32)
            cpu::vrec_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                          reinterpret_cast<float*>(ptr.get()), n);
        else if (dt == Dtype::F64)
            cpu::vrec_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                          reinterpret_cast<double*>(ptr.get()), n);
        else
            ErrorBuilder("cpu_backend::reciprocal").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage square(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vsq_f32, cpu::vsq_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = ip[i] * ip[i];
                        });
    }

    Storage cube(const Storage& a, const Shape& shape, Dtype dt) override {
        // x^3 = x * x^2 via two passes
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            auto* p = reinterpret_cast<const float*>(cs.ptr.get());
            auto* q = reinterpret_cast<float*>(ptr.get());
            cpu::vsq_f32(p, q, n);
            cpu::vmul_f32(p, q, q, n);
        } else if (dt == Dtype::F64) {
            auto* p = reinterpret_cast<const double*>(cs.ptr.get());
            auto* q = reinterpret_cast<double*>(ptr.get());
            cpu::vsq_f64(p, q, n);
            cpu::vmul_f64(p, q, q, n);
        } else {
            ErrorBuilder("cpu_backend::cube").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage cube_root(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(
            a, shape, dt,
            [](const float* ip, float* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = std::cbrt(ip[i]);
            },
            [](const double* ip, double* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = std::cbrt(ip[i]);
            },
            [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                for (std::size_t i = 0; i < n; ++i)
                    op[i] = static_cast<std::int32_t>(std::cbrt(static_cast<float>(ip[i])));
            });
    }

    Storage tan(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vtan_f32, cpu::vtan_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = static_cast<std::int32_t>(
                                    std::tan(static_cast<float>(ip[i])));
                        });
    }

    Storage asin(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vasin_f32, cpu::vasin_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = static_cast<std::int32_t>(
                                    std::asin(static_cast<float>(ip[i])));
                        });
    }

    Storage acos(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vacos_f32, cpu::vacos_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = static_cast<std::int32_t>(
                                    std::acos(static_cast<float>(ip[i])));
                        });
    }

    Storage atan(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vatan_f32, cpu::vatan_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = static_cast<std::int32_t>(
                                    std::atan(static_cast<float>(ip[i])));
                        });
    }

    Storage sinh(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vsinh_f32, cpu::vsinh_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = static_cast<std::int32_t>(
                                    std::sinh(static_cast<float>(ip[i])));
                        });
    }

    Storage cosh(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vcosh_f32, cpu::vcosh_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] = static_cast<std::int32_t>(
                                    std::cosh(static_cast<float>(ip[i])));
                        });
    }

    Storage invert(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::Bool) {
            const std::uint8_t* ip = reinterpret_cast<const std::uint8_t*>(cs.ptr.get());
            std::uint8_t* op = reinterpret_cast<std::uint8_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ip[i] ? 0 : 1;
        } else if (dt == Dtype::I8) {
            const std::int8_t* ip = reinterpret_cast<const std::int8_t*>(cs.ptr.get());
            std::int8_t* op = reinterpret_cast<std::int8_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ~ip[i];
        } else if (dt == Dtype::I16) {
            const std::int16_t* ip = reinterpret_cast<const std::int16_t*>(cs.ptr.get());
            std::int16_t* op = reinterpret_cast<std::int16_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ~ip[i];
        } else if (dt == Dtype::I32) {
            const std::int32_t* ip = reinterpret_cast<const std::int32_t*>(cs.ptr.get());
            std::int32_t* op = reinterpret_cast<std::int32_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ~ip[i];
        } else if (dt == Dtype::I64) {
            const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
            std::int64_t* op = reinterpret_cast<std::int64_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = ~ip[i];
        } else {
            ErrorBuilder("cpu_backend::invert").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage silu(const Storage& a, const Shape& shape, Dtype dt) override {
        // y = x * sigmoid(x)
        auto sig = sigmoid(a, shape, dt);
        return mul(a, sig, shape, dt);
    }

    Storage gelu(const Storage& a, const Shape& shape, Dtype dt) override {
        // gelu(x) = 0.5 * x * (1 + tanh(c1 * (x + c2 * x^3)))
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        constexpr double kC1 = 0.7978845608028654;  // sqrt(2/pi)
        constexpr double kC2 = 0.044715;
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            const float c1 = static_cast<float>(kC1);
            const float c2 = static_cast<float>(kC2);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                q[i] = 0.5f * x * (1.f + std::tanh(c1 * (x + c2 * x * x * x)));
            }
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                q[i] = 0.5 * x * (1.0 + std::tanh(kC1 * (x + kC2 * x * x * x)));
            }
        } else {
            ErrorBuilder("cpu_backend::gelu").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage softplus(const Storage& a, const Shape& shape, Dtype dt) override {
        // softplus(x) = log(1 + exp(x)) — numerically stable: max(x,0) + log1p(exp(-|x|))
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                q[i] = std::max(x, 0.f) + std::log1p(std::exp(-std::abs(x)));
            }
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                q[i] = std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
            }
        } else {
            ErrorBuilder("cpu_backend::softplus").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage selu(const Storage& a, const Shape& shape, Dtype dt) override {
        // SELU(x) = scale * (x if x >= 0 else alpha * (exp(x) - 1))
        constexpr double kScale = 1.0507009873554805;
        constexpr double kAlpha = 1.6732632423543772;
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            const float s = static_cast<float>(kScale);
            const float al = static_cast<float>(kAlpha);
            for (std::size_t i = 0; i < n; ++i)
                q[i] = s * (p[i] >= 0.f ? p[i] : al * (std::exp(p[i]) - 1.f));
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = kScale * (p[i] >= 0.0 ? p[i] : kAlpha * (std::exp(p[i]) - 1.0));
        } else {
            ErrorBuilder("cpu_backend::selu").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage mish(const Storage& a, const Shape& shape, Dtype dt) override {
        // y = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                const float sp = std::max(x, 0.f) + std::log1p(std::exp(-std::abs(x)));
                q[i] = x * std::tanh(sp);
            }
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                const double sp = std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
                q[i] = x * std::tanh(sp);
            }
        } else {
            ErrorBuilder("cpu_backend::mish").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage hard_sigmoid(const Storage& a, const Shape& shape, Dtype dt) override {
        // clip((x + 3) / 6, 0, 1)
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = std::min(std::max((p[i] + 3.f) / 6.f, 0.f), 1.f);
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = std::min(std::max((p[i] + 3.0) / 6.0, 0.0), 1.0);
        } else {
            ErrorBuilder("cpu_backend::hard_sigmoid").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage hard_swish(const Storage& a, const Shape& shape, Dtype dt) override {
        // x * clip((x + 3) / 6, 0, 1)
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = p[i] * std::min(std::max((p[i] + 3.f) / 6.f, 0.f), 1.f);
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = p[i] * std::min(std::max((p[i] + 3.0) / 6.0, 0.0), 1.0);
        } else {
            ErrorBuilder("cpu_backend::hard_swish").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage relu6(const Storage& a, const Shape& shape, Dtype dt) override {
        // clip(x, 0, 6)
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = std::min(std::max(p[i], 0.f), 6.f);
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = std::min(std::max(p[i], 0.0), 6.0);
        } else {
            ErrorBuilder("cpu_backend::relu6").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    // ---- Reduction ----------------------------------------------------

    Storage reduce_sum(const Storage& a,
                       const Shape& in_shape,
                       const ReduceOpts& opts,
                       Dtype dt) override {
        return reduce_axes(a, in_shape, opts, dt, ReduceOp::Sum);
    }

    Storage reduce_mean(const Storage& a,
                        const Shape& in_shape,
                        const ReduceOpts& opts,
                        Dtype dt) override {
        return reduce_axes(a, in_shape, opts, dt, ReduceOp::Mean);
    }

    Storage variance(const Storage& a,
                     const Shape& in_shape,
                     const ReduceOpts& opts,
                     Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        auto is_reduced_axis = [&](std::size_t d) {
            return std::find(opts.axes.begin(), opts.axes.end(), static_cast<int>(d)) !=
                   opts.axes.end();
        };
        Shape out_shape;
        if (opts.keepdims) {
            out_shape = in_shape;
            for (int ax : opts.axes)
                out_shape[static_cast<std::size_t>(ax)] = 1;
        } else {
            for (std::size_t d = 0; d < in_shape.size(); ++d) {
                if (!is_reduced_axis(d))
                    out_shape.push_back(in_shape[d]);
            }
        }

        const std::size_t in_numel = shape_numel(in_shape);
        const std::size_t out_numel = shape_numel(out_shape);
        std::size_t reduced = 1;
        for (int ax : opts.axes)
            reduced *= static_cast<std::size_t>(in_shape[static_cast<std::size_t>(ax)]);
        if (reduced == 0)
            reduced = 1;

        const std::size_t out_nbytes = out_numel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);

        auto compute = [&](auto* out_p, const auto* in_p) {
            using T = std::remove_pointer_t<decltype(out_p)>;
            std::vector<double> means(out_numel, 0.0);
            std::vector<std::int64_t> coords(in_shape.size(), 0);

            Shape kd_shape = in_shape;
            for (int ax : opts.axes)
                kd_shape[static_cast<std::size_t>(ax)] = 1;

            Stride kd_stride(kd_shape.size());
            if (!kd_shape.empty()) {
                kd_stride.back() = 1;
                for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(kd_shape.size()) - 2; i >= 0;
                     --i) {
                    kd_stride[static_cast<std::size_t>(i)] =
                        kd_stride[static_cast<std::size_t>(i) + 1] *
                        kd_shape[static_cast<std::size_t>(i) + 1];
                }
            }

            Stride in_stride(in_shape.size());
            if (!in_shape.empty()) {
                in_stride.back() = 1;
                for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(in_shape.size()) - 2; i >= 0;
                     --i) {
                    in_stride[static_cast<std::size_t>(i)] =
                        in_stride[static_cast<std::size_t>(i) + 1] *
                        in_shape[static_cast<std::size_t>(i) + 1];
                }
            }

            auto flat_to_coord = [&](std::size_t flat) {
                for (std::size_t d = 0; d < in_shape.size(); ++d) {
                    coords[d] = flat / static_cast<std::size_t>(in_stride[d]);
                    flat %= static_cast<std::size_t>(in_stride[d]);
                }
            };

            auto kd_flat = [&]() {
                std::size_t f = 0;
                for (std::size_t d = 0; d < in_shape.size(); ++d) {
                    std::int64_t c = coords[d];
                    if (kd_shape[d] == 1)
                        c = 0;
                    f += c * static_cast<std::size_t>(kd_stride[d]);
                }
                return f;
            };

            for (std::size_t i = 0; i < in_numel; ++i) {
                flat_to_coord(i);
                means[kd_flat()] += static_cast<double>(in_p[i]);
            }
            for (auto& m : means)
                m /= static_cast<double>(reduced);

            std::vector<double> vars(out_numel, 0.0);
            for (std::size_t i = 0; i < in_numel; ++i) {
                flat_to_coord(i);
                const auto kf = kd_flat();
                const double d = static_cast<double>(in_p[i]) - means[kf];
                vars[kf] += d * d;
            }
            for (std::size_t k = 0; k < out_numel; ++k)
                out_p[k] = static_cast<T>(vars[k] / static_cast<double>(reduced));
        };

        if (dt == Dtype::F32)
            compute(reinterpret_cast<float*>(ptr.get()),
                    reinterpret_cast<const float*>(cs.ptr.get()));
        else if (dt == Dtype::F64)
            compute(reinterpret_cast<double*>(ptr.get()),
                    reinterpret_cast<const double*>(cs.ptr.get()));
        else
            ErrorBuilder("cpu_backend::variance").not_implemented("dtype not supported");

        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage reduce_max(const Storage& a,
                       const Shape& in_shape,
                       const ReduceOpts& opts,
                       Dtype dt) override {
        return reduce_axes(a, in_shape, opts, dt, ReduceOp::Max);
    }

    Storage reduce_min(const Storage& a,
                       const Shape& in_shape,
                       const ReduceOpts& opts,
                       Dtype dt) override {
        return reduce_axes(a, in_shape, opts, dt, ReduceOp::Min);
    }

    Storage cumsum(const Storage& a, const Shape& shape, int axis, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t nb = cs.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const int ndim = static_cast<int>(shape.size());
        std::size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        for (int d = axis + 1; d < ndim; ++d)
            inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);

        auto run = [&](auto* dst, const auto* src) {
            using T = std::remove_pointer_t<decltype(dst)>;
            for (std::size_t o = 0; o < outer; ++o)
                for (std::size_t j = 0; j < inner; ++j) {
                    T acc = src[(o * L) * inner + j];
                    dst[(o * L) * inner + j] = acc;
                    for (std::size_t k = 1; k < L; ++k) {
                        acc = acc + src[(o * L + k) * inner + j];
                        dst[(o * L + k) * inner + j] = acc;
                    }
                }
        };

        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(ptr.get()),
                reinterpret_cast<const double*>(cs.ptr.get()));
        else if (dt == Dtype::I32)
            run(reinterpret_cast<std::int32_t*>(ptr.get()),
                reinterpret_cast<const std::int32_t*>(cs.ptr.get()));
        else if (dt == Dtype::I64)
            run(reinterpret_cast<std::int64_t*>(ptr.get()),
                reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
        else
            ErrorBuilder("cpu_backend::cumsum").not_implemented("dtype not supported");

        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage cumprod(const Storage& a, const Shape& shape, int axis, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t nb = cs.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const int ndim = static_cast<int>(shape.size());
        std::size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        for (int d = axis + 1; d < ndim; ++d)
            inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);

        auto run = [&](auto* dst, const auto* src) {
            using T = std::remove_pointer_t<decltype(dst)>;
            for (std::size_t o = 0; o < outer; ++o)
                for (std::size_t j = 0; j < inner; ++j) {
                    T acc = src[(o * L) * inner + j];
                    dst[(o * L) * inner + j] = acc;
                    for (std::size_t k = 1; k < L; ++k) {
                        acc = acc * src[(o * L + k) * inner + j];
                        dst[(o * L + k) * inner + j] = acc;
                    }
                }
        };

        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(ptr.get()),
                reinterpret_cast<const double*>(cs.ptr.get()));
        else if (dt == Dtype::I32)
            run(reinterpret_cast<std::int32_t*>(ptr.get()),
                reinterpret_cast<const std::int32_t*>(cs.ptr.get()));
        else if (dt == Dtype::I64)
            run(reinterpret_cast<std::int64_t*>(ptr.get()),
                reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
        else
            ErrorBuilder("cpu_backend::cumprod").not_implemented("dtype not supported");

        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t nb = cs.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
        std::size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
            inner *= static_cast<std::size_t>(shape[d]);

        if (dt == Dtype::F32) {
            const float* in = reinterpret_cast<const float*>(cs.ptr.get());
            float* out = reinterpret_cast<float*>(ptr.get());
            if (inner == 1) {
                for (std::size_t o = 0; o < outer; ++o) {
                    const float* row = in + o * axis_dim;
                    float* dst = out + o * axis_dim;
                    const float m = cpu::vmaxval_f32(row, axis_dim);
                    const float neg_m = -m;
                    cpu::vsadd_f32(row, neg_m, dst, axis_dim);
                    cpu::vexp_f32(dst, dst, axis_dim);
                    const float inv_s = 1.0f / cpu::vsum_f32(dst, axis_dim);
                    cpu::vsmul_f32(dst, inv_s, dst, axis_dim);
                }
            } else {
                for (std::size_t o = 0; o < outer; ++o) {
                    for (std::size_t i = 0; i < inner; ++i) {
                        const float* base = in + o * axis_dim * inner + i;
                        float m = base[0];
                        for (std::size_t r = 1; r < axis_dim; ++r) {
                            const float v = base[r * inner];
                            if (v > m)
                                m = v;
                        }
                        float* obase = out + o * axis_dim * inner + i;
                        float s = 0.0f;
                        for (std::size_t r = 0; r < axis_dim; ++r) {
                            const float e = std::exp(base[r * inner] - m);
                            obase[r * inner] = e;
                            s += e;
                        }
                        const float inv = 1.0f / s;
                        for (std::size_t r = 0; r < axis_dim; ++r)
                            obase[r * inner] *= inv;
                    }
                }
            }
        } else if (dt == Dtype::F64) {
            const double* in = reinterpret_cast<const double*>(cs.ptr.get());
            double* out = reinterpret_cast<double*>(ptr.get());
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t i = 0; i < inner; ++i) {
                    const double* base = in + o * axis_dim * inner + i;
                    double m = base[0];
                    for (std::size_t r = 1; r < axis_dim; ++r) {
                        const double v = base[r * inner];
                        if (v > m)
                            m = v;
                    }
                    double* obase = out + o * axis_dim * inner + i;
                    double s = 0.0;
                    for (std::size_t r = 0; r < axis_dim; ++r) {
                        const double e = std::exp(base[r * inner] - m);
                        obase[r * inner] = e;
                        s += e;
                    }
                    const double inv = 1.0 / s;
                    for (std::size_t r = 0; r < axis_dim; ++r)
                        obase[r * inner] *= inv;
                }
            }
        } else {
            ErrorBuilder("cpu_backend::softmax").not_implemented("dtype not supported");
        }

        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage softmax_backward(const Storage& z,
                             const Storage& grad_out,
                             const Shape& shape,
                             int axis,
                             Dtype dt) override {
        const auto& z_cpu = std::get<CpuStorage>(z);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);
        std::size_t nb = z_cpu.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
        std::size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
            inner *= static_cast<std::size_t>(shape[d]);

        auto run = [&](auto* dx, const auto* zp, const auto* gp) {
            using T = std::remove_pointer_t<decltype(dx)>;
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t i = 0; i < inner; ++i) {
                    const T* zb = zp + o * axis_dim * inner + i;
                    const T* gb = gp + o * axis_dim * inner + i;
                    T sum = T{};
                    for (std::size_t r = 0; r < axis_dim; ++r)
                        sum += gb[r * inner] * zb[r * inner];
                    T* xb = dx + o * axis_dim * inner + i;
                    for (std::size_t r = 0; r < axis_dim; ++r)
                        xb[r * inner] = zb[r * inner] * (gb[r * inner] - sum);
                }
            }
        };

        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(ptr.get()),
                reinterpret_cast<const float*>(z_cpu.ptr.get()),
                reinterpret_cast<const float*>(g_cpu.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(ptr.get()),
                reinterpret_cast<const double*>(z_cpu.ptr.get()),
                reinterpret_cast<const double*>(g_cpu.ptr.get()));
        else
            ErrorBuilder("cpu_backend::softmax_backward").not_implemented("dtype not supported");

        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage reverse_along_axis(const Storage& a,
                               const Shape& shape,
                               int axis,
                               Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
        const std::size_t elem = dtype_size(dt);
        const int ndim = static_cast<int>(shape.size());
        std::size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        for (int d = axis + 1; d < ndim; ++d)
            inner *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t k = 0; k < L; ++k) {
                std::memcpy(ptr.get() + ((o * L + k) * inner) * elem,
                            cs.ptr.get() + ((o * L + (L - 1 - k)) * inner) * elem,
                            inner * elem);
            }
        }
        return Storage{CpuStorage{ptr, cs.nbytes, dt}};
    }

    Storage trace(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        const std::int64_t M = shape[0];
        const std::int64_t N = shape[1];
        const std::int64_t L = std::min(M, N);
        Shape out_shape(shape.begin() + 2, shape.end());
        const std::size_t out_numel = shape_numel(out_shape);
        const std::size_t out_nbytes = out_numel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        auto run = [&](auto* out_p, const auto* in_p) {
            using T = std::remove_pointer_t<decltype(out_p)>;
            for (std::size_t k = 0; k < out_numel; ++k) {
                T sum{};
                for (std::int64_t i = 0; i < L; ++i) {
                    const std::size_t idx = (i * N + i) * out_numel + k;
                    sum = static_cast<T>(static_cast<double>(sum) + static_cast<double>(in_p[idx]));
                }
                out_p[k] = sum;
            }
        };
        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(ptr.get()),
                reinterpret_cast<const float*>(cs.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(ptr.get()),
                reinterpret_cast<const double*>(cs.ptr.get()));
        else
            ErrorBuilder("cpu_backend::trace").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage trace_backward(const Storage& grad_out, const Shape& input_shape, Dtype dt) override {
        const auto& cg = std::get<CpuStorage>(grad_out);
        const std::int64_t M = input_shape[0];
        const std::int64_t N = input_shape[1];
        const std::int64_t L = std::min(M, N);
        const std::size_t total = static_cast<std::size_t>(M * N);
        const std::size_t nbytes = total * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
        std::memset(ptr.get(), 0, nbytes);
        auto fill = [&](auto* dst, const auto* gp) {
            for (std::int64_t i = 0; i < L; ++i)
                dst[i * N + i] = *gp;
        };
        if (dt == Dtype::F32)
            fill(reinterpret_cast<float*>(ptr.get()),
                 reinterpret_cast<const float*>(cg.ptr.get()));
        else if (dt == Dtype::F64)
            fill(reinterpret_cast<double*>(ptr.get()),
                 reinterpret_cast<const double*>(cg.ptr.get()));
        else
            ErrorBuilder("cpu_backend::trace_backward").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nbytes, dt}};
    }

    std::vector<Storage> meshgrid(const std::vector<Storage>& xs,
                                  const Shape& out_shape,
                                  Dtype dt,
                                  bool indexing_xy) override {
        const std::size_t N = xs.size();
        const std::size_t total = shape_numel(out_shape);
        const std::size_t elem = dtype_size(dt);
        std::vector<Storage> result;
        result.reserve(N);
        for (std::size_t i = 0; i < N; ++i) {
            const auto& cv = std::get<CpuStorage>(xs[i]);
            std::size_t out_nbytes = total * elem;
            auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
            std::size_t carry_axis = i;
            if (indexing_xy && N >= 2 && i < 2)
                carry_axis = 1 - i;

            std::vector<std::int64_t> coord(N, 0);
            for (std::size_t f = 0; f < total; ++f) {
                const std::int64_t k = coord[carry_axis];
                std::memcpy(ptr.get() + f * elem, cv.ptr.get() + static_cast<std::size_t>(k) * elem,
                            elem);
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(N) - 1; d >= 0; --d) {
                    if (++coord[static_cast<std::size_t>(d)] <
                        out_shape[static_cast<std::size_t>(d)]) {
                        break;
                    }
                    coord[static_cast<std::size_t>(d)] = 0;
                }
            }
            result.push_back(Storage{CpuStorage{ptr, out_nbytes, dt}});
        }
        return result;
    }

    // ---- Linear algebra -----------------------------------------------

    Storage matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        const int M = opts.M, K = opts.K, N = opts.N;
        const std::size_t batch = opts.batch;
        std::size_t out_n = batch * static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
        std::size_t nb = out_n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        std::memset(ptr.get(), 0, nb);

        if (dt == Dtype::F32) {
            const float* ap = reinterpret_cast<const float*>(ca.ptr.get());
            const float* bp = reinterpret_cast<const float*>(cb.ptr.get());
            float* cp = reinterpret_cast<float*>(ptr.get());
            for (std::size_t bi = 0; bi < batch; ++bi) {
                cpu::sgemm(opts.transA, opts.transB, M, N, K, 1.0f, ap + bi * M * K,
                           opts.transA ? M : K, bp + bi * K * N, opts.transB ? K : N, 0.0f,
                           cp + bi * M * N, N);
            }
        } else if (dt == Dtype::F64) {
            const double* ap = reinterpret_cast<const double*>(ca.ptr.get());
            const double* bp = reinterpret_cast<const double*>(cb.ptr.get());
            double* cp = reinterpret_cast<double*>(ptr.get());
            for (std::size_t bi = 0; bi < batch; ++bi) {
                cpu::dgemm(opts.transA, opts.transB, M, N, K, 1.0, ap + bi * M * K,
                           opts.transA ? M : K, bp + bi * K * N, opts.transB ? K : N, 0.0,
                           cp + bi * M * N, N);
            }
        } else {
            ErrorBuilder("cpu_backend::matmul").not_implemented("dtype not supported");
        }
        Shape out_shape;
        if (batch > 1)
            out_shape.push_back(static_cast<std::int64_t>(batch));
        out_shape.push_back(M);
        out_shape.push_back(N);
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    // ---- Broadcast / cast -------------------------------------------

    Storage broadcast(const Storage& a,
                      const Shape& src_shape,
                      const Shape& dst_shape,
                      Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t ndim_out = dst_shape.size();
        const std::size_t ndim_in = src_shape.size();
        Shape padded(ndim_out, 1);
        for (std::size_t i = 0; i < ndim_in; ++i)
            padded[ndim_out - ndim_in + i] = src_shape[i];
        std::vector<std::size_t> in_str(ndim_out, 0);
        std::size_t s = 1;
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim_out) - 1; d >= 0; --d) {
            in_str[d] = (padded[d] == 1) ? 0 : s;
            s *= static_cast<std::size_t>(padded[d]);
        }
        const std::size_t out_n = shape_numel(dst_shape);
        std::size_t nb = out_n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* sp = reinterpret_cast<const T*>(cs.ptr.get());
            T* dp = reinterpret_cast<T*>(ptr.get());
            std::vector<std::size_t> coord(ndim_out, 0);
            for (std::size_t f = 0; f < out_n; ++f) {
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim_out; ++d)
                    in_flat += coord[d] * in_str[d];
                dp[f] = sp[in_flat];
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim_out) - 1; d >= 0; --d) {
                    if (++coord[d] < static_cast<std::size_t>(dst_shape[d]))
                        break;
                    coord[d] = 0;
                }
            }
        };

        switch (dt) {
            case Dtype::F32:
                run(float{});
                break;
            case Dtype::F64:
                run(double{});
                break;
            case Dtype::I32:
                run(std::int32_t{});
                break;
            case Dtype::I64:
                run(std::int64_t{});
                break;
            case Dtype::I16:
                run(std::int16_t{});
                break;
            case Dtype::I8:
            case Dtype::Bool:
                run(std::uint8_t{});
                break;
            default:
                ErrorBuilder("cpu_backend::broadcast").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage repeat(const Storage& a,
                   const Shape& shape,
                   Dtype dt,
                   std::int64_t repeats,
                   int axis) override {
        const auto& cs = std::get<CpuStorage>(a);
        Shape out_shape = shape;
        out_shape[static_cast<std::size_t>(axis)] *= repeats;
        std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t elem = dtype_size(dt);
        std::size_t outer = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        std::size_t inner = elem;
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
            inner *= static_cast<std::size_t>(shape[d]);
        const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
        auto* dst = ptr.get();
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t k = 0; k < L; ++k) {
                const auto* src = cs.ptr.get() + (o * L + k) * inner;
                for (std::int64_t r = 0; r < repeats; ++r) {
                    std::memcpy(dst, src, inner);
                    dst += inner;
                }
            }
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage tile(const Storage& a,
                 const Shape& shape,
                 Dtype dt,
                 const std::vector<std::int64_t>& reps) override {
        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t nout = reps.size();
        Shape padded(nout, 1);
        const std::size_t lead = nout - shape.size();
        for (std::size_t d = 0; d < shape.size(); ++d)
            padded[lead + d] = shape[d];
        Shape out_shape(nout);
        for (std::size_t d = 0; d < nout; ++d)
            out_shape[d] = padded[d] * reps[d];

        std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t elem = dtype_size(dt);
        Stride in_stride(nout);
        if (nout > 0) {
            in_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nout) - 2; d >= 0; --d)
                in_stride[static_cast<std::size_t>(d)] =
                    in_stride[static_cast<std::size_t>(d) + 1] * padded[static_cast<std::size_t>(d) + 1];
        }
        const std::size_t total = shape_numel(out_shape);
        std::vector<std::int64_t> coord(nout, 0);
        for (std::size_t f = 0; f < total; ++f) {
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < nout; ++d)
                in_flat += static_cast<std::size_t>(coord[d] % padded[d]) *
                           static_cast<std::size_t>(in_stride[d]);
            std::memcpy(ptr.get() + f * elem, cs.ptr.get() + in_flat * elem, elem);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nout) - 1; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] < out_shape[static_cast<std::size_t>(d)])
                    break;
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage permute(const Storage& a,
                    const Shape& shape,
                    const std::vector<int>& perm,
                    Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        Shape out_shape;
        out_shape.reserve(perm.size());
        for (int p : perm)
            out_shape.push_back(shape[static_cast<std::size_t>(p)]);
        std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (nb == 0)
            return Storage{CpuStorage{ptr, nb, dt}};
        switch (dt) {
            case Dtype::F32:
                cpu::permute_copy_f32(reinterpret_cast<const float*>(cs.ptr.get()),
                                      reinterpret_cast<float*>(ptr.get()), shape, perm);
                break;
            case Dtype::F64:
                cpu::permute_copy_f64(reinterpret_cast<const double*>(cs.ptr.get()),
                                      reinterpret_cast<double*>(ptr.get()), shape, perm);
                break;
            case Dtype::I32:
                cpu::permute_copy_i32(reinterpret_cast<const std::int32_t*>(cs.ptr.get()),
                                      reinterpret_cast<std::int32_t*>(ptr.get()), shape, perm);
                break;
            case Dtype::I64:
                cpu::permute_copy_i64(reinterpret_cast<const std::int64_t*>(cs.ptr.get()),
                                      reinterpret_cast<std::int64_t*>(ptr.get()), shape, perm);
                break;
            default:
                ErrorBuilder("cpu_backend::permute")
                    .not_implemented("dtype not supported (F32/F64/I32/I64)");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage pad(const Storage& a,
                const Shape& shape,
                Dtype dt,
                const std::vector<std::pair<std::int64_t, std::int64_t>>& pad_width,
                double constant) override {
        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t ndim = shape.size();
        Shape out_shape(ndim);
        for (std::size_t d = 0; d < ndim; ++d)
            out_shape[d] = shape[d] + pad_width[d].first + pad_width[d].second;

        std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto fill = [&](auto* dst, std::size_t n, double value) {
            using T = std::remove_pointer_t<decltype(dst)>;
            const T v = static_cast<T>(value);
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = v;
        };
        const std::size_t out_numel = shape_numel(out_shape);
        if (constant != 0.0) {
            switch (dt) {
                case Dtype::F32:
                    fill(reinterpret_cast<float*>(ptr.get()), out_numel, constant);
                    break;
                case Dtype::F64:
                    fill(reinterpret_cast<double*>(ptr.get()), out_numel, constant);
                    break;
                case Dtype::I32:
                    fill(reinterpret_cast<std::int32_t*>(ptr.get()), out_numel, constant);
                    break;
                case Dtype::I64:
                    fill(reinterpret_cast<std::int64_t*>(ptr.get()), out_numel, constant);
                    break;
                default:
                    ErrorBuilder("cpu_backend::pad").not_implemented("dtype not supported");
            }
        }

        const std::size_t elem = dtype_size(dt);
        Stride in_stride(ndim), out_stride(ndim);
        if (ndim > 0) {
            in_stride.back() = 1;
            out_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
                in_stride[static_cast<std::size_t>(d)] =
                    in_stride[static_cast<std::size_t>(d) + 1] * shape[static_cast<std::size_t>(d) + 1];
                out_stride[static_cast<std::size_t>(d)] =
                    out_stride[static_cast<std::size_t>(d) + 1] * out_shape[static_cast<std::size_t>(d) + 1];
            }
        }
        const std::size_t row_in = static_cast<std::size_t>(shape.back());
        const std::size_t row_bytes = row_in * elem;
        const std::size_t in_numel = shape_numel(shape);
        const std::size_t rows = in_numel / row_in;

        std::vector<std::int64_t> coord(ndim - 1, 0);
        for (std::size_t r = 0; r < rows; ++r) {
            std::size_t out_off = static_cast<std::size_t>(pad_width.back().first);
            for (std::size_t d = 0; d + 1 < ndim; ++d)
                out_off += static_cast<std::size_t>(coord[d] + pad_width[d].first) *
                           static_cast<std::size_t>(out_stride[d]);
            std::size_t in_off = 0;
            for (std::size_t d = 0; d + 1 < ndim; ++d)
                in_off += static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
            std::memcpy(ptr.get() + out_off * elem, cs.ptr.get() + in_off * elem, row_bytes);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)])
                    break;
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage pow_scalar(const Storage& a, const Shape& shape, Dtype dt, double exp) override {
        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t numel = shape_numel(shape);
        std::size_t nb = numel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        switch (dt) {
            case Dtype::F32: {
                std::vector<float> exp_buf(numel, static_cast<float>(exp));
                cpu::vpow_f32(reinterpret_cast<const float*>(cs.ptr.get()), exp_buf.data(),
                              reinterpret_cast<float*>(ptr.get()), numel);
                break;
            }
            case Dtype::F64: {
                std::vector<double> exp_buf(numel, exp);
                cpu::vpow_f64(reinterpret_cast<const double*>(cs.ptr.get()), exp_buf.data(),
                              reinterpret_cast<double*>(ptr.get()), numel);
                break;
            }
            default:
                ErrorBuilder("cpu_backend::pow_scalar").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage rpow_scalar(const Storage& a, const Shape& shape, Dtype dt, double base) override {
        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t numel = shape_numel(shape);
        std::size_t nb = numel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        switch (dt) {
            case Dtype::F32: {
                std::vector<float> base_buf(numel, static_cast<float>(base));
                cpu::vpow_f32(base_buf.data(), reinterpret_cast<const float*>(cs.ptr.get()),
                              reinterpret_cast<float*>(ptr.get()), numel);
                break;
            }
            case Dtype::F64: {
                std::vector<double> base_buf(numel, base);
                cpu::vpow_f64(base_buf.data(), reinterpret_cast<const double*>(cs.ptr.get()),
                              reinterpret_cast<double*>(ptr.get()), numel);
                break;
            }
            default:
                ErrorBuilder("cpu_backend::rpow_scalar").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage clip(const Storage& a,
                 const Shape& shape,
                 Dtype dt,
                 double min_v,
                 double max_v) override {
        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t numel = shape_numel(shape);
        std::size_t nb = numel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        switch (dt) {
            case Dtype::F32: {
                const auto* src = reinterpret_cast<const float*>(cs.ptr.get());
                auto* dst = reinterpret_cast<float*>(ptr.get());
                const float lo = static_cast<float>(min_v);
                const float hi = static_cast<float>(max_v);
                for (std::size_t i = 0; i < numel; ++i) {
                    float v = src[i];
                    if (v < lo)
                        v = lo;
                    else if (v > hi)
                        v = hi;
                    dst[i] = v;
                }
                break;
            }
            case Dtype::F64: {
                const auto* src = reinterpret_cast<const double*>(cs.ptr.get());
                auto* dst = reinterpret_cast<double*>(ptr.get());
                for (std::size_t i = 0; i < numel; ++i) {
                    double v = src[i];
                    if (v < min_v)
                        v = min_v;
                    else if (v > max_v)
                        v = max_v;
                    dst[i] = v;
                }
                break;
            }
            default:
                ErrorBuilder("cpu_backend::clip").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage cast(const Storage& a, const Shape& shape, Dtype src_dt, Dtype dst_dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dst_dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        cast_impl(cs.ptr.get(), ptr.get(), n, src_dt, dst_dt);
        return Storage{CpuStorage{ptr, nb, dst_dt}};
    }

private:
    // ---- Helpers -------------------------------------------------------

    void fill_ones(std::byte* ptr, std::size_t n, Dtype dt) {
        switch (dt) {
            case Dtype::F32: {
                float* p = reinterpret_cast<float*>(ptr);
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = 1.f;
                break;
            }
            case Dtype::F64: {
                double* p = reinterpret_cast<double*>(ptr);
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = 1.0;
                break;
            }
            case Dtype::I32: {
                std::int32_t* p = reinterpret_cast<std::int32_t*>(ptr);
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = 1;
                break;
            }
            case Dtype::I64: {
                std::int64_t* p = reinterpret_cast<std::int64_t*>(ptr);
                for (std::size_t i = 0; i < n; ++i)
                    p[i] = 1;
                break;
            }
            default:
                ErrorBuilder("cpu_backend::ones").not_implemented("dtype not supported");
        }
    }

    template <class F32Fn, class F64Fn, class I32Fn>
    Storage unary_op(
        const Storage& a, const Shape& shape, Dtype dt, F32Fn fn32, F64Fn fn64, I32Fn fni32) {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32)
            fn32(reinterpret_cast<const float*>(cs.ptr.get()), reinterpret_cast<float*>(ptr.get()),
                 n);
        else if (dt == Dtype::F64)
            fn64(reinterpret_cast<const double*>(cs.ptr.get()),
                 reinterpret_cast<double*>(ptr.get()), n);
        else if (dt == Dtype::I32)
            fni32(reinterpret_cast<const std::int32_t*>(cs.ptr.get()),
                  reinterpret_cast<std::int32_t*>(ptr.get()), n);
        else if (dt == Dtype::I64) {
            // Promote i64 through double
            const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
            std::int64_t* op = reinterpret_cast<std::int64_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = static_cast<std::int64_t>(static_cast<double>(static_cast<double>(ip[i])));
        } else {
            ErrorBuilder("cpu_backend::unary").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    template <class F32Fn, class F64Fn, class I32Fn, class I64Fn>
    Storage binary_op(const Storage& a,
                      const Storage& b,
                      const Shape& shape,
                      Dtype dt,
                      F32Fn fn32,
                      F64Fn fn64,
                      I32Fn fni32,
                      I64Fn fni64) {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32)
            fn32(reinterpret_cast<const float*>(ca.ptr.get()),
                 reinterpret_cast<const float*>(cb.ptr.get()), reinterpret_cast<float*>(ptr.get()),
                 n);
        else if (dt == Dtype::F64)
            fn64(reinterpret_cast<const double*>(ca.ptr.get()),
                 reinterpret_cast<const double*>(cb.ptr.get()),
                 reinterpret_cast<double*>(ptr.get()), n);
        else if (dt == Dtype::I32)
            fni32(reinterpret_cast<const std::int32_t*>(ca.ptr.get()),
                  reinterpret_cast<const std::int32_t*>(cb.ptr.get()),
                  reinterpret_cast<std::int32_t*>(ptr.get()), n);
        else if (dt == Dtype::I64)
            fni64(reinterpret_cast<const std::int64_t*>(ca.ptr.get()),
                  reinterpret_cast<const std::int64_t*>(cb.ptr.get()),
                  reinterpret_cast<std::int64_t*>(ptr.get()), n);
        else
            ErrorBuilder("cpu_backend::binary").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    enum class ReduceOp { Sum, Mean, Max, Min };

    // Reduce over a sorted list of axes sequentially.
    Storage reduce_axes(
        const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt, ReduceOp op) {
        if (opts.axes.empty()) {
            // Reduce all dims
            std::vector<int> all_axes;
            for (int i = 0; i < static_cast<int>(in_shape.size()); ++i)
                all_axes.push_back(i);
            ReduceOpts all_opts{all_axes, opts.keepdims};
            return reduce_axes(a, in_shape, all_opts, dt, op);
        }

        // Sort axes descending so indices stay stable
        auto axes = opts.axes;
        std::sort(axes.begin(), axes.end(), std::greater<int>());

        Storage cur = a;
        Shape cur_shape = in_shape;

        for (int axis : axes) {
            const auto& cs = std::get<CpuStorage>(cur);
            const int ndim = static_cast<int>(cur_shape.size());
            const int ax = (axis < 0) ? axis + ndim : axis;
            const std::size_t outer = [&] {
                std::size_t o = 1;
                for (int i = 0; i < ax; ++i)
                    o *= static_cast<std::size_t>(cur_shape[i]);
                return o;
            }();
            const std::size_t rd = static_cast<std::size_t>(cur_shape[ax]);
            const std::size_t inner = [&] {
                std::size_t ii = 1;
                for (int i = ax + 1; i < ndim; ++i)
                    ii *= static_cast<std::size_t>(cur_shape[i]);
                return ii;
            }();
            const std::size_t out_n = outer * inner;
            Shape out_shape;
            for (int i = 0; i < ndim; ++i)
                if (i != ax)
                    out_shape.push_back(cur_shape[i]);
            if (out_shape.empty())
                out_shape.push_back(1);

            std::size_t nb = out_n * dtype_size(dt);
            auto ptr = allocate_aligned_bytes(nb, Device::CPU);

            if (dt == Dtype::F32) {
                const float* ip = reinterpret_cast<const float*>(cs.ptr.get());
                float* outp = reinterpret_cast<float*>(ptr.get());
                if (op == ReduceOp::Sum || op == ReduceOp::Mean)
                    cpu::sum_axis_f32(ip, outp, outer, rd, inner);
                else if (op == ReduceOp::Max)
                    cpu::max_axis_f32(ip, outp, outer, rd, inner);
                else
                    cpu::min_axis_f32(ip, outp, outer, rd, inner);
                if (op == ReduceOp::Mean) {
                    float factor = 1.f / static_cast<float>(rd);
                    for (std::size_t i = 0; i < out_n; ++i)
                        outp[i] *= factor;
                }
            } else if (dt == Dtype::F64) {
                const double* ip = reinterpret_cast<const double*>(cs.ptr.get());
                double* outp = reinterpret_cast<double*>(ptr.get());
                if (op == ReduceOp::Sum || op == ReduceOp::Mean)
                    cpu::sum_axis_f64(ip, outp, outer, rd, inner);
                else if (op == ReduceOp::Max)
                    cpu::max_axis_f64(ip, outp, outer, rd, inner);
                else
                    cpu::min_axis_f64(ip, outp, outer, rd, inner);
                if (op == ReduceOp::Mean) {
                    double factor = 1.0 / static_cast<double>(rd);
                    for (std::size_t i = 0; i < out_n; ++i)
                        outp[i] *= factor;
                }
            } else {
                ErrorBuilder("cpu_backend::reduce").not_implemented("dtype not supported");
            }
            cur = Storage{CpuStorage{ptr, nb, dt}};
            cur_shape = out_shape;
        }

        // Apply keepdims if requested
        if (opts.keepdims) {
            Shape kept = in_shape;
            for (int ax : axes)
                kept[(ax < 0) ? ax + static_cast<int>(in_shape.size()) : ax] = 1;
            // cur_shape is already squeezed; just return with kept shape metadata
            // (caller re-shapes if needed — Storage doesn't carry shape)
        }
        return cur;
    }

    void cast_impl(
        const std::byte* src, std::byte* dst, std::size_t n, Dtype src_dt, Dtype dst_dt) {
        auto cast_loop = [&](auto from_tag, auto to_tag) {
            using Src = decltype(from_tag);
            using Dst = decltype(to_tag);
            const Src* sp = reinterpret_cast<const Src*>(src);
            Dst* dp = reinterpret_cast<Dst*>(dst);
            for (std::size_t i = 0; i < n; ++i)
                dp[i] = static_cast<Dst>(sp[i]);
        };

#define CAST_CASE(S, D, st, dt_)                    \
    if (src_dt == Dtype::S && dst_dt == Dtype::D) { \
        cast_loop(st{}, dt_{});                     \
        return;                                     \
    }

        CAST_CASE(F32, F64, float, double)
        CAST_CASE(F64, F32, double, float)
        CAST_CASE(F32, I32, float, std::int32_t)
        CAST_CASE(F32, I64, float, std::int64_t)
        CAST_CASE(F64, I32, double, std::int32_t)
        CAST_CASE(F64, I64, double, std::int64_t)
        CAST_CASE(I32, F32, std::int32_t, float)
        CAST_CASE(I32, F64, std::int32_t, double)
        CAST_CASE(I64, F32, std::int64_t, float)
        CAST_CASE(I64, F64, std::int64_t, double)
        CAST_CASE(I32, I64, std::int32_t, std::int64_t)
        CAST_CASE(I64, I32, std::int64_t, std::int32_t)
#undef CAST_CASE

        // Same type: memcpy
        if (src_dt == dst_dt) {
            std::memcpy(dst, src, n * dtype_size(dst_dt));
            return;
        }
        ErrorBuilder("cpu_backend::cast").not_implemented("unsupported dtype pair");
    }
};

}  // namespace backend
}  // namespace lucid

// ---------------------------------------------------------------------------
// Auto-registration at static init.
// ---------------------------------------------------------------------------
namespace {
struct CpuBackendRegistrar {
    CpuBackendRegistrar() {
        lucid::backend::Dispatcher::register_backend(
            lucid::Device::CPU, std::make_unique<lucid::backend::CpuBackend>());
    }
} g_cpu_registrar;
}  // namespace
