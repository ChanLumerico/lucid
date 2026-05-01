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
#include <numeric>

#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../Dispatcher.h"
#include "../IBackend.h"
#include "Blas.h"
#include "Lapack.h"
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

    Storage bitwise_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        const std::size_t n = shape_numel(shape);
        const std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);

        auto run = [&](auto* dst, const auto* lhs, const auto* rhs) {
            using T = std::remove_pointer_t<decltype(dst)>;
            for (std::size_t i = 0; i < n; ++i) {
                const auto x = static_cast<std::int64_t>(lhs[i]);
                const auto y = static_cast<std::int64_t>(rhs[i]);
                std::int64_t out;
                if (op == 0)
                    out = x & y;
                else if (op == 1)
                    out = x | y;
                else if (op == 2)
                    out = x ^ y;
                else
                    ErrorBuilder("cpu_backend::bitwise_binary").fail("unknown op");
                dst[i] = static_cast<T>(out);
            }
        };

        if (dt == Dtype::I32) {
            run(reinterpret_cast<std::int32_t*>(ptr.get()),
                reinterpret_cast<const std::int32_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int32_t*>(cb.ptr.get()));
        } else if (dt == Dtype::I64) {
            run(reinterpret_cast<std::int64_t*>(ptr.get()),
                reinterpret_cast<const std::int64_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int64_t*>(cb.ptr.get()));
        } else if (dt == Dtype::I16) {
            run(reinterpret_cast<std::int16_t*>(ptr.get()),
                reinterpret_cast<const std::int16_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int16_t*>(cb.ptr.get()));
        } else if (dt == Dtype::I8) {
            run(reinterpret_cast<std::int8_t*>(ptr.get()),
                reinterpret_cast<const std::int8_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int8_t*>(cb.ptr.get()));
        } else if (dt == Dtype::Bool) {
            run(reinterpret_cast<std::uint8_t*>(ptr.get()),
                reinterpret_cast<const std::uint8_t*>(ca.ptr.get()),
                reinterpret_cast<const std::uint8_t*>(cb.ptr.get()));
        } else {
            ErrorBuilder("cpu_backend::bitwise_binary")
                .not_implemented("dtype must be integer or bool");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage compare_binary(
        const Storage& a, const Storage& b, const Shape& shape, Dtype dt, int op) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        const std::size_t n = shape_numel(shape);
        auto ptr = allocate_aligned_bytes(n, Device::CPU);
        auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());

        auto run = [&](const auto* lhs, const auto* rhs) {
            for (std::size_t i = 0; i < n; ++i) {
                bool out;
                if (op == 0)
                    out = lhs[i] == rhs[i];
                else if (op == 1)
                    out = lhs[i] != rhs[i];
                else if (op == 2)
                    out = lhs[i] > rhs[i];
                else if (op == 3)
                    out = lhs[i] >= rhs[i];
                else if (op == 4)
                    out = lhs[i] < rhs[i];
                else if (op == 5)
                    out = lhs[i] <= rhs[i];
                else
                    ErrorBuilder("cpu_backend::compare_binary").fail("unknown op");
                dst[i] = out ? 1u : 0u;
            }
        };

        if (dt == Dtype::F32) {
            run(reinterpret_cast<const float*>(ca.ptr.get()),
                reinterpret_cast<const float*>(cb.ptr.get()));
        } else if (dt == Dtype::F64) {
            run(reinterpret_cast<const double*>(ca.ptr.get()),
                reinterpret_cast<const double*>(cb.ptr.get()));
        } else if (dt == Dtype::I32) {
            run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int32_t*>(cb.ptr.get()));
        } else if (dt == Dtype::I64) {
            run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int64_t*>(cb.ptr.get()));
        } else if (dt == Dtype::I16) {
            run(reinterpret_cast<const std::int16_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int16_t*>(cb.ptr.get()));
        } else if (dt == Dtype::I8) {
            run(reinterpret_cast<const std::int8_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int8_t*>(cb.ptr.get()));
        } else if (dt == Dtype::Bool) {
            run(reinterpret_cast<const std::uint8_t*>(ca.ptr.get()),
                reinterpret_cast<const std::uint8_t*>(cb.ptr.get()));
        } else {
            ErrorBuilder("cpu_backend::compare_binary").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, n, Dtype::Bool}};
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
                                op[i] =
                                    static_cast<std::int32_t>(std::log2(static_cast<float>(ip[i])));
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
                                op[i] =
                                    static_cast<std::int32_t>(std::tan(static_cast<float>(ip[i])));
                        });
    }

    Storage asin(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vasin_f32, cpu::vasin_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::asin(static_cast<float>(ip[i])));
                        });
    }

    Storage acos(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vacos_f32, cpu::vacos_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::acos(static_cast<float>(ip[i])));
                        });
    }

    Storage atan(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vatan_f32, cpu::vatan_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::atan(static_cast<float>(ip[i])));
                        });
    }

    Storage sinh(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vsinh_f32, cpu::vsinh_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::sinh(static_cast<float>(ip[i])));
                        });
    }

    Storage cosh(const Storage& a, const Shape& shape, Dtype dt) override {
        return unary_op(a, shape, dt, cpu::vcosh_f32, cpu::vcosh_f64,
                        [](const std::int32_t* ip, std::int32_t* op, std::size_t n) {
                            for (std::size_t i = 0; i < n; ++i)
                                op[i] =
                                    static_cast<std::int32_t>(std::cosh(static_cast<float>(ip[i])));
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

    Storage gelu_backward(const Storage& a,
                          const Storage& grad,
                          const Shape& shape,
                          Dtype dt) override {
        constexpr double kC1 = 0.7978845608028654;  // sqrt(2/pi)
        constexpr double kC2 = 0.044715;
        const auto& cs = std::get<CpuStorage>(a);
        const auto& gs = std::get<CpuStorage>(grad);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* x = reinterpret_cast<const float*>(cs.ptr.get());
            const float* g = reinterpret_cast<const float*>(gs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            const float c1 = static_cast<float>(kC1);
            const float c2 = static_cast<float>(kC2);
            for (std::size_t i = 0; i < n; ++i) {
                const float xi = x[i];
                const float inner = c1 * (xi + c2 * xi * xi * xi);
                const float t = std::tanh(inner);
                const float dinner = c1 * (1.f + 3.f * c2 * xi * xi);
                const float dx = 0.5f * (1.f + t) + 0.5f * xi * (1.f - t * t) * dinner;
                q[i] = dx * g[i];
            }
        } else if (dt == Dtype::F64) {
            const double* x = reinterpret_cast<const double*>(cs.ptr.get());
            const double* g = reinterpret_cast<const double*>(gs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double xi = x[i];
                const double inner = kC1 * (xi + kC2 * xi * xi * xi);
                const double t = std::tanh(inner);
                const double dinner = kC1 * (1.0 + 3.0 * kC2 * xi * xi);
                const double dx = 0.5 * (1.0 + t) + 0.5 * xi * (1.0 - t * t) * dinner;
                q[i] = dx * g[i];
            }
        } else {
            ErrorBuilder("cpu_backend::gelu_backward").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage leaky_relu(const Storage& a, const Shape& shape, Dtype dt, double slope) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            const float fs = static_cast<float>(slope);
            for (std::size_t i = 0; i < n; ++i)
                q[i] = p[i] >= 0.f ? p[i] : fs * p[i];
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = p[i] >= 0.0 ? p[i] : slope * p[i];
        } else {
            ErrorBuilder("cpu_backend::leaky_relu").not_implemented("dtype not supported");
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

    Storage elu(const Storage& a, const Shape& shape, Dtype dt, double alpha) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            float* q = reinterpret_cast<float*>(ptr.get());
            const float fa = static_cast<float>(alpha);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = p[i];
                q[i] = x >= 0.f ? x : fa * (std::exp(x) - 1.f);
            }
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            double* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = p[i];
                q[i] = x >= 0.0 ? x : alpha * (std::exp(x) - 1.0);
            }
        } else {
            ErrorBuilder("cpu_backend::elu").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage elu_backward(const Storage& a,
                         const Storage& grad,
                         const Shape& shape,
                         Dtype dt,
                         double alpha) override {
        const auto& cs = std::get<CpuStorage>(a);
        const auto& gs = std::get<CpuStorage>(grad);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
            const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
            float* qp = reinterpret_cast<float*>(ptr.get());
            const float fa = static_cast<float>(alpha);
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                qp[i] = (x >= 0.f ? 1.f : fa * std::exp(x)) * gp[i];
            }
        } else if (dt == Dtype::F64) {
            const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
            const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
            double* qp = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = xp[i];
                qp[i] = (x >= 0.0 ? 1.0 : alpha * std::exp(x)) * gp[i];
            }
        } else {
            ErrorBuilder("cpu_backend::elu_backward").not_implemented("dtype not supported");
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

    Storage selu_backward(const Storage& a,
                          const Storage& grad,
                          const Shape& shape,
                          Dtype dt) override {
        constexpr double kScale = 1.0507009873554805;
        constexpr double kAlpha = 1.6732632423543772;
        const auto& cs = std::get<CpuStorage>(a);
        const auto& gs = std::get<CpuStorage>(grad);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
            const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
            float* qp = reinterpret_cast<float*>(ptr.get());
            const float s = static_cast<float>(kScale);
            const float sa = static_cast<float>(kScale * kAlpha);
            for (std::size_t i = 0; i < n; ++i)
                qp[i] = (xp[i] >= 0.f ? s : sa * std::exp(xp[i])) * gp[i];
        } else if (dt == Dtype::F64) {
            const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
            const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
            double* qp = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double dx = xp[i] >= 0.0 ? kScale : kScale * kAlpha * std::exp(xp[i]);
                qp[i] = dx * gp[i];
            }
        } else {
            ErrorBuilder("cpu_backend::selu_backward").not_implemented("dtype not supported");
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

    Storage mish_backward(const Storage& a,
                          const Storage& grad,
                          const Shape& shape,
                          Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        const auto& gs = std::get<CpuStorage>(grad);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
            const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
            float* qp = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                const float sp = std::max(x, 0.f) + std::log1p(std::exp(-std::abs(x)));
                const float t = std::tanh(sp);
                const float sig = 1.f / (1.f + std::exp(-x));
                qp[i] = (t + x * (1.f - t * t) * sig) * gp[i];
            }
        } else if (dt == Dtype::F64) {
            const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
            const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
            double* qp = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = xp[i];
                const double sp = std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
                const double t = std::tanh(sp);
                const double sig = 1.0 / (1.0 + std::exp(-x));
                qp[i] = (t + x * (1.0 - t * t) * sig) * gp[i];
            }
        } else {
            ErrorBuilder("cpu_backend::mish_backward").not_implemented("dtype not supported");
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

    Storage hard_sigmoid_backward(const Storage& a,
                                  const Storage& grad,
                                  const Shape& shape,
                                  Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        const auto& gs = std::get<CpuStorage>(grad);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
            const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
            float* qp = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                qp[i] = (xp[i] > -3.f && xp[i] < 3.f) ? gp[i] / 6.f : 0.f;
        } else if (dt == Dtype::F64) {
            const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
            const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
            double* qp = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                qp[i] = (xp[i] > -3.0 && xp[i] < 3.0) ? gp[i] / 6.0 : 0.0;
        } else {
            ErrorBuilder("cpu_backend::hard_sigmoid_backward")
                .not_implemented("dtype not supported");
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

    Storage hard_swish_backward(const Storage& a,
                                const Storage& grad,
                                const Shape& shape,
                                Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        const auto& gs = std::get<CpuStorage>(grad);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const float* xp = reinterpret_cast<const float*>(cs.ptr.get());
            const float* gp = reinterpret_cast<const float*>(gs.ptr.get());
            float* qp = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const float x = xp[i];
                float dx;
                if (x <= -3.f)
                    dx = 0.f;
                else if (x >= 3.f)
                    dx = 1.f;
                else
                    dx = x / 3.f + 0.5f;
                qp[i] = dx * gp[i];
            }
        } else if (dt == Dtype::F64) {
            const double* xp = reinterpret_cast<const double*>(cs.ptr.get());
            const double* gp = reinterpret_cast<const double*>(gs.ptr.get());
            double* qp = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const double x = xp[i];
                double dx;
                if (x <= -3.0)
                    dx = 0.0;
                else if (x >= 3.0)
                    dx = 1.0;
                else
                    dx = x / 3.0 + 0.5;
                qp[i] = dx * gp[i];
            }
        } else {
            ErrorBuilder("cpu_backend::hard_swish_backward").not_implemented("dtype not supported");
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
        const std::size_t axis_dim =
            static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
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
        const std::size_t axis_dim =
            static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
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

    Storage reverse_along_axis(const Storage& a, const Shape& shape, int axis, Dtype dt) override {
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
                            cs.ptr.get() + ((o * L + (L - 1 - k)) * inner) * elem, inner * elem);
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
            run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cs.ptr.get()));
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
            fill(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cg.ptr.get()));
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

    Storage where_branch(const Storage& grad,
                         const Storage& cond,
                         const Shape& shape,
                         Dtype dt,
                         bool true_branch) override {
        const auto& g = std::get<CpuStorage>(grad);
        const auto& c = std::get<CpuStorage>(cond);
        auto out = zeros(shape, dt);
        const std::size_t n = shape_numel(shape);
        const auto* cp = reinterpret_cast<const std::uint8_t*>(c.ptr.get());
        if (dt == Dtype::F32) {
            const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
            auto* dst = reinterpret_cast<float*>(std::get<CpuStorage>(out).ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const bool take = cp[i] != 0;
                dst[i] = (take == true_branch) ? gp[i] : 0.0f;
            }
        } else if (dt == Dtype::F64) {
            const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
            auto* dst = reinterpret_cast<double*>(std::get<CpuStorage>(out).ptr.get());
            for (std::size_t i = 0; i < n; ++i) {
                const bool take = cp[i] != 0;
                dst[i] = (take == true_branch) ? gp[i] : 0.0;
            }
        } else {
            ErrorBuilder("cpu_backend::where_branch").not_implemented("dtype not supported");
        }
        return out;
    }

    Storage masked_fill(const Storage& a,
                        const Storage& mask,
                        const Shape& shape,
                        Dtype dt,
                        double value) override {
        const auto& as = std::get<CpuStorage>(a);
        const auto& ms = std::get<CpuStorage>(mask);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const auto* mp = reinterpret_cast<const std::uint8_t*>(ms.ptr.get());
        if (dt == Dtype::F32) {
            const auto* src = reinterpret_cast<const float*>(as.ptr.get());
            auto* dst = reinterpret_cast<float*>(ptr.get());
            const float v = static_cast<float>(value);
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = mp[i] ? v : src[i];
        } else if (dt == Dtype::F64) {
            const auto* src = reinterpret_cast<const double*>(as.ptr.get());
            auto* dst = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = mp[i] ? value : src[i];
        } else {
            ErrorBuilder("cpu_backend::masked_fill").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage gather(const Storage& a,
                   const Storage& indices,
                   const Shape& input_shape,
                   const Shape& output_shape,
                   int axis,
                   Dtype index_dtype,
                   Dtype dt) override {
        const auto& as = std::get<CpuStorage>(a);
        const auto& is = std::get<CpuStorage>(indices);
        std::size_t nb = shape_numel(output_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t ndim = input_shape.size();
        const std::size_t elem = dtype_size(dt);

        Stride a_stride(ndim), out_stride(ndim);
        if (ndim > 0) {
            a_stride.back() = 1;
            out_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
                a_stride[static_cast<std::size_t>(d)] =
                    a_stride[static_cast<std::size_t>(d) + 1] *
                    input_shape[static_cast<std::size_t>(d) + 1];
                out_stride[static_cast<std::size_t>(d)] =
                    out_stride[static_cast<std::size_t>(d) + 1] *
                    output_shape[static_cast<std::size_t>(d) + 1];
            }
        }
        const std::size_t total = shape_numel(output_shape);
        auto load_idx = [&](std::size_t flat) -> std::int64_t {
            if (index_dtype == Dtype::I32)
                return reinterpret_cast<const std::int32_t*>(is.ptr.get())[flat];
            if (index_dtype == Dtype::I64)
                return reinterpret_cast<const std::int64_t*>(is.ptr.get())[flat];
            ErrorBuilder("cpu_backend::gather").not_implemented("indices dtype must be I32 or I64");
        };

        std::vector<std::int64_t> coord(ndim, 0);
        for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
            std::int64_t k = load_idx(out_flat);
            if (k < 0)
                k += input_shape[static_cast<std::size_t>(axis)];
            std::size_t a_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                std::int64_t c = (static_cast<int>(d) == axis) ? k : coord[d];
                a_flat += static_cast<std::size_t>(c) * static_cast<std::size_t>(a_stride[d]);
            }
            std::memcpy(ptr.get() + out_flat * elem, as.ptr.get() + a_flat * elem, elem);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] <
                    output_shape[static_cast<std::size_t>(d)]) {
                    break;
                }
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage gather_backward(const Storage& grad,
                            const Storage& indices,
                            const Shape& input_shape,
                            const Shape& output_shape,
                            int axis,
                            Dtype index_dtype,
                            Dtype dt) override {
        const auto& g = std::get<CpuStorage>(grad);
        const auto& idx = std::get<CpuStorage>(indices);
        auto out = zeros(input_shape, dt);
        const std::size_t ndim = input_shape.size();
        Stride input_stride(ndim), output_stride(ndim);
        if (ndim > 0) {
            input_stride.back() = 1;
            output_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
                input_stride[static_cast<std::size_t>(d)] =
                    input_stride[static_cast<std::size_t>(d) + 1] *
                    input_shape[static_cast<std::size_t>(d) + 1];
                output_stride[static_cast<std::size_t>(d)] =
                    output_stride[static_cast<std::size_t>(d) + 1] *
                    output_shape[static_cast<std::size_t>(d) + 1];
            }
        }
        auto load_idx = [&](std::size_t flat) -> std::int64_t {
            if (index_dtype == Dtype::I32)
                return reinterpret_cast<const std::int32_t*>(idx.ptr.get())[flat];
            if (index_dtype == Dtype::I64)
                return reinterpret_cast<const std::int64_t*>(idx.ptr.get())[flat];
            ErrorBuilder("cpu_backend::gather_backward")
                .not_implemented("indices dtype must be I32 or I64");
        };

        const std::size_t total = shape_numel(output_shape);
        std::vector<std::int64_t> coord(ndim, 0);
        if (dt == Dtype::F32) {
            const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
            auto* dst = reinterpret_cast<float*>(std::get<CpuStorage>(out).ptr.get());
            for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
                std::int64_t k = load_idx(out_flat);
                if (k < 0)
                    k += input_shape[static_cast<std::size_t>(axis)];
                std::size_t input_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    const std::int64_t c = (static_cast<int>(d) == axis) ? k : coord[d];
                    input_flat +=
                        static_cast<std::size_t>(c) * static_cast<std::size_t>(input_stride[d]);
                }
                dst[input_flat] += gp[out_flat];
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
                    if (++coord[static_cast<std::size_t>(d)] <
                        output_shape[static_cast<std::size_t>(d)]) {
                        break;
                    }
                    coord[static_cast<std::size_t>(d)] = 0;
                }
            }
        } else if (dt == Dtype::F64) {
            const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
            auto* dst = reinterpret_cast<double*>(std::get<CpuStorage>(out).ptr.get());
            for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
                std::int64_t k = load_idx(out_flat);
                if (k < 0)
                    k += input_shape[static_cast<std::size_t>(axis)];
                std::size_t input_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    const std::int64_t c = (static_cast<int>(d) == axis) ? k : coord[d];
                    input_flat +=
                        static_cast<std::size_t>(c) * static_cast<std::size_t>(input_stride[d]);
                }
                dst[input_flat] += gp[out_flat];
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
                    if (++coord[static_cast<std::size_t>(d)] <
                        output_shape[static_cast<std::size_t>(d)]) {
                        break;
                    }
                    coord[static_cast<std::size_t>(d)] = 0;
                }
            }
        } else {
            ErrorBuilder("cpu_backend::gather_backward").not_implemented("dtype not supported");
        }
        return out;
    }

    Storage diagonal(const Storage& a,
                     const Shape& input_shape,
                     int offset,
                     int axis1,
                     int axis2,
                     Dtype dt) override {
        const auto& as = std::get<CpuStorage>(a);
        const std::size_t ndim = input_shape.size();
        int a1 = axis1;
        int a2 = axis2;
        if (a1 > a2)
            std::swap(a1, a2);

        const std::int64_t M = input_shape[static_cast<std::size_t>(a1)];
        const std::int64_t N = input_shape[static_cast<std::size_t>(a2)];
        const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
        const std::int64_t c0 = (offset >= 0) ? offset : 0;
        const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

        Shape out_shape;
        for (std::size_t d = 0; d < ndim; ++d) {
            if (static_cast<int>(d) == a1 || static_cast<int>(d) == a2)
                continue;
            out_shape.push_back(input_shape[d]);
        }
        out_shape.push_back(L);
        std::size_t nb = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t elem = dtype_size(dt);

        Stride a_stride(ndim);
        if (ndim > 0) {
            a_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d)
                a_stride[static_cast<std::size_t>(d)] =
                    a_stride[static_cast<std::size_t>(d) + 1] *
                    input_shape[static_cast<std::size_t>(d) + 1];
        }

        std::vector<std::size_t> outer_dims;
        for (std::size_t d = 0; d < ndim; ++d)
            if (static_cast<int>(d) != a1 && static_cast<int>(d) != a2)
                outer_dims.push_back(d);

        std::size_t outer_numel = 1;
        for (auto d : outer_dims)
            outer_numel *= static_cast<std::size_t>(input_shape[d]);

        std::vector<std::int64_t> coord(ndim, 0);
        for (std::size_t o = 0; o < outer_numel; ++o) {
            std::size_t rem = o;
            for (auto d : outer_dims) {
                std::size_t prod = 1;
                for (std::size_t e : outer_dims)
                    if (e > d)
                        prod *= static_cast<std::size_t>(input_shape[e]);
                coord[d] = static_cast<std::int64_t>(rem / prod);
                rem %= prod;
            }
            for (std::int64_t i = 0; i < L; ++i) {
                coord[static_cast<std::size_t>(a1)] = r0 + i;
                coord[static_cast<std::size_t>(a2)] = c0 + i;
                std::size_t a_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d)
                    a_flat +=
                        static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(a_stride[d]);
                const std::size_t out_flat =
                    o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
                std::memcpy(ptr.get() + out_flat * elem, as.ptr.get() + a_flat * elem, elem);
            }
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage diagonal_backward(const Storage& grad,
                              const Shape& input_shape,
                              const Shape& output_shape,
                              int offset,
                              int axis1,
                              int axis2,
                              Dtype dt) override {
        const auto& g = std::get<CpuStorage>(grad);
        auto out = zeros(input_shape, dt);
        const std::size_t ndim = input_shape.size();
        int a1 = axis1;
        int a2 = axis2;
        if (a1 > a2)
            std::swap(a1, a2);

        const std::int64_t M = input_shape[static_cast<std::size_t>(a1)];
        const std::int64_t N = input_shape[static_cast<std::size_t>(a2)];
        const std::int64_t r0 = (offset >= 0) ? 0 : -offset;
        const std::int64_t c0 = (offset >= 0) ? offset : 0;
        const std::int64_t L = std::max<std::int64_t>(0, std::min(M - r0, N - c0));

        Stride input_stride(ndim);
        if (ndim > 0) {
            input_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
                input_stride[static_cast<std::size_t>(d)] =
                    input_stride[static_cast<std::size_t>(d) + 1] *
                    input_shape[static_cast<std::size_t>(d) + 1];
            }
        }

        std::vector<std::size_t> outer_dims;
        for (std::size_t d = 0; d < ndim; ++d) {
            if (static_cast<int>(d) != a1 && static_cast<int>(d) != a2)
                outer_dims.push_back(d);
        }
        std::size_t outer_numel = 1;
        for (auto d : outer_dims)
            outer_numel *= static_cast<std::size_t>(input_shape[d]);

        std::vector<std::int64_t> coord(ndim, 0);
        if (dt == Dtype::F32) {
            const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
            auto* dst = reinterpret_cast<float*>(std::get<CpuStorage>(out).ptr.get());
            for (std::size_t o = 0; o < outer_numel; ++o) {
                std::size_t rem = o;
                for (auto d : outer_dims) {
                    std::size_t prod = 1;
                    for (std::size_t e : outer_dims)
                        if (e > d)
                            prod *= static_cast<std::size_t>(input_shape[e]);
                    coord[d] = static_cast<std::int64_t>(rem / prod);
                    rem %= prod;
                }
                for (std::int64_t i = 0; i < L; ++i) {
                    coord[static_cast<std::size_t>(a1)] = r0 + i;
                    coord[static_cast<std::size_t>(a2)] = c0 + i;
                    std::size_t input_flat = 0;
                    for (std::size_t d = 0; d < ndim; ++d)
                        input_flat += static_cast<std::size_t>(coord[d]) *
                                      static_cast<std::size_t>(input_stride[d]);
                    const std::size_t grad_flat =
                        o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
                    dst[input_flat] += gp[grad_flat];
                }
            }
        } else if (dt == Dtype::F64) {
            const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
            auto* dst = reinterpret_cast<double*>(std::get<CpuStorage>(out).ptr.get());
            for (std::size_t o = 0; o < outer_numel; ++o) {
                std::size_t rem = o;
                for (auto d : outer_dims) {
                    std::size_t prod = 1;
                    for (std::size_t e : outer_dims)
                        if (e > d)
                            prod *= static_cast<std::size_t>(input_shape[e]);
                    coord[d] = static_cast<std::int64_t>(rem / prod);
                    rem %= prod;
                }
                for (std::int64_t i = 0; i < L; ++i) {
                    coord[static_cast<std::size_t>(a1)] = r0 + i;
                    coord[static_cast<std::size_t>(a2)] = c0 + i;
                    std::size_t input_flat = 0;
                    for (std::size_t d = 0; d < ndim; ++d)
                        input_flat += static_cast<std::size_t>(coord[d]) *
                                      static_cast<std::size_t>(input_stride[d]);
                    const std::size_t grad_flat =
                        o * static_cast<std::size_t>(L) + static_cast<std::size_t>(i);
                    dst[input_flat] += gp[grad_flat];
                }
            }
        } else {
            ErrorBuilder("cpu_backend::diagonal_backward").not_implemented("dtype not supported");
        }
        return out;
    }

    Storage roll(const Storage& a,
                 const Shape& shape,
                 Dtype dt,
                 const std::vector<std::int64_t>& shifts,
                 const std::vector<int>& axes) override {
        const auto& as = std::get<CpuStorage>(a);
        const std::size_t ndim = shape.size();
        std::vector<std::int64_t> shift_per_dim(ndim, 0);
        for (std::size_t i = 0; i < axes.size(); ++i) {
            int ax = axes[i];
            if (ax < 0)
                ax += static_cast<int>(ndim);
            shift_per_dim[static_cast<std::size_t>(ax)] += shifts[i];
        }
        std::size_t nb = shape_numel(shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t elem = dtype_size(dt);
        Stride stride(ndim);
        if (ndim > 0) {
            stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d)
                stride[static_cast<std::size_t>(d)] = stride[static_cast<std::size_t>(d) + 1] *
                                                      shape[static_cast<std::size_t>(d) + 1];
        }
        const std::size_t total = shape_numel(shape);
        std::vector<std::int64_t> coord(ndim, 0);
        for (std::size_t out_flat = 0; out_flat < total; ++out_flat) {
            std::size_t in_flat = 0;
            for (std::size_t d = 0; d < ndim; ++d) {
                std::int64_t c = coord[d] - shift_per_dim[d];
                std::int64_t L = shape[d];
                c = ((c % L) + L) % L;
                in_flat += static_cast<std::size_t>(c) * static_cast<std::size_t>(stride[d]);
            }
            std::memcpy(ptr.get() + out_flat * elem, as.ptr.get() + in_flat * elem, elem);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 1; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)])
                    break;
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage reshape(const Storage& a,
                    const Shape& /*src_shape*/,
                    const Shape& /*dst_shape*/,
                    Dtype dt) override {
        const auto& as = std::get<CpuStorage>(a);
        auto ptr = allocate_aligned_bytes(as.nbytes, Device::CPU);
        if (as.nbytes > 0)
            std::memcpy(ptr.get(), as.ptr.get(), as.nbytes);
        return Storage{CpuStorage{ptr, as.nbytes, dt}};
    }

    Storage slice_axis(const Storage& a,
                       const Shape& src_shape,
                       const Shape& slice_shape,
                       int axis,
                       std::int64_t offset,
                       Dtype dt) override {
        const auto& as = std::get<CpuStorage>(a);
        std::size_t out_nbytes = shape_numel(slice_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        const std::size_t elem = dtype_size(dt);
        std::size_t outer = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(src_shape[static_cast<std::size_t>(d)]);
        std::size_t inner_bytes = elem;
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < src_shape.size(); ++d)
            inner_bytes *= static_cast<std::size_t>(src_shape[d]);
        const std::size_t src_axis =
            static_cast<std::size_t>(src_shape[static_cast<std::size_t>(axis)]);
        const std::size_t slice_axis =
            static_cast<std::size_t>(slice_shape[static_cast<std::size_t>(axis)]);
        const std::size_t copy_bytes = slice_axis * inner_bytes;
        for (std::size_t o = 0; o < outer; ++o) {
            const auto* src_ptr =
                as.ptr.get() + (o * src_axis + static_cast<std::size_t>(offset)) * inner_bytes;
            auto* dst_ptr = ptr.get() + o * copy_bytes;
            if (copy_bytes > 0)
                std::memcpy(dst_ptr, src_ptr, copy_bytes);
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage insert_axis_slice(const Storage& a,
                              const Shape& src_shape,
                              const Shape& dst_shape,
                              int axis,
                              std::int64_t offset,
                              Dtype dt) override {
        const auto& as = std::get<CpuStorage>(a);
        std::size_t out_nbytes = shape_numel(dst_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        std::memset(ptr.get(), 0, out_nbytes);
        const std::size_t elem = dtype_size(dt);
        std::size_t outer = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(dst_shape[static_cast<std::size_t>(d)]);
        std::size_t inner_bytes = elem;
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < dst_shape.size(); ++d)
            inner_bytes *= static_cast<std::size_t>(dst_shape[d]);
        const std::size_t dst_axis =
            static_cast<std::size_t>(dst_shape[static_cast<std::size_t>(axis)]);
        const std::size_t src_axis =
            static_cast<std::size_t>(src_shape[static_cast<std::size_t>(axis)]);
        const std::size_t copy_bytes = src_axis * inner_bytes;
        for (std::size_t o = 0; o < outer; ++o) {
            const auto* src_ptr = as.ptr.get() + o * copy_bytes;
            auto* dst_ptr =
                ptr.get() + (o * dst_axis + static_cast<std::size_t>(offset)) * inner_bytes;
            if (copy_bytes > 0)
                std::memcpy(dst_ptr, src_ptr, copy_bytes);
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage concatenate(const std::vector<Storage>& xs,
                        const std::vector<Shape>& shapes,
                        int axis,
                        Dtype dt) override {
        Shape out_shape = shapes.front();
        std::int64_t cat_dim = 0;
        for (const auto& shape : shapes)
            cat_dim += shape[static_cast<std::size_t>(axis)];
        out_shape[static_cast<std::size_t>(axis)] = cat_dim;

        std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        const std::size_t elem = dtype_size(dt);
        std::size_t outer = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(out_shape[static_cast<std::size_t>(d)]);
        std::size_t inner_per_unit = elem;
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < out_shape.size(); ++d)
            inner_per_unit *= static_cast<std::size_t>(out_shape[d]);

        auto* dst = ptr.get();
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t i = 0; i < xs.size(); ++i) {
                const auto& cs = std::get<CpuStorage>(xs[i]);
                const std::size_t L =
                    static_cast<std::size_t>(shapes[i][static_cast<std::size_t>(axis)]);
                const std::size_t bytes = L * inner_per_unit;
                std::memcpy(dst, cs.ptr.get() + o * bytes, bytes);
                dst += bytes;
            }
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage stack(const std::vector<Storage>& xs,
                  const Shape& input_shape,
                  int axis,
                  Dtype dt) override {
        Shape out_shape = input_shape;
        out_shape.insert(out_shape.begin() + axis, static_cast<std::int64_t>(xs.size()));
        std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        const std::size_t elem = dtype_size(dt);
        std::size_t outer = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(out_shape[static_cast<std::size_t>(d)]);
        std::size_t inner_bytes = elem;
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < out_shape.size(); ++d)
            inner_bytes *= static_cast<std::size_t>(out_shape[d]);
        const std::size_t block_bytes = static_cast<std::size_t>(xs.size()) * inner_bytes;
        for (std::size_t idx = 0; idx < xs.size(); ++idx) {
            const auto& cs = std::get<CpuStorage>(xs[idx]);
            for (std::size_t o = 0; o < outer; ++o) {
                std::memcpy(ptr.get() + o * block_bytes + idx * inner_bytes,
                            cs.ptr.get() + o * inner_bytes, inner_bytes);
            }
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    std::vector<Storage> split_equal(const Storage& a,
                                     const Shape& shape,
                                     int axis,
                                     std::int64_t num_splits,
                                     Dtype dt) override {
        std::vector<Storage> out;
        out.reserve(static_cast<std::size_t>(num_splits));
        Shape piece_shape = shape;
        const std::int64_t piece = shape[static_cast<std::size_t>(axis)] / num_splits;
        piece_shape[static_cast<std::size_t>(axis)] = piece;
        for (std::int64_t k = 0; k < num_splits; ++k) {
            out.push_back(slice_axis(a, shape, piece_shape, axis, k * piece, dt));
        }
        return out;
    }

    std::vector<Storage> split_at(const Storage& a,
                                  const Shape& shape,
                                  int axis,
                                  const std::vector<std::int64_t>& indices,
                                  Dtype dt) override {
        std::vector<Storage> out;
        out.reserve(indices.size() + 1);
        std::int64_t off = 0;
        for (auto idx : indices) {
            Shape piece_shape = shape;
            piece_shape[static_cast<std::size_t>(axis)] = idx - off;
            out.push_back(slice_axis(a, shape, piece_shape, axis, off, dt));
            off = idx;
        }
        Shape tail_shape = shape;
        tail_shape[static_cast<std::size_t>(axis)] = shape[static_cast<std::size_t>(axis)] - off;
        out.push_back(slice_axis(a, shape, tail_shape, axis, off, dt));
        return out;
    }

    Storage repeat_backward(const Storage& grad_out,
                            const Shape& input_shape,
                            const Shape& output_shape,
                            int axis,
                            std::int64_t repeats,
                            Dtype dt) override {
        const auto& g = std::get<CpuStorage>(grad_out);
        std::size_t out_nbytes = shape_numel(input_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        std::memset(ptr.get(), 0, out_nbytes);

        const std::size_t ndim = output_shape.size();
        Stride out_stride(ndim), in_stride(ndim);
        if (ndim > 0) {
            out_stride.back() = 1;
            in_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
                out_stride[static_cast<std::size_t>(d)] =
                    out_stride[static_cast<std::size_t>(d) + 1] *
                    output_shape[static_cast<std::size_t>(d) + 1];
                in_stride[static_cast<std::size_t>(d)] =
                    in_stride[static_cast<std::size_t>(d) + 1] *
                    input_shape[static_cast<std::size_t>(d) + 1];
            }
        }

        const std::size_t total = shape_numel(output_shape);
        if (dt == Dtype::F32) {
            auto* out = reinterpret_cast<float*>(ptr.get());
            const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    if (static_cast<int>(d) == axis)
                        coord /= static_cast<std::size_t>(repeats);
                    in_flat += coord * static_cast<std::size_t>(in_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else if (dt == Dtype::F64) {
            auto* out = reinterpret_cast<double*>(ptr.get());
            const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    if (static_cast<int>(d) == axis)
                        coord /= static_cast<std::size_t>(repeats);
                    in_flat += coord * static_cast<std::size_t>(in_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else if (dt == Dtype::I32) {
            auto* out = reinterpret_cast<std::int32_t*>(ptr.get());
            const auto* gp = reinterpret_cast<const std::int32_t*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    if (static_cast<int>(d) == axis)
                        coord /= static_cast<std::size_t>(repeats);
                    in_flat += coord * static_cast<std::size_t>(in_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else if (dt == Dtype::I64) {
            auto* out = reinterpret_cast<std::int64_t*>(ptr.get());
            const auto* gp = reinterpret_cast<const std::int64_t*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    if (static_cast<int>(d) == axis)
                        coord /= static_cast<std::size_t>(repeats);
                    in_flat += coord * static_cast<std::size_t>(in_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else {
            ErrorBuilder("cpu_backend::repeat_backward").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage tile_backward(const Storage& grad_out,
                          const Shape& input_shape,
                          const Shape& padded_shape,
                          const Shape& output_shape,
                          const std::vector<std::int64_t>& /*reps*/,
                          Dtype dt) override {
        const auto& g = std::get<CpuStorage>(grad_out);
        std::size_t out_nbytes = shape_numel(input_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        std::memset(ptr.get(), 0, out_nbytes);

        const std::size_t ndim = output_shape.size();
        Stride out_stride(ndim), padded_stride(ndim);
        if (ndim > 0) {
            out_stride.back() = 1;
            padded_stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d) {
                out_stride[static_cast<std::size_t>(d)] =
                    out_stride[static_cast<std::size_t>(d) + 1] *
                    output_shape[static_cast<std::size_t>(d) + 1];
                padded_stride[static_cast<std::size_t>(d)] =
                    padded_stride[static_cast<std::size_t>(d) + 1] *
                    padded_shape[static_cast<std::size_t>(d) + 1];
            }
        }

        const std::size_t total = shape_numel(output_shape);
        if (dt == Dtype::F32) {
            auto* out = reinterpret_cast<float*>(ptr.get());
            const auto* gp = reinterpret_cast<const float*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                    in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else if (dt == Dtype::F64) {
            auto* out = reinterpret_cast<double*>(ptr.get());
            const auto* gp = reinterpret_cast<const double*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                    in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else if (dt == Dtype::I32) {
            auto* out = reinterpret_cast<std::int32_t*>(ptr.get());
            const auto* gp = reinterpret_cast<const std::int32_t*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                    in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else if (dt == Dtype::I64) {
            auto* out = reinterpret_cast<std::int64_t*>(ptr.get());
            const auto* gp = reinterpret_cast<const std::int64_t*>(g.ptr.get());
            for (std::size_t flat = 0; flat < total; ++flat) {
                std::size_t rem = flat;
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < ndim; ++d) {
                    const std::size_t coord = rem / static_cast<std::size_t>(out_stride[d]);
                    rem -= coord * static_cast<std::size_t>(out_stride[d]);
                    const std::size_t in_coord = coord % static_cast<std::size_t>(padded_shape[d]);
                    in_flat += in_coord * static_cast<std::size_t>(padded_stride[d]);
                }
                out[in_flat] += gp[flat];
            }
        } else {
            ErrorBuilder("cpu_backend::tile_backward").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    std::pair<Storage, Storage> sort_select(const Storage& a,
                                            const Shape& input_shape,
                                            const Shape& output_shape,
                                            int axis,
                                            Dtype dt,
                                            bool descending) override {
        const auto& ca = std::get<CpuStorage>(a);
        CpuStorage values;
        CpuStorage indices;
        if (dt == Dtype::F32) {
            std::tie(values, indices) =
                sort_select_cpu<float>(ca, input_shape, output_shape, axis, dt, descending);
        } else if (dt == Dtype::F64) {
            std::tie(values, indices) =
                sort_select_cpu<double>(ca, input_shape, output_shape, axis, dt, descending);
        } else if (dt == Dtype::I32) {
            std::tie(values, indices) =
                sort_select_cpu<std::int32_t>(ca, input_shape, output_shape, axis, dt, descending);
        } else if (dt == Dtype::I64) {
            std::tie(values, indices) =
                sort_select_cpu<std::int64_t>(ca, input_shape, output_shape, axis, dt, descending);
        } else {
            ErrorBuilder("cpu_backend::sort_select").not_implemented("dtype not supported");
        }
        return {Storage{std::move(values)}, Storage{std::move(indices)}};
    }

    Storage argsort(const Storage& a, const Shape& shape, int axis, Dtype dt) override {
        auto result = sort_select(a, shape, shape, axis, dt, /*descending=*/false);
        return std::move(result.second);
    }

    Storage arg_reduce_index(const Storage& a,
                             const Shape& shape,
                             int axis,
                             bool keepdims,
                             Dtype dt,
                             bool is_min) override {
        Shape out_shape = shape;
        if (keepdims)
            out_shape[static_cast<std::size_t>(axis)] = 1;
        else
            out_shape.erase(out_shape.begin() + axis);
        const std::size_t nbytes = shape_numel(out_shape) * dtype_size(Dtype::I64);
        auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
        auto* dst = reinterpret_cast<std::int64_t*>(ptr.get());
        const auto& ca = std::get<CpuStorage>(a);

        std::size_t outer = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        std::size_t inner = 1;
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
            inner *= static_cast<std::size_t>(shape[d]);
        const std::size_t L = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);

        auto run = [&](const auto* src) {
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t j = 0; j < inner; ++j) {
                    std::int64_t best = 0;
                    auto best_v = src[o * L * inner + j];
                    for (std::size_t k = 1; k < L; ++k) {
                        const auto v = src[(o * L + k) * inner + j];
                        if (is_min ? (v < best_v) : (v > best_v)) {
                            best_v = v;
                            best = static_cast<std::int64_t>(k);
                        }
                    }
                    dst[o * inner + j] = best;
                }
            }
        };

        if (dt == Dtype::F32) {
            run(reinterpret_cast<const float*>(ca.ptr.get()));
        } else if (dt == Dtype::F64) {
            run(reinterpret_cast<const double*>(ca.ptr.get()));
        } else if (dt == Dtype::I32) {
            run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()));
        } else if (dt == Dtype::I64) {
            run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()));
        } else {
            ErrorBuilder("cpu_backend::arg_reduce_index").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nbytes, Dtype::I64}};
    }

    Storage scatter_add_axis(const Storage& grad,
                             const Storage& indices,
                             const Shape& output_shape,
                             const Shape& grad_shape,
                             int axis,
                             Dtype dt) override {
        const auto& g = std::get<CpuStorage>(grad);
        const auto& idx_storage = std::get<CpuStorage>(indices);
        const std::size_t nbytes = shape_numel(output_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
        std::memset(ptr.get(), 0, nbytes);

        std::size_t outer = 1;
        std::size_t inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(output_shape[static_cast<std::size_t>(d)]);
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < output_shape.size(); ++d)
            inner *= static_cast<std::size_t>(output_shape[d]);
        const std::size_t L =
            static_cast<std::size_t>(output_shape[static_cast<std::size_t>(axis)]);
        const std::size_t K = static_cast<std::size_t>(grad_shape[static_cast<std::size_t>(axis)]);
        const auto* idx = reinterpret_cast<const std::int32_t*>(idx_storage.ptr.get());

        auto run = [&](auto* dst, const auto* gp) {
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t j = 0; j < inner; ++j) {
                    for (std::size_t k = 0; k < K; ++k) {
                        const std::size_t grad_flat = (o * K + k) * inner + j;
                        std::int32_t src_k = idx[grad_flat];
                        if (src_k < 0)
                            src_k += static_cast<std::int32_t>(L);
                        const std::size_t dst_flat =
                            (o * L + static_cast<std::size_t>(src_k)) * inner + j;
                        dst[dst_flat] += gp[grad_flat];
                    }
                }
            }
        };

        if (dt == Dtype::F32) {
            run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(g.ptr.get()));
        } else if (dt == Dtype::F64) {
            run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(g.ptr.get()));
        } else {
            ErrorBuilder("cpu_backend::scatter_add_axis").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nbytes, dt}};
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

    Storage linear(const Storage& x,
                   const Storage& weight,
                   const Storage& bias,
                   const Shape& x_shape,
                   const Shape& weight_shape,
                   const Shape& out_shape,
                   Dtype dt) override {
        const auto [M, K] = flatten_linear_x(x_shape);
        const std::size_t N = static_cast<std::size_t>(weight_shape[0]);
        CpuStorage out{allocate_aligned_bytes(M * N * dtype_size(dt), Device::CPU),
                       M * N * dtype_size(dt), dt};
        if (M > 0 && N > 0 && K > 0) {
            MatmulOpts opts;
            opts.transA = false;
            opts.transB = true;
            opts.M = static_cast<int>(M);
            opts.K = static_cast<int>(K);
            opts.N = static_cast<int>(N);
            opts.batch = 1;
            out = std::get<CpuStorage>(matmul(x, weight, opts, dt));
        } else if (out.nbytes) {
            std::memset(out.ptr.get(), 0, out.nbytes);
        }
        if (M > 0 && N > 0)
            add_linear_bias(out, std::get<CpuStorage>(bias), M, N, dt);
        (void)out_shape;
        return Storage{std::move(out)};
    }

    std::vector<Storage> linear_backward(const Storage& grad,
                                         const Storage& x,
                                         const Storage& weight,
                                         const Shape& x_shape,
                                         const Shape& weight_shape,
                                         const Shape& bias_shape,
                                         Dtype dt) override {
        const auto [M, K] = flatten_linear_x(x_shape);
        const std::size_t N = static_cast<std::size_t>(weight_shape[0]);
        const std::size_t elem = dtype_size(dt);
        CpuStorage dx{allocate_aligned_bytes(M * K * elem, Device::CPU), M * K * elem, dt};
        CpuStorage dW{allocate_aligned_bytes(N * K * elem, Device::CPU), N * K * elem, dt};
        CpuStorage db{allocate_aligned_bytes(shape_numel(bias_shape) * elem, Device::CPU),
                      shape_numel(bias_shape) * elem, dt};

        if (M > 0 && N > 0 && K > 0) {
            MatmulOpts dx_opts;
            dx_opts.M = static_cast<int>(M);
            dx_opts.K = static_cast<int>(N);
            dx_opts.N = static_cast<int>(K);
            dx_opts.batch = 1;
            dx = std::get<CpuStorage>(matmul(grad, weight, dx_opts, dt));

            MatmulOpts dW_opts;
            dW_opts.transA = true;
            dW_opts.M = static_cast<int>(N);
            dW_opts.K = static_cast<int>(M);
            dW_opts.N = static_cast<int>(K);
            dW_opts.batch = 1;
            dW = std::get<CpuStorage>(matmul(grad, x, dW_opts, dt));

            sum_linear_rows(std::get<CpuStorage>(grad), db, M, N, dt);
        } else {
            if (dx.nbytes)
                std::memset(dx.ptr.get(), 0, dx.nbytes);
            if (dW.nbytes)
                std::memset(dW.ptr.get(), 0, dW.nbytes);
            if (db.nbytes)
                std::memset(db.ptr.get(), 0, db.nbytes);
        }
        return {Storage{std::move(dx)}, Storage{std::move(dW)}, Storage{std::move(db)}};
    }

    Storage linalg_norm(const Storage& a,
                        const Shape& shape,
                        double ord,
                        const std::vector<int>& axes,
                        bool keepdims,
                        Dtype dt) override {
        Shape out_shape = reduced_norm_shape(shape, axes, keepdims);
        const std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        const auto& cs = std::get<CpuStorage>(a);
        if (dt == Dtype::F32) {
            norm_typed(reinterpret_cast<const float*>(cs.ptr.get()),
                       reinterpret_cast<float*>(ptr.get()), shape, axes, ord, out_shape);
        } else if (dt == Dtype::F64) {
            norm_typed(reinterpret_cast<const double*>(cs.ptr.get()),
                       reinterpret_cast<double*>(ptr.get()), shape, axes, ord, out_shape);
        } else {
            ErrorBuilder("cpu_backend::linalg_norm").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage linalg_cholesky(const Storage& a, const Shape& shape, bool upper, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
        if (cs.nbytes > 0)
            std::memcpy(ptr.get(), cs.ptr.get(), cs.nbytes);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(shape, /*mat_dims=*/2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;
        const bool lower = !upper;
        int info = 0;
        if (dt == Dtype::F32) {
            auto* p = reinterpret_cast<float*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_cholesky_f32(p + b * per_mat, n, lower, &info);
                check_lapack_info(info, "cholesky");
            }
        } else if (dt == Dtype::F64) {
            auto* p = reinterpret_cast<double*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_cholesky_f64(p + b * per_mat, n, lower, &info);
                check_lapack_info(info, "cholesky");
            }
        } else {
            ErrorBuilder("cpu_backend::linalg_cholesky").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, cs.nbytes, dt}};
    }

    Storage linalg_inv(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
        if (cs.nbytes > 0)
            std::memcpy(ptr.get(), cs.ptr.get(), cs.nbytes);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(shape, /*mat_dims=*/2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;
        int info = 0;
        if (dt == Dtype::F32) {
            auto* p = reinterpret_cast<float*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_inv_f32(p + b * per_mat, n, &info);
                check_lapack_info(info, "inv");
            }
        } else if (dt == Dtype::F64) {
            auto* p = reinterpret_cast<double*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_inv_f64(p + b * per_mat, n, &info);
                check_lapack_info(info, "inv");
            }
        } else {
            ErrorBuilder("cpu_backend::linalg_inv").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, cs.nbytes, dt}};
    }

    Storage linalg_solve(const Storage& a,
                         const Storage& b,
                         const Shape& a_shape,
                         const Shape& b_shape,
                         Dtype dt) override {
        const auto& a_cpu = std::get<CpuStorage>(a);
        const auto& b_cpu = std::get<CpuStorage>(b);
        auto out_ptr = allocate_aligned_bytes(b_cpu.nbytes, Device::CPU);
        if (b_cpu.nbytes > 0)
            std::memcpy(out_ptr.get(), b_cpu.ptr.get(), b_cpu.nbytes);

        const int n = static_cast<int>(a_shape[a_shape.size() - 1]);
        const bool b_is_vec = (b_shape.size() == a_shape.size() - 1);
        const int nrhs = b_is_vec ? 1 : static_cast<int>(b_shape[b_shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(a_shape, /*mat_dims=*/2);
        const std::size_t a_per = static_cast<std::size_t>(n) * n;
        const std::size_t b_per = static_cast<std::size_t>(n) * nrhs;
        int info = 0;
        if (dt == Dtype::F32) {
            std::vector<float> A_local(a_per);
            const auto* a_p = reinterpret_cast<const float*>(a_cpu.ptr.get());
            auto* x_p = reinterpret_cast<float*>(out_ptr.get());
            for (std::int64_t bi = 0; bi < batch; ++bi) {
                std::memcpy(A_local.data(), a_p + bi * a_per, a_per * sizeof(float));
                cpu::lapack_solve_f32(A_local.data(), x_p + bi * b_per, n, nrhs, &info);
                check_lapack_info(info, "solve");
            }
        } else if (dt == Dtype::F64) {
            std::vector<double> A_local(a_per);
            const auto* a_p = reinterpret_cast<const double*>(a_cpu.ptr.get());
            auto* x_p = reinterpret_cast<double*>(out_ptr.get());
            for (std::int64_t bi = 0; bi < batch; ++bi) {
                std::memcpy(A_local.data(), a_p + bi * a_per, a_per * sizeof(double));
                cpu::lapack_solve_f64(A_local.data(), x_p + bi * b_per, n, nrhs, &info);
                check_lapack_info(info, "solve");
            }
        } else {
            ErrorBuilder("cpu_backend::linalg_solve").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{out_ptr, b_cpu.nbytes, dt}};
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

    Storage repeat(
        const Storage& a, const Shape& shape, Dtype dt, std::int64_t repeats, int axis) override {
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
                    in_stride[static_cast<std::size_t>(d) + 1] *
                    padded[static_cast<std::size_t>(d) + 1];
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
                    in_stride[static_cast<std::size_t>(d) + 1] *
                    shape[static_cast<std::size_t>(d) + 1];
                out_stride[static_cast<std::size_t>(d)] =
                    out_stride[static_cast<std::size_t>(d) + 1] *
                    out_shape[static_cast<std::size_t>(d) + 1];
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
                in_off +=
                    static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
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

    Storage clip(
        const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) override {
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

    Storage mse_loss(const Storage& input,
                     const Storage& target,
                     const Shape& shape,
                     Dtype dt,
                     int reduction) override {
        const std::size_t n = shape_numel(shape);
        if (reduction == 0) {
            const std::size_t nb = n * dtype_size(dt);
            auto ptr = allocate_aligned_bytes(nb, Device::CPU);
            compute_mse_loss_values(input, target, ptr.get(), n, dt);
            return Storage{CpuStorage{ptr, nb, dt}};
        }

        auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
        if (dt == Dtype::F32) {
            float sum = mse_loss_sum<float>(input, target, n);
            if (reduction == 1)
                sum /= static_cast<float>(n);
            *reinterpret_cast<float*>(scalar.get()) = sum;
        } else if (dt == Dtype::F64) {
            double sum = mse_loss_sum<double>(input, target, n);
            if (reduction == 1)
                sum /= static_cast<double>(n);
            *reinterpret_cast<double*>(scalar.get()) = sum;
        } else {
            ErrorBuilder("cpu_backend::mse_loss").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
    }

    std::pair<Storage, Storage> mse_loss_backward(const Storage& input,
                                                  const Storage& target,
                                                  const Storage& grad,
                                                  const Shape& shape,
                                                  Dtype dt,
                                                  int reduction) override {
        const std::size_t n = shape_numel(shape);
        const std::size_t nb = n * dtype_size(dt);
        auto dx = allocate_aligned_bytes(nb, Device::CPU);
        auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& gs = std::get<CpuStorage>(grad);

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
            auto* dxp = reinterpret_cast<T*>(dx.get());
            auto* dtp = reinterpret_cast<T*>(dtarget.get());
            const bool elem = reduction == 0;
            const T mean_scale = static_cast<T>(n);
            for (std::size_t i = 0; i < n; ++i) {
                const T go = elem ? gp[i] : gp[0];
                const T scale = (reduction == 1) ? (go / mean_scale) : go;
                const T d = xp[i] - tp[i];
                dxp[i] = static_cast<T>(2) * d * scale;
                dtp[i] = -dxp[i];
            }
        };

        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::mse_loss_backward").not_implemented("dtype not supported");

        return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}}};
    }

    Storage huber_loss(const Storage& input,
                       const Storage& target,
                       const Shape& shape,
                       Dtype dt,
                       double delta,
                       int reduction) override {
        const std::size_t n = shape_numel(shape);
        if (reduction == 0) {
            const std::size_t nb = n * dtype_size(dt);
            auto ptr = allocate_aligned_bytes(nb, Device::CPU);
            compute_huber_loss_values(input, target, ptr.get(), n, dt, delta);
            return Storage{CpuStorage{ptr, nb, dt}};
        }

        auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
        if (dt == Dtype::F32) {
            float sum = huber_loss_sum<float>(input, target, n, static_cast<float>(delta));
            if (reduction == 1)
                sum /= static_cast<float>(n);
            *reinterpret_cast<float*>(scalar.get()) = sum;
        } else if (dt == Dtype::F64) {
            double sum = huber_loss_sum<double>(input, target, n, delta);
            if (reduction == 1)
                sum /= static_cast<double>(n);
            *reinterpret_cast<double*>(scalar.get()) = sum;
        } else {
            ErrorBuilder("cpu_backend::huber_loss").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
    }

    std::pair<Storage, Storage> huber_loss_backward(const Storage& input,
                                                    const Storage& target,
                                                    const Storage& grad,
                                                    const Shape& shape,
                                                    Dtype dt,
                                                    double delta,
                                                    int reduction) override {
        const std::size_t n = shape_numel(shape);
        const std::size_t nb = n * dtype_size(dt);
        auto dx = allocate_aligned_bytes(nb, Device::CPU);
        auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& gs = std::get<CpuStorage>(grad);

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
            auto* dxp = reinterpret_cast<T*>(dx.get());
            auto* dtp = reinterpret_cast<T*>(dtarget.get());
            const T d = static_cast<T>(delta);
            const bool elem = reduction == 0;
            const T mean_scale = static_cast<T>(n);
            for (std::size_t i = 0; i < n; ++i) {
                const T go = elem ? gp[i] : gp[0];
                const T scale = (reduction == 1) ? (go / mean_scale) : go;
                const T r = xp[i] - tp[i];
                T dr;
                if (std::abs(r) <= d)
                    dr = r;
                else
                    dr = (r > T{0}) ? d : -d;
                dxp[i] = dr * scale;
                dtp[i] = -dxp[i];
            }
        };

        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::huber_loss_backward").not_implemented("dtype not supported");

        return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}}};
    }

    Storage bce_loss(const Storage& input,
                     const Storage& target,
                     const Storage& weight,
                     const Shape& shape,
                     Dtype dt,
                     double eps,
                     int reduction) override {
        const std::size_t n = shape_numel(shape);
        if (reduction == 0) {
            const std::size_t nb = n * dtype_size(dt);
            auto ptr = allocate_aligned_bytes(nb, Device::CPU);
            compute_bce_loss_values(input, target, weight, ptr.get(), n, dt, eps);
            return Storage{CpuStorage{ptr, nb, dt}};
        }

        auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
        if (dt == Dtype::F32) {
            float sum = bce_loss_sum<float>(input, target, weight, n, static_cast<float>(eps));
            if (reduction == 1)
                sum /= static_cast<float>(n);
            *reinterpret_cast<float*>(scalar.get()) = sum;
        } else if (dt == Dtype::F64) {
            double sum = bce_loss_sum<double>(input, target, weight, n, eps);
            if (reduction == 1)
                sum /= static_cast<double>(n);
            *reinterpret_cast<double*>(scalar.get()) = sum;
        } else {
            ErrorBuilder("cpu_backend::bce_loss").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
    }

    std::vector<Storage> bce_loss_backward(const Storage& input,
                                           const Storage& target,
                                           const Storage& weight,
                                           const Storage& grad,
                                           const Shape& shape,
                                           Dtype dt,
                                           double eps,
                                           int reduction) override {
        const std::size_t n = shape_numel(shape);
        const std::size_t nb = n * dtype_size(dt);
        auto dx = allocate_aligned_bytes(nb, Device::CPU);
        auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
        auto dweight = allocate_aligned_bytes(nb, Device::CPU);
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& ws = std::get<CpuStorage>(weight);
        const auto& gs = std::get<CpuStorage>(grad);

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
            const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
            auto* dxp = reinterpret_cast<T*>(dx.get());
            auto* dtp = reinterpret_cast<T*>(dtarget.get());
            auto* dwp = reinterpret_cast<T*>(dweight.get());
            const T e = static_cast<T>(eps);
            const bool elem = reduction == 0;
            const T mean_scale = static_cast<T>(n);
            for (std::size_t i = 0; i < n; ++i) {
                const T go = elem ? gp[i] : gp[0];
                const T scale = (reduction == 1) ? (go / mean_scale) : go;
                const T p = std::min(std::max(xp[i], e), static_cast<T>(1) - e);
                const T y = tp[i];
                const T w = wp[i];
                const T one = static_cast<T>(1);
                const T dlp = -y / p + (one - y) / (one - p);
                dxp[i] = w * dlp * scale;
                const T dly = -std::log(p) + std::log(one - p);
                dtp[i] = w * dly * scale;
                const T l = -(y * std::log(p) + (one - y) * std::log(one - p));
                dwp[i] = l * scale;
            }
        };

        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::bce_loss_backward").not_implemented("dtype not supported");

        return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}},
                Storage{CpuStorage{dweight, nb, dt}}};
    }

    Storage bce_with_logits_loss(const Storage& input,
                                 const Storage& target,
                                 const Storage& weight,
                                 const Storage& pos_weight,
                                 const Shape& shape,
                                 const Shape& weight_shape,
                                 const Shape& pos_weight_shape,
                                 Dtype dt,
                                 int reduction) override {
        const std::size_t n = shape_numel(shape);
        Storage weight_storage =
            weight_shape == shape ? weight : broadcast(weight, weight_shape, shape, dt);
        Storage pos_weight_storage = pos_weight_shape == shape
                                         ? pos_weight
                                         : broadcast(pos_weight, pos_weight_shape, shape, dt);
        if (reduction == 0) {
            const std::size_t nb = n * dtype_size(dt);
            auto ptr = allocate_aligned_bytes(nb, Device::CPU);
            compute_bce_logits_values(input, target, weight_storage, pos_weight_storage, ptr.get(),
                                      n, dt);
            return Storage{CpuStorage{ptr, nb, dt}};
        }

        auto scalar = allocate_aligned_bytes(dtype_size(dt), Device::CPU);
        if (dt == Dtype::F32) {
            float sum = bce_logits_sum<float>(input, target, weight_storage, pos_weight_storage, n);
            if (reduction == 1)
                sum /= static_cast<float>(n);
            *reinterpret_cast<float*>(scalar.get()) = sum;
        } else if (dt == Dtype::F64) {
            double sum =
                bce_logits_sum<double>(input, target, weight_storage, pos_weight_storage, n);
            if (reduction == 1)
                sum /= static_cast<double>(n);
            *reinterpret_cast<double*>(scalar.get()) = sum;
        } else {
            ErrorBuilder("cpu_backend::bce_with_logits_loss")
                .not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{scalar, dtype_size(dt), dt}};
    }

    std::vector<Storage> bce_with_logits_backward(const Storage& input,
                                                  const Storage& target,
                                                  const Storage& weight,
                                                  const Storage& pos_weight,
                                                  const Storage& grad,
                                                  const Shape& shape,
                                                  Dtype dt,
                                                  int reduction) override {
        const std::size_t n = shape_numel(shape);
        const std::size_t nb = n * dtype_size(dt);
        auto dx = allocate_aligned_bytes(nb, Device::CPU);
        auto dtarget = allocate_aligned_bytes(nb, Device::CPU);
        auto dweight = allocate_aligned_bytes(nb, Device::CPU);
        auto dpos_weight = allocate_aligned_bytes(nb, Device::CPU);
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& ws = std::get<CpuStorage>(weight);
        const auto& pws = std::get<CpuStorage>(pos_weight);
        const auto& gs = std::get<CpuStorage>(grad);

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
            const auto* pwp = reinterpret_cast<const T*>(pws.ptr.get());
            const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
            auto* dxp = reinterpret_cast<T*>(dx.get());
            auto* dtp = reinterpret_cast<T*>(dtarget.get());
            auto* dwp = reinterpret_cast<T*>(dweight.get());
            auto* dpwp = reinterpret_cast<T*>(dpos_weight.get());
            const bool elem = reduction == 0;
            const T mean_scale = static_cast<T>(n);
            const T one = static_cast<T>(1);
            for (std::size_t i = 0; i < n; ++i) {
                const T go = elem ? gp[i] : gp[0];
                const T scale = (reduction == 1) ? (go / mean_scale) : go;
                const T x = xp[i];
                const T y = tp[i];
                const T pw = pwp[i];
                const T sigm = one / (one + std::exp(-x));
                const T log_weight = (pw - one) * y + one;
                const T log1pexp = std::log1p(std::exp(-std::abs(x)));
                const T loss = std::max(x, T{0}) - x * y + log_weight * log1pexp;
                const T w = wp[i];
                dxp[i] = w * (log_weight * sigm - y) * scale;
                dtp[i] = w * (-x + (pw - one) * log1pexp) * scale;
                dwp[i] = loss * scale;
                dpwp[i] = w * y * log1pexp * scale;
            }
        };

        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::bce_with_logits_backward")
                .not_implemented("dtype not supported");

        return {Storage{CpuStorage{dx, nb, dt}}, Storage{CpuStorage{dtarget, nb, dt}},
                Storage{CpuStorage{dweight, nb, dt}}, Storage{CpuStorage{dpos_weight, nb, dt}}};
    }

    ClassLossForwardResult cross_entropy_loss(const Storage& input,
                                              const Storage& target,
                                              const Storage* weight,
                                              const Shape& input_shape,
                                              const Shape& /*target_shape*/,
                                              Dtype dt,
                                              double eps,
                                              int ignore_index,
                                              int reduction) override {
        const int n_batch = static_cast<int>(input_shape[0]);
        const int channels = static_cast<int>(input_shape[1]);
        const int spatial = class_loss_spatial(input_shape);
        const std::size_t samples = static_cast<std::size_t>(n_batch) * spatial;
        const std::size_t input_numel = samples * static_cast<std::size_t>(channels);
        CpuStorage softmax{allocate_aligned_bytes(input_numel * dtype_size(dt), Device::CPU),
                           input_numel * dtype_size(dt), dt};
        CpuStorage losses{allocate_aligned_bytes(samples * dtype_size(dt), Device::CPU),
                          samples * dtype_size(dt), dt};
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;
        std::size_t valid_count = 0;

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            auto* sp = reinterpret_cast<T*>(softmax.ptr.get());
            auto* lp = reinterpret_cast<T*>(losses.ptr.get());
            const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
            for (int n = 0; n < n_batch; ++n) {
                for (int s = 0; s < spatial; ++s) {
                    T mx = -std::numeric_limits<T>::infinity();
                    for (int c = 0; c < channels; ++c)
                        mx = std::max(mx, xp[(n * channels + c) * spatial + s]);
                    T sum = T{0};
                    for (int c = 0; c < channels; ++c) {
                        const T e = std::exp(xp[(n * channels + c) * spatial + s] - mx);
                        sp[(n * channels + c) * spatial + s] = e;
                        sum += e;
                    }
                    const T inv = T{1} / sum;
                    for (int c = 0; c < channels; ++c)
                        sp[(n * channels + c) * spatial + s] *= inv;
                    const std::int64_t y = read_target_index(ts, n * spatial + s);
                    if (static_cast<int>(y) == ignore_index) {
                        lp[n * spatial + s] = T{0};
                        continue;
                    }
                    ++valid_count;
                    const T pred = sp[(n * channels + static_cast<int>(y)) * spatial + s];
                    const T w = wp ? wp[static_cast<int>(y)] : T{1};
                    lp[n * spatial + s] = -w * std::log(pred + static_cast<T>(eps));
                }
            }
        };
        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::cross_entropy_loss").not_implemented("dtype not supported");
        if (valid_count == 0)
            valid_count = 1;

        return {reduce_class_losses(losses, samples, valid_count, dt, reduction),
                Storage{std::move(softmax)},
                Storage{make_cpu_scalar(static_cast<double>(valid_count), dt)}};
    }

    Storage cross_entropy_backward(const Storage& saved_softmax,
                                   const Storage& target,
                                   const Storage* weight,
                                   const Storage& valid_count,
                                   const Storage& grad,
                                   const Shape& input_shape,
                                   Dtype dt,
                                   int ignore_index,
                                   int reduction) override {
        const int n_batch = static_cast<int>(input_shape[0]);
        const int channels = static_cast<int>(input_shape[1]);
        const int spatial = class_loss_spatial(input_shape);
        const std::size_t numel = static_cast<std::size_t>(n_batch) * channels * spatial;
        CpuStorage dx{allocate_aligned_bytes(numel * dtype_size(dt), Device::CPU),
                      numel * dtype_size(dt), dt};
        if (dx.nbytes)
            std::memset(dx.ptr.get(), 0, dx.nbytes);
        const auto& sm = std::get<CpuStorage>(saved_softmax);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& gs = std::get<CpuStorage>(grad);
        const auto& vc = std::get<CpuStorage>(valid_count);
        const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* sp = reinterpret_cast<const T*>(sm.ptr.get());
            const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
            auto* dxp = reinterpret_cast<T*>(dx.ptr.get());
            const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
            const T valid = *reinterpret_cast<const T*>(vc.ptr.get());
            const bool elem = reduction == 0;
            for (int n = 0; n < n_batch; ++n) {
                for (int s = 0; s < spatial; ++s) {
                    const std::int64_t y = read_target_index(ts, n * spatial + s);
                    if (static_cast<int>(y) == ignore_index)
                        continue;
                    const T go = elem ? gp[n * spatial + s] : gp[0];
                    const T scale = (reduction == 1) ? (go / valid) : go;
                    const T w = wp ? wp[static_cast<int>(y)] : T{1};
                    for (int c = 0; c < channels; ++c) {
                        T v = sp[(n * channels + c) * spatial + s];
                        if (c == static_cast<int>(y))
                            v -= T{1};
                        dxp[(n * channels + c) * spatial + s] = w * v * scale;
                    }
                }
            }
        };
        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::cross_entropy_backward")
                .not_implemented("dtype not supported");
        return Storage{std::move(dx)};
    }

    ClassLossForwardResult nll_loss(const Storage& input,
                                    const Storage& target,
                                    const Storage* weight,
                                    const Shape& input_shape,
                                    const Shape& /*target_shape*/,
                                    Dtype dt,
                                    int ignore_index,
                                    int reduction) override {
        const int n_batch = static_cast<int>(input_shape[0]);
        const int channels = static_cast<int>(input_shape[1]);
        const int spatial = class_loss_spatial(input_shape);
        const std::size_t samples = static_cast<std::size_t>(n_batch) * spatial;
        CpuStorage losses{allocate_aligned_bytes(samples * dtype_size(dt), Device::CPU),
                          samples * dtype_size(dt), dt};
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;
        std::size_t valid_count = 0;

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            auto* lp = reinterpret_cast<T*>(losses.ptr.get());
            const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
            for (int n = 0; n < n_batch; ++n) {
                for (int s = 0; s < spatial; ++s) {
                    const std::int64_t y = read_target_index(ts, n * spatial + s);
                    if (static_cast<int>(y) == ignore_index) {
                        lp[n * spatial + s] = T{0};
                        continue;
                    }
                    ++valid_count;
                    const T w = wp ? wp[static_cast<int>(y)] : T{1};
                    lp[n * spatial + s] =
                        -w * xp[(n * channels + static_cast<int>(y)) * spatial + s];
                }
            }
        };
        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::nll_loss").not_implemented("dtype not supported");
        if (valid_count == 0)
            valid_count = 1;
        return {reduce_class_losses(losses, samples, valid_count, dt, reduction), Storage{},
                Storage{make_cpu_scalar(static_cast<double>(valid_count), dt)}};
    }

    Storage nll_loss_backward(const Storage& target,
                              const Storage* weight,
                              const Storage& valid_count,
                              const Storage& grad,
                              const Shape& input_shape,
                              Dtype dt,
                              int ignore_index,
                              int reduction) override {
        const int n_batch = static_cast<int>(input_shape[0]);
        const int channels = static_cast<int>(input_shape[1]);
        const int spatial = class_loss_spatial(input_shape);
        const std::size_t numel = static_cast<std::size_t>(n_batch) * channels * spatial;
        CpuStorage dx{allocate_aligned_bytes(numel * dtype_size(dt), Device::CPU),
                      numel * dtype_size(dt), dt};
        if (dx.nbytes)
            std::memset(dx.ptr.get(), 0, dx.nbytes);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& gs = std::get<CpuStorage>(grad);
        const auto& vc = std::get<CpuStorage>(valid_count);
        const CpuStorage* ws = weight ? &std::get<CpuStorage>(*weight) : nullptr;

        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* gp = reinterpret_cast<const T*>(gs.ptr.get());
            auto* dxp = reinterpret_cast<T*>(dx.ptr.get());
            const T* wp = ws ? reinterpret_cast<const T*>(ws->ptr.get()) : nullptr;
            const T valid = *reinterpret_cast<const T*>(vc.ptr.get());
            const bool elem = reduction == 0;
            for (int n = 0; n < n_batch; ++n) {
                for (int s = 0; s < spatial; ++s) {
                    const std::int64_t y = read_target_index(ts, n * spatial + s);
                    if (static_cast<int>(y) == ignore_index)
                        continue;
                    const T go = elem ? gp[n * spatial + s] : gp[0];
                    const T scale = (reduction == 1) ? (go / valid) : go;
                    const T w = wp ? wp[static_cast<int>(y)] : T{1};
                    dxp[(n * channels + static_cast<int>(y)) * spatial + s] = -w * scale;
                }
            }
        };
        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::nll_loss_backward").not_implemented("dtype not supported");
        return Storage{std::move(dx)};
    }

    CpuStorage to_cpu(const Storage& a, const Shape& /*shape*/) override {
        return std::get<CpuStorage>(a);
    }

private:
    // ---- Helpers -------------------------------------------------------

    template <typename T>
    static std::pair<CpuStorage, CpuStorage> sort_select_cpu(const CpuStorage& input,
                                                             const Shape& input_shape,
                                                             const Shape& output_shape,
                                                             int axis,
                                                             Dtype dt,
                                                             bool descending) {
        const std::size_t value_nbytes = shape_numel(output_shape) * dtype_size(dt);
        CpuStorage values{allocate_aligned_bytes(value_nbytes, Device::CPU), value_nbytes, dt};
        const std::size_t index_nbytes = shape_numel(output_shape) * dtype_size(Dtype::I32);
        CpuStorage indices{allocate_aligned_bytes(index_nbytes, Device::CPU), index_nbytes,
                           Dtype::I32};

        const auto* src = reinterpret_cast<const T*>(input.ptr.get());
        auto* dst = reinterpret_cast<T*>(values.ptr.get());
        auto* idx_dst = reinterpret_cast<std::int32_t*>(indices.ptr.get());

        std::size_t outer = 1;
        std::size_t inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(input_shape[static_cast<std::size_t>(d)]);
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < input_shape.size(); ++d)
            inner *= static_cast<std::size_t>(input_shape[d]);
        const std::size_t L = static_cast<std::size_t>(input_shape[static_cast<std::size_t>(axis)]);
        const std::size_t K =
            static_cast<std::size_t>(output_shape[static_cast<std::size_t>(axis)]);

        std::vector<std::int32_t> order(L);
        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t j = 0; j < inner; ++j) {
                std::iota(order.begin(), order.end(), 0);
                auto cmp = [&](std::int32_t lhs, std::int32_t rhs) {
                    const T lv = src[(o * L + static_cast<std::size_t>(lhs)) * inner + j];
                    const T rv = src[(o * L + static_cast<std::size_t>(rhs)) * inner + j];
                    return descending ? (lv > rv) : (lv < rv);
                };
                if (K == L) {
                    std::sort(order.begin(), order.end(), cmp);
                } else {
                    std::partial_sort(order.begin(), order.begin() + K, order.end(), cmp);
                }
                for (std::size_t k = 0; k < K; ++k) {
                    const std::int32_t src_k = order[k];
                    const std::size_t out_flat = (o * K + k) * inner + j;
                    const std::size_t src_flat =
                        (o * L + static_cast<std::size_t>(src_k)) * inner + j;
                    dst[out_flat] = src[src_flat];
                    idx_dst[out_flat] = src_k;
                }
            }
        }
        return {std::move(values), std::move(indices)};
    }

    void compute_mse_loss_values(
        const Storage& input, const Storage& target, std::byte* out, std::size_t n, Dtype dt) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        if (dt == Dtype::F32) {
            const auto* xp = reinterpret_cast<const float*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const float*>(ts.ptr.get());
            auto* lp = reinterpret_cast<float*>(out);
            for (std::size_t i = 0; i < n; ++i) {
                const float d = xp[i] - tp[i];
                lp[i] = d * d;
            }
        } else if (dt == Dtype::F64) {
            const auto* xp = reinterpret_cast<const double*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const double*>(ts.ptr.get());
            auto* lp = reinterpret_cast<double*>(out);
            for (std::size_t i = 0; i < n; ++i) {
                const double d = xp[i] - tp[i];
                lp[i] = d * d;
            }
        } else {
            ErrorBuilder("cpu_backend::mse_loss").not_implemented("dtype not supported");
        }
    }

    template <typename T>
    T mse_loss_sum(const Storage& input, const Storage& target, std::size_t n) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i) {
            const T d = xp[i] - tp[i];
            sum += d * d;
        }
        return sum;
    }

    void compute_huber_loss_values(const Storage& input,
                                   const Storage& target,
                                   std::byte* out,
                                   std::size_t n,
                                   Dtype dt,
                                   double delta) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            auto* lp = reinterpret_cast<T*>(out);
            const T d = static_cast<T>(delta);
            for (std::size_t i = 0; i < n; ++i) {
                const T r = xp[i] - tp[i];
                const T ar = std::abs(r);
                lp[i] = (ar <= d) ? T{0.5} * r * r : d * (ar - T{0.5} * d);
            }
        };
        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::huber_loss").not_implemented("dtype not supported");
    }

    template <typename T>
    T huber_loss_sum(const Storage& input, const Storage& target, std::size_t n, T delta) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i) {
            const T r = xp[i] - tp[i];
            const T ar = std::abs(r);
            sum += (ar <= delta) ? T{0.5} * r * r : delta * (ar - T{0.5} * delta);
        }
        return sum;
    }

    void compute_bce_loss_values(const Storage& input,
                                 const Storage& target,
                                 const Storage& weight,
                                 std::byte* out,
                                 std::size_t n,
                                 Dtype dt,
                                 double eps) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& ws = std::get<CpuStorage>(weight);
        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
            auto* lp = reinterpret_cast<T*>(out);
            const T e = static_cast<T>(eps);
            const T one = static_cast<T>(1);
            for (std::size_t i = 0; i < n; ++i) {
                const T p = std::min(std::max(xp[i], e), one - e);
                const T l = -(tp[i] * std::log(p) + (one - tp[i]) * std::log(one - p));
                lp[i] = wp[i] * l;
            }
        };
        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::bce_loss").not_implemented("dtype not supported");
    }

    template <typename T>
    T bce_loss_sum(
        const Storage& input, const Storage& target, const Storage& weight, std::size_t n, T eps) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& ws = std::get<CpuStorage>(weight);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        const T one = static_cast<T>(1);
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i) {
            const T p = std::min(std::max(xp[i], eps), one - eps);
            const T l = -(tp[i] * std::log(p) + (one - tp[i]) * std::log(one - p));
            sum += wp[i] * l;
        }
        return sum;
    }

    void compute_bce_logits_values(const Storage& input,
                                   const Storage& target,
                                   const Storage& weight,
                                   const Storage& pos_weight,
                                   std::byte* out,
                                   std::size_t n,
                                   Dtype dt) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& ws = std::get<CpuStorage>(weight);
        const auto& pws = std::get<CpuStorage>(pos_weight);
        auto compute = [&](auto type_tag) {
            using T = decltype(type_tag);
            const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
            const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
            const auto* pwp = reinterpret_cast<const T*>(pws.ptr.get());
            auto* lp = reinterpret_cast<T*>(out);
            const T one = static_cast<T>(1);
            for (std::size_t i = 0; i < n; ++i) {
                const T x = xp[i];
                const T y = tp[i];
                const T log_weight = (pwp[i] - one) * y + one;
                const T log1pexp = std::log1p(std::exp(-std::abs(x)));
                const T loss = std::max(x, T{0}) - x * y + log_weight * log1pexp;
                lp[i] = wp[i] * loss;
            }
        };
        if (dt == Dtype::F32)
            compute(float{});
        else if (dt == Dtype::F64)
            compute(double{});
        else
            ErrorBuilder("cpu_backend::bce_with_logits_loss")
                .not_implemented("dtype not supported");
    }

    template <typename T>
    T bce_logits_sum(const Storage& input,
                     const Storage& target,
                     const Storage& weight,
                     const Storage& pos_weight,
                     std::size_t n) {
        const auto& xs = std::get<CpuStorage>(input);
        const auto& ts = std::get<CpuStorage>(target);
        const auto& ws = std::get<CpuStorage>(weight);
        const auto& pws = std::get<CpuStorage>(pos_weight);
        const auto* xp = reinterpret_cast<const T*>(xs.ptr.get());
        const auto* tp = reinterpret_cast<const T*>(ts.ptr.get());
        const auto* wp = reinterpret_cast<const T*>(ws.ptr.get());
        const auto* pwp = reinterpret_cast<const T*>(pws.ptr.get());
        const T one = static_cast<T>(1);
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i) {
            const T x = xp[i];
            const T y = tp[i];
            const T log_weight = (pwp[i] - one) * y + one;
            const T log1pexp = std::log1p(std::exp(-std::abs(x)));
            const T loss = std::max(x, T{0}) - x * y + log_weight * log1pexp;
            sum += wp[i] * loss;
        }
        return sum;
    }

    static int class_loss_spatial(const Shape& input_shape) {
        int spatial = 1;
        for (std::size_t i = 2; i < input_shape.size(); ++i)
            spatial *= static_cast<int>(input_shape[i]);
        return spatial;
    }

    static CpuStorage make_cpu_scalar(double value, Dtype dt) {
        CpuStorage out{allocate_aligned_bytes(dtype_size(dt), Device::CPU), dtype_size(dt), dt};
        if (dt == Dtype::F32)
            *reinterpret_cast<float*>(out.ptr.get()) = static_cast<float>(value);
        else if (dt == Dtype::F64)
            *reinterpret_cast<double*>(out.ptr.get()) = value;
        else
            ErrorBuilder("cpu_backend::class_loss_scalar").not_implemented("dtype not supported");
        return out;
    }

    static std::int64_t read_target_index(const CpuStorage& target, std::size_t i) {
        switch (target.dtype) {
            case Dtype::I8:
                return static_cast<std::int64_t>(
                    reinterpret_cast<const std::int8_t*>(target.ptr.get())[i]);
            case Dtype::I16:
                return static_cast<std::int64_t>(
                    reinterpret_cast<const std::int16_t*>(target.ptr.get())[i]);
            case Dtype::I32:
                return static_cast<std::int64_t>(
                    reinterpret_cast<const std::int32_t*>(target.ptr.get())[i]);
            case Dtype::I64:
                return reinterpret_cast<const std::int64_t*>(target.ptr.get())[i];
            case Dtype::F32:
                return static_cast<std::int64_t>(
                    reinterpret_cast<const float*>(target.ptr.get())[i]);
            case Dtype::F64:
                return static_cast<std::int64_t>(
                    reinterpret_cast<const double*>(target.ptr.get())[i]);
            default:
                ErrorBuilder("cpu_backend::class_loss_target")
                    .not_implemented("dtype not supported");
        }
    }

    template <typename T>
    static T sum_values(const T* values, std::size_t n) {
        T sum = T{0};
        for (std::size_t i = 0; i < n; ++i)
            sum += values[i];
        return sum;
    }

    static Storage reduce_class_losses(const CpuStorage& losses,
                                       std::size_t samples,
                                       std::size_t valid_count,
                                       Dtype dt,
                                       int reduction) {
        if (reduction == 0) {
            CpuStorage out{allocate_aligned_bytes(samples * dtype_size(dt), Device::CPU),
                           samples * dtype_size(dt), dt};
            std::memcpy(out.ptr.get(), losses.ptr.get(), out.nbytes);
            return Storage{std::move(out)};
        }
        if (dt == Dtype::F32) {
            const auto* lp = reinterpret_cast<const float*>(losses.ptr.get());
            float sum = sum_values(lp, samples);
            if (reduction == 1)
                sum /= static_cast<float>(valid_count);
            return Storage{make_cpu_scalar(static_cast<double>(sum), dt)};
        }
        if (dt == Dtype::F64) {
            const auto* lp = reinterpret_cast<const double*>(losses.ptr.get());
            double sum = sum_values(lp, samples);
            if (reduction == 1)
                sum /= static_cast<double>(valid_count);
            return Storage{make_cpu_scalar(sum, dt)};
        }
        ErrorBuilder("cpu_backend::class_loss_reduce").not_implemented("dtype not supported");
    }

    static std::pair<std::size_t, std::size_t> flatten_linear_x(const Shape& x_shape) {
        std::size_t M = 1;
        for (std::size_t d = 0; d + 1 < x_shape.size(); ++d)
            M *= static_cast<std::size_t>(x_shape[d]);
        return {M, static_cast<std::size_t>(x_shape.back())};
    }

    template <typename T>
    static void add_linear_bias_typed(T* y, const T* b, std::size_t M, std::size_t N) {
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t n = 0; n < N; ++n)
                y[m * N + n] += b[n];
        }
    }

    static void add_linear_bias(
        CpuStorage& y, const CpuStorage& bias, std::size_t M, std::size_t N, Dtype dt) {
        if (dt == Dtype::F32)
            add_linear_bias_typed(reinterpret_cast<float*>(y.ptr.get()),
                                  reinterpret_cast<const float*>(bias.ptr.get()), M, N);
        else if (dt == Dtype::F64)
            add_linear_bias_typed(reinterpret_cast<double*>(y.ptr.get()),
                                  reinterpret_cast<const double*>(bias.ptr.get()), M, N);
        else
            ErrorBuilder("cpu_backend::linear").not_implemented("dtype not supported");
    }

    template <typename T>
    static void sum_linear_rows_typed(const T* grad, T* db, std::size_t M, std::size_t N) {
        for (std::size_t n = 0; n < N; ++n)
            db[n] = T{};
        for (std::size_t m = 0; m < M; ++m) {
            for (std::size_t n = 0; n < N; ++n)
                db[n] += grad[m * N + n];
        }
    }

    static void sum_linear_rows(
        const CpuStorage& grad, CpuStorage& db, std::size_t M, std::size_t N, Dtype dt) {
        if (dt == Dtype::F32)
            sum_linear_rows_typed(reinterpret_cast<const float*>(grad.ptr.get()),
                                  reinterpret_cast<float*>(db.ptr.get()), M, N);
        else if (dt == Dtype::F64)
            sum_linear_rows_typed(reinterpret_cast<const double*>(grad.ptr.get()),
                                  reinterpret_cast<double*>(db.ptr.get()), M, N);
        else
            ErrorBuilder("cpu_backend::linear_backward").not_implemented("dtype not supported");
    }

    static Shape reduced_norm_shape(const Shape& shape,
                                    const std::vector<int>& axes,
                                    bool keepdims) {
        if (axes.empty()) {
            if (keepdims)
                return Shape(shape.size(), 1);
            return Shape{};
        }
        std::vector<bool> mask(shape.size(), false);
        for (int axis : axes) {
            const int wrapped = axis < 0 ? axis + static_cast<int>(shape.size()) : axis;
            mask[static_cast<std::size_t>(wrapped)] = true;
        }
        Shape out;
        for (std::size_t d = 0; d < shape.size(); ++d) {
            if (mask[d]) {
                if (keepdims)
                    out.push_back(1);
            } else {
                out.push_back(shape[d]);
            }
        }
        return out;
    }

    template <typename T, typename Fn>
    static void norm_elementwise_loop(const T* in,
                                      const Shape& shape,
                                      const std::vector<bool>& reduce_mask,
                                      const Shape& out_shape,
                                      Fn fn) {
        const std::size_t nd = shape.size();
        Stride in_stride(nd), out_stride(out_shape.size());
        if (nd > 0) {
            in_stride[nd - 1] = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 2; i >= 0; --i)
                in_stride[static_cast<std::size_t>(i)] =
                    in_stride[static_cast<std::size_t>(i) + 1] *
                    shape[static_cast<std::size_t>(i) + 1];
        }
        if (!out_shape.empty()) {
            const std::size_t ond = out_shape.size();
            out_stride[ond - 1] = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(ond) - 2; i >= 0; --i)
                out_stride[static_cast<std::size_t>(i)] =
                    out_stride[static_cast<std::size_t>(i) + 1] *
                    out_shape[static_cast<std::size_t>(i) + 1];
        }

        std::vector<std::int64_t> coord(nd, 0);
        const std::size_t in_numel = shape_numel(shape);
        for (std::size_t f = 0; f < in_numel; ++f) {
            std::size_t in_flat = 0;
            std::size_t out_flat = 0;
            std::size_t out_axis = 0;
            for (std::size_t d = 0; d < nd; ++d) {
                in_flat +=
                    static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
                if (!reduce_mask[d]) {
                    out_flat += static_cast<std::size_t>(coord[d]) *
                                static_cast<std::size_t>(out_stride[out_axis]);
                    ++out_axis;
                }
            }
            fn(in[in_flat], out_flat);
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nd) - 1; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)])
                    break;
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
    }

    template <typename T>
    static void norm_typed(const T* in,
                           T* out,
                           const Shape& shape,
                           const std::vector<int>& axes,
                           double ord,
                           const Shape& out_shape) {
        std::vector<bool> reduce_mask(shape.size(), false);
        if (axes.empty()) {
            for (std::size_t i = 0; i < shape.size(); ++i)
                reduce_mask[i] = true;
        } else {
            for (int axis : axes) {
                const int wrapped = axis < 0 ? axis + static_cast<int>(shape.size()) : axis;
                reduce_mask[static_cast<std::size_t>(wrapped)] = true;
            }
        }
        const std::size_t out_numel = std::max<std::size_t>(1, shape_numel(out_shape));
        if (std::isinf(ord)) {
            const bool positive = ord > 0;
            std::vector<T> acc(out_numel, positive ? T{0} : std::numeric_limits<T>::infinity());
            norm_elementwise_loop<T>(in, shape, reduce_mask, out_shape, [&](T v, std::size_t o) {
                const T av = std::abs(v);
                acc[o] = positive ? std::max(acc[o], av) : std::min(acc[o], av);
            });
            std::memcpy(out, acc.data(), out_numel * sizeof(T));
            return;
        }
        if (ord == 0.0) {
            std::vector<T> acc(out_numel, T{0});
            norm_elementwise_loop<T>(in, shape, reduce_mask, out_shape, [&](T v, std::size_t o) {
                if (v != T{0})
                    acc[o] += T{1};
            });
            std::memcpy(out, acc.data(), out_numel * sizeof(T));
            return;
        }
        std::vector<T> acc(out_numel, T{0});
        norm_elementwise_loop<T>(in, shape, reduce_mask, out_shape, [&](T v, std::size_t o) {
            const T av = std::abs(v);
            if (ord == 2.0)
                acc[o] += v * v;
            else if (ord == 1.0)
                acc[o] += av;
            else
                acc[o] += std::pow(av, static_cast<T>(ord));
        });
        if (ord == 2.0) {
            for (std::size_t i = 0; i < out_numel; ++i)
                acc[i] = std::sqrt(acc[i]);
        } else if (ord != 1.0) {
            const T inv_ord = static_cast<T>(1.0 / ord);
            for (std::size_t i = 0; i < out_numel; ++i)
                acc[i] = std::pow(acc[i], inv_ord);
        }
        std::memcpy(out, acc.data(), out_numel * sizeof(T));
    }

    static std::int64_t leading_matrix_batch_count(const Shape& shape, std::size_t mat_dims) {
        std::int64_t batch = 1;
        for (std::size_t i = 0; i + mat_dims < shape.size(); ++i)
            batch *= shape[i];
        return batch;
    }

    static void check_lapack_info(int info, const char* op) {
        if (info < 0)
            ErrorBuilder(op).fail("LAPACK invalid argument index" + std::to_string(-info));
        if (info > 0)
            ErrorBuilder(op).fail("LAPACK numerical failure (info=" + std::to_string(info) + ")");
    }

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
