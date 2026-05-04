// lucid/_C/backend/cpu/CpuBackend.h
//
// Concrete IBackend implementation for Apple Silicon CPU using the Apple
// Accelerate framework.  All compute is performed on CpuStorage (host-side
// aligned allocations); no raw loops are used where a vDSP or vForce vector
// intrinsic exists.
//
// Design overview:
//   Elementwise ops (add, sub, mul, exp, …) delegate to cpu::vadd_f32 and
//   friends (vDSP / vForce wrappers) through the private unary_op / binary_op
//   template helpers, which dispatch on the runtime Dtype.
//
//   Axis reductions (reduce_sum, reduce_mean, reduce_max, reduce_min) use the
//   private reduce_axes helper, which iterates axes from highest to lowest
//   (sorted descending) and calls the corresponding cpu::sum_axis_f32 etc.
//   primitives from Reduce.h.  For mean reduction the result is scaled by
//   1/reduce_dim after the sum.
//
//   Convolution uses the im2col + GEMM strategy: for each batch element the
//   input patches are unrolled into a column matrix by conv_nd_im2col_f32/f64,
//   then cblas_sgemm/dgemm computes the full output in a single call.  The
//   backward pass uses col2im for the input gradient and a second GEMM for the
//   weight gradient.
//
//   Normalization (BatchNorm, LayerNorm, GroupNorm, RMSNorm) delegates to the
//   Norm.h wrappers which use per-row vDSP operations for f32 throughput.
//
//   Linear algebra (eig, eigh, svd, qr, chol, inv, solve) delegates to the
//   Lapack.h wrappers which handle the row-major ↔ column-major conversion
//   required by Accelerate's LAPACK interface.
//
//   Pooling (MaxPool, AvgPool) delegates to Pool.h for 1-D, 2-D, and 3-D.
//
// Private helpers:
//   fill_ones(ptr, n, dt)           — fills a raw buffer with the value 1.
//   unary_op(a, shape, dt, …)       — allocates output, dispatches f32/f64/i32.
//   binary_op(a, b, shape, dt, …)   — allocates output, dispatches f32/f64/i32/i64.
//   reduce_axes(a, in_shape, opts, dt, op) — single-axis reduction loop.
//   cast_impl(src, dst, n, src_dt, dst_dt) — element-by-element type cast.
//
// Self-registration:
//   An anonymous-namespace CpuBackendRegistrar struct at the bottom of the
//   file registers a CpuBackend singleton with the Dispatcher at static-init
//   time.  BackendInit.cpp includes this header to trigger registration.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Shape.h"
#include "../Dispatcher.h"
#include "../IBackend.h"
#include "Blas.h"
#include "Im2Col.h"
#include "Lapack.h"
#include "Norm.h"
#include "Pool.h"
#include "Reduce.h"
#include "Shape.h"
#include "Vdsp.h"
#include "Vforce.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace lucid {
namespace backend {

// CPU (Accelerate-backed) concrete backend.
//
// All public methods satisfy the IBackend contract.  The private section
// contains low-level type-dispatch helpers (unary_op, binary_op, reduce_axes)
// and per-op compute routines that delegate to the Accelerate helpers in
// Blas.h, Lapack.h, Norm.h, Pool.h, Reduce.h, Vdsp.h, and Vforce.h.
class CpuBackend final : public IBackend {
public:
    // Registers this backend with the Dispatcher for Device::CPU.
    static void register_self() {
        Dispatcher::register_backend(Device::CPU, std::make_unique<CpuBackend>());
    }

    Device device() const noexcept override { return Device::CPU; }

    // CpuStorage is already in the correct form; move it into Storage directly.
    Storage from_cpu(CpuStorage cpu, const Shape&) override { return Storage{std::move(cpu)}; }

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

    Storage contiguous(const Storage& src,
                       const Shape& shape,
                       const Stride& stride,
                       std::size_t storage_offset,
                       bool already_contiguous,
                       Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(src);
        const std::size_t elem = dtype_size(dt);
        const std::size_t n = shape_numel(shape);
        const std::size_t nbytes = n * elem;
        auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
        if (nbytes == 0)
            return Storage{CpuStorage{ptr, nbytes, dt}};

        if (already_contiguous && storage_offset == 0) {
            std::memcpy(ptr.get(), cs.ptr.get(), nbytes);
            return Storage{CpuStorage{ptr, nbytes, dt}};
        }

        const int ndim = static_cast<int>(shape.size());
        const auto* base = reinterpret_cast<const std::uint8_t*>(cs.ptr.get()) + storage_offset;
        auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
        std::vector<std::size_t> coord(static_cast<std::size_t>(ndim), 0);
        for (std::size_t f = 0; f < n; ++f) {
            std::ptrdiff_t byte_offset = 0;
            for (int d = 0; d < ndim; ++d)
                byte_offset +=
                    static_cast<std::ptrdiff_t>(coord[d]) * static_cast<std::ptrdiff_t>(stride[d]);
            std::memcpy(dst + f * elem, base + byte_offset, elem);
            for (int d = ndim - 1; d >= 0; --d) {
                if (++coord[d] < static_cast<std::size_t>(shape[d]))
                    break;
                coord[d] = 0;
            }
        }
        return Storage{CpuStorage{ptr, nbytes, dt}};
    }

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
        auto neg_a = neg(a, shape, dt);
        auto exp_neg = exp(neg_a, shape, dt);
        auto one = ones(shape, dt);
        auto denom = add(one, exp_neg, shape, dt);

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
        auto sig = sigmoid(a, shape, dt);
        return mul(a, sig, shape, dt);
    }

    Storage gelu(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        constexpr double kC1 = 0.7978845608028654;
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

    Storage
    gelu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) override {
        constexpr double kC1 = 0.7978845608028654;
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

    Storage
    selu_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) override {
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

    Storage
    mish_backward(const Storage& a, const Storage& grad, const Shape& shape, Dtype dt) override {
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

    Storage isinf(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        auto ptr = allocate_aligned_bytes(n, Device::CPU);
        auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i) dst[i] = std::isinf(p[i]) ? 1u : 0u;
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i) dst[i] = std::isinf(p[i]) ? 1u : 0u;
        } else {
            for (std::size_t i = 0; i < n; ++i) dst[i] = 0u;
        }
        return Storage{CpuStorage{ptr, n, Dtype::Bool}};
    }

    Storage isnan(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        auto ptr = allocate_aligned_bytes(n, Device::CPU);
        auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i) dst[i] = std::isnan(p[i]) ? 1u : 0u;
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i) dst[i] = std::isnan(p[i]) ? 1u : 0u;
        } else {
            for (std::size_t i = 0; i < n; ++i) dst[i] = 0u;
        }
        return Storage{CpuStorage{ptr, n, Dtype::Bool}};
    }

    Storage isfinite(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        auto ptr = allocate_aligned_bytes(n, Device::CPU);
        auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i) dst[i] = std::isfinite(p[i]) ? 1u : 0u;
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            for (std::size_t i = 0; i < n; ++i) dst[i] = std::isfinite(p[i]) ? 1u : 0u;
        } else {
            for (std::size_t i = 0; i < n; ++i) dst[i] = 1u;
        }
        return Storage{CpuStorage{ptr, n, Dtype::Bool}};
    }

    Storage any(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        auto ptr = allocate_aligned_bytes(1, Device::CPU);
        auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
        bool found = false;
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && !found; ++i) found = (p[i] != 0.f);
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && !found; ++i) found = (p[i] != 0.0);
        } else if (dt == Dtype::I32) {
            const std::int32_t* p = reinterpret_cast<const std::int32_t*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && !found; ++i) found = (p[i] != 0);
        } else if (dt == Dtype::Bool) {
            const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && !found; ++i) found = (p[i] != 0u);
        } else {
            for (std::size_t i = 0; i < n && !found; ++i)
                found = (reinterpret_cast<const std::uint8_t*>(cs.ptr.get())[i] != 0u);
        }
        dst[0] = found ? 1u : 0u;
        return Storage{CpuStorage{ptr, 1, Dtype::Bool}};
    }

    Storage all(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        auto ptr = allocate_aligned_bytes(1, Device::CPU);
        auto* dst = reinterpret_cast<std::uint8_t*>(ptr.get());
        bool all_nz = true;
        if (dt == Dtype::F32) {
            const float* p = reinterpret_cast<const float*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && all_nz; ++i) all_nz = (p[i] != 0.f);
        } else if (dt == Dtype::F64) {
            const double* p = reinterpret_cast<const double*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && all_nz; ++i) all_nz = (p[i] != 0.0);
        } else if (dt == Dtype::I32) {
            const std::int32_t* p = reinterpret_cast<const std::int32_t*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && all_nz; ++i) all_nz = (p[i] != 0);
        } else if (dt == Dtype::Bool) {
            const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(cs.ptr.get());
            for (std::size_t i = 0; i < n && all_nz; ++i) all_nz = (p[i] != 0u);
        } else {
            for (std::size_t i = 0; i < n && all_nz; ++i)
                all_nz = (reinterpret_cast<const std::uint8_t*>(cs.ptr.get())[i] != 0u);
        }
        dst[0] = all_nz ? 1u : 0u;
        return Storage{CpuStorage{ptr, 1, Dtype::Bool}};
    }

    Storage nan_to_num(const Storage& a, const Shape& shape, Dtype dt,
                       double nan_val, double posinf_val, double neginf_val) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto replace = [&](auto* dst, const auto* src, auto nan_v, auto pi_v, auto ni_v) {
            for (std::size_t i = 0; i < n; ++i) {
                auto v = src[i];
                if (std::isnan(v))       dst[i] = nan_v;
                else if (v ==  std::numeric_limits<decltype(v)>::infinity()) dst[i] = pi_v;
                else if (v == -std::numeric_limits<decltype(v)>::infinity()) dst[i] = ni_v;
                else                    dst[i] = v;
            }
        };
        if (dt == Dtype::F32) {
            replace(reinterpret_cast<float*>(ptr.get()),
                    reinterpret_cast<const float*>(cs.ptr.get()),
                    static_cast<float>(nan_val),
                    static_cast<float>(posinf_val),
                    static_cast<float>(neginf_val));
        } else if (dt == Dtype::F64) {
            replace(reinterpret_cast<double*>(ptr.get()),
                    reinterpret_cast<const double*>(cs.ptr.get()),
                    nan_val, posinf_val, neginf_val);
        } else {
            std::memcpy(ptr.get(), cs.ptr.get(), nb);
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage
    reduce_sum(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override {
        return reduce_axes(a, in_shape, opts, dt, ReduceOp::Sum);
    }

    Storage reduce_mean(const Storage& a,
                        const Shape& in_shape,
                        const ReduceOpts& opts,
                        Dtype dt) override {
        return reduce_axes(a, in_shape, opts, dt, ReduceOp::Mean);
    }

    Storage
    variance(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override {
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

    Storage
    reduce_max(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override {
        return reduce_axes(a, in_shape, opts, dt, ReduceOp::Max);
    }

    Storage
    reduce_min(const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt) override {
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

    Storage log_softmax_backward(const Storage& y, const Storage& grad_out,
                                  const Shape& shape, int axis, Dtype dt) override {
        const auto& yc = std::get<CpuStorage>(y);
        const auto& gc = std::get<CpuStorage>(grad_out);
        std::size_t nb = gc.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
        std::size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
            inner *= static_cast<std::size_t>(shape[d]);

        auto kernel = [&](auto* dx, const auto* gy, const auto* g) {
            using T = std::remove_pointer_t<decltype(dx)>;
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t i = 0; i < inner; ++i) {
                    T sum_g = T(0);
                    for (std::size_t r = 0; r < axis_dim; ++r)
                        sum_g += g[o * axis_dim * inner + r * inner + i];
                    for (std::size_t r = 0; r < axis_dim; ++r) {
                        std::size_t idx = o * axis_dim * inner + r * inner + i;
                        dx[idx] = g[idx] - std::exp(gy[idx]) * sum_g;
                    }
                }
            }
        };

        if (dt == Dtype::F32)
            kernel(reinterpret_cast<float*>(ptr.get()),
                   reinterpret_cast<const float*>(yc.ptr.get()),
                   reinterpret_cast<const float*>(gc.ptr.get()));
        else if (dt == Dtype::F64)
            kernel(reinterpret_cast<double*>(ptr.get()),
                   reinterpret_cast<const double*>(yc.ptr.get()),
                   reinterpret_cast<const double*>(gc.ptr.get()));
        else
            ErrorBuilder("cpu_backend::log_softmax_backward").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage log_softmax(const Storage& a, const Shape& shape, int axis, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        std::size_t nb = cs.nbytes;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::size_t axis_dim = static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
        std::size_t outer = 1, inner = 1;
        for (int d = 0; d < axis; ++d)
            outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
        for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
            inner *= static_cast<std::size_t>(shape[d]);

        auto kernel = [&](auto* out, const auto* in) {
            using T = std::remove_pointer_t<decltype(out)>;
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t i = 0; i < inner; ++i) {
                    const T* base = in + o * axis_dim * inner + i;
                    T m = base[0];
                    for (std::size_t r = 1; r < axis_dim; ++r) {
                        T v = base[r * inner];
                        if (v > m) m = v;
                    }
                    T s = T(0);
                    for (std::size_t r = 0; r < axis_dim; ++r)
                        s += std::exp(base[r * inner] - m);
                    const T log_sum = m + std::log(s);
                    T* obase = out + o * axis_dim * inner + i;
                    for (std::size_t r = 0; r < axis_dim; ++r)
                        obase[r * inner] = base[r * inner] - log_sum;
                }
            }
        };

        if (dt == Dtype::F32)
            kernel(reinterpret_cast<float*>(ptr.get()),
                   reinterpret_cast<const float*>(cs.ptr.get()));
        else if (dt == Dtype::F64)
            kernel(reinterpret_cast<double*>(ptr.get()),
                   reinterpret_cast<const double*>(cs.ptr.get()));
        else
            ErrorBuilder("cpu_backend::log_softmax").not_implemented("dtype not supported");
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

    Storage reshape(const Storage& a, const Shape&, const Shape&, Dtype dt) override {
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

    Storage
    stack(const std::vector<Storage>& xs, const Shape& input_shape, int axis, Dtype dt) override {
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
                          const std::vector<std::int64_t>&,
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
        auto result = sort_select(a, shape, shape, axis, dt, false);
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

    // User-facing scatter-add: copies base then adds src at positions given by indices.
    Storage scatter_add(const Storage& base,
                        const Storage& indices,
                        const Storage& src,
                        const Shape& base_shape,
                        const Shape& idx_shape,
                        int dim,
                        Dtype dt) override {
        const auto& cb = std::get<CpuStorage>(base);
        const auto& ci = std::get<CpuStorage>(indices);
        const auto& cs = std::get<CpuStorage>(src);

        const std::size_t base_n = shape_numel(base_shape);
        const std::size_t nbytes = base_n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);
        // Start from a copy of base
        std::memcpy(ptr.get(), cb.ptr.get(), std::min(nbytes, cb.nbytes));

        const int ndim = static_cast<int>(base_shape.size());
        if (dim < 0) dim += ndim;

        // outer = product of dims before dim
        std::size_t outer = 1;
        for (int d = 0; d < dim; ++d)
            outer *= static_cast<std::size_t>(base_shape[static_cast<std::size_t>(d)]);
        // inner = product of dims after dim
        std::size_t inner = 1;
        for (int d = dim + 1; d < ndim; ++d)
            inner *= static_cast<std::size_t>(base_shape[static_cast<std::size_t>(d)]);
        const std::size_t base_dim = static_cast<std::size_t>(base_shape[static_cast<std::size_t>(dim)]);
        const std::size_t idx_dim  = static_cast<std::size_t>(idx_shape[static_cast<std::size_t>(dim)]);
        const auto* ip = reinterpret_cast<const std::int32_t*>(ci.ptr.get());

        auto run = [&](auto* dst, const auto* sp) {
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t j = 0; j < inner; ++j) {
                    for (std::size_t k = 0; k < idx_dim; ++k) {
                        const std::size_t src_flat = (o * idx_dim + k) * inner + j;
                        std::int32_t tgt = ip[src_flat];
                        if (tgt < 0) tgt += static_cast<std::int32_t>(base_dim);
                        const std::size_t dst_flat =
                            (o * base_dim + static_cast<std::size_t>(tgt)) * inner + j;
                        dst[dst_flat] += sp[src_flat];
                    }
                }
            }
        };

        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(ptr.get()),
                reinterpret_cast<const float*>(cs.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(ptr.get()),
                reinterpret_cast<const double*>(cs.ptr.get()));
        else
            ErrorBuilder("cpu_backend::scatter_add").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nbytes, dt}};
    }

    // Sliding-window view: out shape is (*in_shape[:dim], L, *in_shape[dim+1:], size)
    Storage unfold_dim(const Storage& a,
                       const Shape& in_shape,
                       int dim,
                       int size,
                       int step,
                       Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const int ndim = static_cast<int>(in_shape.size());
        if (dim < 0) dim += ndim;

        const std::size_t dim_size = static_cast<std::size_t>(in_shape[static_cast<std::size_t>(dim)]);
        const std::size_t L = (dim_size - static_cast<std::size_t>(size)) /
                               static_cast<std::size_t>(step) + 1;

        // out_shape = in_shape[:dim] + [L] + in_shape[dim+1:] + [size]
        Shape out_shape;
        for (int d = 0; d < dim; ++d) out_shape.push_back(in_shape[static_cast<std::size_t>(d)]);
        out_shape.push_back(static_cast<std::int64_t>(L));
        for (int d = dim + 1; d < ndim; ++d) out_shape.push_back(in_shape[static_cast<std::size_t>(d)]);
        out_shape.push_back(static_cast<std::int64_t>(size));

        const std::size_t out_n = shape_numel(out_shape);
        const std::size_t nbytes = out_n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nbytes, Device::CPU);

        // outer = product of dims before dim
        std::size_t outer = 1;
        for (int d = 0; d < dim; ++d)
            outer *= static_cast<std::size_t>(in_shape[static_cast<std::size_t>(d)]);
        // inner = product of dims after dim (in the original tensor)
        std::size_t inner = 1;
        for (int d = dim + 1; d < ndim; ++d)
            inner *= static_cast<std::size_t>(in_shape[static_cast<std::size_t>(d)]);

        const std::size_t esz = dtype_size(dt);
        const std::uint8_t* src_ptr = static_cast<const std::uint8_t*>(
            static_cast<const void*>(ca.ptr.get()));
        std::uint8_t* dst_ptr = static_cast<std::uint8_t*>(static_cast<void*>(ptr.get()));

        for (std::size_t o = 0; o < outer; ++o) {
            for (std::size_t l = 0; l < L; ++l) {
                for (std::size_t j = 0; j < inner; ++j) {
                    for (std::size_t s = 0; s < static_cast<std::size_t>(size); ++s) {
                        // Source element: [o, l*step+s, j] in the input
                        const std::size_t src_dim_pos = l * static_cast<std::size_t>(step) + s;
                        const std::size_t src_flat =
                            (o * dim_size + src_dim_pos) * inner + j;
                        // Destination: [o, l, j, s] in the output
                        const std::size_t dst_flat =
                            ((o * L + l) * inner + j) * static_cast<std::size_t>(size) + s;
                        std::memcpy(dst_ptr + dst_flat * esz, src_ptr + src_flat * esz, esz);
                    }
                }
            }
        }
        return Storage{CpuStorage{ptr, nbytes, dt}};
    }

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

    StoragePair rms_norm_forward(const Storage& x,
                                 const Storage& gamma,
                                 std::size_t outer,
                                 std::size_t normalized_size,
                                 double eps,
                                 const Shape&,
                                 Dtype dt) override {
        const std::size_t y_nbytes = outer * normalized_size * dtype_size(dt);
        const std::size_t rstd_nbytes = outer * dtype_size(dt);
        auto y_ptr = allocate_aligned_bytes(y_nbytes, Device::CPU);
        auto rstd_ptr = allocate_aligned_bytes(rstd_nbytes, Device::CPU);
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& g_cpu = std::get<CpuStorage>(gamma);

        if (outer * normalized_size > 0) {
            if (dt == Dtype::F32) {
                cpu::rms_norm_forward_f32(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                          reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                          reinterpret_cast<float*>(y_ptr.get()),
                                          reinterpret_cast<float*>(rstd_ptr.get()), outer,
                                          normalized_size, eps);
            } else if (dt == Dtype::F64) {
                cpu::rms_norm_forward_f64(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                          reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                          reinterpret_cast<double*>(y_ptr.get()),
                                          reinterpret_cast<double*>(rstd_ptr.get()), outer,
                                          normalized_size, eps);
            } else {
                ErrorBuilder("cpu_backend::rms_norm_forward")
                    .not_implemented("dtype not supported");
            }
        }
        return {Storage{CpuStorage{y_ptr, y_nbytes, dt}},
                Storage{CpuStorage{rstd_ptr, rstd_nbytes, dt}}};
    }

    StoragePair rms_norm_backward(const Storage& x,
                                  const Storage& gamma,
                                  const Storage& saved_rstd,
                                  const Storage& grad,
                                  std::size_t outer,
                                  std::size_t normalized_size,
                                  const Shape&,
                                  const Shape&,
                                  Dtype dt) override {
        const std::size_t dx_nbytes = outer * normalized_size * dtype_size(dt);
        const std::size_t dgamma_nbytes = normalized_size * dtype_size(dt);
        auto dx_ptr = allocate_aligned_bytes(dx_nbytes, Device::CPU);
        auto dgamma_ptr = allocate_aligned_bytes(dgamma_nbytes, Device::CPU);

        if (outer * normalized_size > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x);
            const auto& gamma_cpu = std::get<CpuStorage>(gamma);
            const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
            const auto& grad_cpu = std::get<CpuStorage>(grad);
            if (dt == Dtype::F32) {
                cpu::rms_norm_backward_f32(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                           reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                                           reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                                           reinterpret_cast<const float*>(grad_cpu.ptr.get()),
                                           reinterpret_cast<float*>(dx_ptr.get()),
                                           reinterpret_cast<float*>(dgamma_ptr.get()), outer,
                                           normalized_size);
            } else if (dt == Dtype::F64) {
                cpu::rms_norm_backward_f64(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                           reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                                           reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                                           reinterpret_cast<const double*>(grad_cpu.ptr.get()),
                                           reinterpret_cast<double*>(dx_ptr.get()),
                                           reinterpret_cast<double*>(dgamma_ptr.get()), outer,
                                           normalized_size);
            } else {
                ErrorBuilder("cpu_backend::rms_norm_backward")
                    .not_implemented("dtype not supported");
            }
        } else {
            if (dx_nbytes)
                std::memset(dx_ptr.get(), 0, dx_nbytes);
            if (dgamma_nbytes)
                std::memset(dgamma_ptr.get(), 0, dgamma_nbytes);
        }
        return {Storage{CpuStorage{dx_ptr, dx_nbytes, dt}},
                Storage{CpuStorage{dgamma_ptr, dgamma_nbytes, dt}}};
    }

    std::vector<Storage> layer_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            std::size_t outer,
                                            std::size_t normalized_size,
                                            double eps,
                                            const Shape&,
                                            Dtype dt) override {
        const std::size_t y_nbytes = outer * normalized_size * dtype_size(dt);
        const std::size_t saved_nbytes = outer * dtype_size(dt);
        auto y_ptr = allocate_aligned_bytes(y_nbytes, Device::CPU);
        auto mean_ptr = allocate_aligned_bytes(saved_nbytes, Device::CPU);
        auto rstd_ptr = allocate_aligned_bytes(saved_nbytes, Device::CPU);
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& g_cpu = std::get<CpuStorage>(gamma);
        const auto& b_cpu = std::get<CpuStorage>(beta);

        if (outer * normalized_size > 0) {
            if (dt == Dtype::F32) {
                cpu::layer_norm_forward_f32(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<const float*>(b_cpu.ptr.get()),
                    reinterpret_cast<float*>(y_ptr.get()), reinterpret_cast<float*>(mean_ptr.get()),
                    reinterpret_cast<float*>(rstd_ptr.get()), outer, normalized_size, eps);
            } else if (dt == Dtype::F64) {
                cpu::layer_norm_forward_f64(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                            reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                            reinterpret_cast<const double*>(b_cpu.ptr.get()),
                                            reinterpret_cast<double*>(y_ptr.get()),
                                            reinterpret_cast<double*>(mean_ptr.get()),
                                            reinterpret_cast<double*>(rstd_ptr.get()), outer,
                                            normalized_size, eps);
            } else {
                ErrorBuilder("cpu_backend::layer_norm_forward")
                    .not_implemented("dtype not supported");
            }
        }
        return {Storage{CpuStorage{y_ptr, y_nbytes, dt}},
                Storage{CpuStorage{mean_ptr, saved_nbytes, dt}},
                Storage{CpuStorage{rstd_ptr, saved_nbytes, dt}}};
    }

    std::vector<Storage> layer_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             std::size_t outer,
                                             std::size_t normalized_size,
                                             const Shape&,
                                             const Shape&,
                                             const Shape&,
                                             Dtype dt) override {
        const std::size_t dx_nbytes = outer * normalized_size * dtype_size(dt);
        const std::size_t param_nbytes = normalized_size * dtype_size(dt);
        auto dx_ptr = allocate_aligned_bytes(dx_nbytes, Device::CPU);
        auto dgamma_ptr = allocate_aligned_bytes(param_nbytes, Device::CPU);
        auto dbeta_ptr = allocate_aligned_bytes(param_nbytes, Device::CPU);

        if (outer * normalized_size > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x);
            const auto& gamma_cpu = std::get<CpuStorage>(gamma);
            const auto& mean_cpu = std::get<CpuStorage>(saved_mean);
            const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
            const auto& g_cpu = std::get<CpuStorage>(grad);
            if (dt == Dtype::F32) {
                cpu::layer_norm_backward_f32(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                             reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                                             reinterpret_cast<const float*>(mean_cpu.ptr.get()),
                                             reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                                             reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                             reinterpret_cast<float*>(dx_ptr.get()),
                                             reinterpret_cast<float*>(dgamma_ptr.get()),
                                             reinterpret_cast<float*>(dbeta_ptr.get()), outer,
                                             normalized_size);
            } else if (dt == Dtype::F64) {
                cpu::layer_norm_backward_f64(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                             reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                                             reinterpret_cast<const double*>(mean_cpu.ptr.get()),
                                             reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                                             reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                             reinterpret_cast<double*>(dx_ptr.get()),
                                             reinterpret_cast<double*>(dgamma_ptr.get()),
                                             reinterpret_cast<double*>(dbeta_ptr.get()), outer,
                                             normalized_size);
            } else {
                ErrorBuilder("cpu_backend::layer_norm_backward")
                    .not_implemented("dtype not supported");
            }
        } else {
            if (dx_nbytes)
                std::memset(dx_ptr.get(), 0, dx_nbytes);
            if (param_nbytes) {
                std::memset(dgamma_ptr.get(), 0, param_nbytes);
                std::memset(dbeta_ptr.get(), 0, param_nbytes);
            }
        }
        return {Storage{CpuStorage{dx_ptr, dx_nbytes, dt}},
                Storage{CpuStorage{dgamma_ptr, param_nbytes, dt}},
                Storage{CpuStorage{dbeta_ptr, param_nbytes, dt}}};
    }

    std::vector<Storage> batch_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int batch,
                                            int channels,
                                            int spatial,
                                            int,
                                            double eps,
                                            const Shape&,
                                            Dtype dt) override {
        const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
        auto y_ptr = allocate_aligned_bytes(total * dtype_size(dt), Device::CPU);
        auto mean_ptr = allocate_aligned_bytes(static_cast<std::size_t>(channels) * dtype_size(dt),
                                               Device::CPU);
        auto rstd_ptr = allocate_aligned_bytes(static_cast<std::size_t>(channels) * dtype_size(dt),
                                               Device::CPU);
        if (total > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x);
            const auto& g_cpu = std::get<CpuStorage>(gamma);
            const auto& b_cpu = std::get<CpuStorage>(beta);
            if (dt == Dtype::F32) {
                batch_norm_forward_f32_fast(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<const float*>(b_cpu.ptr.get()),
                    reinterpret_cast<float*>(y_ptr.get()), reinterpret_cast<float*>(mean_ptr.get()),
                    reinterpret_cast<float*>(rstd_ptr.get()), batch, channels, spatial, eps);
            } else if (dt == Dtype::F64) {
                batch_norm_forward_typed<double>(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                                 reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                                 reinterpret_cast<const double*>(b_cpu.ptr.get()),
                                                 reinterpret_cast<double*>(y_ptr.get()),
                                                 reinterpret_cast<double*>(mean_ptr.get()),
                                                 reinterpret_cast<double*>(rstd_ptr.get()), batch,
                                                 channels, spatial, eps);
            } else {
                ErrorBuilder("cpu_backend::batch_norm_forward")
                    .not_implemented("dtype not supported");
            }
        }
        const std::size_t param_nbytes = static_cast<std::size_t>(channels) * dtype_size(dt);
        return {Storage{CpuStorage{y_ptr, total * dtype_size(dt), dt}},
                Storage{CpuStorage{mean_ptr, param_nbytes, dt}},
                Storage{CpuStorage{rstd_ptr, param_nbytes, dt}}};
    }

    std::vector<Storage> batch_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             int batch,
                                             int channels,
                                             int spatial,
                                             int,
                                             const Shape&,
                                             Dtype dt) override {
        const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
        const std::size_t param_nbytes = static_cast<std::size_t>(channels) * dtype_size(dt);
        auto dx_ptr = allocate_aligned_bytes(total * dtype_size(dt), Device::CPU);
        auto dgamma_ptr = allocate_aligned_bytes(param_nbytes, Device::CPU);
        auto dbeta_ptr = allocate_aligned_bytes(param_nbytes, Device::CPU);
        if (dx_ptr && total > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x);
            const auto& gamma_cpu = std::get<CpuStorage>(gamma);
            const auto& mean_cpu = std::get<CpuStorage>(saved_mean);
            const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
            const auto& g_cpu = std::get<CpuStorage>(grad);
            if (dt == Dtype::F32) {
                batch_norm_backward_typed<float>(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const float*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<float*>(dx_ptr.get()),
                    reinterpret_cast<float*>(dgamma_ptr.get()),
                    reinterpret_cast<float*>(dbeta_ptr.get()), batch, channels, spatial);
            } else if (dt == Dtype::F64) {
                batch_norm_backward_typed<double>(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const double*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const double*>(g_cpu.ptr.get()),
                    reinterpret_cast<double*>(dx_ptr.get()),
                    reinterpret_cast<double*>(dgamma_ptr.get()),
                    reinterpret_cast<double*>(dbeta_ptr.get()), batch, channels, spatial);
            } else {
                ErrorBuilder("cpu_backend::batch_norm_backward")
                    .not_implemented("dtype not supported");
            }
        } else {
            if (total)
                std::memset(dx_ptr.get(), 0, total * dtype_size(dt));
            if (param_nbytes) {
                std::memset(dgamma_ptr.get(), 0, param_nbytes);
                std::memset(dbeta_ptr.get(), 0, param_nbytes);
            }
        }
        return {Storage{CpuStorage{dx_ptr, total * dtype_size(dt), dt}},
                Storage{CpuStorage{dgamma_ptr, param_nbytes, dt}},
                Storage{CpuStorage{dbeta_ptr, param_nbytes, dt}}};
    }

    std::vector<Storage> group_norm_forward(const Storage& x,
                                            const Storage& gamma,
                                            const Storage& beta,
                                            int batch,
                                            int channels,
                                            int spatial,
                                            int groups,
                                            const std::vector<int>&,
                                            double eps,
                                            const Shape&,
                                            Dtype dt) override {
        const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
        const std::size_t saved_numel = static_cast<std::size_t>(batch) * groups;
        auto y_ptr = allocate_aligned_bytes(total * dtype_size(dt), Device::CPU);
        auto mean_ptr = allocate_aligned_bytes(saved_numel * dtype_size(dt), Device::CPU);
        auto rstd_ptr = allocate_aligned_bytes(saved_numel * dtype_size(dt), Device::CPU);
        if (total > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x);
            const auto& g_cpu = std::get<CpuStorage>(gamma);
            const auto& b_cpu = std::get<CpuStorage>(beta);
            if (dt == Dtype::F32) {
                group_norm_forward_typed<float>(reinterpret_cast<const float*>(x_cpu.ptr.get()),
                                                reinterpret_cast<const float*>(g_cpu.ptr.get()),
                                                reinterpret_cast<const float*>(b_cpu.ptr.get()),
                                                reinterpret_cast<float*>(y_ptr.get()),
                                                reinterpret_cast<float*>(mean_ptr.get()),
                                                reinterpret_cast<float*>(rstd_ptr.get()), batch,
                                                channels, spatial, groups, eps);
            } else if (dt == Dtype::F64) {
                group_norm_forward_typed<double>(reinterpret_cast<const double*>(x_cpu.ptr.get()),
                                                 reinterpret_cast<const double*>(g_cpu.ptr.get()),
                                                 reinterpret_cast<const double*>(b_cpu.ptr.get()),
                                                 reinterpret_cast<double*>(y_ptr.get()),
                                                 reinterpret_cast<double*>(mean_ptr.get()),
                                                 reinterpret_cast<double*>(rstd_ptr.get()), batch,
                                                 channels, spatial, groups, eps);
            } else {
                ErrorBuilder("cpu_backend::group_norm_forward")
                    .not_implemented("dtype not supported");
            }
        }
        return {Storage{CpuStorage{y_ptr, total * dtype_size(dt), dt}},
                Storage{CpuStorage{mean_ptr, saved_numel * dtype_size(dt), dt}},
                Storage{CpuStorage{rstd_ptr, saved_numel * dtype_size(dt), dt}}};
    }

    std::vector<Storage> group_norm_backward(const Storage& x,
                                             const Storage& gamma,
                                             const Storage& saved_mean,
                                             const Storage& saved_rstd,
                                             const Storage& grad,
                                             int batch,
                                             int channels,
                                             int spatial,
                                             int groups,
                                             const std::vector<int>&,
                                             const Shape&,
                                             Dtype dt) override {
        const std::size_t total = static_cast<std::size_t>(batch) * channels * spatial;
        const std::size_t param_nbytes = static_cast<std::size_t>(channels) * dtype_size(dt);
        auto dx_ptr = allocate_aligned_bytes(total * dtype_size(dt), Device::CPU);
        auto dgamma_ptr = allocate_aligned_bytes(param_nbytes, Device::CPU);
        auto dbeta_ptr = allocate_aligned_bytes(param_nbytes, Device::CPU);
        if (dx_ptr && total > 0) {
            const auto& x_cpu = std::get<CpuStorage>(x);
            const auto& gamma_cpu = std::get<CpuStorage>(gamma);
            const auto& mean_cpu = std::get<CpuStorage>(saved_mean);
            const auto& rstd_cpu = std::get<CpuStorage>(saved_rstd);
            const auto& g_cpu = std::get<CpuStorage>(grad);
            if (dt == Dtype::F32) {
                group_norm_backward_typed<float>(
                    reinterpret_cast<const float*>(x_cpu.ptr.get()),
                    reinterpret_cast<const float*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const float*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const float*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const float*>(g_cpu.ptr.get()),
                    reinterpret_cast<float*>(dx_ptr.get()),
                    reinterpret_cast<float*>(dgamma_ptr.get()),
                    reinterpret_cast<float*>(dbeta_ptr.get()), batch, channels, spatial, groups);
            } else if (dt == Dtype::F64) {
                group_norm_backward_typed<double>(
                    reinterpret_cast<const double*>(x_cpu.ptr.get()),
                    reinterpret_cast<const double*>(gamma_cpu.ptr.get()),
                    reinterpret_cast<const double*>(mean_cpu.ptr.get()),
                    reinterpret_cast<const double*>(rstd_cpu.ptr.get()),
                    reinterpret_cast<const double*>(g_cpu.ptr.get()),
                    reinterpret_cast<double*>(dx_ptr.get()),
                    reinterpret_cast<double*>(dgamma_ptr.get()),
                    reinterpret_cast<double*>(dbeta_ptr.get()), batch, channels, spatial, groups);
            } else {
                ErrorBuilder("cpu_backend::group_norm_backward")
                    .not_implemented("dtype not supported");
            }
        } else if (param_nbytes) {
            std::memset(dgamma_ptr.get(), 0, param_nbytes);
            std::memset(dbeta_ptr.get(), 0, param_nbytes);
        }
        return {Storage{CpuStorage{dx_ptr, total * dtype_size(dt), dt}},
                Storage{CpuStorage{dgamma_ptr, param_nbytes, dt}},
                Storage{CpuStorage{dbeta_ptr, param_nbytes, dt}}};
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
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
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
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
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
        const std::int64_t batch = leading_matrix_batch_count(a_shape, 2);
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

    Storage
    linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) override {
        const auto& cs = std::get<CpuStorage>(a);
        auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;
        const int reps = std::abs(power);

        auto run = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* in_p = reinterpret_cast<const T*>(cs.ptr.get());
            T* out_p = reinterpret_cast<T*>(ptr.get());
            std::vector<T> base(per_mat), tmp(per_mat);
            for (std::int64_t b = 0; b < batch; ++b) {
                T* result = out_p + b * per_mat;
                if (power == 0) {
                    set_matrix_identity(result, n);
                    continue;
                }
                std::memcpy(base.data(), in_p + b * per_mat, per_mat * sizeof(T));
                if (power < 0) {
                    int info = 0;
                    if constexpr (std::is_same_v<T, float>)
                        cpu::lapack_inv_f32(base.data(), n, &info);
                    else
                        cpu::lapack_inv_f64(base.data(), n, &info);
                    check_lapack_info(info, "matrix_power");
                }
                std::memcpy(result, base.data(), per_mat * sizeof(T));
                for (int i = 1; i < reps; ++i) {
                    if constexpr (std::is_same_v<T, float>)
                        cpu::sgemm(false, false, n, n, n, 1.0f, result, n, base.data(), n, 0.0f,
                                   tmp.data(), n);
                    else
                        cpu::dgemm(false, false, n, n, n, 1.0, result, n, base.data(), n, 0.0,
                                   tmp.data(), n);
                    std::memcpy(result, tmp.data(), per_mat * sizeof(T));
                }
            }
        };

        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu_backend::linalg_matrix_power").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, cs.nbytes, dt}};
    }

    Storage linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) override {
        const int m = static_cast<int>(shape[shape.size() - 2]);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        Shape out_shape(shape.begin(), shape.end() - 2);
        out_shape.push_back(n);
        out_shape.push_back(m);

        const auto& cs = std::get<CpuStorage>(a);
        const std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t in_per = static_cast<std::size_t>(m) * n;
        const std::size_t out_per = static_cast<std::size_t>(n) * m;

        if (dt == Dtype::F32) {
            const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
            auto* out_p = reinterpret_cast<float*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b)
                pinv_one(in_p + b * in_per, m, n, out_p + b * out_per);
        } else if (dt == Dtype::F64) {
            const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
            auto* out_p = reinterpret_cast<double*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b)
                pinv_one(in_p + b * in_per, m, n, out_p + b * out_per);
        } else {
            ErrorBuilder("cpu_backend::linalg_pinv").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    Storage linalg_det(const Storage& a, const Shape& shape, Dtype dt) override {
        const int n = static_cast<int>(shape[shape.size() - 1]);
        Shape out_shape(shape.begin(), shape.end() - 2);
        const std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);

        const auto& cs = std::get<CpuStorage>(a);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;
        std::vector<int> ipiv(n);
        int info = 0;

        if (dt == Dtype::F32) {
            std::vector<float> l(per_mat), u(per_mat);
            const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
            auto* out_p = reinterpret_cast<float*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_lu_f32(in_p + b * per_mat, n, ipiv.data(), l.data(), u.data(), &info);
                if (info < 0)
                    check_lapack_info(info, "det");
                float det = ipiv_sign(ipiv.data(), n);
                if (info > 0) {
                    det = 0.0f;
                } else {
                    for (int i = 0; i < n; ++i)
                        det *= u[i * n + i];
                }
                out_p[b] = det;
            }
        } else if (dt == Dtype::F64) {
            std::vector<double> l(per_mat), u(per_mat);
            const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
            auto* out_p = reinterpret_cast<double*>(ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_lu_f64(in_p + b * per_mat, n, ipiv.data(), l.data(), u.data(), &info);
                if (info < 0)
                    check_lapack_info(info, "det");
                double det = ipiv_sign(ipiv.data(), n);
                if (info > 0) {
                    det = 0.0;
                } else {
                    for (int i = 0; i < n; ++i)
                        det *= u[i * n + i];
                }
                out_p[b] = det;
            }
        } else {
            ErrorBuilder("cpu_backend::linalg_det").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, out_nbytes, dt}};
    }

    StoragePair linalg_qr(const Storage& a,
                          const Shape& shape,
                          const Shape& q_shape,
                          const Shape& r_shape,
                          Dtype dt) override {
        const int m = static_cast<int>(shape[shape.size() - 2]);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const int k = std::min(m, n);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t in_per = static_cast<std::size_t>(m) * n;
        const std::size_t q_per = static_cast<std::size_t>(m) * k;
        const std::size_t r_per = static_cast<std::size_t>(k) * n;

        const std::size_t q_nbytes = shape_numel(q_shape) * dtype_size(dt);
        const std::size_t r_nbytes = shape_numel(r_shape) * dtype_size(dt);
        auto q_ptr = allocate_aligned_bytes(q_nbytes, Device::CPU);
        auto r_ptr = allocate_aligned_bytes(r_nbytes, Device::CPU);
        const auto& cs = std::get<CpuStorage>(a);

        int info = 0;
        if (dt == Dtype::F32) {
            const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
            auto* q_p = reinterpret_cast<float*>(q_ptr.get());
            auto* r_p = reinterpret_cast<float*>(r_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_qr_f32(in_p + b * in_per, m, n, q_p + b * q_per, r_p + b * r_per,
                                   &info);
                check_lapack_info(info, "qr");
            }
        } else if (dt == Dtype::F64) {
            const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
            auto* q_p = reinterpret_cast<double*>(q_ptr.get());
            auto* r_p = reinterpret_cast<double*>(r_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_qr_f64(in_p + b * in_per, m, n, q_p + b * q_per, r_p + b * r_per,
                                   &info);
                check_lapack_info(info, "qr");
            }
        } else {
            ErrorBuilder("cpu_backend::linalg_qr").not_implemented("dtype not supported");
        }

        return {Storage{CpuStorage{q_ptr, q_nbytes, dt}}, Storage{CpuStorage{r_ptr, r_nbytes, dt}}};
    }

    StoragePair linalg_eig(const Storage& a,
                           const Shape& shape,
                           const Shape& values_shape,
                           const Shape& vectors_shape,
                           Dtype dt) override {
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;
        const std::size_t per_w = static_cast<std::size_t>(n);
        const std::size_t values_nbytes = shape_numel(values_shape) * dtype_size(dt);
        const std::size_t vectors_nbytes = shape_numel(vectors_shape) * dtype_size(dt);
        auto values_ptr = allocate_aligned_bytes(values_nbytes, Device::CPU);
        auto vectors_ptr = allocate_aligned_bytes(vectors_nbytes, Device::CPU);
        const auto& cs = std::get<CpuStorage>(a);

        int info = 0;
        if (dt == Dtype::F32) {
            const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
            auto* w_p = reinterpret_cast<float*>(values_ptr.get());
            auto* v_p = reinterpret_cast<float*>(vectors_ptr.get());
            std::vector<float> wr(n), wi(n);
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_eig_f32(in_p + b * per_mat, n, wr.data(), wi.data(), v_p + b * per_mat,
                                    &info);
                check_lapack_info(info, "eig");
                std::memcpy(w_p + b * per_w, wr.data(), per_w * sizeof(float));
            }
        } else if (dt == Dtype::F64) {
            const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
            auto* w_p = reinterpret_cast<double*>(values_ptr.get());
            auto* v_p = reinterpret_cast<double*>(vectors_ptr.get());
            std::vector<double> wr(n), wi(n);
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_eig_f64(in_p + b * per_mat, n, wr.data(), wi.data(), v_p + b * per_mat,
                                    &info);
                check_lapack_info(info, "eig");
                std::memcpy(w_p + b * per_w, wr.data(), per_w * sizeof(double));
            }
        } else {
            ErrorBuilder("cpu_backend::linalg_eig").not_implemented("dtype not supported");
        }

        return {Storage{CpuStorage{values_ptr, values_nbytes, dt}},
                Storage{CpuStorage{vectors_ptr, vectors_nbytes, dt}}};
    }

    StoragePair linalg_eigh(const Storage& a,
                            const Shape& shape,
                            const Shape& values_shape,
                            const Shape& vectors_shape,
                            Dtype dt) override {
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;
        const std::size_t per_w = static_cast<std::size_t>(n);
        const std::size_t w_nb = shape_numel(values_shape) * dtype_size(dt);
        const std::size_t v_nb = shape_numel(vectors_shape) * dtype_size(dt);
        auto w_ptr = allocate_aligned_bytes(w_nb, Device::CPU);
        auto v_ptr = allocate_aligned_bytes(v_nb, Device::CPU);
        const auto& cs = std::get<CpuStorage>(a);
        int info = 0;
        if (dt == Dtype::F32) {
            const auto* in = reinterpret_cast<const float*>(cs.ptr.get());
            auto* wp = reinterpret_cast<float*>(w_ptr.get());
            auto* vp = reinterpret_cast<float*>(v_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b)
                cpu::lapack_eigh_f32(in + b * per_mat, n, wp + b * per_w, vp + b * per_mat, &info);
            check_lapack_info(info, "eigh");
        } else if (dt == Dtype::F64) {
            const auto* in = reinterpret_cast<const double*>(cs.ptr.get());
            auto* wp = reinterpret_cast<double*>(w_ptr.get());
            auto* vp = reinterpret_cast<double*>(v_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b)
                cpu::lapack_eigh_f64(in + b * per_mat, n, wp + b * per_w, vp + b * per_mat, &info);
            check_lapack_info(info, "eigh");
        } else {
            ErrorBuilder("cpu_backend::linalg_eigh").not_implemented("dtype not supported");
        }
        return {Storage{CpuStorage{w_ptr, w_nb, dt}}, Storage{CpuStorage{v_ptr, v_nb, dt}}};
    }

    std::vector<Storage> linalg_svd(const Storage& a,
                                    const Shape& shape,
                                    bool compute_uv,
                                    const Shape& u_shape,
                                    const Shape& s_shape,
                                    const Shape& vt_shape,
                                    Dtype dt) override {
        const int m = static_cast<int>(shape[shape.size() - 2]);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const int k = std::min(m, n);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t in_per = static_cast<std::size_t>(m) * n;
        const std::size_t s_per = static_cast<std::size_t>(k);
        const std::size_t u_per = static_cast<std::size_t>(m) * k;
        const std::size_t vt_per = static_cast<std::size_t>(k) * n;
        const std::size_t s_nbytes = shape_numel(s_shape) * dtype_size(dt);
        auto s_ptr = allocate_aligned_bytes(s_nbytes, Device::CPU);
        const auto& cs = std::get<CpuStorage>(a);

        if (!compute_uv) {
            int info = 0;
            if (dt == Dtype::F32) {
                std::vector<float> u(u_per), vt(vt_per);
                const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
                auto* s_p = reinterpret_cast<float*>(s_ptr.get());
                for (std::int64_t b = 0; b < batch; ++b) {
                    cpu::lapack_svd_f32(in_p + b * in_per, m, n, false, u.data(), s_p + b * s_per,
                                        vt.data(), &info);
                    check_lapack_info(info, "svd");
                }
            } else if (dt == Dtype::F64) {
                std::vector<double> u(u_per), vt(vt_per);
                const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
                auto* s_p = reinterpret_cast<double*>(s_ptr.get());
                for (std::int64_t b = 0; b < batch; ++b) {
                    cpu::lapack_svd_f64(in_p + b * in_per, m, n, false, u.data(), s_p + b * s_per,
                                        vt.data(), &info);
                    check_lapack_info(info, "svd");
                }
            } else {
                ErrorBuilder("cpu_backend::linalg_svd").not_implemented("dtype not supported");
            }
            return {Storage{CpuStorage{s_ptr, s_nbytes, dt}}};
        }

        const std::size_t u_nbytes = shape_numel(u_shape) * dtype_size(dt);
        const std::size_t vt_nbytes = shape_numel(vt_shape) * dtype_size(dt);
        auto u_ptr = allocate_aligned_bytes(u_nbytes, Device::CPU);
        auto vt_ptr = allocate_aligned_bytes(vt_nbytes, Device::CPU);
        int info = 0;
        if (dt == Dtype::F32) {
            const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
            auto* u_p = reinterpret_cast<float*>(u_ptr.get());
            auto* s_p = reinterpret_cast<float*>(s_ptr.get());
            auto* vt_p = reinterpret_cast<float*>(vt_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_svd_f32(in_p + b * in_per, m, n, false, u_p + b * u_per,
                                    s_p + b * s_per, vt_p + b * vt_per, &info);
                check_lapack_info(info, "svd");
            }
        } else if (dt == Dtype::F64) {
            const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
            auto* u_p = reinterpret_cast<double*>(u_ptr.get());
            auto* s_p = reinterpret_cast<double*>(s_ptr.get());
            auto* vt_p = reinterpret_cast<double*>(vt_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                cpu::lapack_svd_f64(in_p + b * in_per, m, n, false, u_p + b * u_per,
                                    s_p + b * s_per, vt_p + b * vt_per, &info);
                check_lapack_info(info, "svd");
            }
        } else {
            ErrorBuilder("cpu_backend::linalg_svd").not_implemented("dtype not supported");
        }
        return {Storage{CpuStorage{u_ptr, u_nbytes, dt}}, Storage{CpuStorage{s_ptr, s_nbytes, dt}},
                Storage{CpuStorage{vt_ptr, vt_nbytes, dt}}};
    }

    StoragePair linalg_lu_factor(const Storage& a, const Shape& shape, Dtype dt) override {
        if (dt != Dtype::F32 && dt != Dtype::F64)
            ErrorBuilder("cpu_backend::linalg_lu_factor").not_implemented("only F32/F64 supported");
        if (shape.size() < 2 || shape[shape.size()-1] != shape[shape.size()-2])
            ErrorBuilder("cpu_backend::linalg_lu_factor").fail("input must be square 2-D");
        const auto& cs = std::get<CpuStorage>(a);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;
        // LU output (same size as A)
        auto lu_ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
        // pivots: n int32_t per batch element
        const std::size_t ipiv_nbytes = static_cast<std::size_t>(batch) * n * sizeof(std::int32_t);
        auto ipiv_ptr = allocate_aligned_bytes(ipiv_nbytes, Device::CPU);
        int info = 0;
        auto* ipiv_out = reinterpret_cast<std::int32_t*>(ipiv_ptr.get());
        if (dt == Dtype::F32) {
            const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
            auto* lu_p = reinterpret_cast<float*>(lu_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                std::vector<int> ipiv_local(n);
                cpu::lapack_lu_factor_f32(in_p + b * per_mat, n,
                                          lu_p + b * per_mat,
                                          ipiv_local.data(), &info);
                check_lapack_info(info < 0 ? info : 0, "lu_factor");
                for (int i = 0; i < n; ++i)
                    ipiv_out[b * n + i] = static_cast<std::int32_t>(ipiv_local[i]);
            }
        } else {
            const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
            auto* lu_p = reinterpret_cast<double*>(lu_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                std::vector<int> ipiv_local(n);
                cpu::lapack_lu_factor_f64(in_p + b * per_mat, n,
                                          lu_p + b * per_mat,
                                          ipiv_local.data(), &info);
                check_lapack_info(info < 0 ? info : 0, "lu_factor");
                for (int i = 0; i < n; ++i)
                    ipiv_out[b * n + i] = static_cast<std::int32_t>(ipiv_local[i]);
            }
        }
        Storage lu_storage{CpuStorage{lu_ptr, cs.nbytes, dt}};
        Storage ipiv_storage{CpuStorage{ipiv_ptr, ipiv_nbytes, Dtype::I32}};
        return {lu_storage, ipiv_storage};
    }

    Storage linalg_solve_triangular(const Storage& a,
                                     const Storage& b,
                                     const Shape& a_shape,
                                     const Shape& b_shape,
                                     bool upper,
                                     bool unitriangular,
                                     Dtype dt) override {
        if (dt != Dtype::F32 && dt != Dtype::F64)
            ErrorBuilder("cpu_backend::linalg_solve_triangular").not_implemented("only F32/F64");
        const auto& a_cpu = std::get<CpuStorage>(a);
        const auto& b_cpu = std::get<CpuStorage>(b);
        const int n = static_cast<int>(a_shape[a_shape.size() - 1]);
        const bool b_is_vec = (b_shape.size() == a_shape.size() - 1);
        const int nrhs = b_is_vec ? 1 : static_cast<int>(b_shape[b_shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(a_shape, 2);
        const std::size_t b_per = static_cast<std::size_t>(n) * nrhs;
        // Copy B for in-place overwrite
        auto out_ptr = allocate_aligned_bytes(b_cpu.nbytes, Device::CPU);
        if (b_cpu.nbytes > 0)
            std::memcpy(out_ptr.get(), b_cpu.ptr.get(), b_cpu.nbytes);
        int info = 0;
        if (dt == Dtype::F32) {
            const auto* a_p = reinterpret_cast<const float*>(a_cpu.ptr.get());
            auto* x_p = reinterpret_cast<float*>(out_ptr.get());
            const std::size_t a_per = static_cast<std::size_t>(n) * n;
            for (std::int64_t bi = 0; bi < batch; ++bi) {
                cpu::lapack_solve_triangular_f32(a_p + bi * a_per, x_p + bi * b_per,
                                                 n, nrhs, upper, unitriangular, &info);
                check_lapack_info(info, "solve_triangular");
            }
        } else {
            const auto* a_p = reinterpret_cast<const double*>(a_cpu.ptr.get());
            auto* x_p = reinterpret_cast<double*>(out_ptr.get());
            const std::size_t a_per = static_cast<std::size_t>(n) * n;
            for (std::int64_t bi = 0; bi < batch; ++bi) {
                cpu::lapack_solve_triangular_f64(a_p + bi * a_per, x_p + bi * b_per,
                                                 n, nrhs, upper, unitriangular, &info);
                check_lapack_info(info, "solve_triangular");
            }
        }
        return Storage{CpuStorage{out_ptr, b_cpu.nbytes, dt}};
    }

    std::vector<Storage> linalg_lstsq(const Storage& a,
                                       const Storage& b,
                                       const Shape& a_shape,
                                       const Shape& b_shape,
                                       Dtype dt) override {
        if (dt != Dtype::F32 && dt != Dtype::F64)
            ErrorBuilder("cpu_backend::linalg_lstsq").not_implemented("only F32/F64");
        const auto& a_cpu = std::get<CpuStorage>(a);
        const auto& b_cpu = std::get<CpuStorage>(b);
        const int m = static_cast<int>(a_shape[a_shape.size() - 2]);
        const int n = static_cast<int>(a_shape[a_shape.size() - 1]);
        const int nrhs = (b_shape.size() > 1) ? static_cast<int>(b_shape[b_shape.size() - 1]) : 1;
        const int ldb = std::max(m, n);

        // A copy (sgels overwrites A)
        std::size_t a_nb = static_cast<std::size_t>(m) * n * dtype_size(dt);
        std::size_t b_nb = static_cast<std::size_t>(ldb) * nrhs * dtype_size(dt);
        auto a_ptr = allocate_aligned_bytes(a_nb, Device::CPU);
        auto b_ptr = allocate_aligned_bytes(b_nb, Device::CPU);
        std::memset(b_ptr.get(), 0, b_nb);  // zero-pad to max(m,n)

        int info = 0;
        if (dt == Dtype::F32) {
            std::memcpy(a_ptr.get(), a_cpu.ptr.get(), a_nb);
            // Copy B into first m rows
            auto* bp = reinterpret_cast<float*>(b_ptr.get());
            const auto* bsrc = reinterpret_cast<const float*>(b_cpu.ptr.get());
            for (int r = 0; r < m; ++r)
                for (int c = 0; c < nrhs; ++c)
                    bp[r * nrhs + c] = bsrc[r * nrhs + c];
            cpu::lapack_lstsq_f32(reinterpret_cast<float*>(a_ptr.get()), bp, m, n, nrhs, &info);
        } else {
            std::memcpy(a_ptr.get(), a_cpu.ptr.get(), a_nb);
            auto* bp = reinterpret_cast<double*>(b_ptr.get());
            const auto* bsrc = reinterpret_cast<const double*>(b_cpu.ptr.get());
            for (int r = 0; r < m; ++r)
                for (int c = 0; c < nrhs; ++c)
                    bp[r * nrhs + c] = bsrc[r * nrhs + c];
            cpu::lapack_lstsq_f64(reinterpret_cast<double*>(a_ptr.get()), bp, m, n, nrhs, &info);
        }
        check_lapack_info(info, "lstsq");

        // Solution shape: (n, nrhs) — first n rows of B
        std::size_t sol_nb = static_cast<std::size_t>(n) * nrhs * dtype_size(dt);
        auto sol_ptr = allocate_aligned_bytes(sol_nb, Device::CPU);
        std::memcpy(sol_ptr.get(), b_ptr.get(), sol_nb);

        return {Storage{CpuStorage{sol_ptr, sol_nb, dt}}};
    }

    Storage linalg_lu_solve(const Storage& LU,
                             const Storage& pivots,
                             const Storage& b,
                             const Shape& lu_shape,
                             const Shape& b_shape,
                             Dtype dt) override {
        if (dt != Dtype::F32 && dt != Dtype::F64)
            ErrorBuilder("cpu_backend::linalg_lu_solve").not_implemented("only F32/F64");
        const auto& lu_cpu  = std::get<CpuStorage>(LU);
        const auto& piv_cpu = std::get<CpuStorage>(pivots);
        const auto& b_cpu   = std::get<CpuStorage>(b);
        const int n    = static_cast<int>(lu_shape[lu_shape.size() - 1]);
        const int nrhs = (b_shape.size() > 1) ? static_cast<int>(b_shape[b_shape.size() - 1]) : 1;
        const std::int64_t batch = leading_matrix_batch_count(lu_shape, 2);

        auto out_ptr = allocate_aligned_bytes(b_cpu.nbytes, Device::CPU);
        std::memcpy(out_ptr.get(), b_cpu.ptr.get(), b_cpu.nbytes);

        const std::size_t lu_per = static_cast<std::size_t>(n) * n;
        const std::size_t b_per  = static_cast<std::size_t>(n) * nrhs;
        const auto* ipiv = reinterpret_cast<const int*>(piv_cpu.ptr.get());
        int info = 0;
        if (dt == Dtype::F32) {
            const auto* lup = reinterpret_cast<const float*>(lu_cpu.ptr.get());
            auto* xp = reinterpret_cast<float*>(out_ptr.get());
            for (std::int64_t bi = 0; bi < batch; ++bi)
                cpu::lapack_lu_solve_f32(lup + bi * lu_per, ipiv + bi * n,
                                          xp + bi * b_per, n, nrhs, &info);
        } else {
            const auto* lup = reinterpret_cast<const double*>(lu_cpu.ptr.get());
            auto* xp = reinterpret_cast<double*>(out_ptr.get());
            for (std::int64_t bi = 0; bi < batch; ++bi)
                cpu::lapack_lu_solve_f64(lup + bi * lu_per, ipiv + bi * n,
                                          xp + bi * b_per, n, nrhs, &info);
        }
        check_lapack_info(info, "lu_solve");
        return Storage{CpuStorage{out_ptr, b_cpu.nbytes, dt}};
    }

    Storage linalg_householder_product(const Storage& H,
                                        const Storage& tau,
                                        const Shape& h_shape,
                                        Dtype dt) override {
        if (dt != Dtype::F32 && dt != Dtype::F64)
            ErrorBuilder("cpu_backend::linalg_householder_product").not_implemented("only F32/F64");
        const auto& h_cpu   = std::get<CpuStorage>(H);
        const auto& tau_cpu = std::get<CpuStorage>(tau);
        const int m = static_cast<int>(h_shape[h_shape.size() - 2]);
        const int n = static_cast<int>(h_shape[h_shape.size() - 1]);
        const int k = std::min(m, n);

        // Q shape: m × k
        const std::size_t q_nb = static_cast<std::size_t>(m) * k * dtype_size(dt);
        auto q_ptr = allocate_aligned_bytes(q_nb, Device::CPU);
        int info = 0;
        if (dt == Dtype::F32) {
            cpu::lapack_householder_product_f32(
                reinterpret_cast<const float*>(h_cpu.ptr.get()),
                reinterpret_cast<const float*>(tau_cpu.ptr.get()),
                reinterpret_cast<float*>(q_ptr.get()), m, n, k, &info);
        } else {
            cpu::lapack_householder_product_f64(
                reinterpret_cast<const double*>(h_cpu.ptr.get()),
                reinterpret_cast<const double*>(tau_cpu.ptr.get()),
                reinterpret_cast<double*>(q_ptr.get()), m, n, k, &info);
        }
        check_lapack_info(info, "householder_product");
        return Storage{CpuStorage{q_ptr, q_nb, dt}};
    }

    StoragePair linalg_ldl_factor(const Storage& a, const Shape& shape, Dtype dt) override {
        if (dt != Dtype::F32 && dt != Dtype::F64)
            ErrorBuilder("cpu_backend::linalg_ldl_factor").not_implemented("only F32/F64");
        const auto& cs = std::get<CpuStorage>(a);
        const int n = static_cast<int>(shape[shape.size() - 1]);
        const std::int64_t batch = leading_matrix_batch_count(shape, 2);
        const std::size_t per_mat = static_cast<std::size_t>(n) * n;

        auto ld_ptr   = allocate_aligned_bytes(cs.nbytes, Device::CPU);
        auto piv_nb   = static_cast<std::size_t>(batch) * n * sizeof(std::int32_t);
        auto piv_ptr  = allocate_aligned_bytes(piv_nb, Device::CPU);
        auto* piv_out = reinterpret_cast<int*>(piv_ptr.get());

        int info = 0;
        if (dt == Dtype::F32) {
            const auto* src = reinterpret_cast<const float*>(cs.ptr.get());
            auto* dst = reinterpret_cast<float*>(ld_ptr.get());
            for (std::int64_t bi = 0; bi < batch; ++bi)
                cpu::lapack_ldl_factor_f32(src + bi * per_mat, dst + bi * per_mat,
                                            piv_out + bi * n, n, &info);
        } else {
            const auto* src = reinterpret_cast<const double*>(cs.ptr.get());
            auto* dst = reinterpret_cast<double*>(ld_ptr.get());
            for (std::int64_t bi = 0; bi < batch; ++bi)
                cpu::lapack_ldl_factor_f64(src + bi * per_mat, dst + bi * per_mat,
                                            piv_out + bi * n, n, &info);
        }
        check_lapack_info(info, "ldl_factor");
        return {Storage{CpuStorage{ld_ptr,  cs.nbytes, dt}},
                Storage{CpuStorage{piv_ptr, piv_nb,    Dtype::I32}}};
    }

    // fold (col2im): scatter-add (N, C*kH*kW, L) → (N, C, outH, outW)
    Storage nn_fold(const Storage& x,
                     const Shape& x_shape,
                     const Shape& out_shape,
                     const std::vector<int>& kernel_size,
                     const std::vector<int>& stride,
                     const std::vector<int>& padding,
                     const std::vector<int>& dilation,
                     Dtype dt) override {
        const auto& cx = std::get<CpuStorage>(x);
        const int N    = static_cast<int>(x_shape[0]);
        const int CKK  = static_cast<int>(x_shape[1]);
        const int L    = static_cast<int>(x_shape[2]);
        const int kH   = kernel_size[0], kW = kernel_size[1];
        const int sH   = stride[0],      sW = stride[1];
        const int pH   = padding[0],     pW = padding[1];
        const int dH   = dilation[0],    dW = dilation[1];
        const int C    = CKK / (kH * kW);
        const int outH = static_cast<int>(out_shape[2]);
        const int outW = static_cast<int>(out_shape[3]);

        // Number of output positions
        const int H_pad = outH + 2 * pH;
        const int W_pad = outW + 2 * pW;
        const std::size_t total_out = static_cast<std::size_t>(N) * C * outH * outW;
        auto out_ptr = allocate_aligned_bytes(total_out * dtype_size(dt), Device::CPU);
        std::memset(out_ptr.get(), 0, total_out * dtype_size(dt));

        auto run = [&](auto* op, const auto* xp) {
            int l_idx = 0;
            for (int oh = 0; oh < (H_pad - kH) / sH + 1; ++oh) {
                for (int ow = 0; ow < (W_pad - kW) / sW + 1; ++ow) {
                    for (int n = 0; n < N; ++n) {
                        for (int c = 0; c < C; ++c) {
                            for (int ki = 0; ki < kH; ++ki) {
                                for (int kj = 0; kj < kW; ++kj) {
                                    const int ri = oh * sH + ki * dH - pH;
                                    const int ci = ow * sW + kj * dW - pW;
                                    if (ri < 0 || ri >= outH || ci < 0 || ci >= outW) continue;
                                    const int src_c = c * kH * kW + ki * kW + kj;
                                    const int src_flat = n * CKK * L + src_c * L + l_idx;
                                    const int dst_flat = n * C * outH * outW + c * outH * outW
                                                        + ri * outW + ci;
                                    op[dst_flat] += xp[src_flat];
                                }
                            }
                        }
                    }
                    ++l_idx;
                }
            }
        };

        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(out_ptr.get()),
                reinterpret_cast<const float*>(cx.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(out_ptr.get()),
                reinterpret_cast<const double*>(cx.ptr.get()));
        else ErrorBuilder("nn_fold").not_implemented("only F32/F64");
        return Storage{CpuStorage{out_ptr, total_out * dtype_size(dt), dt}};
    }

    // EmbeddingBag: gather + reduce per bag
    Storage embedding_bag_forward(const Storage& weight,
                                   const Storage& indices,
                                   const Storage& offsets,
                                   const Shape& weight_shape,
                                   const Shape& indices_shape,
                                   int mode,
                                   int padding_idx,
                                   bool include_last_offset,
                                   Dtype dt) override {
        const auto& cw = std::get<CpuStorage>(weight);
        const auto& ci = std::get<CpuStorage>(indices);
        const auto& co = std::get<CpuStorage>(offsets);

        const int num_emb = static_cast<int>(weight_shape[0]);
        const int D       = static_cast<int>(weight_shape[1]);
        const int n_idx   = static_cast<int>(shape_numel(indices_shape));
        const auto* idx_p = reinterpret_cast<const std::int32_t*>(ci.ptr.get());
        const auto* off_p = reinterpret_cast<const std::int32_t*>(co.ptr.get());
        const int B       = static_cast<int>(co.nbytes / sizeof(std::int32_t));

        // Determine bag boundaries
        std::vector<int> starts(static_cast<std::size_t>(B));
        std::vector<int> ends(static_cast<std::size_t>(B));
        for (int b = 0; b < B; ++b) {
            starts[static_cast<std::size_t>(b)] = static_cast<int>(off_p[b]);
            ends[static_cast<std::size_t>(b)] = (b + 1 < B && !include_last_offset)
                                                  ? static_cast<int>(off_p[b + 1])
                                                  : n_idx;
        }
        if (include_last_offset && B > 0)
            ends[static_cast<std::size_t>(B - 1)] = static_cast<int>(off_p[B - 1]);

        std::size_t out_nb = static_cast<std::size_t>(B) * D * dtype_size(dt);
        auto out_ptr = allocate_aligned_bytes(out_nb, Device::CPU);
        std::memset(out_ptr.get(), 0, out_nb);

        const std::size_t esz = dtype_size(dt);
        const std::uint8_t* wp = static_cast<const std::uint8_t*>(
            static_cast<const void*>(cw.ptr.get()));

        auto run = [&](auto* op) {
            using T = std::remove_pointer_t<decltype(op)>;
            for (int b = 0; b < B; ++b) {
                int s = starts[static_cast<std::size_t>(b)];
                int e = ends[static_cast<std::size_t>(b)];
                T* row = op + b * D;
                int count = 0;
                for (int k = s; k < e; ++k) {
                    int emb = static_cast<int>(idx_p[k]);
                    if (emb == padding_idx) continue;
                    if (emb < 0 || emb >= num_emb) continue;
                    const T* src = reinterpret_cast<const T*>(wp + emb * D * esz);
                    if (mode == 2) {  // max
                        for (int d = 0; d < D; ++d)
                            if (src[d] > row[d]) row[d] = src[d];
                    } else {          // sum / mean
                        for (int d = 0; d < D; ++d) row[d] += src[d];
                    }
                    ++count;
                }
                if (mode == 1 && count > 0) {  // mean
                    for (int d = 0; d < D; ++d) row[d] /= static_cast<T>(count);
                }
            }
        };

        if (dt == Dtype::F32) run(reinterpret_cast<float*>(out_ptr.get()));
        else if (dt == Dtype::F64) run(reinterpret_cast<double*>(out_ptr.get()));
        else ErrorBuilder("embedding_bag").not_implemented("only F32/F64");
        return Storage{CpuStorage{out_ptr, out_nb, dt}};
    }

    // ── astype ────────────────────────────────────────────────────────────────
    Storage astype(const Storage& a, const Shape& shape,
                   Dtype src_dt, Dtype dst_dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const std::size_t n   = shape_numel(shape);
        const std::size_t dsz = dtype_size(dst_dt);
        auto out_ptr = allocate_aligned_bytes(n * dsz, Device::CPU);

        // Template cast: read as From, write as To.
        auto run = [&]<typename From, typename To>() {
            const From* src = reinterpret_cast<const From*>(ca.ptr.get());
            To*         dst = reinterpret_cast<To*>(out_ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = static_cast<To>(src[i]);
        };

        // Dispatch on (src_dt, dst_dt) pair.
#define CPU_CAST(F, T) run.template operator()<F, T>()
        using I8 = std::int8_t; using I16 = std::int16_t;
        using I32 = std::int32_t; using I64 = std::int64_t;
        bool ok = true;
        switch (src_dt) {
            case Dtype::F32:
                switch (dst_dt) {
                    case Dtype::F64: CPU_CAST(float,double);  break;
                    case Dtype::I8:  CPU_CAST(float,I8);      break;
                    case Dtype::I16: CPU_CAST(float,I16);     break;
                    case Dtype::I32: CPU_CAST(float,I32);     break;
                    case Dtype::I64: CPU_CAST(float,I64);     break;
                    case Dtype::Bool:CPU_CAST(float,bool);    break;
                    default: ok=false;
                } break;
            case Dtype::F64:
                switch (dst_dt) {
                    case Dtype::F32: CPU_CAST(double,float);  break;
                    case Dtype::I32: CPU_CAST(double,I32);    break;
                    case Dtype::I64: CPU_CAST(double,I64);    break;
                    default: ok=false;
                } break;
            case Dtype::I32:
                switch (dst_dt) {
                    case Dtype::F32: CPU_CAST(I32,float);     break;
                    case Dtype::F64: CPU_CAST(I32,double);    break;
                    case Dtype::I64: CPU_CAST(I32,I64);       break;
                    case Dtype::I16: CPU_CAST(I32,I16);       break;
                    case Dtype::I8:  CPU_CAST(I32,I8);        break;
                    case Dtype::Bool:CPU_CAST(I32,bool);      break;
                    default: ok=false;
                } break;
            case Dtype::I64:
                switch (dst_dt) {
                    case Dtype::F32: CPU_CAST(I64,float);     break;
                    case Dtype::F64: CPU_CAST(I64,double);    break;
                    case Dtype::I32: CPU_CAST(I64,I32);       break;
                    case Dtype::I16: CPU_CAST(I64,I16);       break;
                    default: ok=false;
                } break;
            case Dtype::I16:
                switch (dst_dt) {
                    case Dtype::F32: CPU_CAST(I16,float);     break;
                    case Dtype::I32: CPU_CAST(I16,I32);       break;
                    default: ok=false;
                } break;
            case Dtype::I8:
                switch (dst_dt) {
                    case Dtype::F32: CPU_CAST(I8,float);      break;
                    case Dtype::I32: CPU_CAST(I8,I32);        break;
                    default: ok=false;
                } break;
            case Dtype::Bool:
                switch (dst_dt) {
                    case Dtype::F32: CPU_CAST(bool,float);    break;
                    case Dtype::I32: CPU_CAST(bool,I32);      break;
                    default: ok=false;
                } break;
            default: ok=false;
        }
#undef CPU_CAST
        if (!ok)
            ErrorBuilder("astype").not_implemented(
                std::string(dtype_name(src_dt)) + " -> " + std::string(dtype_name(dst_dt)));
        return Storage{CpuStorage{out_ptr, n * dsz, dst_dt}};
    }

    // ── flip ──────────────────────────────────────────────────────────────────
    Storage flip(const Storage& a, const Shape& shape,
                 const std::vector<int>& dims, Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const std::size_t n     = shape_numel(shape);
        const std::size_t esz   = dtype_size(dt);
        const std::size_t nb    = n * esz;
        auto out_ptr = allocate_aligned_bytes(nb, Device::CPU);

        // Compute strides (row-major).
        const int ndim = static_cast<int>(shape.size());
        std::vector<std::size_t> strides(ndim, 1);
        for (int d = ndim - 2; d >= 0; --d)
            strides[d] = strides[d + 1] * static_cast<std::size_t>(shape[d + 1]);

        // Build a set of dims to flip.
        std::vector<bool> flip_dim(ndim, false);
        for (int d : dims) flip_dim[d] = true;

        const std::uint8_t* src = static_cast<const std::uint8_t*>(
            static_cast<const void*>(ca.ptr.get()));
        std::uint8_t* dst = static_cast<std::uint8_t*>(
            static_cast<void*>(out_ptr.get()));

        // Iterate over all elements; for each, compute destination index after flipping.
        for (std::size_t flat = 0; flat < n; ++flat) {
            std::size_t rem = flat;
            std::size_t src_idx = 0;
            for (int d = 0; d < ndim; ++d) {
                std::size_t coord = rem / strides[d];
                rem %= strides[d];
                std::size_t flipped = flip_dim[d]
                    ? static_cast<std::size_t>(shape[d]) - 1 - coord
                    : coord;
                src_idx += flipped * strides[d];
            }
            std::memcpy(dst + flat * esz, src + src_idx * esz, esz);
        }
        return Storage{CpuStorage{out_ptr, nb, dt}};
    }

    // ── masked_select ─────────────────────────────────────────────────────────
    Storage masked_select_count(const Storage& mask,
                                const Shape& shape, Dtype /*dt*/) override {
        const auto& cm = std::get<CpuStorage>(mask);
        const std::size_t n = shape_numel(shape);
        const bool* mp = reinterpret_cast<const bool*>(cm.ptr.get());
        std::int64_t count = 0;
        for (std::size_t i = 0; i < n; ++i) if (mp[i]) ++count;
        auto out_ptr = allocate_aligned_bytes(sizeof(std::int64_t), Device::CPU);
        std::memcpy(out_ptr.get(), &count, sizeof(std::int64_t));
        return Storage{CpuStorage{out_ptr, sizeof(std::int64_t), Dtype::I64}};
    }

    Storage masked_select(const Storage& a, const Storage& mask,
                           const Shape& a_shape, const Shape& /*mask_shape*/,
                           std::int64_t n_true, Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cm = std::get<CpuStorage>(mask);
        const std::size_t n   = shape_numel(a_shape);
        const std::size_t esz = dtype_size(dt);
        const std::size_t nb  = static_cast<std::size_t>(n_true) * esz;
        auto out_ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::uint8_t* src = static_cast<const std::uint8_t*>(
            static_cast<const void*>(ca.ptr.get()));
        const bool* mp = reinterpret_cast<const bool*>(cm.ptr.get());
        std::uint8_t* dst = static_cast<std::uint8_t*>(
            static_cast<void*>(out_ptr.get()));
        std::size_t out_idx = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (mp[i]) {
                std::memcpy(dst + out_idx * esz, src + i * esz, esz);
                ++out_idx;
            }
        }
        return Storage{CpuStorage{out_ptr, nb, dt}};
    }

    // ── ctc_loss ──────────────────────────────────────────────────────────────
    Storage ctc_loss_forward(const Storage& log_probs,
                              const Storage& targets,
                              const Storage& input_lengths,
                              const Storage& target_lengths,
                              const Shape& lp_shape,
                              int blank,
                              bool zero_infinity,
                              Dtype dt) override {
        // lp_shape: [T, N, C]
        const int N = static_cast<int>(lp_shape[1]);
        const int C = static_cast<int>(lp_shape[2]);

        const auto& clp  = std::get<CpuStorage>(log_probs);
        const auto& ctgt = std::get<CpuStorage>(targets);
        const auto& cil  = std::get<CpuStorage>(input_lengths);
        const auto& ctl  = std::get<CpuStorage>(target_lengths);

        // Input lengths and target lengths as int arrays.
        auto get_i32 = [](const CpuStorage& s, int b) -> int {
            return static_cast<int>(reinterpret_cast<const std::int32_t*>(s.ptr.get())[b]);
        };

        // log_probs layout: [T * N * C], row-major.
        auto lp = [&](int t, int b, int c) -> double {
            if (dt == Dtype::F32) {
                return static_cast<double>(
                    reinterpret_cast<const float*>(clp.ptr.get())[t * N * C + b * C + c]);
            }
            return reinterpret_cast<const double*>(clp.ptr.get())[t * N * C + b * C + c];
        };
        // targets layout: flat (sum_S,) as int32.
        auto tgt_ptr = reinterpret_cast<const std::int32_t*>(ctgt.ptr.get());

        constexpr double NEG_INF = -1e30;
        auto logaddexp = [](double a, double b) -> double {
            if (a <= NEG_INF / 2) return b;
            if (b <= NEG_INF / 2) return a;
            double hi = std::max(a, b);
            return hi + std::log1p(std::exp(std::min(a, b) - hi));
        };

        const std::size_t out_nb = static_cast<std::size_t>(N) * dtype_size(dt);
        auto out_ptr = allocate_aligned_bytes(out_nb, Device::CPU);

        int tgt_offset = 0;
        for (int b = 0; b < N; ++b) {
            const int T_b = get_i32(cil, b);
            const int S   = get_i32(ctl, b);
            const int L   = 2 * S + 1;

            // Extended target: blank, t[0], blank, t[1], ..., blank
            std::vector<int> ext(L, blank);
            for (int s = 0; s < S; ++s)
                ext[2 * s + 1] = static_cast<int>(tgt_ptr[tgt_offset + s]);
            tgt_offset += S;

            // Forward variable alpha[T_b x L] in log-domain.
            std::vector<double> alpha(T_b * L, NEG_INF);
            auto at = [&](int t, int s) -> double& { return alpha[t * L + s]; };

            at(0, 0) = lp(0, b, ext[0]);
            if (L > 1) at(0, 1) = lp(0, b, ext[1]);

            for (int t = 1; t < T_b; ++t) {
                for (int s = 0; s < L; ++s) {
                    double a = at(t - 1, s);
                    if (s > 0)
                        a = logaddexp(a, at(t - 1, s - 1));
                    if (s > 1 && ext[s] != ext[s - 2])
                        a = logaddexp(a, at(t - 1, s - 2));
                    at(t, s) = a + lp(t, b, ext[s]);
                }
            }

            double end = at(T_b - 1, L - 1);
            if (L >= 2) end = logaddexp(end, at(T_b - 1, L - 2));
            double v = -end;
            if (zero_infinity && (!std::isfinite(v))) v = 0.0;

            if (dt == Dtype::F32) {
                reinterpret_cast<float*>(out_ptr.get())[b] = static_cast<float>(v);
            } else {
                reinterpret_cast<double*>(out_ptr.get())[b] = v;
            }
        }
        return Storage{CpuStorage{out_ptr, out_nb, dt}};
    }

    Storage
    broadcast(const Storage& a, const Shape& src_shape, const Shape& dst_shape, Dtype dt) override {
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

    Storage
    permute(const Storage& a, const Shape& shape, const std::vector<int>& perm, Dtype dt) override {
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
        if (constant == 0.0) {
            std::memset(ptr.get(), 0, nb);
        } else {
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

    Storage
    clip(const Storage& a, const Shape& shape, Dtype dt, double min_v, double max_v) override {
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
                                              const Shape&,
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
                                    const Shape&,
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

    CpuStorage to_cpu(const Storage& a, const Shape&) override { return std::get<CpuStorage>(a); }

    std::vector<Storage> sdpa_forward(const Storage& q,
                                      const Storage& k,
                                      const Storage& v,
                                      const Storage* attn_mask,
                                      const Shape& q_shape,
                                      const Shape& k_shape,
                                      const Shape& v_shape,
                                      Dtype mask_dtype,
                                      std::size_t mask_numel,
                                      double scale,
                                      bool is_causal,
                                      Dtype dt) override {
        std::size_t B = 1;
        for (std::size_t i = 0; i + 2 < q_shape.size(); ++i)
            B *= static_cast<std::size_t>(q_shape[i]);
        const std::size_t Lq = static_cast<std::size_t>(q_shape[q_shape.size() - 2]);
        const std::size_t Dk = static_cast<std::size_t>(q_shape.back());
        const std::size_t Lk = static_cast<std::size_t>(k_shape[k_shape.size() - 2]);
        const std::size_t Dv = static_cast<std::size_t>(v_shape.back());

        const std::size_t weights_numel = B * Lq * Lk;
        const std::size_t output_numel = B * Lq * Dv;

        CpuStorage weights_cpu = alloc_cpu(weights_numel, dt);
        CpuStorage output_cpu = alloc_cpu(output_numel, dt);

        const auto& qs = std::get<CpuStorage>(q);
        const auto& ks = std::get<CpuStorage>(k);
        const auto& vs = std::get<CpuStorage>(v);

        auto apply_masks = [&](auto* Wp_t) {
            using T = std::remove_pointer_t<decltype(Wp_t)>;
            const T neg_inf = -std::numeric_limits<T>::infinity();
            const std::size_t pb = Lq * Lk;
            if (attn_mask) {
                const auto& ms = std::get<CpuStorage>(*attn_mask);
                if (mask_dtype == Dtype::Bool) {
                    const auto* mp = reinterpret_cast<const std::uint8_t*>(ms.ptr.get());
                    if (mask_numel == pb) {
                        for (std::size_t bb = 0; bb < B; ++bb) {
                            T* sb = Wp_t + bb * pb;
                            for (std::size_t i = 0; i < pb; ++i)
                                if (mp[i])
                                    sb[i] = neg_inf;
                        }
                    } else {
                        for (std::size_t i = 0; i < B * pb; ++i)
                            if (mp[i])
                                Wp_t[i] = neg_inf;
                    }
                } else {
                    const auto* mp = reinterpret_cast<const T*>(ms.ptr.get());
                    if (mask_numel == pb) {
                        for (std::size_t bb = 0; bb < B; ++bb) {
                            T* sb = Wp_t + bb * pb;
                            for (std::size_t i = 0; i < pb; ++i)
                                sb[i] += mp[i];
                        }
                    } else {
                        for (std::size_t i = 0; i < B * pb; ++i)
                            Wp_t[i] += mp[i];
                    }
                }
            }
            if (is_causal) {
                for (std::size_t b = 0; b < B; ++b)
                    for (std::size_t i = 0; i < Lq; ++i) {
                        T* row = Wp_t + (b * Lq + i) * Lk;
                        for (std::size_t j = i + 1; j < Lk; ++j)
                            row[j] = neg_inf;
                    }
            }
        };

        if (dt == Dtype::F32) {
            const float* Qp = reinterpret_cast<const float*>(qs.ptr.get());
            const float* Kp = reinterpret_cast<const float*>(ks.ptr.get());
            const float* Vp = reinterpret_cast<const float*>(vs.ptr.get());
            float* Wp = reinterpret_cast<float*>(weights_cpu.ptr.get());
            float* Op = reinterpret_cast<float*>(output_cpu.ptr.get());
            const float sc = static_cast<float>(scale);

            for (std::size_t b = 0; b < B; ++b)
                cpu::sgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                           static_cast<int>(Dk), sc, Qp + b * Lq * Dk, static_cast<int>(Dk),
                           Kp + b * Lk * Dk, static_cast<int>(Dk), 0.0f, Wp + b * Lq * Lk,
                           static_cast<int>(Lk));
            apply_masks(Wp);

            for (std::size_t r = 0; r < B * Lq; ++r) {
                float* row = Wp + r * Lk;
                const float m = cpu::vmaxval_f32(row, Lk);
                if (!std::isfinite(m)) {
                    std::memset(row, 0, Lk * sizeof(float));
                    continue;
                }
                cpu::vsadd_f32(row, -m, row, Lk);
                cpu::vexp_f32(row, row, Lk);
                const float s = cpu::vsum_f32(row, Lk);
                cpu::vsmul_f32(row, s > 0.f ? 1.f / s : 0.f, row, Lk);
            }

            for (std::size_t b = 0; b < B; ++b)
                cpu::sgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dv),
                           static_cast<int>(Lk), 1.0f, Wp + b * Lq * Lk, static_cast<int>(Lk),
                           Vp + b * Lk * Dv, static_cast<int>(Dv), 0.0f, Op + b * Lq * Dv,
                           static_cast<int>(Dv));
        } else if (dt == Dtype::F64) {
            const double* Qp = reinterpret_cast<const double*>(qs.ptr.get());
            const double* Kp = reinterpret_cast<const double*>(ks.ptr.get());
            const double* Vp = reinterpret_cast<const double*>(vs.ptr.get());
            double* Wp = reinterpret_cast<double*>(weights_cpu.ptr.get());
            double* Op = reinterpret_cast<double*>(output_cpu.ptr.get());
            const double sc = scale;
            for (std::size_t b = 0; b < B; ++b)
                cpu::dgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                           static_cast<int>(Dk), sc, Qp + b * Lq * Dk, static_cast<int>(Dk),
                           Kp + b * Lk * Dk, static_cast<int>(Dk), 0.0, Wp + b * Lq * Lk,
                           static_cast<int>(Lk));
            apply_masks(Wp);

            for (std::size_t r = 0; r < B * Lq; ++r) {
                double* row = Wp + r * Lk;
                double m = row[0];
                for (std::size_t j = 1; j < Lk; ++j)
                    if (row[j] > m)
                        m = row[j];
                if (!std::isfinite(m)) {
                    for (std::size_t j = 0; j < Lk; ++j)
                        row[j] = 0.0;
                    continue;
                }
                double s = 0.0;
                for (std::size_t j = 0; j < Lk; ++j) {
                    row[j] = std::exp(row[j] - m);
                    s += row[j];
                }
                const double inv = s > 0.0 ? 1.0 / s : 0.0;
                for (std::size_t j = 0; j < Lk; ++j)
                    row[j] *= inv;
            }
            for (std::size_t b = 0; b < B; ++b)
                cpu::dgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dv),
                           static_cast<int>(Lk), 1.0, Wp + b * Lq * Lk, static_cast<int>(Lk),
                           Vp + b * Lk * Dv, static_cast<int>(Dv), 0.0, Op + b * Lq * Dv,
                           static_cast<int>(Dv));
        } else {
            ErrorBuilder("cpu_backend::sdpa_forward")
                .not_implemented("dtype not supported (F32/F64 only)");
        }

        return {Storage{std::move(weights_cpu)}, Storage{std::move(output_cpu)}};
    }

    std::vector<Storage> sdpa_backward(const Storage& grad_out,
                                       const Storage& q,
                                       const Storage& k,
                                       const Storage& v,
                                       const Storage& saved_weights,
                                       const Shape& q_shape,
                                       const Shape& k_shape,
                                       const Shape& v_shape,
                                       double scale,
                                       Dtype dt) override {
        std::size_t B = 1;
        for (std::size_t i = 0; i + 2 < q_shape.size(); ++i)
            B *= static_cast<std::size_t>(q_shape[i]);
        const std::size_t Lq = static_cast<std::size_t>(q_shape[q_shape.size() - 2]);
        const std::size_t Dk = static_cast<std::size_t>(q_shape.back());
        const std::size_t Lk = static_cast<std::size_t>(k_shape[k_shape.size() - 2]);
        const std::size_t Dv = static_cast<std::size_t>(v_shape.back());

        CpuStorage dQ_cpu = alloc_cpu(B * Lq * Dk, dt);
        CpuStorage dK_cpu = alloc_cpu(B * Lk * Dk, dt);
        CpuStorage dV_cpu = alloc_cpu(B * Lk * Dv, dt);

        const auto& qs = std::get<CpuStorage>(q);
        const auto& ks = std::get<CpuStorage>(k);
        const auto& vs = std::get<CpuStorage>(v);
        const auto& ws = std::get<CpuStorage>(saved_weights);
        const auto& gs = std::get<CpuStorage>(grad_out);

        auto softmax_bwd = [&](auto* Wp_t, const auto* Gp_t, auto* out_t, std::size_t lq,
                               std::size_t lk) {
            using T = std::remove_pointer_t<decltype(Wp_t)>;
            std::vector<T> dw(lq * lk), ds(lq * lk);

            (void)lq;
            (void)lk;
            (void)out_t;
        };
        (void)softmax_bwd;

        auto do_backward = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T sc = static_cast<T>(scale);
            const T* Qp = reinterpret_cast<const T*>(qs.ptr.get());
            const T* Kp = reinterpret_cast<const T*>(ks.ptr.get());
            const T* Vp = reinterpret_cast<const T*>(vs.ptr.get());
            const T* Wp = reinterpret_cast<const T*>(ws.ptr.get());
            const T* Gp = reinterpret_cast<const T*>(gs.ptr.get());
            T* dQp = reinterpret_cast<T*>(dQ_cpu.ptr.get());
            T* dKp = reinterpret_cast<T*>(dK_cpu.ptr.get());
            T* dVp = reinterpret_cast<T*>(dV_cpu.ptr.get());
            std::vector<T> dweights(Lq * Lk), dscores(Lq * Lk);

            for (std::size_t b = 0; b < B; ++b) {
                const T* Qb = Qp + b * Lq * Dk;
                const T* Kb = Kp + b * Lk * Dk;
                const T* Vb = Vp + b * Lk * Dv;
                const T* Wb = Wp + b * Lq * Lk;
                const T* Gb = Gp + b * Lq * Dv;
                T* dQb = dQp + b * Lq * Dk;
                T* dKb = dKp + b * Lk * Dk;
                T* dVb = dVp + b * Lk * Dv;

                if constexpr (std::is_same_v<T, float>) {
                    cpu::sgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dv),
                               static_cast<int>(Lq), 1.0f, Wb, static_cast<int>(Lk), Gb,
                               static_cast<int>(Dv), 0.0f, dVb, static_cast<int>(Dv));

                    cpu::sgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                               static_cast<int>(Dv), 1.0f, Gb, static_cast<int>(Dv), Vb,
                               static_cast<int>(Dv), 0.0f, dweights.data(), static_cast<int>(Lk));

                    for (std::size_t r = 0; r < Lq; ++r) {
                        const float* wr = Wb + r * Lk;
                        const float* dwr = dweights.data() + r * Lk;
                        float sum = 0.f;
                        for (std::size_t j = 0; j < Lk; ++j)
                            sum += wr[j] * dwr[j];
                        float* dr = dscores.data() + r * Lk;
                        for (std::size_t j = 0; j < Lk; ++j)
                            dr[j] = wr[j] * (dwr[j] - sum);
                    }

                    cpu::sgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dk),
                               static_cast<int>(Lk), sc, dscores.data(), static_cast<int>(Lk), Kb,
                               static_cast<int>(Dk), 0.0f, dQb, static_cast<int>(Dk));

                    cpu::sgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dk),
                               static_cast<int>(Lq), sc, dscores.data(), static_cast<int>(Lk), Qb,
                               static_cast<int>(Dk), 0.0f, dKb, static_cast<int>(Dk));
                } else {
                    cpu::dgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dv),
                               static_cast<int>(Lq), 1.0, Wb, static_cast<int>(Lk), Gb,
                               static_cast<int>(Dv), 0.0, dVb, static_cast<int>(Dv));
                    cpu::dgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                               static_cast<int>(Dv), 1.0, Gb, static_cast<int>(Dv), Vb,
                               static_cast<int>(Dv), 0.0, dweights.data(), static_cast<int>(Lk));
                    for (std::size_t r = 0; r < Lq; ++r) {
                        const double* wr = Wb + r * Lk;
                        const double* dwr = dweights.data() + r * Lk;
                        double sum = 0.0;
                        for (std::size_t j = 0; j < Lk; ++j)
                            sum += wr[j] * dwr[j];
                        double* dr = dscores.data() + r * Lk;
                        for (std::size_t j = 0; j < Lk; ++j)
                            dr[j] = wr[j] * (dwr[j] - sum);
                    }
                    cpu::dgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dk),
                               static_cast<int>(Lk), sc, dscores.data(), static_cast<int>(Lk), Kb,
                               static_cast<int>(Dk), 0.0, dQb, static_cast<int>(Dk));
                    cpu::dgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dk),
                               static_cast<int>(Lq), sc, dscores.data(), static_cast<int>(Lk), Qb,
                               static_cast<int>(Dk), 0.0, dKb, static_cast<int>(Dk));
                }
            }
        };

        if (dt == Dtype::F32)
            do_backward(float{});
        else if (dt == Dtype::F64)
            do_backward(double{});
        else
            ErrorBuilder("cpu_backend::sdpa_backward").not_implemented("dtype not supported");

        return {Storage{std::move(dQ_cpu)}, Storage{std::move(dK_cpu)}, Storage{std::move(dV_cpu)}};
    }

    Storage conv_transpose_nd_forward(const Storage& x,
                                      const Storage& W,
                                      const Storage& b,
                                      int B,
                                      int Cin,
                                      int Cout,
                                      const int* S,
                                      const int* K,
                                      const int* O,
                                      const int* stride,
                                      const int* pad,
                                      const int* opad,
                                      int N,
                                      const Shape& out_shape,
                                      Dtype dt) override {
        int S_total = 1, K_total = 1, O_total = 1;
        for (int i = 0; i < N; ++i) {
            S_total *= S[i];
            K_total *= K[i];
            O_total *= O[i];
        }
        const int K_flat = Cout * K_total;

        CpuStorage out_cpu = alloc_cpu(static_cast<std::size_t>(B) * Cout * O_total, dt);
        CpuStorage cols_cpu = alloc_cpu(static_cast<std::size_t>(K_flat) * S_total, dt);
        if (out_cpu.nbytes)
            std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);

        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& W_cpu = std::get<CpuStorage>(W);
        const auto& b_cpu = std::get<CpuStorage>(b);

        for (int bi = 0; bi < B; ++bi) {
            if (dt == Dtype::F32) {
                const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                                  static_cast<std::size_t>(bi) * Cin * S_total;
                const float* Wp = reinterpret_cast<const float*>(W_cpu.ptr.get());
                float* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());
                float* yp = reinterpret_cast<float*>(out_cpu.ptr.get()) +
                            static_cast<std::size_t>(bi) * Cout * O_total;
                cpu::sgemm(true, false, K_flat, S_total, Cin, 1.0f, Wp, K_flat, xp, S_total, 0.0f,
                           cp, S_total);
                ctnd_col2im_f32(cp, yp, Cout, O, K, S, stride, pad, N);
                {
                    const float* bp = reinterpret_cast<const float*>(b_cpu.ptr.get());
                    for (int c = 0; c < Cout; ++c) {
                        float* row = yp + static_cast<std::size_t>(c) * O_total;
                        const float bv = bp[c];
                        for (int i = 0; i < O_total; ++i)
                            row[i] += bv;
                    }
                }
            } else if (dt == Dtype::F64) {
                const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                                   static_cast<std::size_t>(bi) * Cin * S_total;
                const double* Wp = reinterpret_cast<const double*>(W_cpu.ptr.get());
                double* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());
                double* yp = reinterpret_cast<double*>(out_cpu.ptr.get()) +
                             static_cast<std::size_t>(bi) * Cout * O_total;
                cpu::dgemm(true, false, K_flat, S_total, Cin, 1.0, Wp, K_flat, xp, S_total, 0.0, cp,
                           S_total);
                ctnd_col2im_f64(cp, yp, Cout, O, K, S, stride, pad, N);
                {
                    const double* bp = reinterpret_cast<const double*>(b_cpu.ptr.get());
                    for (int c = 0; c < Cout; ++c) {
                        double* row = yp + static_cast<std::size_t>(c) * O_total;
                        const double bv = bp[c];
                        for (int i = 0; i < O_total; ++i)
                            row[i] += bv;
                    }
                }
            } else {
                ErrorBuilder("cpu_backend::conv_transpose_nd_forward")
                    .not_implemented("dtype not supported (F32/F64)");
            }
        }
        return Storage{std::move(out_cpu)};
    }

    std::vector<Storage> conv_transpose_nd_backward(const Storage& grad_out,
                                                    const Storage& x,
                                                    const Storage& W,
                                                    int B,
                                                    int Cin,
                                                    int Cout,
                                                    const int* S,
                                                    const int* K,
                                                    const int* O,
                                                    const int* stride,
                                                    const int* pad,
                                                    int N,
                                                    Dtype dt) override {
        int S_total = 1, K_total = 1, O_total = 1;
        for (int i = 0; i < N; ++i) {
            S_total *= S[i];
            K_total *= K[i];
            O_total *= O[i];
        }
        const int K_flat = Cout * K_total;

        CpuStorage dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * Cin * S_total, dt);
        CpuStorage dW_cpu = alloc_cpu(static_cast<std::size_t>(Cin) * K_flat, dt);
        CpuStorage db_cpu = alloc_cpu(static_cast<std::size_t>(Cout), dt);
        CpuStorage cols_cpu = alloc_cpu(static_cast<std::size_t>(K_flat) * S_total, dt);
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        if (dW_cpu.nbytes)
            std::memset(dW_cpu.ptr.get(), 0, dW_cpu.nbytes);
        if (db_cpu.nbytes)
            std::memset(db_cpu.ptr.get(), 0, db_cpu.nbytes);

        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& W_cpu = std::get<CpuStorage>(W);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);

        for (int bi = 0; bi < B; ++bi) {
            if (dt == Dtype::F32) {
                const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                                  static_cast<std::size_t>(bi) * Cin * S_total;
                const float* gp = reinterpret_cast<const float*>(g_cpu.ptr.get()) +
                                  static_cast<std::size_t>(bi) * Cout * O_total;
                float* dxp = reinterpret_cast<float*>(dx_cpu.ptr.get()) +
                             static_cast<std::size_t>(bi) * Cin * S_total;
                float* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());

                ctnd_im2col_f32(gp, cp, Cout, O, K, S, stride, pad, N);
                cpu::sgemm(false, false, Cin, S_total, K_flat, 1.0f,
                           reinterpret_cast<const float*>(W_cpu.ptr.get()), K_flat, cp, S_total,
                           0.0f, dxp, S_total);
                cpu::sgemm(false, true, Cin, K_flat, S_total, 1.0f, xp, S_total, cp, S_total, 1.0f,
                           reinterpret_cast<float*>(dW_cpu.ptr.get()), K_flat);
                {
                    float* dbp = reinterpret_cast<float*>(db_cpu.ptr.get());
                    for (int co = 0; co < Cout; ++co) {
                        const float* row = gp + co * O_total;
                        float s = 0.f;
                        for (int j = 0; j < O_total; ++j)
                            s += row[j];
                        dbp[co] += s;
                    }
                }
            } else if (dt == Dtype::F64) {
                const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                                   static_cast<std::size_t>(bi) * Cin * S_total;
                const double* gp = reinterpret_cast<const double*>(g_cpu.ptr.get()) +
                                   static_cast<std::size_t>(bi) * Cout * O_total;
                double* dxp = reinterpret_cast<double*>(dx_cpu.ptr.get()) +
                              static_cast<std::size_t>(bi) * Cin * S_total;
                double* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());

                ctnd_im2col_f64(gp, cp, Cout, O, K, S, stride, pad, N);
                cpu::dgemm(false, false, Cin, S_total, K_flat, 1.0,
                           reinterpret_cast<const double*>(W_cpu.ptr.get()), K_flat, cp, S_total,
                           0.0, dxp, S_total);
                cpu::dgemm(false, true, Cin, K_flat, S_total, 1.0, xp, S_total, cp, S_total, 1.0,
                           reinterpret_cast<double*>(dW_cpu.ptr.get()), K_flat);
                {
                    double* dbp = reinterpret_cast<double*>(db_cpu.ptr.get());
                    for (int co = 0; co < Cout; ++co) {
                        const double* row = gp + co * O_total;
                        double s = 0.0;
                        for (int j = 0; j < O_total; ++j)
                            s += row[j];
                        dbp[co] += s;
                    }
                }
            } else {
                ErrorBuilder("cpu_backend::conv_transpose_nd_backward")
                    .not_implemented("dtype not supported");
            }
        }
        return {Storage{std::move(dx_cpu)}, Storage{std::move(dW_cpu)}, Storage{std::move(db_cpu)}};
    }

    Storage conv_nd_forward(const Storage& x,
                            const Storage& W,
                            const Storage& b,
                            int B,
                            int Cin,
                            int Cout,
                            int Cin_g,
                            int Cout_g,
                            const int* S,
                            const int* K,
                            const int* O,
                            const IBackend::ConvNdOpts& opts,
                            const Shape&,
                            Dtype dt) override {
        const int N = opts.N;
        int O_total = 1, K_total = 1, S_total = 1;
        for (int i = 0; i < N; ++i) {
            O_total *= O[i];
            K_total *= K[i];
            S_total *= S[i];
        }

        const int K_flat = Cin_g * K_total;
        const int M_out = O_total;
        const int W_per_group = Cout_g * K_flat;

        CpuStorage out_cpu = alloc_cpu(static_cast<std::size_t>(B) * Cout * O_total, dt);
        CpuStorage cols_cpu = alloc_cpu(static_cast<std::size_t>(K_flat) * O_total, dt);
        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& W_cpu = std::get<CpuStorage>(W);
        const auto& b_cpu = std::get<CpuStorage>(b);

#ifdef __APPLE__

        if ((N == 1 || N == 2) && dt == Dtype::F32 && opts.dilation[0] == 1 &&
            (N == 1 || opts.dilation[1] == 1) && opts.groups == 1) {
            const int H_in = S[0];
            const int W_in = (N == 2) ? S[1] : 1;
            const int KH = K[0];
            const int KW = (N == 2) ? K[1] : 1;
            const int OH = O[0];
            const int OW = (N == 2) ? O[1] : 1;
            const int stride_h = opts.stride[0];
            const int stride_w = (N == 2) ? opts.stride[1] : 1;
            const int pad_h = opts.pad[0];
            const int pad_w = (N == 2) ? opts.pad[1] : 0;

            const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get());
            const float* wp = reinterpret_cast<const float*>(W_cpu.ptr.get());
            const float* bp = reinterpret_cast<const float*>(b_cpu.ptr.get());
            float* yp = reinterpret_cast<float*>(out_cpu.ptr.get());

            BNNSNDArrayDescriptor in_desc = {};
            in_desc.layout = BNNSDataLayoutImageCHW;
            in_desc.size[0] = static_cast<std::size_t>(W_in);
            in_desc.size[1] = static_cast<std::size_t>(H_in);
            in_desc.size[2] = static_cast<std::size_t>(Cin);
            in_desc.data_type = BNNSDataTypeFloat32;
            in_desc.data = nullptr;

            BNNSNDArrayDescriptor w_desc = {};
            w_desc.layout = BNNSDataLayoutConvolutionWeightsOIHW;
            w_desc.size[0] = static_cast<std::size_t>(KW);
            w_desc.size[1] = static_cast<std::size_t>(KH);
            w_desc.size[2] = static_cast<std::size_t>(Cin);
            w_desc.size[3] = static_cast<std::size_t>(Cout);
            w_desc.data_type = BNNSDataTypeFloat32;
            w_desc.data = const_cast<float*>(wp);

            BNNSNDArrayDescriptor b_desc = {};
            b_desc.layout = BNNSDataLayout1DLastMajor;
            b_desc.size[0] = static_cast<std::size_t>(Cout);
            b_desc.data_type = BNNSDataTypeFloat32;
            b_desc.data = const_cast<float*>(bp);

            BNNSNDArrayDescriptor out_desc = {};
            out_desc.layout = BNNSDataLayoutImageCHW;
            out_desc.size[0] = static_cast<std::size_t>(OW);
            out_desc.size[1] = static_cast<std::size_t>(OH);
            out_desc.size[2] = static_cast<std::size_t>(Cout);
            out_desc.data_type = BNNSDataTypeFloat32;
            out_desc.data = nullptr;

            BNNSLayerParametersConvolution conv_params = {};
            conv_params.i_desc = in_desc;
            conv_params.w_desc = w_desc;
            conv_params.o_desc = out_desc;
            conv_params.bias = b_desc;
            conv_params.x_stride = static_cast<std::size_t>(stride_w);
            conv_params.y_stride = static_cast<std::size_t>(stride_h);
            conv_params.x_padding = static_cast<std::size_t>(pad_w);
            conv_params.y_padding = static_cast<std::size_t>(pad_h);
            conv_params.x_dilation_stride = 1;
            conv_params.y_dilation_stride = 1;
            conv_params.groups = 1;
            BNNSActivation act = {};
            act.function = BNNSActivationFunctionIdentity;
            conv_params.activation = act;

            BNNSFilter filter = BNNSFilterCreateLayerConvolution(&conv_params, nullptr);
            if (filter) {
                const std::size_t in_per_sample = static_cast<std::size_t>(Cin) * H_in * W_in;
                const std::size_t out_per_sample = static_cast<std::size_t>(Cout) * OH * OW;
                int ret = 0;
                for (int bi = 0; bi < B && ret == 0; ++bi) {
                    ret =
                        BNNSFilterApply(filter, xp + bi * in_per_sample, yp + bi * out_per_sample);
                }
                BNNSFilterDestroy(filter);
                if (ret == 0) {
                    return Storage{std::move(out_cpu)};
                }
            }
        }
#endif

        for (int bi = 0; bi < B; ++bi) {
            for (int g = 0; g < opts.groups; ++g) {
                if (dt == Dtype::F32) {
                    const float* xp =
                        reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                        (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                            S_total;
                    float* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());
                    const float* wp = reinterpret_cast<const float*>(W_cpu.ptr.get()) +
                                      static_cast<std::size_t>(g) * W_per_group;
                    float* yp = reinterpret_cast<float*>(out_cpu.ptr.get()) +
                                (static_cast<std::size_t>(bi) * Cout +
                                 static_cast<std::size_t>(g) * Cout_g) *
                                    O_total;
                    conv_nd_im2col_f32(xp, cp, Cin_g, S, K, O, opts.stride, opts.pad, opts.dilation,
                                       N);
                    cpu::sgemm(false, false, Cout_g, M_out, K_flat, 1.0f, wp, K_flat, cp, M_out,
                               0.0f, yp, M_out);
                } else if (dt == Dtype::F64) {
                    const double* xp =
                        reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                        (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                            S_total;
                    double* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());
                    const double* wp = reinterpret_cast<const double*>(W_cpu.ptr.get()) +
                                       static_cast<std::size_t>(g) * W_per_group;
                    double* yp = reinterpret_cast<double*>(out_cpu.ptr.get()) +
                                 (static_cast<std::size_t>(bi) * Cout +
                                  static_cast<std::size_t>(g) * Cout_g) *
                                     O_total;
                    conv_nd_im2col_f64(xp, cp, Cin_g, S, K, O, opts.stride, opts.pad, opts.dilation,
                                       N);
                    cpu::dgemm(false, false, Cout_g, M_out, K_flat, 1.0, wp, K_flat, cp, M_out, 0.0,
                               yp, M_out);
                } else {
                    ErrorBuilder("cpu_backend::conv_nd_forward")
                        .not_implemented("dtype not supported (F32/F64)");
                }
            }

            if (dt == Dtype::F32) {
                float* yp = reinterpret_cast<float*>(out_cpu.ptr.get()) +
                            static_cast<std::size_t>(bi) * Cout * O_total;
                const float* bp = reinterpret_cast<const float*>(b_cpu.ptr.get());
                for (int c = 0; c < Cout; ++c) {
                    const float bv = bp[c];
                    float* row = yp + c * O_total;
                    for (int i = 0; i < O_total; ++i)
                        row[i] += bv;
                }
            } else if (dt == Dtype::F64) {
                double* yp = reinterpret_cast<double*>(out_cpu.ptr.get()) +
                             static_cast<std::size_t>(bi) * Cout * O_total;
                const double* bp = reinterpret_cast<const double*>(b_cpu.ptr.get());
                for (int c = 0; c < Cout; ++c) {
                    const double bv = bp[c];
                    double* row = yp + c * O_total;
                    for (int i = 0; i < O_total; ++i)
                        row[i] += bv;
                }
            }
        }
        return Storage{std::move(out_cpu)};
    }

    std::vector<Storage> conv_nd_backward(const Storage& grad_out,
                                          const Storage& x,
                                          const Storage& W,
                                          int B,
                                          int Cin,
                                          int Cout,
                                          int Cin_g,
                                          int Cout_g,
                                          const int* S,
                                          const int* K,
                                          const int* O,
                                          const IBackend::ConvNdOpts& opts,
                                          Dtype dt) override {
        const int N = opts.N;
        const int G = opts.groups;
        int O_total = 1, K_total = 1, S_total = 1;
        for (int i = 0; i < N; ++i) {
            O_total *= O[i];
            K_total *= K[i];
            S_total *= S[i];
        }

        const int K_flat = Cin_g * K_total;
        const int M_out = O_total;
        const int W_per_group = Cout_g * K_flat;

        CpuStorage dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * Cin * S_total, dt);
        CpuStorage dW_cpu = alloc_cpu(static_cast<std::size_t>(Cout) * K_flat, dt);
        CpuStorage db_cpu = alloc_cpu(static_cast<std::size_t>(Cout), dt);
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        if (dW_cpu.nbytes)
            std::memset(dW_cpu.ptr.get(), 0, dW_cpu.nbytes);
        if (db_cpu.nbytes)
            std::memset(db_cpu.ptr.get(), 0, db_cpu.nbytes);
        CpuStorage cols_cpu = alloc_cpu(static_cast<std::size_t>(K_flat) * M_out, dt);
        CpuStorage col_grad_cpu = alloc_cpu(static_cast<std::size_t>(K_flat) * M_out, dt);

        const auto& x_cpu = std::get<CpuStorage>(x);
        const auto& W_cpu = std::get<CpuStorage>(W);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);

        for (int bi = 0; bi < B; ++bi) {
            for (int g = 0; g < G; ++g) {
                if (dt == Dtype::F32) {
                    const float* xp =
                        reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                        (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                            S_total;
                    const float* gp = reinterpret_cast<const float*>(g_cpu.ptr.get()) +
                                      (static_cast<std::size_t>(bi) * Cout +
                                       static_cast<std::size_t>(g) * Cout_g) *
                                          O_total;
                    float* dxp =
                        reinterpret_cast<float*>(dx_cpu.ptr.get()) +
                        (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                            S_total;
                    const float* wp = reinterpret_cast<const float*>(W_cpu.ptr.get()) +
                                      static_cast<std::size_t>(g) * W_per_group;
                    float* dwp = reinterpret_cast<float*>(dW_cpu.ptr.get()) +
                                 static_cast<std::size_t>(g) * W_per_group;
                    float* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());
                    float* cgp = reinterpret_cast<float*>(col_grad_cpu.ptr.get());
                    conv_nd_im2col_f32(xp, cp, Cin_g, S, K, O, opts.stride, opts.pad, opts.dilation,
                                       N);
                    cpu::sgemm(false, true, Cout_g, K_flat, M_out, 1.0f, gp, M_out, cp, M_out, 1.0f,
                               dwp, K_flat);
                    cpu::sgemm(true, false, K_flat, M_out, Cout_g, 1.0f, wp, K_flat, gp, M_out,
                               0.0f, cgp, M_out);
                    conv_nd_col2im_f32(cgp, dxp, Cin_g, S, K, O, opts.stride, opts.pad,
                                       opts.dilation, N);
                    float* dbp = reinterpret_cast<float*>(db_cpu.ptr.get()) +
                                 static_cast<std::size_t>(g) * Cout_g;
                    for (int co = 0; co < Cout_g; ++co) {
                        const float* row = gp + co * O_total;
                        float s = 0.f;
                        for (int j = 0; j < O_total; ++j)
                            s += row[j];
                        dbp[co] += s;
                    }
                } else if (dt == Dtype::F64) {
                    const double* xp =
                        reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                        (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                            S_total;
                    const double* gp = reinterpret_cast<const double*>(g_cpu.ptr.get()) +
                                       (static_cast<std::size_t>(bi) * Cout +
                                        static_cast<std::size_t>(g) * Cout_g) *
                                           O_total;
                    double* dxp =
                        reinterpret_cast<double*>(dx_cpu.ptr.get()) +
                        (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                            S_total;
                    const double* wp = reinterpret_cast<const double*>(W_cpu.ptr.get()) +
                                       static_cast<std::size_t>(g) * W_per_group;
                    double* dwp = reinterpret_cast<double*>(dW_cpu.ptr.get()) +
                                  static_cast<std::size_t>(g) * W_per_group;
                    double* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());
                    double* cgp = reinterpret_cast<double*>(col_grad_cpu.ptr.get());
                    conv_nd_im2col_f64(xp, cp, Cin_g, S, K, O, opts.stride, opts.pad, opts.dilation,
                                       N);
                    cpu::dgemm(false, true, Cout_g, K_flat, M_out, 1.0, gp, M_out, cp, M_out, 1.0,
                               dwp, K_flat);
                    cpu::dgemm(true, false, K_flat, M_out, Cout_g, 1.0, wp, K_flat, gp, M_out, 0.0,
                               cgp, M_out);
                    conv_nd_col2im_f64(cgp, dxp, Cin_g, S, K, O, opts.stride, opts.pad,
                                       opts.dilation, N);
                    double* dbp = reinterpret_cast<double*>(db_cpu.ptr.get()) +
                                  static_cast<std::size_t>(g) * Cout_g;
                    for (int co = 0; co < Cout_g; ++co) {
                        const double* row = gp + co * O_total;
                        double s = 0.0;
                        for (int j = 0; j < O_total; ++j)
                            s += row[j];
                        dbp[co] += s;
                    }
                } else {
                    ErrorBuilder("cpu_backend::conv_nd_backward")
                        .not_implemented("dtype not supported");
                }
            }
        }
        return {Storage{std::move(dx_cpu)}, Storage{std::move(dW_cpu)}, Storage{std::move(db_cpu)}};
    }

    Storage unfold_forward(const Storage& x,
                           int B,
                           int C,
                           const std::vector<int>& S,
                           const std::vector<int>& K,
                           const std::vector<int>& O,
                           const std::vector<int>& stride,
                           const std::vector<int>& pad,
                           const std::vector<int>& dilation,
                           const Shape&,
                           Dtype dt) override {
        const int N = static_cast<int>(K.size());
        int K_total = 1, O_total = 1, S_total = 1;
        for (int i = 0; i < N; ++i) {
            K_total *= K[i];
            O_total *= O[i];
            S_total *= S[i];
        }

        CpuStorage out_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * K_total * O_total, dt);
        const auto& x_cpu = std::get<CpuStorage>(x);

        for (int bi = 0; bi < B; ++bi) {
            if (dt == Dtype::F32) {
                const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                                  static_cast<std::size_t>(bi) * C * S_total;
                float* yp = reinterpret_cast<float*>(out_cpu.ptr.get()) +
                            static_cast<std::size_t>(bi) * C * K_total * O_total;
                unfold_im2col_f32(xp, yp, C, S.data(), K.data(), O.data(), stride.data(),
                                  pad.data(), dilation.data(), N);
            } else if (dt == Dtype::F64) {
                const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                                   static_cast<std::size_t>(bi) * C * S_total;
                double* yp = reinterpret_cast<double*>(out_cpu.ptr.get()) +
                             static_cast<std::size_t>(bi) * C * K_total * O_total;
                unfold_im2col_f64(xp, yp, C, S.data(), K.data(), O.data(), stride.data(),
                                  pad.data(), dilation.data(), N);
            } else {
                ErrorBuilder("cpu_backend::unfold_forward").not_implemented("dtype not supported");
            }
        }
        return Storage{std::move(out_cpu)};
    }

    Storage unfold_backward(const Storage& grad_out,
                            int B,
                            int C,
                            const std::vector<int>& S,
                            const std::vector<int>& K,
                            const std::vector<int>& O,
                            const std::vector<int>& stride,
                            const std::vector<int>& pad,
                            const std::vector<int>& dilation,
                            Dtype dt) override {
        const int N = static_cast<int>(K.size());
        int K_total = 1, O_total = 1, S_total = 1;
        for (int i = 0; i < N; ++i) {
            K_total *= K[i];
            O_total *= O[i];
            S_total *= S[i];
        }

        CpuStorage dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * S_total, dt);
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        const auto& g_cpu = std::get<CpuStorage>(grad_out);

        for (int bi = 0; bi < B; ++bi) {
            if (dt == Dtype::F32) {
                const float* gp = reinterpret_cast<const float*>(g_cpu.ptr.get()) +
                                  static_cast<std::size_t>(bi) * C * K_total * O_total;
                float* dxp = reinterpret_cast<float*>(dx_cpu.ptr.get()) +
                             static_cast<std::size_t>(bi) * C * S_total;
                unfold_col2im_f32(gp, dxp, C, S.data(), K.data(), O.data(), stride.data(),
                                  pad.data(), dilation.data(), N);
            } else if (dt == Dtype::F64) {
                const double* gp = reinterpret_cast<const double*>(g_cpu.ptr.get()) +
                                   static_cast<std::size_t>(bi) * C * K_total * O_total;
                double* dxp = reinterpret_cast<double*>(dx_cpu.ptr.get()) +
                              static_cast<std::size_t>(bi) * C * S_total;
                unfold_col2im_f64(gp, dxp, C, S.data(), K.data(), O.data(), stride.data(),
                                  pad.data(), dilation.data(), N);
            } else {
                ErrorBuilder("cpu_backend::unfold_backward").not_implemented("dtype not supported");
            }
        }
        return Storage{std::move(dx_cpu)};
    }

    std::pair<Storage, Storage> expand_and_multiply(const Storage& mask,
                                                    const Storage& x,
                                                    const Shape& mask_shape,
                                                    const Shape& x_shape,
                                                    Dtype dt) override {
        const std::size_t B_sz = static_cast<std::size_t>(x_shape[0]);

        const bool is_channel_mask = (mask_shape.size() >= 2 && mask_shape[1] > 1);
        const std::size_t C_sz = is_channel_mask ? static_cast<std::size_t>(mask_shape[1])
                                                 : static_cast<std::size_t>(x_shape[1]);
        std::size_t spatial = 1;
        for (std::size_t i = 2; i < x_shape.size(); ++i)
            spatial *= static_cast<std::size_t>(x_shape[i]);

        const std::size_t numel = shape_numel(x_shape);
        CpuStorage exp_out = alloc_cpu(numel, dt);
        const auto& m_cpu = std::get<CpuStorage>(mask);
        const auto& x_cpu = std::get<CpuStorage>(x);
        CpuStorage y_out = alloc_cpu(numel, dt);

        if (is_channel_mask) {
            if (dt == Dtype::F32) {
                const float* mp = reinterpret_cast<const float*>(m_cpu.ptr.get());
                const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get());
                float* ep = reinterpret_cast<float*>(exp_out.ptr.get());
                float* yp = reinterpret_cast<float*>(y_out.ptr.get());
                for (std::size_t b = 0; b < B_sz; ++b)
                    for (std::size_t c = 0; c < C_sz; ++c) {
                        const float v = mp[b * C_sz + c];
                        for (std::size_t s = 0; s < spatial; ++s) {
                            const std::size_t idx = (b * C_sz + c) * spatial + s;
                            ep[idx] = v;
                            yp[idx] = xp[idx] * v;
                        }
                    }
            } else if (dt == Dtype::F64) {
                const double* mp = reinterpret_cast<const double*>(m_cpu.ptr.get());
                const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get());
                double* ep = reinterpret_cast<double*>(exp_out.ptr.get());
                double* yp = reinterpret_cast<double*>(y_out.ptr.get());
                for (std::size_t b = 0; b < B_sz; ++b)
                    for (std::size_t c = 0; c < C_sz; ++c) {
                        const double v = mp[b * C_sz + c];
                        for (std::size_t s = 0; s < spatial; ++s) {
                            const std::size_t idx = (b * C_sz + c) * spatial + s;
                            ep[idx] = v;
                            yp[idx] = xp[idx] * v;
                        }
                    }
            } else {
                ErrorBuilder("cpu_backend::expand_and_multiply")
                    .not_implemented("dtype not supported");
            }
        } else {
            const std::size_t per = numel / B_sz;
            if (dt == Dtype::F32) {
                const float* mp = reinterpret_cast<const float*>(m_cpu.ptr.get());
                const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get());
                float* ep = reinterpret_cast<float*>(exp_out.ptr.get());
                float* yp = reinterpret_cast<float*>(y_out.ptr.get());
                for (std::size_t b = 0; b < B_sz; ++b) {
                    const float v = mp[b];
                    for (std::size_t s = 0; s < per; ++s) {
                        const std::size_t idx = b * per + s;
                        ep[idx] = v;
                        yp[idx] = xp[idx] * v;
                    }
                }
            } else if (dt == Dtype::F64) {
                const double* mp = reinterpret_cast<const double*>(m_cpu.ptr.get());
                const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get());
                double* ep = reinterpret_cast<double*>(exp_out.ptr.get());
                double* yp = reinterpret_cast<double*>(y_out.ptr.get());
                for (std::size_t b = 0; b < B_sz; ++b) {
                    const double v = mp[b];
                    for (std::size_t s = 0; s < per; ++s) {
                        const std::size_t idx = b * per + s;
                        ep[idx] = v;
                        yp[idx] = xp[idx] * v;
                    }
                }
            } else {
                ErrorBuilder("cpu_backend::expand_and_multiply")
                    .not_implemented("dtype not supported");
            }
        }
        return {Storage{std::move(exp_out)}, Storage{std::move(y_out)}};
    }

    Storage drop_block_mask(const Storage& seed,
                            double drop_prob,
                            std::int64_t block_size,
                            const Shape& x_shape,
                            Dtype dt) override {
        const std::size_t B = static_cast<std::size_t>(x_shape[0]);
        const std::size_t C = static_cast<std::size_t>(x_shape[1]);
        const std::size_t H = static_cast<std::size_t>(x_shape[2]);
        const std::size_t W = static_cast<std::size_t>(x_shape[3]);
        const std::size_t total = B * C * H * W;
        const std::int64_t bsz = block_size;
        const std::int64_t pad_h = bsz / 2;

        const auto& s_cpu = std::get<CpuStorage>(seed);
        CpuStorage block_mask = alloc_cpu(total, dt);

        if (dt == Dtype::F32) {
            const float* sp = reinterpret_cast<const float*>(s_cpu.ptr.get());
            float* mp = reinterpret_cast<float*>(block_mask.ptr.get());
            for (std::size_t b = 0; b < B; ++b)
                for (std::size_t c = 0; c < C; ++c) {
                    const float* s = sp + (b * C + c) * H * W;
                    float* m = mp + (b * C + c) * H * W;
                    for (std::size_t y = 0; y < H; ++y)
                        for (std::size_t x = 0; x < W; ++x) {
                            float mx = 0.f;
                            for (std::int64_t dy = 0; dy < bsz; ++dy) {
                                const std::int64_t yy = static_cast<std::int64_t>(y) + dy - pad_h;
                                if (yy < 0 || yy >= static_cast<std::int64_t>(H))
                                    continue;
                                for (std::int64_t dx = 0; dx < bsz; ++dx) {
                                    const std::int64_t xx =
                                        static_cast<std::int64_t>(x) + dx - pad_h;
                                    if (xx < 0 || xx >= static_cast<std::int64_t>(W))
                                        continue;
                                    const float v = s[yy * W + xx];
                                    if (v > mx)
                                        mx = v;
                                }
                            }
                            m[y * W + x] = mx;
                        }
                }
            const float fs = static_cast<float>(1.0 / (1.0 - drop_prob));
            for (std::size_t i = 0; i < total; ++i)
                mp[i] = (1.f - mp[i]) * fs;
        } else if (dt == Dtype::F64) {
            const double* sp = reinterpret_cast<const double*>(s_cpu.ptr.get());
            double* mp = reinterpret_cast<double*>(block_mask.ptr.get());
            for (std::size_t b = 0; b < B; ++b)
                for (std::size_t c = 0; c < C; ++c) {
                    const double* s = sp + (b * C + c) * H * W;
                    double* m = mp + (b * C + c) * H * W;
                    for (std::size_t y = 0; y < H; ++y)
                        for (std::size_t x = 0; x < W; ++x) {
                            double mx = 0.0;
                            for (std::int64_t dy = 0; dy < bsz; ++dy) {
                                const std::int64_t yy = static_cast<std::int64_t>(y) + dy - pad_h;
                                if (yy < 0 || yy >= static_cast<std::int64_t>(H))
                                    continue;
                                for (std::int64_t dx = 0; dx < bsz; ++dx) {
                                    const std::int64_t xx =
                                        static_cast<std::int64_t>(x) + dx - pad_h;
                                    if (xx < 0 || xx >= static_cast<std::int64_t>(W))
                                        continue;
                                    const double v = s[yy * W + xx];
                                    if (v > mx)
                                        mx = v;
                                }
                            }
                            m[y * W + x] = mx;
                        }
                }
            const double scale = 1.0 / (1.0 - drop_prob);
            for (std::size_t i = 0; i < total; ++i)
                mp[i] = (1.0 - mp[i]) * scale;
        } else {
            ErrorBuilder("cpu_backend::drop_block_mask").not_implemented("dtype not supported");
        }
        return Storage{std::move(block_mask)};
    }

    std::vector<Storage> max_pool_nd_forward(const Storage& x,
                                             const Shape& x_shape,
                                             const Shape& out_shape,
                                             const PoolOpts& opts,
                                             Dtype dt) override {
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3];
        int O_total = 1;
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            O_total *= O[i];
        }
        const auto& xc = std::get<CpuStorage>(x);
        auto y_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * O_total, dt);
        auto am_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * O_total, Dtype::I32);
        auto dispatch = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xc.ptr.get());
            T* yp = reinterpret_cast<T*>(y_cpu.ptr.get());
            auto* ap = reinterpret_cast<std::int32_t*>(am_cpu.ptr.get());
            if constexpr (std::is_same_v<T, float>) {
                if (N == 1)
                    cpu::max_pool1d_forward_f32(xp, yp, ap, B, C, S[0], opts.K[0], O[0],
                                                opts.stride[0], opts.pad[0]);
                else if (N == 2)
                    cpu::max_pool2d_forward_f32(xp, yp, ap, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                                O[0], O[1], opts.stride[0], opts.stride[1],
                                                opts.pad[0], opts.pad[1]);
                else
                    cpu::max_pool3d_forward_f32(xp, yp, ap, B, C, S[0], S[1], S[2], opts.K[0],
                                                opts.K[1], opts.K[2], O[0], O[1], O[2],
                                                opts.stride[0], opts.stride[1], opts.stride[2],
                                                opts.pad[0], opts.pad[1], opts.pad[2]);
            } else {
                if (N == 1)
                    cpu::max_pool1d_forward_f64(xp, yp, ap, B, C, S[0], opts.K[0], O[0],
                                                opts.stride[0], opts.pad[0]);
                else if (N == 2)
                    cpu::max_pool2d_forward_f64(xp, yp, ap, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                                O[0], O[1], opts.stride[0], opts.stride[1],
                                                opts.pad[0], opts.pad[1]);
                else
                    cpu::max_pool3d_forward_f64(xp, yp, ap, B, C, S[0], S[1], S[2], opts.K[0],
                                                opts.K[1], opts.K[2], O[0], O[1], O[2],
                                                opts.stride[0], opts.stride[1], opts.stride[2],
                                                opts.pad[0], opts.pad[1], opts.pad[2]);
            }
        };
        if (dt == Dtype::F32)
            dispatch(float{});
        else if (dt == Dtype::F64)
            dispatch(double{});
        else
            ErrorBuilder("max_pool_nd_forward").not_implemented("dtype");
        return {Storage{std::move(y_cpu)}, Storage{std::move(am_cpu)}};
    }

    Storage max_pool_nd_backward(const Storage& grad_out,
                                 const Storage& saved_argmax,
                                 const Shape& x_shape,
                                 const Shape& out_shape,
                                 const PoolOpts& opts,
                                 Dtype dt) override {
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3];
        int S_total = 1, O_total = 1;
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            S_total *= S[i];
            O_total *= O[i];
        }
        (void)O_total;
        const auto& gc = std::get<CpuStorage>(grad_out);
        const auto& ac = std::get<CpuStorage>(saved_argmax);
        auto dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * S_total, dt);
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        auto dispatch = [&](auto tag) {
            using T = decltype(tag);
            const T* gp = reinterpret_cast<const T*>(gc.ptr.get());
            const auto* ap = reinterpret_cast<const std::int32_t*>(ac.ptr.get());
            T* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
            if constexpr (std::is_same_v<T, float>) {
                if (N == 1)
                    cpu::max_pool1d_backward_f32(gp, ap, dxp, B, C, S[0], O[0]);
                else if (N == 2)
                    cpu::max_pool2d_backward_f32(gp, ap, dxp, B, C, S[0], S[1], O[0], O[1]);
                else
                    cpu::max_pool3d_backward_f32(gp, ap, dxp, B, C, S[0], S[1], S[2], O[0], O[1],
                                                 O[2]);
            } else {
                if (N == 1)
                    cpu::max_pool1d_backward_f64(gp, ap, dxp, B, C, S[0], O[0]);
                else if (N == 2)
                    cpu::max_pool2d_backward_f64(gp, ap, dxp, B, C, S[0], S[1], O[0], O[1]);
                else
                    cpu::max_pool3d_backward_f64(gp, ap, dxp, B, C, S[0], S[1], S[2], O[0], O[1],
                                                 O[2]);
            }
        };
        if (dt == Dtype::F32)
            dispatch(float{});
        else if (dt == Dtype::F64)
            dispatch(double{});
        else
            ErrorBuilder("max_pool_nd_backward").not_implemented("dtype");
        return Storage{std::move(dx_cpu)};
    }

    Storage avg_pool_nd_forward(const Storage& x,
                                const Shape& x_shape,
                                const Shape& out_shape,
                                const PoolOpts& opts,
                                Dtype dt) override {
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3];
        int O_total = 1;
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            O_total *= O[i];
        }
        const auto& xc = std::get<CpuStorage>(x);
        auto y_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * O_total, dt);
        auto dispatch = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xc.ptr.get());
            T* yp = reinterpret_cast<T*>(y_cpu.ptr.get());
            if constexpr (std::is_same_v<T, float>) {
                if (N == 1)
                    cpu::avg_pool1d_forward_f32(xp, yp, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                                opts.pad[0]);
                else if (N == 2)
                    cpu::avg_pool2d_forward_f32(xp, yp, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                                O[0], O[1], opts.stride[0], opts.stride[1],
                                                opts.pad[0], opts.pad[1]);
                else
                    cpu::avg_pool3d_forward_f32(xp, yp, B, C, S[0], S[1], S[2], opts.K[0],
                                                opts.K[1], opts.K[2], O[0], O[1], O[2],
                                                opts.stride[0], opts.stride[1], opts.stride[2],
                                                opts.pad[0], opts.pad[1], opts.pad[2]);
            } else {
                if (N == 1)
                    cpu::avg_pool1d_forward_f64(xp, yp, B, C, S[0], opts.K[0], O[0], opts.stride[0],
                                                opts.pad[0]);
                else if (N == 2)
                    cpu::avg_pool2d_forward_f64(xp, yp, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                                O[0], O[1], opts.stride[0], opts.stride[1],
                                                opts.pad[0], opts.pad[1]);
                else
                    cpu::avg_pool3d_forward_f64(xp, yp, B, C, S[0], S[1], S[2], opts.K[0],
                                                opts.K[1], opts.K[2], O[0], O[1], O[2],
                                                opts.stride[0], opts.stride[1], opts.stride[2],
                                                opts.pad[0], opts.pad[1], opts.pad[2]);
            }
        };
        if (dt == Dtype::F32)
            dispatch(float{});
        else if (dt == Dtype::F64)
            dispatch(double{});
        else
            ErrorBuilder("avg_pool_nd_forward").not_implemented("dtype");
        return Storage{std::move(y_cpu)};
    }

    Storage avg_pool_nd_backward(const Storage& grad_out,
                                 const Shape& x_shape,
                                 const Shape& out_shape,
                                 const PoolOpts& opts,
                                 Dtype dt) override {
        const int N = opts.N;
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        int S[3], O[3];
        int S_total = 1, O_total = 1;
        for (int i = 0; i < N; ++i) {
            S[i] = static_cast<int>(x_shape[2 + i]);
            O[i] = static_cast<int>(out_shape[2 + i]);
            S_total *= S[i];
            O_total *= O[i];
        }
        (void)O_total;
        const auto& gc = std::get<CpuStorage>(grad_out);
        auto dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * S_total, dt);
        if (dx_cpu.nbytes)
            std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
        auto dispatch = [&](auto tag) {
            using T = decltype(tag);
            const T* gp = reinterpret_cast<const T*>(gc.ptr.get());
            T* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
            if constexpr (std::is_same_v<T, float>) {
                if (N == 1)
                    cpu::avg_pool1d_backward_f32(gp, dxp, B, C, S[0], opts.K[0], O[0],
                                                 opts.stride[0], opts.pad[0]);
                else if (N == 2)
                    cpu::avg_pool2d_backward_f32(gp, dxp, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                                 O[0], O[1], opts.stride[0], opts.stride[1],
                                                 opts.pad[0], opts.pad[1]);
                else
                    cpu::avg_pool3d_backward_f32(gp, dxp, B, C, S[0], S[1], S[2], opts.K[0],
                                                 opts.K[1], opts.K[2], O[0], O[1], O[2],
                                                 opts.stride[0], opts.stride[1], opts.stride[2],
                                                 opts.pad[0], opts.pad[1], opts.pad[2]);
            } else {
                if (N == 1)
                    cpu::avg_pool1d_backward_f64(gp, dxp, B, C, S[0], opts.K[0], O[0],
                                                 opts.stride[0], opts.pad[0]);
                else if (N == 2)
                    cpu::avg_pool2d_backward_f64(gp, dxp, B, C, S[0], S[1], opts.K[0], opts.K[1],
                                                 O[0], O[1], opts.stride[0], opts.stride[1],
                                                 opts.pad[0], opts.pad[1]);
                else
                    cpu::avg_pool3d_backward_f64(gp, dxp, B, C, S[0], S[1], S[2], opts.K[0],
                                                 opts.K[1], opts.K[2], O[0], O[1], O[2],
                                                 opts.stride[0], opts.stride[1], opts.stride[2],
                                                 opts.pad[0], opts.pad[1], opts.pad[2]);
            }
        };
        if (dt == Dtype::F32)
            dispatch(float{});
        else if (dt == Dtype::F64)
            dispatch(double{});
        else
            ErrorBuilder("avg_pool_nd_backward").not_implemented("dtype");
        return Storage{std::move(dx_cpu)};
    }

    Storage embedding_forward(const Storage& weight,
                              const Storage& indices,
                              const Shape& weight_shape,
                              const Shape& indices_shape,
                              const Shape& out_shape,
                              int padding_idx,
                              Dtype dt) override {
        const std::int64_t N = weight_shape[0];
        const std::int64_t D = weight_shape[1];
        std::size_t M = 1;
        for (auto d : indices_shape)
            M *= static_cast<std::size_t>(d);
        (void)N;
        (void)out_shape;
        const auto& ws = std::get<CpuStorage>(weight);
        const auto& is = std::get<CpuStorage>(indices);
        auto out_cpu = alloc_cpu(M * static_cast<std::size_t>(D), dt);
        const std::size_t row_bytes = static_cast<std::size_t>(D) * dtype_size(dt);
        auto read_idx = [&](std::size_t i) -> std::int64_t {
            const auto* ip = is.ptr.get();
            switch (is.dtype) {
            case Dtype::I32:
                return static_cast<std::int64_t>(reinterpret_cast<const std::int32_t*>(ip)[i]);
            case Dtype::I64:
                return reinterpret_cast<const std::int64_t*>(ip)[i];
            default:
                return static_cast<std::int64_t>(reinterpret_cast<const std::int32_t*>(ip)[i]);
            }
        };
        for (std::size_t i = 0; i < M; ++i) {
            const std::int64_t id = read_idx(i);
            std::byte* dst = out_cpu.ptr.get() + i * row_bytes;
            if (padding_idx >= 0 && id == static_cast<std::int64_t>(padding_idx)) {
                std::memset(dst, 0, row_bytes);
            } else {
                const std::byte* src = ws.ptr.get() + static_cast<std::size_t>(id) * row_bytes;
                std::memcpy(dst, src, row_bytes);
            }
        }
        return Storage{std::move(out_cpu)};
    }

    Storage embedding_backward(const Storage& grad_out,
                               const Storage& indices,
                               const Shape& weight_shape,
                               const Shape& indices_shape,
                               int padding_idx,
                               Dtype dt) override {
        const std::int64_t N = weight_shape[0];
        const std::int64_t D = weight_shape[1];
        std::size_t M = 1;
        for (auto d : indices_shape)
            M *= static_cast<std::size_t>(d);
        const auto& gs = std::get<CpuStorage>(grad_out);
        const auto& is = std::get<CpuStorage>(indices);
        auto dW = alloc_cpu(static_cast<std::size_t>(N) * static_cast<std::size_t>(D), dt);
        std::memset(dW.ptr.get(), 0, dW.nbytes);
        auto read_idx = [&](std::size_t i) -> std::int64_t {
            const auto* ip = is.ptr.get();
            switch (is.dtype) {
            case Dtype::I32:
                return static_cast<std::int64_t>(reinterpret_cast<const std::int32_t*>(ip)[i]);
            case Dtype::I64:
                return reinterpret_cast<const std::int64_t*>(ip)[i];
            default:
                return static_cast<std::int64_t>(reinterpret_cast<const std::int32_t*>(ip)[i]);
            }
        };
        const std::size_t row_bytes = static_cast<std::size_t>(D) * dtype_size(dt);
        for (std::size_t i = 0; i < M; ++i) {
            const std::int64_t id = read_idx(i);
            if (padding_idx >= 0 && id == static_cast<std::int64_t>(padding_idx))
                continue;
            const std::byte* src = gs.ptr.get() + i * row_bytes;
            std::byte* dst = dW.ptr.get() + static_cast<std::size_t>(id) * row_bytes;
            if (dt == Dtype::F32) {
                const float* sp = reinterpret_cast<const float*>(src);
                float* dp = reinterpret_cast<float*>(dst);
                for (std::int64_t d = 0; d < D; ++d)
                    dp[d] += sp[d];
            } else {
                const double* sp = reinterpret_cast<const double*>(src);
                double* dp = reinterpret_cast<double*>(dst);
                for (std::int64_t d = 0; d < D; ++d)
                    dp[d] += sp[d];
            }
        }
        return Storage{std::move(dW)};
    }

    Storage
    sinusoidal_pos_embedding(std::int64_t seq_len, std::int64_t embed_dim, Dtype dt) override {
        const std::size_t L = static_cast<std::size_t>(seq_len);
        const std::size_t D = static_cast<std::size_t>(embed_dim);
        const std::size_t Dh = D / 2;
        auto out = alloc_cpu(L * D, dt);
        if (L * D == 0)
            return Storage{std::move(out)};
        const double inv_d = -std::log(10000.0) / static_cast<double>(D);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            T* op = reinterpret_cast<T*>(out.ptr.get());
            for (std::size_t i = 0; i < L; ++i)
                for (std::size_t k = 0; k < Dh; ++k) {
                    const double angle =
                        static_cast<double>(i) * std::exp(2.0 * static_cast<double>(k) * inv_d);
                    op[i * D + 2 * k] = static_cast<T>(std::sin(angle));
                    if (2 * k + 1 < D)
                        op[i * D + 2 * k + 1] = static_cast<T>(std::cos(angle));
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("sinusoidal_pos_embedding").not_implemented("dtype");
        return Storage{std::move(out)};
    }

    std::vector<Storage> rope_forward(const Storage& x,
                                      const Storage* position_ids,
                                      const Shape& x_shape,
                                      bool interleaved,
                                      Dtype pos_dtype,
                                      Dtype dt) override {
        (void)pos_dtype;
        const std::size_t ndim = x_shape.size();
        const std::size_t L = static_cast<std::size_t>(x_shape[ndim - 2]);
        const std::size_t D = static_cast<std::size_t>(x_shape.back());
        const std::size_t Dh = D / 2;
        std::size_t batch = 1;
        for (std::size_t i = 0; i + 2 < ndim; ++i)
            batch *= static_cast<std::size_t>(x_shape[i]);
        const auto& xs = std::get<CpuStorage>(x);
        auto out = alloc_cpu(batch * L * D, dt);
        auto cos_t = alloc_cpu(L * Dh, dt);
        auto sin_t = alloc_cpu(L * Dh, dt);
        std::vector<double> pos(L);
        if (position_ids) {
            const auto& ps = std::get<CpuStorage>(*position_ids);
            for (std::size_t i = 0; i < L; ++i)
                pos[i] =
                    static_cast<double>(reinterpret_cast<const std::int64_t*>(ps.ptr.get())[i]);
        } else {
            for (std::size_t i = 0; i < L; ++i)
                pos[i] = static_cast<double>(i);
        }
        std::vector<double> theta(Dh);
        const double base = std::log(10000.0);
        for (std::size_t k = 0; k < Dh; ++k)
            theta[k] = std::exp(-2.0 * static_cast<double>(k) * base / static_cast<double>(D));
        auto run = [&](auto tag) {
            using T = decltype(tag);
            T* cp = reinterpret_cast<T*>(cos_t.ptr.get());
            T* sp = reinterpret_cast<T*>(sin_t.ptr.get());
            for (std::size_t i = 0; i < L; ++i)
                for (std::size_t k = 0; k < Dh; ++k) {
                    const double a = pos[i] * theta[k];
                    cp[i * Dh + k] = static_cast<T>(std::cos(a));
                    sp[i * Dh + k] = static_cast<T>(std::sin(a));
                }
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            T* op = reinterpret_cast<T*>(out.ptr.get());
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t i = 0; i < L; ++i) {
                    const T* xrow = xp + (b * L + i) * D;
                    T* orow = op + (b * L + i) * D;
                    const T* crow = cp + i * Dh;
                    const T* srow = sp + i * Dh;
                    if (interleaved) {
                        for (std::size_t k = 0; k < Dh; ++k) {
                            const T xe = xrow[2 * k], xo = xrow[2 * k + 1];
                            orow[2 * k] = xe * crow[k] - xo * srow[k];
                            orow[2 * k + 1] = xo * crow[k] + xe * srow[k];
                        }
                    } else {
                        for (std::size_t k = 0; k < Dh; ++k) {
                            orow[k] = xrow[k] * crow[k] - xrow[k + Dh] * srow[k];
                            orow[k + Dh] = xrow[k + Dh] * crow[k] + xrow[k] * srow[k];
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("rope_forward").not_implemented("dtype");
        return {Storage{std::move(out)}, Storage{std::move(cos_t)}, Storage{std::move(sin_t)}};
    }

    std::vector<Storage> batch_norm_eval_forward(const Storage& x,
                                                 const Storage& mean,
                                                 const Storage& var,
                                                 const Storage& gamma,
                                                 const Storage& beta,
                                                 const Shape& x_shape,
                                                 int C,
                                                 int spatial,
                                                 double eps,
                                                 Dtype dt) override {
        const int B = static_cast<int>(x_shape[0]);
        const auto& xs = std::get<CpuStorage>(x);
        const auto& ms = std::get<CpuStorage>(mean);
        const auto& vs = std::get<CpuStorage>(var);
        const auto& gs = std::get<CpuStorage>(gamma);
        const auto& bs = std::get<CpuStorage>(beta);
        auto rstd_cpu = alloc_cpu(static_cast<std::size_t>(C), dt);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * spatial, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const T* mp = reinterpret_cast<const T*>(ms.ptr.get());
            const T* vp = reinterpret_cast<const T*>(vs.ptr.get());
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            const T* bp = reinterpret_cast<const T*>(bs.ptr.get());
            T* yp = reinterpret_cast<T*>(out_cpu.ptr.get());
            T* rp = reinterpret_cast<T*>(rstd_cpu.ptr.get());
            for (int c = 0; c < C; ++c)
                rp[c] = T{1} / static_cast<T>(std::sqrt(static_cast<double>(vp[c]) + eps));
            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C; ++c) {
                    const T m = mp[c], r = rp[c], g = gp[c], bb = bp[c];
                    const T* xrow = xp + (b * C + c) * spatial;
                    T* yrow = yp + (b * C + c) * spatial;
                    for (int s = 0; s < spatial; ++s)
                        yrow[s] = g * (xrow[s] - m) * r + bb;
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("batch_norm_eval_forward").not_implemented("dtype");
        return {Storage{std::move(out_cpu)}, Storage{std::move(rstd_cpu)}};
    }

    std::vector<Storage> batch_norm_eval_backward(const Storage& x,
                                                  const Storage& mean,
                                                  const Storage& gamma,
                                                  const Storage& rstd,
                                                  const Storage& grad_out,
                                                  const Shape& x_shape,
                                                  int C,
                                                  int spatial,
                                                  Dtype dt) override {
        const int B = static_cast<int>(x_shape[0]);
        const auto& xs = std::get<CpuStorage>(x);
        const auto& ms = std::get<CpuStorage>(mean);
        const auto& gs = std::get<CpuStorage>(gamma);
        const auto& rs = std::get<CpuStorage>(rstd);
        const auto& go = std::get<CpuStorage>(grad_out);
        auto dx = alloc_cpu(static_cast<std::size_t>(B) * C * spatial, dt);
        auto dm = alloc_cpu(static_cast<std::size_t>(C), dt);
        auto dv = alloc_cpu(static_cast<std::size_t>(C), dt);
        auto dg = alloc_cpu(static_cast<std::size_t>(C), dt);
        auto db = alloc_cpu(static_cast<std::size_t>(C), dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const T* mp = reinterpret_cast<const T*>(ms.ptr.get());
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            const T* rp = reinterpret_cast<const T*>(rs.ptr.get());
            const T* gop = reinterpret_cast<const T*>(go.ptr.get());
            T* dxp = reinterpret_cast<T*>(dx.ptr.get());
            T* dmp = reinterpret_cast<T*>(dm.ptr.get());
            T* dvp = reinterpret_cast<T*>(dv.ptr.get());
            T* dgp = reinterpret_cast<T*>(dg.ptr.get());
            T* dbp = reinterpret_cast<T*>(db.ptr.get());
            for (int c = 0; c < C; ++c) {
                dmp[c] = dvp[c] = dgp[c] = dbp[c] = T{0};
                const T r = rp[c], g = gp[c], m = mp[c];
                T sg = T{0}, sxmg = T{0};
                for (int b = 0; b < B; ++b) {
                    const T* xr = xp + (b * C + c) * spatial;
                    const T* gor = gop + (b * C + c) * spatial;
                    T* dxr = dxp + (b * C + c) * spatial;
                    for (int s = 0; s < spatial; ++s) {
                        dxr[s] = g * r * gor[s];
                        sg += gor[s];
                        sxmg += (xr[s] - m) * gor[s];
                    }
                }
                dgp[c] = sxmg * r;
                dbp[c] = sg;
                dmp[c] = -g * r * sg;
                dvp[c] = static_cast<T>(-0.5) * g * r * r * r * sxmg;
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("batch_norm_eval_backward").not_implemented("dtype");
        return {Storage{std::move(dx)}, Storage{std::move(dm)}, Storage{std::move(dv)},
                Storage{std::move(dg)}, Storage{std::move(db)}};
    }

    std::vector<Storage> lp_normalize_forward(const Storage& x,
                                              const Shape& x_shape,
                                              double ord,
                                              int axis,
                                              double eps,
                                              Dtype dt) override {
        const auto& xs = std::get<CpuStorage>(x);
        const std::size_t numel = shape_numel(x_shape);
        int outer = 1, inner = 1;
        const int rank = static_cast<int>(x_shape.size());
        for (int i = 0; i < axis; ++i)
            outer *= static_cast<int>(x_shape[i]);
        for (int i = axis + 1; i < rank; ++i)
            inner *= static_cast<int>(x_shape[i]);
        const int axis_len = static_cast<int>(x_shape[axis]);
        auto y_cpu = alloc_cpu(numel, dt);
        auto norm_cpu = alloc_cpu(static_cast<std::size_t>(outer) * inner, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            T* yp = reinterpret_cast<T*>(y_cpu.ptr.get());
            T* np = reinterpret_cast<T*>(norm_cpu.ptr.get());
            for (int o = 0; o < outer; ++o)
                for (int n = 0; n < inner; ++n) {
                    T acc = T{0};
                    for (int a = 0; a < axis_len; ++a)
                        acc += static_cast<T>(std::pow(
                            std::abs(static_cast<double>(xp[(o * axis_len + a) * inner + n])),
                            ord));
                    const T nm = static_cast<T>(std::pow(static_cast<double>(acc), 1.0 / ord));
                    const T denom = nm > static_cast<T>(eps) ? nm : static_cast<T>(eps);
                    np[o * inner + n] = denom;
                    for (int a = 0; a < axis_len; ++a) {
                        const std::size_t idx = (o * axis_len + a) * inner + n;
                        yp[idx] = xp[idx] / denom;
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("lp_normalize_forward").not_implemented("dtype");
        return {Storage{std::move(y_cpu)}, Storage{std::move(norm_cpu)}};
    }

    Storage lp_normalize_backward(const Storage& x,
                                  const Storage& saved_norm,
                                  const Storage& grad_out,
                                  const Shape& x_shape,
                                  double ord,
                                  int axis,
                                  Dtype dt) override {
        const auto& xs = std::get<CpuStorage>(x);
        const auto& ns = std::get<CpuStorage>(saved_norm);
        const auto& gs = std::get<CpuStorage>(grad_out);
        const std::size_t numel = shape_numel(x_shape);
        int outer = 1, inner = 1;
        const int rank = static_cast<int>(x_shape.size());
        for (int i = 0; i < axis; ++i)
            outer *= static_cast<int>(x_shape[i]);
        for (int i = axis + 1; i < rank; ++i)
            inner *= static_cast<int>(x_shape[i]);
        const int axis_len = static_cast<int>(x_shape[axis]);
        auto dx_cpu = alloc_cpu(numel, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const T* np = reinterpret_cast<const T*>(ns.ptr.get());
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            T* dxp = reinterpret_cast<T*>(dx_cpu.ptr.get());
            for (int o = 0; o < outer; ++o)
                for (int n = 0; n < inner; ++n) {
                    const T N = np[o * inner + n];
                    T proj = T{0};
                    for (int a = 0; a < axis_len; ++a) {
                        const std::size_t idx = (o * axis_len + a) * inner + n;
                        proj += gp[idx] * xp[idx];
                    }
                    const T N_pp1 = static_cast<T>(std::pow(static_cast<double>(N), ord + 1.0));
                    for (int a = 0; a < axis_len; ++a) {
                        const std::size_t idx = (o * axis_len + a) * inner + n;
                        const T xi = xp[idx];
                        const T sgn = (xi > T{0}) ? T{1} : (xi < T{0} ? T{-1} : T{0});
                        const T abs_pm1 =
                            static_cast<T>(std::pow(std::abs(static_cast<double>(xi)), ord - 1.0));
                        dxp[idx] = gp[idx] / N - sgn * abs_pm1 * proj / N_pp1;
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("lp_normalize_backward").not_implemented("dtype");
        return Storage{std::move(dx_cpu)};
    }

    std::vector<Storage> global_response_norm_forward(const Storage& x,
                                                      const Storage& gamma,
                                                      const Storage& beta,
                                                      const Shape& x_shape,
                                                      double eps,
                                                      Dtype dt) override {
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        const int H = static_cast<int>(x_shape[2]);
        const int W = static_cast<int>(x_shape[3]);
        const int spatial = H * W;
        const auto& xs = std::get<CpuStorage>(x);
        const auto& gs = std::get<CpuStorage>(gamma);
        const auto& bs = std::get<CpuStorage>(beta);
        auto y_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * spatial, dt);
        auto nx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            const T* bp = reinterpret_cast<const T*>(bs.ptr.get());
            T* yp = reinterpret_cast<T*>(y_cpu.ptr.get());
            T* nxp = reinterpret_cast<T*>(nx_cpu.ptr.get());
            std::vector<T> Gx(static_cast<std::size_t>(B) * C);
            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C; ++c) {
                    T acc = T{0};
                    const T* xr = xp + (b * C + c) * spatial;
                    for (int s = 0; s < spatial; ++s)
                        acc += xr[s] * xr[s];
                    Gx[b * C + c] = static_cast<T>(std::sqrt(static_cast<double>(acc)));
                }
            for (int b = 0; b < B; ++b) {
                T mean = T{0};
                for (int c = 0; c < C; ++c)
                    mean += Gx[b * C + c];
                mean /= static_cast<T>(C);
                const T denom = mean + static_cast<T>(eps);
                for (int c = 0; c < C; ++c) {
                    const T nx = Gx[b * C + c] / denom;
                    nxp[b * C + c] = nx;
                    const T* xr = xp + (b * C + c) * spatial;
                    T* yr = yp + (b * C + c) * spatial;
                    for (int s = 0; s < spatial; ++s)
                        yr[s] = gp[c] * (xr[s] * nx) + bp[c] * xr[s];
                }
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("grn_forward").not_implemented("dtype");
        return {Storage{std::move(y_cpu)}, Storage{std::move(nx_cpu)}};
    }

    std::vector<Storage> global_response_norm_backward(const Storage& x,
                                                       const Storage& gamma,
                                                       const Storage& beta,
                                                       const Storage& saved_Nx,
                                                       const Storage& grad_out,
                                                       const Shape& x_shape,
                                                       double eps,
                                                       Dtype dt) override {
        const int B = static_cast<int>(x_shape[0]);
        const int C = static_cast<int>(x_shape[1]);
        const int H = static_cast<int>(x_shape[2]);
        const int W = static_cast<int>(x_shape[3]);
        const int spatial = H * W;
        const auto& xs = std::get<CpuStorage>(x);
        const auto& gs = std::get<CpuStorage>(gamma);
        const auto& bts = std::get<CpuStorage>(beta);
        const auto& nxs = std::get<CpuStorage>(saved_Nx);
        const auto& gos = std::get<CpuStorage>(grad_out);
        auto dx = alloc_cpu(static_cast<std::size_t>(B) * C * spatial, dt);
        auto dg = alloc_cpu(static_cast<std::size_t>(C), dt);
        auto db = alloc_cpu(static_cast<std::size_t>(C), dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            const T* bp = reinterpret_cast<const T*>(bts.ptr.get());
            const T* nxp = reinterpret_cast<const T*>(nxs.ptr.get());
            const T* gop = reinterpret_cast<const T*>(gos.ptr.get());
            T* dxp = reinterpret_cast<T*>(dx.ptr.get());
            T* dgp = reinterpret_cast<T*>(dg.ptr.get());
            T* dbp = reinterpret_cast<T*>(db.ptr.get());
            for (int c = 0; c < C; ++c) {
                dgp[c] = dbp[c] = T{0};
            }

            std::vector<T> Gx(static_cast<std::size_t>(B) * C, T{0});
            std::vector<T> mv(static_cast<std::size_t>(B), T{0});
            for (int b = 0; b < B; ++b) {
                for (int c = 0; c < C; ++c) {
                    T acc = T{0};
                    const T* xr = xp + (b * C + c) * spatial;
                    for (int s = 0; s < spatial; ++s)
                        acc += xr[s] * xr[s];
                    Gx[b * C + c] = static_cast<T>(std::sqrt(static_cast<double>(acc)));
                }
                T s = T{0};
                for (int c = 0; c < C; ++c)
                    s += Gx[b * C + c];
                mv[b] = s / static_cast<T>(C);
            }

            std::vector<T> A(static_cast<std::size_t>(B) * C, T{0});
            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C; ++c) {
                    const T* xr = xp + (b * C + c) * spatial;
                    const T* gr = gop + (b * C + c) * spatial;
                    T sgx = T{0};
                    for (int s = 0; s < spatial; ++s)
                        sgx += gr[s] * xr[s];
                    A[b * C + c] = sgx * gp[c];
                    dgp[c] += sgx * nxp[b * C + c];
                    dbp[c] += sgx;
                }

            std::vector<T> dG(static_cast<std::size_t>(B) * C, T{0});
            for (int b = 0; b < B; ++b) {
                const T denom = mv[b] + static_cast<T>(eps);
                const T denom2 = denom * denom;
                T sum_AG = T{0};
                for (int c = 0; c < C; ++c)
                    sum_AG += A[b * C + c] * Gx[b * C + c];
                const T common = -sum_AG / denom2 / static_cast<T>(C);
                for (int c = 0; c < C; ++c)
                    dG[b * C + c] = A[b * C + c] / denom + common;
            }

            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C; ++c) {
                    const T nbc = nxp[b * C + c];
                    const T gG = Gx[b * C + c];
                    const T invG = (gG > T{0}) ? T{1} / gG : T{0};
                    const T dGbc = dG[b * C + c];
                    const T gc = gp[c];

                    const T* xr = xp + (b * C + c) * spatial;
                    const T* gr = gop + (b * C + c) * spatial;
                    T* dxr = dxp + (b * C + c) * spatial;
                    const T bc = bp[c];
                    for (int s = 0; s < spatial; ++s)
                        dxr[s] = gc * nbc * gr[s] + bc * gr[s] + dGbc * xr[s] * invG;
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("grn_backward").not_implemented("dtype");
        return {Storage{std::move(dx)}, Storage{std::move(dg)}, Storage{std::move(db)}};
    }

    Storage interpolate_nearest_2d_forward(
        const Storage& input, const Shape& in_shape, int H_out, int W_out, Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int H_in = static_cast<int>(in_shape[2]);
        const int W_in = static_cast<int>(in_shape[3]);
        const auto& xs = std::get<CpuStorage>(input);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * H_out * W_out, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) {
                    const T* base = xp + ((n * C + c) * H_in) * W_in;
                    T* obase = op + ((n * C + c) * H_out) * W_out;
                    for (int h = 0; h < H_out; ++h) {
                        int yh =
                            static_cast<int>(std::floor(static_cast<double>(h) * H_in / H_out));
                        yh = std::clamp(yh, 0, H_in - 1);
                        for (int w = 0; w < W_out; ++w) {
                            int xw =
                                static_cast<int>(std::floor(static_cast<double>(w) * W_in / W_out));
                            xw = std::clamp(xw, 0, W_in - 1);
                            obase[h * W_out + w] = base[yh * W_in + xw];
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::interpolate_nearest_2d_forward")
                .not_implemented("dtype must be F32/F64");
        return Storage{std::move(out_cpu)};
    }

    Storage interpolate_nearest_3d_forward(const Storage& input,
                                           const Shape& in_shape,
                                           int D_out,
                                           int H_out,
                                           int W_out,
                                           Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int D_in = static_cast<int>(in_shape[2]);
        const int H_in = static_cast<int>(in_shape[3]);
        const int W_in = static_cast<int>(in_shape[4]);
        const auto& xs = std::get<CpuStorage>(input);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * D_out * H_out * W_out, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) {
                    const T* base = xp + (n * C + c) * D_in * H_in * W_in;
                    T* obase = op + (n * C + c) * D_out * H_out * W_out;
                    for (int d = 0; d < D_out; ++d) {
                        int dz = std::clamp(
                            static_cast<int>(std::floor(static_cast<double>(d) * D_in / D_out)), 0,
                            D_in - 1);
                        for (int h = 0; h < H_out; ++h) {
                            int yh = std::clamp(
                                static_cast<int>(std::floor(static_cast<double>(h) * H_in / H_out)),
                                0, H_in - 1);
                            for (int w = 0; w < W_out; ++w) {
                                int xw = std::clamp(static_cast<int>(std::floor(
                                                        static_cast<double>(w) * W_in / W_out)),
                                                    0, W_in - 1);
                                obase[(d * H_out + h) * W_out + w] =
                                    base[(dz * H_in + yh) * W_in + xw];
                            }
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::interpolate_nearest_3d_forward")
                .not_implemented("dtype must be F32/F64");
        return Storage{std::move(out_cpu)};
    }

    Storage interpolate_bilinear_forward(const Storage& input,
                                         const Shape& in_shape,
                                         int H_out,
                                         int W_out,
                                         bool align_corners,
                                         Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int H_in = static_cast<int>(in_shape[2]);
        const int W_in = static_cast<int>(in_shape[3]);
        const auto& xs = std::get<CpuStorage>(input);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * H_out * W_out, dt);

        auto src_coord_fn = [](int out_idx, int in_dim, int out_dim, bool align, auto T_tag) {
            using T = decltype(T_tag);
            if (align) {
                if (out_dim <= 1)
                    return T{0};
                return static_cast<T>(out_idx) * static_cast<T>(in_dim - 1) /
                       static_cast<T>(out_dim - 1);
            }
            const T x = (static_cast<T>(out_idx) + T{0.5}) * static_cast<T>(in_dim) /
                            static_cast<T>(out_dim) -
                        T{0.5};
            return x;
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            const std::size_t in_chan = static_cast<std::size_t>(H_in) * W_in;
            const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H_out; ++h) {
                        T iy = src_coord_fn(h, H_in, H_out, align_corners, T{});
                        if (iy < T{0})
                            iy = T{0};
                        if (iy > static_cast<T>(H_in - 1))
                            iy = static_cast<T>(H_in - 1);
                        int y0 = static_cast<int>(std::floor(iy));
                        int y1 = std::min(y0 + 1, H_in - 1);
                        const T dy = iy - static_cast<T>(y0);
                        for (int w = 0; w < W_out; ++w) {
                            T ix = src_coord_fn(w, W_in, W_out, align_corners, T{});
                            if (ix < T{0})
                                ix = T{0};
                            if (ix > static_cast<T>(W_in - 1))
                                ix = static_cast<T>(W_in - 1);
                            int x0 = static_cast<int>(std::floor(ix));
                            int x1 = std::min(x0 + 1, W_in - 1);
                            const T dx = ix - static_cast<T>(x0);
                            const T w00 = (T{1} - dy) * (T{1} - dx);
                            const T w01 = (T{1} - dy) * dx;
                            const T w10 = dy * (T{1} - dx);
                            const T w11 = dy * dx;
                            const T* base = xp + (n * C + c) * in_chan;
                            op[(n * C + c) * out_chan + h * W_out + w] =
                                base[y0 * W_in + x0] * w00 + base[y0 * W_in + x1] * w01 +
                                base[y1 * W_in + x0] * w10 + base[y1 * W_in + x1] * w11;
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::interpolate_bilinear_forward")
                .not_implemented("dtype must be F32/F64");
        return Storage{std::move(out_cpu)};
    }

    Storage interpolate_bilinear_backward(const Storage& grad_out,
                                          const Shape& in_shape,
                                          int H_out,
                                          int W_out,
                                          bool align_corners,
                                          Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int H_in = static_cast<int>(in_shape[2]);
        const int W_in = static_cast<int>(in_shape[3]);
        const auto& go = std::get<CpuStorage>(grad_out);
        auto dx_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * H_in * W_in, dt);

        auto src_coord_fn = [](int out_idx, int in_dim, int out_dim, bool align, auto T_tag) {
            using T = decltype(T_tag);
            if (align) {
                if (out_dim <= 1)
                    return T{0};
                return static_cast<T>(out_idx) * static_cast<T>(in_dim - 1) /
                       static_cast<T>(out_dim - 1);
            }
            const T x = (static_cast<T>(out_idx) + T{0.5}) * static_cast<T>(in_dim) /
                            static_cast<T>(out_dim) -
                        T{0.5};
            return x;
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* go_p = reinterpret_cast<const T*>(go.ptr.get());
            T* dx_p = reinterpret_cast<T*>(dx_cpu.ptr.get());
            const std::size_t in_chan = static_cast<std::size_t>(H_in) * W_in;
            const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;
            std::memset(dx_p, 0, sizeof(T) * static_cast<std::size_t>(N) * C * in_chan);
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) {
                    for (int h = 0; h < H_out; ++h) {
                        T iy = src_coord_fn(h, H_in, H_out, align_corners, T{});
                        if (iy < T{0})
                            iy = T{0};
                        if (iy > static_cast<T>(H_in - 1))
                            iy = static_cast<T>(H_in - 1);
                        int y0 = static_cast<int>(std::floor(iy));
                        int y1 = std::min(y0 + 1, H_in - 1);
                        const T dy = iy - static_cast<T>(y0);
                        for (int w = 0; w < W_out; ++w) {
                            T ix = src_coord_fn(w, W_in, W_out, align_corners, T{});
                            if (ix < T{0})
                                ix = T{0};
                            if (ix > static_cast<T>(W_in - 1))
                                ix = static_cast<T>(W_in - 1);
                            int x0 = static_cast<int>(std::floor(ix));
                            int x1 = std::min(x0 + 1, W_in - 1);
                            const T dx = ix - static_cast<T>(x0);
                            const T w00 = (T{1} - dy) * (T{1} - dx);
                            const T w01 = (T{1} - dy) * dx;
                            const T w10 = dy * (T{1} - dx);
                            const T w11 = dy * dx;
                            const T g = go_p[(n * C + c) * out_chan + h * W_out + w];
                            T* base = dx_p + (n * C + c) * in_chan;
                            base[y0 * W_in + x0] += g * w00;
                            base[y0 * W_in + x1] += g * w01;
                            base[y1 * W_in + x0] += g * w10;
                            base[y1 * W_in + x1] += g * w11;
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::interpolate_bilinear_backward")
                .not_implemented("dtype must be F32/F64");
        return Storage{std::move(dx_cpu)};
    }

    Storage interpolate_trilinear_forward(const Storage& input,
                                          const Shape& in_shape,
                                          int D_out,
                                          int H_out,
                                          int W_out,
                                          bool align_corners,
                                          Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int D_in = static_cast<int>(in_shape[2]);
        const int H_in = static_cast<int>(in_shape[3]);
        const int W_in = static_cast<int>(in_shape[4]);
        const auto& xs = std::get<CpuStorage>(input);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * D_out * H_out * W_out, dt);

        auto src_coord_fn = [](int o, int in_n, int out_n, bool align, auto T_tag) {
            using T = decltype(T_tag);
            T v;
            if (align) {
                if (out_n <= 1) {
                    v = T{0};
                } else {
                    v = static_cast<T>(o) * static_cast<T>(in_n - 1) / static_cast<T>(out_n - 1);
                }
            } else {
                v = (static_cast<T>(o) + T{0.5}) * static_cast<T>(in_n) / static_cast<T>(out_n) -
                    T{0.5};
            }
            if (v < T{0})
                v = T{0};
            if (v > static_cast<T>(in_n - 1))
                v = static_cast<T>(in_n - 1);
            return v;
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            const std::size_t in_chan = static_cast<std::size_t>(D_in) * H_in * W_in;
            const std::size_t out_chan = static_cast<std::size_t>(D_out) * H_out * W_out;
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) {
                    const T* base = xp + (n * C + c) * in_chan;
                    T* obase = op + (n * C + c) * out_chan;
                    for (int d = 0; d < D_out; ++d) {
                        T iz = src_coord_fn(d, D_in, D_out, align_corners, T{});
                        int z0 = static_cast<int>(std::floor(iz));
                        int z1 = std::min(z0 + 1, D_in - 1);
                        const T dz = iz - static_cast<T>(z0);
                        for (int h = 0; h < H_out; ++h) {
                            T iy = src_coord_fn(h, H_in, H_out, align_corners, T{});
                            int y0 = static_cast<int>(std::floor(iy));
                            int y1 = std::min(y0 + 1, H_in - 1);
                            const T dy = iy - static_cast<T>(y0);
                            for (int w = 0; w < W_out; ++w) {
                                T ix = src_coord_fn(w, W_in, W_out, align_corners, T{});
                                int x0 = static_cast<int>(std::floor(ix));
                                int x1 = std::min(x0 + 1, W_in - 1);
                                const T dx = ix - static_cast<T>(x0);
                                auto idx = [&](int z, int y, int x) -> std::size_t {
                                    return static_cast<std::size_t>(z) * H_in * W_in +
                                           static_cast<std::size_t>(y) * W_in + x;
                                };
                                const T c000 = base[idx(z0, y0, x0)];
                                const T c001 = base[idx(z0, y0, x1)];
                                const T c010 = base[idx(z0, y1, x0)];
                                const T c011 = base[idx(z0, y1, x1)];
                                const T c100 = base[idx(z1, y0, x0)];
                                const T c101 = base[idx(z1, y0, x1)];
                                const T c110 = base[idx(z1, y1, x0)];
                                const T c111 = base[idx(z1, y1, x1)];
                                const T c00 = c000 * (T{1} - dx) + c001 * dx;
                                const T c01 = c010 * (T{1} - dx) + c011 * dx;
                                const T c10 = c100 * (T{1} - dx) + c101 * dx;
                                const T c11 = c110 * (T{1} - dx) + c111 * dx;
                                const T c0 = c00 * (T{1} - dy) + c01 * dy;
                                const T c1 = c10 * (T{1} - dy) + c11 * dy;
                                obase[d * H_out * W_out + h * W_out + w] =
                                    c0 * (T{1} - dz) + c1 * dz;
                            }
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::interpolate_trilinear_forward")
                .not_implemented("dtype must be F32/F64");
        return Storage{std::move(out_cpu)};
    }

    Storage interpolate_trilinear_backward(const Storage& grad_out,
                                           const Shape& in_shape,
                                           int D_out,
                                           int H_out,
                                           int W_out,
                                           bool align_corners,
                                           Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int D_in = static_cast<int>(in_shape[2]);
        const int H_in = static_cast<int>(in_shape[3]);
        const int W_in = static_cast<int>(in_shape[4]);
        const auto& go = std::get<CpuStorage>(grad_out);
        auto dx_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * D_in * H_in * W_in, dt);

        auto src_coord_fn = [](int o, int in_n, int out_n, bool align, auto T_tag) {
            using T = decltype(T_tag);
            T v;
            if (align) {
                if (out_n <= 1) {
                    v = T{0};
                } else {
                    v = static_cast<T>(o) * static_cast<T>(in_n - 1) / static_cast<T>(out_n - 1);
                }
            } else {
                v = (static_cast<T>(o) + T{0.5}) * static_cast<T>(in_n) / static_cast<T>(out_n) -
                    T{0.5};
            }
            if (v < T{0})
                v = T{0};
            if (v > static_cast<T>(in_n - 1))
                v = static_cast<T>(in_n - 1);
            return v;
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* go_p = reinterpret_cast<const T*>(go.ptr.get());
            T* dx_p = reinterpret_cast<T*>(dx_cpu.ptr.get());
            const std::size_t in_chan = static_cast<std::size_t>(D_in) * H_in * W_in;
            const std::size_t out_chan = static_cast<std::size_t>(D_out) * H_out * W_out;
            std::memset(dx_p, 0, sizeof(T) * static_cast<std::size_t>(N) * C * in_chan);
            for (int n = 0; n < N; ++n)
                for (int c = 0; c < C; ++c) {
                    T* base = dx_p + (n * C + c) * in_chan;
                    const T* go_base = go_p + (n * C + c) * out_chan;
                    for (int d = 0; d < D_out; ++d) {
                        T iz = src_coord_fn(d, D_in, D_out, align_corners, T{});
                        int z0 = static_cast<int>(std::floor(iz));
                        int z1 = std::min(z0 + 1, D_in - 1);
                        const T dz = iz - static_cast<T>(z0);
                        for (int h = 0; h < H_out; ++h) {
                            T iy = src_coord_fn(h, H_in, H_out, align_corners, T{});
                            int y0 = static_cast<int>(std::floor(iy));
                            int y1 = std::min(y0 + 1, H_in - 1);
                            const T dy = iy - static_cast<T>(y0);
                            for (int w = 0; w < W_out; ++w) {
                                T ix = src_coord_fn(w, W_in, W_out, align_corners, T{});
                                int x0 = static_cast<int>(std::floor(ix));
                                int x1 = std::min(x0 + 1, W_in - 1);
                                const T dx = ix - static_cast<T>(x0);
                                const T g = go_base[d * H_out * W_out + h * W_out + w];
                                auto add = [&](int z, int y, int x, T weight) {
                                    base[static_cast<std::size_t>(z) * H_in * W_in +
                                         static_cast<std::size_t>(y) * W_in + x] += g * weight;
                                };
                                const T omdx = T{1} - dx, omdy = T{1} - dy, omdz = T{1} - dz;
                                add(z0, y0, x0, omdz * omdy * omdx);
                                add(z0, y0, x1, omdz * omdy * dx);
                                add(z0, y1, x0, omdz * dy * omdx);
                                add(z0, y1, x1, omdz * dy * dx);
                                add(z1, y0, x0, dz * omdy * omdx);
                                add(z1, y0, x1, dz * omdy * dx);
                                add(z1, y1, x0, dz * dy * omdx);
                                add(z1, y1, x1, dz * dy * dx);
                            }
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::interpolate_trilinear_backward")
                .not_implemented("dtype must be F32/F64");
        return Storage{std::move(dx_cpu)};
    }

    Storage affine_grid_forward(
        const Storage& theta, int N, int H, int W, bool align_corners, Dtype dt) override {
        const auto& th = std::get<CpuStorage>(theta);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * H * W * 2, dt);

        auto x_norm_fn = [](int w, int W2, bool align, auto T_tag) {
            using T = decltype(T_tag);
            if (align)
                return W2 > 1 ? T{-1} + (T{2} * static_cast<T>(w)) / static_cast<T>(W2 - 1) : T{0};
            return T{-1} + (T{2} * static_cast<T>(w) + T{1}) / static_cast<T>(W2);
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* tp = reinterpret_cast<const T*>(th.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            for (int n = 0; n < N; ++n) {
                const T t00 = tp[n * 6 + 0], t01 = tp[n * 6 + 1], t02 = tp[n * 6 + 2];
                const T t10 = tp[n * 6 + 3], t11 = tp[n * 6 + 4], t12 = tp[n * 6 + 5];
                for (int h = 0; h < H; ++h) {
                    const T y = x_norm_fn(h, H, align_corners, T{});
                    for (int w = 0; w < W; ++w) {
                        const T x = x_norm_fn(w, W, align_corners, T{});
                        const std::size_t base =
                            ((static_cast<std::size_t>(n) * H + h) * W + w) * 2;
                        op[base + 0] = t00 * x + t01 * y + t02;
                        op[base + 1] = t10 * x + t11 * y + t12;
                    }
                }
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::affine_grid_forward").not_implemented("dtype must be F32/F64");
        return Storage{std::move(out_cpu)};
    }

    Storage affine_grid_backward(
        const Storage& grad_grid, int N, int H, int W, bool align_corners, Dtype dt) override {
        const auto& gs = std::get<CpuStorage>(grad_grid);
        auto dtheta = alloc_cpu(static_cast<std::size_t>(N) * 2 * 3, dt);
        std::memset(dtheta.ptr.get(), 0, dtheta.nbytes);

        auto x_norm_fn = [](int w, int W2, bool align, auto T_tag) {
            using T = decltype(T_tag);
            if (align)
                return W2 > 1 ? T{-1} + (T{2} * static_cast<T>(w)) / static_cast<T>(W2 - 1) : T{0};
            return T{-1} + (T{2} * static_cast<T>(w) + T{1}) / static_cast<T>(W2);
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            T* dp = reinterpret_cast<T*>(dtheta.ptr.get());
            for (int n = 0; n < N; ++n) {
                T s00 = T{0}, s01 = T{0}, s02 = T{0};
                T s10 = T{0}, s11 = T{0}, s12 = T{0};
                for (int h = 0; h < H; ++h) {
                    const T y = x_norm_fn(h, H, align_corners, T{});
                    for (int w = 0; w < W; ++w) {
                        const T x = x_norm_fn(w, W, align_corners, T{});
                        const std::size_t base =
                            ((static_cast<std::size_t>(n) * H + h) * W + w) * 2;
                        const T gx = gp[base + 0];
                        const T gy = gp[base + 1];
                        s00 += gx * x;
                        s01 += gx * y;
                        s02 += gx;
                        s10 += gy * x;
                        s11 += gy * y;
                        s12 += gy;
                    }
                }
                dp[n * 6 + 0] = s00;
                dp[n * 6 + 1] = s01;
                dp[n * 6 + 2] = s02;
                dp[n * 6 + 3] = s10;
                dp[n * 6 + 4] = s11;
                dp[n * 6 + 5] = s12;
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::affine_grid_backward").not_implemented("dtype must be F32/F64");
        return Storage{std::move(dtheta)};
    }

    Storage grid_sample_forward(const Storage& input,
                                const Storage& grid,
                                const Shape& in_shape,
                                const Shape& grid_shape,
                                int mode,
                                int padding_mode,
                                bool align_corners,
                                Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int H_in = static_cast<int>(in_shape[2]);
        const int W_in = static_cast<int>(in_shape[3]);
        const int H_out = static_cast<int>(grid_shape[1]);
        const int W_out = static_cast<int>(grid_shape[2]);
        const auto& xs = std::get<CpuStorage>(input);
        const auto& gs = std::get<CpuStorage>(grid);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * H_out * W_out, dt);

        auto denorm_fn = [](auto g, int dim, bool align) {
            using T = decltype(g);
            if (align)
                return (g + T{1}) * static_cast<T>(dim - 1) / T{2};
            return (g + T{1}) * static_cast<T>(dim) / T{2} - static_cast<T>(0.5);
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            const std::size_t in_chan = static_cast<std::size_t>(H_in) * W_in;
            const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;
            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        const std::size_t gidx =
                            ((static_cast<std::size_t>(n) * H_out + h) * W_out + w) * 2;
                        T ix = denorm_fn(gp[gidx + 0], W_in, align_corners);
                        T iy = denorm_fn(gp[gidx + 1], H_in, align_corners);
                        if (mode == 1) {
                            int ixr = static_cast<int>(std::nearbyint(static_cast<double>(ix)));
                            int iyr = static_cast<int>(std::nearbyint(static_cast<double>(iy)));
                            bool oob = false;
                            if (padding_mode == 1) {
                                ixr = std::clamp(ixr, 0, W_in - 1);
                                iyr = std::clamp(iyr, 0, H_in - 1);
                            } else {
                                if (ixr < 0 || ixr > W_in - 1 || iyr < 0 || iyr > H_in - 1)
                                    oob = true;
                                ixr = std::clamp(ixr, 0, W_in - 1);
                                iyr = std::clamp(iyr, 0, H_in - 1);
                            }
                            for (int c = 0; c < C; ++c) {
                                T v = oob ? T{0}
                                          : xp[((static_cast<std::size_t>(n) * C + c) * in_chan) +
                                               iyr * W_in + ixr];
                                op[((static_cast<std::size_t>(n) * C + c) * out_chan) + h * W_out +
                                   w] = v;
                            }
                        } else {
                            if (padding_mode == 1) {
                                ix = std::clamp<T>(ix, T{0}, static_cast<T>(W_in - 1));
                                iy = std::clamp<T>(iy, T{0}, static_cast<T>(H_in - 1));
                            }
                            const T x0f = std::floor(ix);
                            const T y0f = std::floor(iy);
                            const int x0 = static_cast<int>(x0f);
                            const int y0 = static_cast<int>(y0f);
                            const int x1 = x0 + 1, y1 = y0 + 1;
                            const T wa = (static_cast<T>(x1) - ix) * (static_cast<T>(y1) - iy);
                            const T wb = (static_cast<T>(x1) - ix) * (iy - static_cast<T>(y0));
                            const T wc = (ix - static_cast<T>(x0)) * (static_cast<T>(y1) - iy);
                            const T wd = (ix - static_cast<T>(x0)) * (iy - static_cast<T>(y0));
                            auto fetch = [&](int yi, int xi, int c) -> T {
                                const bool oob =
                                    (xi < 0 || xi > W_in - 1 || yi < 0 || yi > H_in - 1);
                                if (oob && padding_mode == 0)
                                    return T{0};
                                const int ycl = std::clamp(yi, 0, H_in - 1);
                                const int xcl = std::clamp(xi, 0, W_in - 1);
                                return xp[((static_cast<std::size_t>(n) * C + c) * in_chan) +
                                          ycl * W_in + xcl];
                            };
                            for (int c = 0; c < C; ++c) {
                                const T Ia = fetch(y0, x0, c);
                                const T Ib = fetch(y1, x0, c);
                                const T Ic = fetch(y0, x1, c);
                                const T Id = fetch(y1, x1, c);
                                op[((static_cast<std::size_t>(n) * C + c) * out_chan) + h * W_out +
                                   w] = Ia * wa + Ib * wb + Ic * wc + Id * wd;
                            }
                        }
                    }
                }
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::grid_sample_forward").not_implemented("dtype must be F32/F64");
        return Storage{std::move(out_cpu)};
    }

    std::vector<Storage> grid_sample_backward(const Storage& grad_out,
                                              const Storage& input,
                                              const Storage& grid,
                                              const Shape& in_shape,
                                              const Shape& grid_shape,
                                              int mode,
                                              int padding_mode,
                                              bool align_corners,
                                              Dtype dt) override {
        const int N = static_cast<int>(in_shape[0]);
        const int C = static_cast<int>(in_shape[1]);
        const int H_in = static_cast<int>(in_shape[2]);
        const int W_in = static_cast<int>(in_shape[3]);
        const int H_out = static_cast<int>(grid_shape[1]);
        const int W_out = static_cast<int>(grid_shape[2]);
        const auto& xs = std::get<CpuStorage>(input);
        const auto& gs = std::get<CpuStorage>(grid);
        const auto& gout = std::get<CpuStorage>(grad_out);
        auto dx = alloc_cpu(static_cast<std::size_t>(N) * C * H_in * W_in, dt);
        auto dg = alloc_cpu(static_cast<std::size_t>(N) * H_out * W_out * 2, dt);
        std::memset(dx.ptr.get(), 0, dx.nbytes);
        std::memset(dg.ptr.get(), 0, dg.nbytes);

        auto denorm_fn = [](auto g, int dim, bool align) {
            using T = decltype(g);
            if (align)
                return (g + T{1}) * static_cast<T>(dim - 1) / T{2};
            return (g + T{1}) * static_cast<T>(dim) / T{2} - static_cast<T>(0.5);
        };
        auto denorm_grad_factor_fn = [](int dim, bool align, auto T_tag) {
            using T = decltype(T_tag);
            return align ? static_cast<T>(dim - 1) / T{2} : static_cast<T>(dim) / T{2};
        };

        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            const T* op = reinterpret_cast<const T*>(gout.ptr.get());
            T* dxp = reinterpret_cast<T*>(dx.ptr.get());
            T* dgp = reinterpret_cast<T*>(dg.ptr.get());
            const std::size_t in_chan = static_cast<std::size_t>(H_in) * W_in;
            const std::size_t out_chan = static_cast<std::size_t>(H_out) * W_out;
            const T sx = denorm_grad_factor_fn(W_in, align_corners, T{});
            const T sy = denorm_grad_factor_fn(H_in, align_corners, T{});
            for (int n = 0; n < N; ++n) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        const std::size_t gidx =
                            ((static_cast<std::size_t>(n) * H_out + h) * W_out + w) * 2;
                        const T gx_norm = gp[gidx + 0];
                        const T gy_norm = gp[gidx + 1];
                        T ix = denorm_fn(gx_norm, W_in, align_corners);
                        T iy = denorm_fn(gy_norm, H_in, align_corners);
                        bool clipped_x = false, clipped_y = false;
                        if (mode == 0 && padding_mode == 1) {
                            if (ix < T{0}) {
                                ix = T{0};
                                clipped_x = true;
                            }
                            if (ix > static_cast<T>(W_in - 1)) {
                                ix = static_cast<T>(W_in - 1);
                                clipped_x = true;
                            }
                            if (iy < T{0}) {
                                iy = T{0};
                                clipped_y = true;
                            }
                            if (iy > static_cast<T>(H_in - 1)) {
                                iy = static_cast<T>(H_in - 1);
                                clipped_y = true;
                            }
                        }
                        if (mode == 1) {
                            int ixr = static_cast<int>(std::nearbyint(static_cast<double>(ix)));
                            int iyr = static_cast<int>(std::nearbyint(static_cast<double>(iy)));
                            bool oob = false;
                            if (padding_mode == 1) {
                                ixr = std::clamp(ixr, 0, W_in - 1);
                                iyr = std::clamp(iyr, 0, H_in - 1);
                            } else {
                                if (ixr < 0 || ixr > W_in - 1 || iyr < 0 || iyr > H_in - 1)
                                    oob = true;
                                ixr = std::clamp(ixr, 0, W_in - 1);
                                iyr = std::clamp(iyr, 0, H_in - 1);
                            }
                            if (!oob) {
                                for (int c = 0; c < C; ++c) {
                                    const T go =
                                        op[((static_cast<std::size_t>(n) * C + c) * out_chan) +
                                           h * W_out + w];
                                    dxp[((static_cast<std::size_t>(n) * C + c) * in_chan) +
                                        iyr * W_in + ixr] += go;
                                }
                            }
                            continue;
                        }
                        const T x0f = std::floor(ix);
                        const T y0f = std::floor(iy);
                        const int x0 = static_cast<int>(x0f);
                        const int y0 = static_cast<int>(y0f);
                        const int x1 = x0 + 1, y1 = y0 + 1;
                        const T wa = (static_cast<T>(x1) - ix) * (static_cast<T>(y1) - iy);
                        const T wb = (static_cast<T>(x1) - ix) * (iy - static_cast<T>(y0));
                        const T wc = (ix - static_cast<T>(x0)) * (static_cast<T>(y1) - iy);
                        const T wd = (ix - static_cast<T>(x0)) * (iy - static_cast<T>(y0));
                        auto in_bounds = [&](int yi, int xi) -> bool {
                            return xi >= 0 && xi <= W_in - 1 && yi >= 0 && yi <= H_in - 1;
                        };
                        auto fetch_for_dgrid = [&](int yi, int xi, int c) -> T {
                            const bool oob = !in_bounds(yi, xi);
                            if (oob && padding_mode == 0)
                                return T{0};
                            const int ycl = std::clamp(yi, 0, H_in - 1);
                            const int xcl = std::clamp(xi, 0, W_in - 1);
                            return xp[((static_cast<std::size_t>(n) * C + c) * in_chan) +
                                      ycl * W_in + xcl];
                        };
                        auto scatter_dx = [&](int yi, int xi, int c, T contrib) {
                            const bool oob = !in_bounds(yi, xi);
                            if (oob && padding_mode == 0)
                                return;
                            const int ycl = std::clamp(yi, 0, H_in - 1);
                            const int xcl = std::clamp(xi, 0, W_in - 1);
                            dxp[((static_cast<std::size_t>(n) * C + c) * in_chan) + ycl * W_in +
                                xcl] += contrib;
                        };
                        T dix_acc = T{0}, diy_acc = T{0};
                        for (int c = 0; c < C; ++c) {
                            const T go = op[((static_cast<std::size_t>(n) * C + c) * out_chan) +
                                            h * W_out + w];
                            scatter_dx(y0, x0, c, go * wa);
                            scatter_dx(y1, x0, c, go * wb);
                            scatter_dx(y0, x1, c, go * wc);
                            scatter_dx(y1, x1, c, go * wd);
                            const T Ia = fetch_for_dgrid(y0, x0, c);
                            const T Ib = fetch_for_dgrid(y1, x0, c);
                            const T Ic = fetch_for_dgrid(y0, x1, c);
                            const T Id = fetch_for_dgrid(y1, x1, c);
                            const T dy_t1 = static_cast<T>(y1) - iy;
                            const T dy_t2 = iy - static_cast<T>(y0);
                            const T dx_t1 = static_cast<T>(x1) - ix;
                            const T dx_t2 = ix - static_cast<T>(x0);
                            dix_acc += go * ((Ic - Ia) * dy_t1 + (Id - Ib) * dy_t2);
                            diy_acc += go * ((Ib - Ia) * dx_t1 + (Id - Ic) * dx_t2);
                        }
                        if (clipped_x)
                            dix_acc = T{0};
                        if (clipped_y)
                            diy_acc = T{0};
                        dgp[gidx + 0] = dix_acc * sx;
                        dgp[gidx + 1] = diy_acc * sy;
                    }
                }
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::grid_sample_backward").not_implemented("dtype must be F32/F64");
        return {Storage{std::move(dx)}, Storage{std::move(dg)}};
    }

    Storage bilinear_layer_forward(const Storage& x1,
                                   const Storage& x2,
                                   const Storage& weight,
                                   const Storage& bias,
                                   bool has_bias,
                                   const Shape& x1_shape,
                                   const Shape& x2_shape,
                                   const Shape& w_shape,
                                   Dtype dt) override {
        std::size_t B = 1;
        for (std::size_t i = 0; i + 1 < x1_shape.size(); ++i)
            B *= static_cast<std::size_t>(x1_shape[i]);
        const std::size_t D1 = static_cast<std::size_t>(x1_shape.back());
        const std::size_t D2 = static_cast<std::size_t>(x2_shape.back());
        const std::size_t Dout = static_cast<std::size_t>(w_shape[0]);
        const auto& xs1 = std::get<CpuStorage>(x1);
        const auto& xs2 = std::get<CpuStorage>(x2);
        const auto& ws = std::get<CpuStorage>(weight);
        const CpuStorage* bs = has_bias ? &std::get<CpuStorage>(bias) : nullptr;
        auto out_cpu = alloc_cpu(B * Dout, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* x1p = reinterpret_cast<const T*>(xs1.ptr.get());
            const T* x2p = reinterpret_cast<const T*>(xs2.ptr.get());
            const T* wp = reinterpret_cast<const T*>(ws.ptr.get());
            const T* bp = bs ? reinterpret_cast<const T*>(bs->ptr.get()) : nullptr;
            T* yp = reinterpret_cast<T*>(out_cpu.ptr.get());
            for (std::size_t bi = 0; bi < B; ++bi) {
                const T* x1b = x1p + bi * D1;
                const T* x2b = x2p + bi * D2;
                T* yb = yp + bi * Dout;
                for (std::size_t k = 0; k < Dout; ++k) {
                    const T* Wk = wp + k * D1 * D2;
                    T acc = T{0};
                    for (std::size_t i = 0; i < D1; ++i) {
                        T row = T{0};
                        for (std::size_t j = 0; j < D2; ++j)
                            row += Wk[i * D2 + j] * x2b[j];
                        acc += x1b[i] * row;
                    }
                    yb[k] = acc + (bp ? bp[k] : T{0});
                }
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::bilinear_layer_forward").not_implemented("dtype");
        return Storage{std::move(out_cpu)};
    }

    std::vector<Storage> bilinear_layer_backward(const Storage& grad_out,
                                                 const Storage& x1,
                                                 const Storage& x2,
                                                 const Storage& weight,
                                                 const Shape& x1_shape,
                                                 const Shape& x2_shape,
                                                 const Shape& w_shape,
                                                 bool has_bias,
                                                 Dtype dt) override {
        std::size_t B = 1;
        for (std::size_t i = 0; i + 1 < x1_shape.size(); ++i)
            B *= static_cast<std::size_t>(x1_shape[i]);
        const std::size_t D1 = static_cast<std::size_t>(x1_shape.back());
        const std::size_t D2 = static_cast<std::size_t>(x2_shape.back());
        const std::size_t Dout = static_cast<std::size_t>(w_shape[0]);
        const auto& xs1 = std::get<CpuStorage>(x1);
        const auto& xs2 = std::get<CpuStorage>(x2);
        const auto& ws = std::get<CpuStorage>(weight);
        const auto& gys = std::get<CpuStorage>(grad_out);
        auto dx1_cpu = alloc_cpu(B * D1, dt);
        auto dx2_cpu = alloc_cpu(B * D2, dt);
        auto dW_cpu = alloc_cpu(Dout * D1 * D2, dt);
        auto db_cpu = has_bias ? alloc_cpu(Dout, dt) : CpuStorage{};
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* x1p = reinterpret_cast<const T*>(xs1.ptr.get());
            const T* x2p = reinterpret_cast<const T*>(xs2.ptr.get());
            const T* wp = reinterpret_cast<const T*>(ws.ptr.get());
            const T* gyp = reinterpret_cast<const T*>(gys.ptr.get());
            T* dx1p = reinterpret_cast<T*>(dx1_cpu.ptr.get());
            T* dx2p = reinterpret_cast<T*>(dx2_cpu.ptr.get());
            T* dwp = reinterpret_cast<T*>(dW_cpu.ptr.get());
            T* dbp = has_bias ? reinterpret_cast<T*>(db_cpu.ptr.get()) : nullptr;
            std::memset(dx1p, 0, sizeof(T) * B * D1);
            std::memset(dx2p, 0, sizeof(T) * B * D2);
            std::memset(dwp, 0, sizeof(T) * Dout * D1 * D2);
            if (dbp)
                std::memset(dbp, 0, sizeof(T) * Dout);
            for (std::size_t bi = 0; bi < B; ++bi) {
                const T* x1b = x1p + bi * D1;
                const T* x2b = x2p + bi * D2;
                const T* gb = gyp + bi * Dout;
                T* dx1b = dx1p + bi * D1;
                T* dx2b = dx2p + bi * D2;
                for (std::size_t k = 0; k < Dout; ++k) {
                    const T g = gb[k];
                    if (dbp)
                        dbp[k] += g;
                    const T* Wk = wp + k * D1 * D2;
                    T* dWk = dwp + k * D1 * D2;
                    for (std::size_t i = 0; i < D1; ++i) {
                        T row = T{0};
                        for (std::size_t j = 0; j < D2; ++j) {
                            row += Wk[i * D2 + j] * x2b[j];
                            dx2b[j] += g * x1b[i] * Wk[i * D2 + j];
                            dWk[i * D2 + j] += g * x1b[i] * x2b[j];
                        }
                        dx1b[i] += g * row;
                    }
                }
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::bilinear_layer_backward").not_implemented("dtype");
        std::vector<Storage> out;
        out.push_back(Storage{std::move(dx1_cpu)});
        out.push_back(Storage{std::move(dx2_cpu)});
        out.push_back(Storage{std::move(dW_cpu)});
        if (has_bias)
            out.push_back(Storage{std::move(db_cpu)});
        else {
            CpuStorage empty;
            empty.dtype = dt;
            empty.nbytes = 0;
            out.push_back(Storage{std::move(empty)});
        }
        return out;
    }

    Storage one_hot_forward(const Storage& indices,
                            const Shape& indices_shape,
                            int num_classes,
                            Dtype out_dtype) override {
        const std::size_t M = shape_numel(indices_shape);
        const auto& is_ = std::get<CpuStorage>(indices);
        auto out_cpu = alloc_cpu(M * static_cast<std::size_t>(num_classes), out_dtype);
        std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);
        auto read_idx = [&](std::size_t i) -> std::int64_t {
            switch (is_.dtype) {
            case Dtype::I8:
                return reinterpret_cast<const std::int8_t*>(is_.ptr.get())[i];
            case Dtype::I16:
                return reinterpret_cast<const std::int16_t*>(is_.ptr.get())[i];
            case Dtype::I32:
                return reinterpret_cast<const std::int32_t*>(is_.ptr.get())[i];
            case Dtype::I64:
                return reinterpret_cast<const std::int64_t*>(is_.ptr.get())[i];
            default:
                return reinterpret_cast<const std::int32_t*>(is_.ptr.get())[i];
            }
        };
        for (std::size_t i = 0; i < M; ++i) {
            const std::int64_t cls = read_idx(i);
            if (cls < 0 || cls >= num_classes)
                continue;
            const std::size_t pos = i * static_cast<std::size_t>(num_classes) + cls;
            switch (out_dtype) {
            case Dtype::F32:
                reinterpret_cast<float*>(out_cpu.ptr.get())[pos] = 1.f;
                break;
            case Dtype::F64:
                reinterpret_cast<double*>(out_cpu.ptr.get())[pos] = 1.0;
                break;
            case Dtype::I8:
            case Dtype::Bool:
                reinterpret_cast<std::uint8_t*>(out_cpu.ptr.get())[pos] = 1;
                break;
            case Dtype::I16:
                reinterpret_cast<std::int16_t*>(out_cpu.ptr.get())[pos] = 1;
                break;
            case Dtype::I32:
                reinterpret_cast<std::int32_t*>(out_cpu.ptr.get())[pos] = 1;
                break;
            case Dtype::I64:
                reinterpret_cast<std::int64_t*>(out_cpu.ptr.get())[pos] = 1;
                break;
            default:
                ErrorBuilder("one_hot").not_implemented("out dtype not supported");
            }
        }
        return Storage{std::move(out_cpu)};
    }

    Storage rotate_forward(const Storage& input,
                           const Shape& shape,
                           double angle_rad_neg,
                           double cx,
                           double cy,
                           Dtype dt) override {
        const int N = static_cast<int>(shape[0]);
        const int C = static_cast<int>(shape[1]);
        const int H = static_cast<int>(shape[2]);
        const int W = static_cast<int>(shape[3]);
        const auto& xs = std::get<CpuStorage>(input);
        auto out_cpu = alloc_cpu(static_cast<std::size_t>(N) * C * H * W, dt);
        const double cosv = std::cos(angle_rad_neg);
        const double sinv = std::sin(angle_rad_neg);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
            T* op = reinterpret_cast<T*>(out_cpu.ptr.get());
            for (int n = 0; n < N; ++n)
                for (int ch = 0; ch < C; ++ch) {
                    const T* base = xp + (n * C + ch) * H * W;
                    T* obase = op + (n * C + ch) * H * W;
                    for (int y = 0; y < H; ++y)
                        for (int x = 0; x < W; ++x) {
                            const double xsd = cosv * (x - cx) - sinv * (y - cy) + cx;
                            const double ysd = sinv * (x - cx) + cosv * (y - cy) + cy;
                            int xs2 = static_cast<int>(std::floor(xsd + 0.5));
                            int ys2 = static_cast<int>(std::floor(ysd + 0.5));
                            if (xs2 < 0 || xs2 >= W || ys2 < 0 || ys2 >= H)
                                obase[y * W + x] = T{0};
                            else
                                obase[y * W + x] = base[ys2 * W + xs2];
                        }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu::rotate_forward").not_implemented("dtype");
        return Storage{std::move(out_cpu)};
    }

    Storage rope_backward(const Storage& grad_out,
                          const Storage& saved_cos,
                          const Storage& saved_sin,
                          const Shape& x_shape,
                          bool interleaved,
                          Dtype dt) override {
        const std::size_t ndim = x_shape.size();
        const std::size_t L = static_cast<std::size_t>(x_shape[ndim - 2]);
        const std::size_t D = static_cast<std::size_t>(x_shape.back());
        const std::size_t Dh = D / 2;
        std::size_t batch = 1;
        for (std::size_t i = 0; i + 2 < ndim; ++i)
            batch *= static_cast<std::size_t>(x_shape[i]);
        const auto& gs = std::get<CpuStorage>(grad_out);
        const auto& cs = std::get<CpuStorage>(saved_cos);
        const auto& ss = std::get<CpuStorage>(saved_sin);
        auto dx = alloc_cpu(batch * L * D, dt);
        auto run = [&](auto tag) {
            using T = decltype(tag);
            const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
            const T* cp = reinterpret_cast<const T*>(cs.ptr.get());
            const T* sp = reinterpret_cast<const T*>(ss.ptr.get());
            T* dxp = reinterpret_cast<T*>(dx.ptr.get());
            for (std::size_t b = 0; b < batch; ++b)
                for (std::size_t i = 0; i < L; ++i) {
                    const T* grow = gp + (b * L + i) * D;
                    T* drow = dxp + (b * L + i) * D;
                    const T* crow = cp + i * Dh;
                    const T* srow = sp + i * Dh;
                    if (interleaved) {
                        for (std::size_t k = 0; k < Dh; ++k) {
                            const T ge = grow[2 * k], go = grow[2 * k + 1];
                            drow[2 * k] = ge * crow[k] + go * srow[k];
                            drow[2 * k + 1] = go * crow[k] - ge * srow[k];
                        }
                    } else {
                        for (std::size_t k = 0; k < Dh; ++k) {
                            drow[k] = grow[k] * crow[k] + grow[k + Dh] * srow[k];
                            drow[k + Dh] = grow[k + Dh] * crow[k] - grow[k] * srow[k];
                        }
                    }
                }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("rope_backward").not_implemented("dtype");
        return Storage{std::move(dx)};
    }

private:
    // Dispatches im2col to the 1-D, 2-D, or 3-D variant based on N.
    // S/K/O are input spatial sizes, kernel sizes, and output sizes indexed 0..N-1.
    static void conv_nd_im2col_f32(const float* x,
                                   float* cols,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
        if (N == 1)
            cpu::im2col_1d_f32(x, cols, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
        else if (N == 2)
            cpu::im2col_f32(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                            pad[0], pad[1], dilation[0], dilation[1]);
        else
            cpu::im2col_3d_f32(x, cols, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                               dilation[1], dilation[2]);
    }
    static void conv_nd_im2col_f64(const double* x,
                                   double* cols,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
        if (N == 1)
            cpu::im2col_1d_f64(x, cols, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
        else if (N == 2)
            cpu::im2col_f64(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                            pad[0], pad[1], dilation[0], dilation[1]);
        else
            cpu::im2col_3d_f64(x, cols, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                               dilation[1], dilation[2]);
    }
    static void conv_nd_col2im_f32(const float* cols,
                                   float* dx,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
        if (N == 1)
            cpu::col2im_1d_f32(cols, dx, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
        else if (N == 2)
            cpu::col2im_f32(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                            pad[0], pad[1], dilation[0], dilation[1]);
        else
            cpu::col2im_3d_f32(cols, dx, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                               dilation[1], dilation[2]);
    }
    static void conv_nd_col2im_f64(const double* cols,
                                   double* dx,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
        if (N == 1)
            cpu::col2im_1d_f64(cols, dx, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
        else if (N == 2)
            cpu::col2im_f64(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                            pad[0], pad[1], dilation[0], dilation[1]);
        else
            cpu::col2im_3d_f64(cols, dx, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                               dilation[1], dilation[2]);
    }
    static void unfold_im2col_f32(const float* x,
                                  float* y,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N) {
        conv_nd_im2col_f32(x, y, C, S, K, O, stride, pad, dilation, N);
    }
    static void unfold_im2col_f64(const double* x,
                                  double* y,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N) {
        conv_nd_im2col_f64(x, y, C, S, K, O, stride, pad, dilation, N);
    }
    static void unfold_col2im_f32(const float* cols,
                                  float* dx,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N) {
        conv_nd_col2im_f32(cols, dx, C, S, K, O, stride, pad, dilation, N);
    }
    static void unfold_col2im_f64(const double* cols,
                                  double* dx,
                                  int C,
                                  const int* S,
                                  const int* K,
                                  const int* O,
                                  const int* stride,
                                  const int* pad,
                                  const int* dilation,
                                  int N) {
        conv_nd_col2im_f64(cols, dx, C, S, K, O, stride, pad, dilation, N);
    }

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
            return static_cast<std::int64_t>(reinterpret_cast<const float*>(target.ptr.get())[i]);
        case Dtype::F64:
            return static_cast<std::int64_t>(reinterpret_cast<const double*>(target.ptr.get())[i]);
        default:
            ErrorBuilder("cpu_backend::class_loss_target").not_implemented("dtype not supported");
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

    static void
    add_linear_bias(CpuStorage& y, const CpuStorage& bias, std::size_t M, std::size_t N, Dtype dt) {
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

    static Shape
    reduced_norm_shape(const Shape& shape, const std::vector<int>& axes, bool keepdims) {
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

    template <typename T>
    static void set_matrix_identity(T* out, int n) {
        const std::size_t total = static_cast<std::size_t>(n) * n;
        std::memset(out, 0, total * sizeof(T));
        for (int i = 0; i < n; ++i)
            out[i * n + i] = T{1};
    }

    template <typename T>
    static void pinv_one(const T* a, int m, int n, T* aplus) {
        const int k = std::min(m, n);
        std::vector<T> u(static_cast<std::size_t>(m) * k);
        std::vector<T> s(k);
        std::vector<T> vt(static_cast<std::size_t>(k) * n);
        int info = 0;
        if constexpr (std::is_same_v<T, float>)
            cpu::lapack_svd_f32(a, m, n, false, u.data(), s.data(), vt.data(), &info);
        else
            cpu::lapack_svd_f64(a, m, n, false, u.data(), s.data(), vt.data(), &info);
        if (info != 0)
            ErrorBuilder("pinv").fail("SVD did not converge");

        const T smax = (k > 0) ? *std::max_element(s.begin(), s.end()) : T{0};
        const T rcond = std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(m, n));
        const T cutoff = rcond * smax;

        std::vector<T> s_inv_ut(static_cast<std::size_t>(k) * m);
        for (int i = 0; i < k; ++i) {
            const T inv = (s[i] > cutoff) ? T{1} / s[i] : T{0};
            for (int j = 0; j < m; ++j)
                s_inv_ut[i * m + j] = inv * u[j * k + i];
        }

        if constexpr (std::is_same_v<T, float>) {
            cpu::sgemm(true, false, n, m, k, 1.0f, vt.data(), n, s_inv_ut.data(), m, 0.0f, aplus,
                       m);
        } else {
            cpu::dgemm(true, false, n, m, k, 1.0, vt.data(), n, s_inv_ut.data(), m, 0.0, aplus, m);
        }
    }

    static float ipiv_sign(const int* ipiv, int n) {
        int swaps = 0;
        for (int i = 0; i < n; ++i)
            if (ipiv[i] != i + 1)
                ++swaps;
        return (swaps % 2 == 0) ? 1.0f : -1.0f;
    }

    static void batch_norm_forward_f32_fast(const float* x,
                                            const float* gamma,
                                            const float* beta,
                                            float* y,
                                            float* mean_per_c,
                                            float* rstd_per_c,
                                            int batch,
                                            int channels,
                                            int spatial,
                                            double eps) {
#ifdef __APPLE__

        {
            BNNSNDArrayDescriptor i_desc = {};
            i_desc.layout = BNNSDataLayoutImageCHW;
            i_desc.size[0] = static_cast<std::size_t>(spatial);
            i_desc.size[1] = 1;
            i_desc.size[2] = static_cast<std::size_t>(channels);
            i_desc.data_type = BNNSDataTypeFloat32;
            i_desc.data = nullptr;

            BNNSNDArrayDescriptor o_desc = i_desc;

            BNNSNDArrayDescriptor gamma_desc = {};
            gamma_desc.layout = BNNSDataLayout1DLastMajor;
            gamma_desc.size[0] = static_cast<std::size_t>(channels);
            gamma_desc.data_type = BNNSDataTypeFloat32;
            gamma_desc.data = const_cast<float*>(gamma);

            BNNSNDArrayDescriptor beta_desc = {};
            beta_desc.layout = BNNSDataLayout1DLastMajor;
            beta_desc.size[0] = static_cast<std::size_t>(channels);
            beta_desc.data_type = BNNSDataTypeFloat32;
            beta_desc.data = const_cast<float*>(beta);

            BNNSNDArrayDescriptor mm_desc = {};
            mm_desc.layout = BNNSDataLayout1DLastMajor;
            mm_desc.size[0] = static_cast<std::size_t>(channels);
            mm_desc.data_type = BNNSDataTypeFloat32;
            mm_desc.data = mean_per_c;

            BNNSNDArrayDescriptor mv_desc = mm_desc;
            mv_desc.data = rstd_per_c;

            BNNSLayerParametersNormalization norm_params = {};
            norm_params.i_desc = i_desc;
            norm_params.o_desc = o_desc;
            norm_params.gamma_desc = gamma_desc;
            norm_params.beta_desc = beta_desc;
            norm_params.moving_mean_desc = mm_desc;
            norm_params.moving_variance_desc = mv_desc;
            norm_params.momentum = 1.0f;
            norm_params.epsilon = static_cast<float>(eps);
            BNNSActivation act = {};
            act.function = BNNSActivationFunctionIdentity;
            norm_params.activation = act;

            BNNSFilter filter =
                BNNSFilterCreateLayerNormalization(BNNSBatchNorm, &norm_params, nullptr);
            if (filter) {
                const std::size_t sample_elems = static_cast<std::size_t>(channels) * spatial;
                int ret = BNNSNormalizationFilterApplyBatch(filter, static_cast<std::size_t>(batch),
                                                            x, sample_elems, y, sample_elems, true);
                BNNSFilterDestroy(filter);
                if (ret == 0) {
                    const float eps_f = static_cast<float>(eps);
                    for (int c = 0; c < channels; ++c)
                        rstd_per_c[c] = 1.0f / std::sqrt(rstd_per_c[c] + eps_f);

                    return;
                }
            }
        }
#endif

        const std::size_t S = static_cast<std::size_t>(spatial);
        const float inv_M = 1.0f / static_cast<float>(batch * spatial);
        for (int c = 0; c < channels; ++c) {
            float sum = 0.f;
            for (int b = 0; b < batch; ++b)
                sum += cpu::vsum_f32(x + (static_cast<std::size_t>(b) * channels + c) * S, S);
            const float mean = sum * inv_M;
            mean_per_c[c] = mean;

            float sumsq = 0.f;
            for (int b = 0; b < batch; ++b) {
                const float* xb = x + (static_cast<std::size_t>(b) * channels + c) * S;
                sumsq += cpu::vdotpr_f32(xb, xb, S);
            }
            const float var = sumsq * inv_M - mean * mean;
            const float rstd = 1.0f / std::sqrt(var + static_cast<float>(eps));
            rstd_per_c[c] = rstd;

            const float scale = gamma[c] * rstd;
            const float bias = beta[c] - mean * scale;
            for (int b = 0; b < batch; ++b) {
                const float* xb = x + (static_cast<std::size_t>(b) * channels + c) * S;
                float* yb = y + (static_cast<std::size_t>(b) * channels + c) * S;
                cpu::vsmul_f32(xb, scale, yb, S);
                cpu::vsadd_f32(yb, bias, yb, S);
            }
        }
    }

    template <typename T>
    static void batch_norm_forward_typed(const T* x,
                                         const T* gamma,
                                         const T* beta,
                                         T* y,
                                         T* mean_per_c,
                                         T* rstd_per_c,
                                         int batch,
                                         int channels,
                                         int spatial,
                                         double eps) {
        const std::size_t M = static_cast<std::size_t>(batch) * spatial;
        const T inv_M = T{1} / static_cast<T>(M);
        for (int c = 0; c < channels; ++c) {
            T mean = T{};
            for (int b = 0; b < batch; ++b) {
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i)
                    mean += xb[i];
            }
            mean *= inv_M;

            T var = T{};
            for (int b = 0; b < batch; ++b) {
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const T d = xb[i] - mean;
                    var += d * d;
                }
            }
            var *= inv_M;
            const T rstd = T{1} / std::sqrt(var + static_cast<T>(eps));
            const T g = gamma[c];
            const T be = beta[c];
            for (int b = 0; b < batch; ++b) {
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                T* yb = y + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i)
                    yb[i] = g * (xb[i] - mean) * rstd + be;
            }
            mean_per_c[c] = mean;
            rstd_per_c[c] = rstd;
        }
    }

    template <typename T>
    static void batch_norm_backward_typed(const T* x,
                                          const T* gamma,
                                          const T* mean,
                                          const T* rstd,
                                          const T* g,
                                          T* dx,
                                          T* dgamma,
                                          T* dbeta,
                                          int batch,
                                          int channels,
                                          int spatial) {
        const std::size_t M = static_cast<std::size_t>(batch) * spatial;
        const T inv_M = T{1} / static_cast<T>(M);
        for (int c = 0; c < channels; ++c) {
            const T m = mean[c];
            const T r = rstd[c];
            const T gc = gamma[c];
            T sum_dxn = T{};
            T sum_dxn_xn = T{};
            T sum_g = T{};
            T sum_g_xn = T{};
            for (int b = 0; b < batch; ++b) {
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const T xn_i = (xb[i] - m) * r;
                    const T dxn_i = gc * gb[i];
                    sum_dxn += dxn_i;
                    sum_dxn_xn += dxn_i * xn_i;
                    sum_g += gb[i];
                    sum_g_xn += gb[i] * xn_i;
                }
            }
            for (int b = 0; b < batch; ++b) {
                const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
                T* dxb = dx + (static_cast<std::size_t>(b) * channels + c) * spatial;
                for (int i = 0; i < spatial; ++i) {
                    const T xn_i = (xb[i] - m) * r;
                    const T dxn_i = gc * gb[i];
                    dxb[i] = inv_M * r * (static_cast<T>(M) * dxn_i - sum_dxn - xn_i * sum_dxn_xn);
                }
            }
            dgamma[c] = sum_g_xn;
            dbeta[c] = sum_g;
        }
    }

    template <typename T>
    static void group_norm_forward_typed(const T* x,
                                         const T* gamma,
                                         const T* beta,
                                         T* y,
                                         T* mean_bg,
                                         T* rstd_bg,
                                         int batch,
                                         int channels,
                                         int spatial,
                                         int groups,
                                         double eps) {
        const int Cg = channels / groups;
        const std::size_t per_group = static_cast<std::size_t>(Cg) * spatial;
        const T inv_pg = T{1} / static_cast<T>(per_group);
        for (int b = 0; b < batch; ++b) {
            for (int g = 0; g < groups; ++g) {
                T mean = T{};
                for (int cc = 0; cc < Cg; ++cc) {
                    const int c = g * Cg + cc;
                    const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    for (int i = 0; i < spatial; ++i)
                        mean += xb[i];
                }
                mean *= inv_pg;
                T var = T{};
                for (int cc = 0; cc < Cg; ++cc) {
                    const int c = g * Cg + cc;
                    const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    for (int i = 0; i < spatial; ++i) {
                        const T d = xb[i] - mean;
                        var += d * d;
                    }
                }
                var *= inv_pg;
                const T rstd = T{1} / std::sqrt(var + static_cast<T>(eps));
                for (int cc = 0; cc < Cg; ++cc) {
                    const int c = g * Cg + cc;
                    const T gc = gamma[c];
                    const T bc = beta[c];
                    const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    T* yb = y + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    for (int i = 0; i < spatial; ++i)
                        yb[i] = gc * (xb[i] - mean) * rstd + bc;
                }
                mean_bg[b * groups + g] = mean;
                rstd_bg[b * groups + g] = rstd;
            }
        }
    }

    template <typename T>
    static void group_norm_backward_typed(const T* x,
                                          const T* gamma,
                                          const T* mean_bg,
                                          const T* rstd_bg,
                                          const T* g,
                                          T* dx,
                                          T* dgamma,
                                          T* dbeta,
                                          int batch,
                                          int channels,
                                          int spatial,
                                          int groups) {
        const int Cg = channels / groups;
        const std::size_t per_group = static_cast<std::size_t>(Cg) * spatial;
        const T inv_pg = T{1} / static_cast<T>(per_group);
        for (int co = 0; co < channels; ++co) {
            dgamma[co] = T{};
            dbeta[co] = T{};
        }
        for (int b = 0; b < batch; ++b) {
            for (int gi = 0; gi < groups; ++gi) {
                const T m = mean_bg[b * groups + gi];
                const T r = rstd_bg[b * groups + gi];
                T sum_dxn = T{};
                T sum_dxn_xn = T{};
                for (int cc = 0; cc < Cg; ++cc) {
                    const int c = gi * Cg + cc;
                    const T gc = gamma[c];
                    const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    for (int i = 0; i < spatial; ++i) {
                        const T xn_i = (xb[i] - m) * r;
                        const T dxn_i = gc * gb[i];
                        sum_dxn += dxn_i;
                        sum_dxn_xn += dxn_i * xn_i;
                        dgamma[c] += gb[i] * xn_i;
                        dbeta[c] += gb[i];
                    }
                }
                for (int cc = 0; cc < Cg; ++cc) {
                    const int c = gi * Cg + cc;
                    const T gc = gamma[c];
                    const T* xb = x + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    const T* gb = g + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    T* dxb = dx + (static_cast<std::size_t>(b) * channels + c) * spatial;
                    for (int i = 0; i < spatial; ++i) {
                        const T xn_i = (xb[i] - m) * r;
                        const T dxn_i = gc * gb[i];
                        dxb[i] = inv_pg * r *
                                 (static_cast<T>(per_group) * dxn_i - sum_dxn - xn_i * sum_dxn_xn);
                    }
                }
            }
        }
    }

    // Fills ptr with the scalar value 1 for dtype dt.  Used by ones().
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

    // Generic unary dispatch: allocates output, calls fn32/fn64/fni32 based on dt.
    // I64 falls back to a scalar loop via static_cast<double>.
    template <class F32Fn, class F64Fn, class I32Fn>
    Storage
    unary_op(const Storage& a, const Shape& shape, Dtype dt, F32Fn fn32, F64Fn fn64, I32Fn fni32) {
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
            const std::int64_t* ip = reinterpret_cast<const std::int64_t*>(cs.ptr.get());
            std::int64_t* op = reinterpret_cast<std::int64_t*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = static_cast<std::int64_t>(static_cast<double>(static_cast<double>(ip[i])));
        } else {
            ErrorBuilder("cpu_backend::unary").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    // Generic binary dispatch: allocates output, calls fn32/fn64/fni32/fni64 based on dt.
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

    // Tags used by reduce_axes to select the accumulation operation.
    enum class ReduceOp { Sum, Mean, Max, Min };

    // Reduces a over the axes in opts by iterating axes from largest to smallest
    // (descending sort ensures that axis indices remain valid after each step).
    // Each single-axis pass calls the appropriate cpu::sum_axis / max_axis /
    // min_axis primitive.  Mean divides by the reduce dimension after summing.
    Storage reduce_axes(
        const Storage& a, const Shape& in_shape, const ReduceOpts& opts, Dtype dt, ReduceOp op) {
        if (opts.axes.empty()) {
            std::vector<int> all_axes;
            for (int i = 0; i < static_cast<int>(in_shape.size()); ++i)
                all_axes.push_back(i);
            ReduceOpts all_opts{all_axes, opts.keepdims};
            return reduce_axes(a, in_shape, all_opts, dt, op);
        }

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

        if (opts.keepdims) {
            Shape kept = in_shape;
            for (int ax : axes)
                kept[(ax < 0) ? ax + static_cast<int>(in_shape.size()) : ax] = 1;
        }
        return cur;
    }

    void
    cast_impl(const std::byte* src, std::byte* dst, std::size_t n, Dtype src_dt, Dtype dst_dt) {
        auto cast_loop = [&](auto from_tag, auto to_tag) {
            using Src = decltype(from_tag);
            using Dst = decltype(to_tag);
            const Src* sp = reinterpret_cast<const Src*>(src);
            Dst* dp = reinterpret_cast<Dst*>(dst);
            for (std::size_t i = 0; i < n; ++i)
                dp[i] = static_cast<Dst>(sp[i]);
        };

#define CAST_CASE(S, D, st, dt_)                                                                   \
    if (src_dt == Dtype::S && dst_dt == Dtype::D) {                                                \
        cast_loop(st{}, dt_{});                                                                    \
        return;                                                                                    \
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

        if (src_dt == dst_dt) {
            std::memcpy(dst, src, n * dtype_size(dst_dt));
            return;
        }
        ErrorBuilder("cpu_backend::cast").not_implemented("unsupported dtype pair");
    }

    static CpuStorage alloc_cpu(std::size_t numel, Dtype dt) {
        const std::size_t nb = numel * dtype_size(dt);
        return CpuStorage{allocate_aligned_bytes(nb, Device::CPU), nb, dt};
    }

    static void ctnd_im2col_f32(const float* src,
                                float* cols,
                                int C,
                                const int* O,
                                const int* K,
                                const int* S,
                                const int* stride,
                                const int* pad,
                                int N) {
        if (N == 1)
            cpu::im2col_1d_f32(src, cols, C, O[0], K[0], S[0], stride[0], pad[0], 1);
        else if (N == 2)
            cpu::im2col_f32(src, cols, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                            pad[0], pad[1], 1, 1);
        else
            cpu::im2col_3d_f32(src, cols, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
    }

    static void ctnd_im2col_f64(const double* src,
                                double* cols,
                                int C,
                                const int* O,
                                const int* K,
                                const int* S,
                                const int* stride,
                                const int* pad,
                                int N) {
        if (N == 1)
            cpu::im2col_1d_f64(src, cols, C, O[0], K[0], S[0], stride[0], pad[0], 1);
        else if (N == 2)
            cpu::im2col_f64(src, cols, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                            pad[0], pad[1], 1, 1);
        else
            cpu::im2col_3d_f64(src, cols, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
    }

    static void ctnd_col2im_f32(const float* cols,
                                float* dst,
                                int C,
                                const int* O,
                                const int* K,
                                const int* S,
                                const int* stride,
                                const int* pad,
                                int N) {
        if (N == 1)
            cpu::col2im_1d_f32(cols, dst, C, O[0], K[0], S[0], stride[0], pad[0], 1);
        else if (N == 2)
            cpu::col2im_f32(cols, dst, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                            pad[0], pad[1], 1, 1);
        else
            cpu::col2im_3d_f32(cols, dst, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
    }

    static void ctnd_col2im_f64(const double* cols,
                                double* dst,
                                int C,
                                const int* O,
                                const int* K,
                                const int* S,
                                const int* stride,
                                const int* pad,
                                int N) {
        if (N == 1)
            cpu::col2im_1d_f64(cols, dst, C, O[0], K[0], S[0], stride[0], pad[0], 1);
        else if (N == 2)
            cpu::col2im_f64(cols, dst, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                            pad[0], pad[1], 1, 1);
        else
            cpu::col2im_3d_f64(cols, dst, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                               stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], 1, 1, 1);
    }

    Storage ge_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto run_f = [&](auto tag_a, auto tag_b, auto tag_out) {
            using A = decltype(tag_a);
            using B = decltype(tag_b);
            using O = decltype(tag_out);
            const A* ap = reinterpret_cast<const A*>(ca.ptr.get());
            const B* bp = reinterpret_cast<const B*>(cb.ptr.get());
            O* op = reinterpret_cast<O*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = (ap[i] >= bp[i]) ? O{1} : O{0};
        };
        if (dt == Dtype::F32)
            run_f(float{}, float{}, float{});
        else if (dt == Dtype::F64)
            run_f(double{}, double{}, double{});
        else
            ErrorBuilder("cpu_backend::ge_mask").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage lt_mask(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto run_f = [&](auto tag_a, auto tag_b, auto tag_out) {
            using A = decltype(tag_a);
            using B = decltype(tag_b);
            using O = decltype(tag_out);
            const A* ap = reinterpret_cast<const A*>(ca.ptr.get());
            const B* bp = reinterpret_cast<const B*>(cb.ptr.get());
            O* op = reinterpret_cast<O*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                op[i] = (ap[i] < bp[i]) ? O{1} : O{0};
        };
        if (dt == Dtype::F32)
            run_f(float{}, float{}, float{});
        else if (dt == Dtype::F64)
            run_f(double{}, double{}, double{});
        else
            ErrorBuilder("cpu_backend::lt_mask").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage add_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) override {
        const auto& ca = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32)
            cpu::vsadd_f32(reinterpret_cast<const float*>(ca.ptr.get()), static_cast<float>(scalar),
                           reinterpret_cast<float*>(ptr.get()), n);
        else if (dt == Dtype::F64)
            cpu::vsadd_f64(reinterpret_cast<const double*>(ca.ptr.get()), scalar,
                           reinterpret_cast<double*>(ptr.get()), n);
        else
            ErrorBuilder("cpu_backend::add_scalar").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage mul_scalar(const Storage& a, const Shape& shape, Dtype dt, double scalar) override {
        const auto& ca = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32)
            cpu::vsmul_f32(reinterpret_cast<const float*>(ca.ptr.get()), static_cast<float>(scalar),
                           reinterpret_cast<float*>(ptr.get()), n);
        else if (dt == Dtype::F64)
            cpu::vsmul_f64(reinterpret_cast<const double*>(ca.ptr.get()), scalar,
                           reinterpret_cast<double*>(ptr.get()), n);
        else
            ErrorBuilder("cpu_backend::mul_scalar").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage
    in_range_mask(const Storage& a, const Shape& shape, Dtype dt, double lo, double hi) override {
        const auto& ca = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const auto* p = reinterpret_cast<const float*>(ca.ptr.get());
            auto* q = reinterpret_cast<float*>(ptr.get());
            const auto flo = static_cast<float>(lo), fhi = static_cast<float>(hi);
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] >= flo && p[i] <= fhi) ? 1.f : 0.f;
        } else if (dt == Dtype::F64) {
            const auto* p = reinterpret_cast<const double*>(ca.ptr.get());
            auto* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] >= lo && p[i] <= hi) ? 1.0 : 0.0;
        } else {
            ErrorBuilder("cpu_backend::in_range_mask").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage leaky_mask(const Storage& a, const Shape& shape, Dtype dt, double slope) override {
        const auto& ca = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const auto* p = reinterpret_cast<const float*>(ca.ptr.get());
            auto* q = reinterpret_cast<float*>(ptr.get());
            const float fs = static_cast<float>(slope);
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] >= 0.f) ? 1.f : fs;
        } else if (dt == Dtype::F64) {
            const auto* p = reinterpret_cast<const double*>(ca.ptr.get());
            auto* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] >= 0.0) ? 1.0 : slope;
        } else {
            ErrorBuilder("cpu_backend::leaky_mask").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage positive_mask(const Storage& a, const Shape& shape, Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        if (dt == Dtype::F32) {
            const auto* p = reinterpret_cast<const float*>(ca.ptr.get());
            auto* q = reinterpret_cast<float*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] > 0.f) ? 1.f : 0.f;
        } else if (dt == Dtype::F64) {
            const auto* p = reinterpret_cast<const double*>(ca.ptr.get());
            auto* q = reinterpret_cast<double*>(ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                q[i] = (p[i] > 0.0) ? 1.0 : 0.0;
        } else {
            ErrorBuilder("cpu_backend::positive_mask").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage reduce_grad_to_shape(const Storage& grad,
                                 const Shape& grad_shape,
                                 const Shape& target_shape,
                                 Dtype dt) override {
        const std::size_t gn = grad_shape.size();
        const std::size_t tn = target_shape.size();
        if (grad_shape == target_shape) {
            const auto& src = std::get<CpuStorage>(grad);
            std::size_t nb = src.nbytes;
            auto ptr = allocate_aligned_bytes(nb, Device::CPU);
            if (nb > 0)
                std::memcpy(ptr.get(), src.ptr.get(), nb);
            return Storage{CpuStorage{ptr, nb, dt}};
        }

        std::vector<std::size_t> axes;
        const std::size_t lead = gn - tn;
        for (std::size_t i = 0; i < lead; ++i)
            axes.push_back(i);
        for (std::size_t i = 0; i < tn; ++i) {
            if (grad_shape[lead + i] != target_shape[i] && target_shape[i] == 1)
                axes.push_back(lead + i);
        }
        const auto& src = std::get<CpuStorage>(grad);
        std::size_t tnumel = shape_numel(target_shape);
        std::size_t nb = tnumel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);

        Stride grad_stride(gn);
        if (gn > 0) {
            grad_stride[gn - 1] = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(gn) - 2; i >= 0; --i)
                grad_stride[i] = grad_stride[i + 1] * grad_shape[i + 1];
        }
        Stride target_stride(tn);
        if (tn > 0) {
            target_stride[tn - 1] = 1;
            for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(tn) - 2; i >= 0; --i)
                target_stride[i] = target_stride[i + 1] * target_shape[i + 1];
        }
        std::vector<bool> reduce_mask(gn, false);
        for (auto a : axes)
            reduce_mask[a] = true;

        auto do_reduce = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* src_p = reinterpret_cast<const T*>(src.ptr.get());
            T* dst_p = reinterpret_cast<T*>(ptr.get());
            std::fill_n(dst_p, tnumel, T{});
            const std::size_t gnumel = shape_numel(grad_shape);
            for (std::size_t flat = 0; flat < gnumel; ++flat) {
                std::size_t rem = flat;
                std::size_t target_flat = 0;
                for (std::size_t d = 0; d < gn; ++d) {
                    const std::size_t coord = rem / static_cast<std::size_t>(grad_stride[d]);
                    rem -= coord * static_cast<std::size_t>(grad_stride[d]);
                    if (d < lead || reduce_mask[d])
                        continue;
                    target_flat += coord * static_cast<std::size_t>(target_stride[d - lead]);
                }
                dst_p[target_flat] += src_p[flat];
            }
        };
        if (dt == Dtype::F32)
            do_reduce(float{});
        else if (dt == Dtype::F64)
            do_reduce(double{});
        else
            ErrorBuilder("cpu_backend::reduce_grad_to_shape")
                .not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage broadcast_back_for_reduce(const Storage& grad,
                                      const Shape& grad_shape,
                                      const Shape& input_shape,
                                      const std::vector<int>& axes,
                                      bool keepdims,
                                      Dtype dt) override {
        Shape kept_shape = input_shape;
        for (int a : axes)
            kept_shape[a] = 1;

        Shape expected_grad;
        {
            std::vector<bool> rm(input_shape.size(), false);
            for (int a : axes)
                rm[a] = true;
            for (std::size_t i = 0; i < input_shape.size(); ++i) {
                if (rm[i]) {
                    if (keepdims)
                        expected_grad.push_back(1);
                } else
                    expected_grad.push_back(input_shape[i]);
            }
        }
        (void)grad_shape;
        (void)expected_grad;
        const auto& g_cpu = std::get<CpuStorage>(grad);
        std::size_t in_numel = shape_numel(input_shape);
        std::size_t nb = in_numel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);

        auto do_bcast = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* gp = reinterpret_cast<const T*>(g_cpu.ptr.get());
            T* dst = reinterpret_cast<T*>(ptr.get());

            const std::size_t nd = input_shape.size();
            for (std::size_t flat = 0; flat < in_numel; ++flat) {
                std::size_t kept_flat = 0;
                std::size_t stride = 1;
                for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nd) - 1; d >= 0; --d) {
                    std::size_t di = static_cast<std::size_t>(d);

                    std::size_t dstride = 1;
                    for (std::size_t e = di + 1; e < nd; ++e)
                        dstride *= static_cast<std::size_t>(input_shape[e]);
                    std::size_t coord =
                        (flat / dstride) % static_cast<std::size_t>(input_shape[di]);
                    std::int64_t kd = kept_shape[di];
                    std::int64_t ii = (kd == 1) ? 0 : static_cast<std::int64_t>(coord);
                    kept_flat += static_cast<std::size_t>(ii) * stride;
                    stride *= static_cast<std::size_t>(kd);
                }
                dst[flat] = gp[kept_flat];
            }
        };
        if (dt == Dtype::F32)
            do_bcast(float{});
        else if (dt == Dtype::F64)
            do_bcast(double{});
        else
            ErrorBuilder("cpu_backend::broadcast_back_for_reduce")
                .not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage full(const Shape& shape, Dtype dt, double fill_value) override {
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto* p = ptr.get();
        switch (dt) {
        case Dtype::Bool: {
            auto v = fill_value != 0.0 ? std::uint8_t{1} : std::uint8_t{0};
            for (std::size_t i = 0; i < n; ++i)
                reinterpret_cast<std::uint8_t*>(p)[i] = v;
            break;
        }
        case Dtype::I8: {
            auto v = static_cast<std::int8_t>(fill_value);
            for (std::size_t i = 0; i < n; ++i)
                reinterpret_cast<std::int8_t*>(p)[i] = v;
            break;
        }
        case Dtype::I16: {
            auto v = static_cast<std::int16_t>(fill_value);
            for (std::size_t i = 0; i < n; ++i)
                reinterpret_cast<std::int16_t*>(p)[i] = v;
            break;
        }
        case Dtype::I32: {
            auto v = static_cast<std::int32_t>(fill_value);
            for (std::size_t i = 0; i < n; ++i)
                reinterpret_cast<std::int32_t*>(p)[i] = v;
            break;
        }
        case Dtype::I64: {
            auto v = static_cast<std::int64_t>(fill_value);
            for (std::size_t i = 0; i < n; ++i)
                reinterpret_cast<std::int64_t*>(p)[i] = v;
            break;
        }
        case Dtype::F32: {
            auto v = static_cast<float>(fill_value);
            for (std::size_t i = 0; i < n; ++i)
                reinterpret_cast<float*>(p)[i] = v;
            break;
        }
        case Dtype::F64: {
            auto v = fill_value;
            for (std::size_t i = 0; i < n; ++i)
                reinterpret_cast<double*>(p)[i] = v;
            break;
        }
        default:
            ErrorBuilder("cpu_backend::full").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) override {
        Shape shape{N, M};
        std::size_t nb = static_cast<std::size_t>(N * M) * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        std::memset(ptr.get(), 0, nb);
        auto set_one = [&](auto* p) {
            using T = std::remove_pointer_t<decltype(p)>;
            for (std::int64_t i = 0; i < N; ++i) {
                const std::int64_t j = i + k;
                if (j < 0 || j >= M)
                    continue;
                p[i * M + j] = static_cast<T>(1);
            }
        };
        switch (dt) {
        case Dtype::Bool:
            set_one(reinterpret_cast<std::uint8_t*>(ptr.get()));
            break;
        case Dtype::I8:
            set_one(reinterpret_cast<std::int8_t*>(ptr.get()));
            break;
        case Dtype::I16:
            set_one(reinterpret_cast<std::int16_t*>(ptr.get()));
            break;
        case Dtype::I32:
            set_one(reinterpret_cast<std::int32_t*>(ptr.get()));
            break;
        case Dtype::I64:
            set_one(reinterpret_cast<std::int64_t*>(ptr.get()));
            break;
        case Dtype::F32:
            set_one(reinterpret_cast<float*>(ptr.get()));
            break;
        case Dtype::F64:
            set_one(reinterpret_cast<double*>(ptr.get()));
            break;
        default:
            ErrorBuilder("cpu_backend::eye").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage diag(const Storage& v,
                 const Shape& v_shape,
                 std::int64_t k,
                 Dtype dt,
                 Shape& out_shape) override {
        const auto& cv = std::get<CpuStorage>(v);
        const std::size_t elem = dtype_size(dt);
        if (v_shape.size() == 1) {
            const std::int64_t L = v_shape[0];
            const std::int64_t side = L + std::abs(k);
            out_shape = {side, side};
            std::size_t nb = static_cast<std::size_t>(side * side) * elem;
            auto ptr = allocate_aligned_bytes(nb, Device::CPU);
            std::memset(ptr.get(), 0, nb);
            auto fill = [&](auto* dst, const auto* src) {
                using T = std::remove_pointer_t<decltype(dst)>;
                for (std::int64_t i = 0; i < L; ++i) {
                    const std::int64_t row = (k >= 0) ? i : (i - k);
                    const std::int64_t col = (k >= 0) ? (i + k) : i;
                    dst[row * side + col] = static_cast<T>(src[i]);
                }
            };
            switch (dt) {
            case Dtype::F32:
                fill(reinterpret_cast<float*>(ptr.get()),
                     reinterpret_cast<const float*>(cv.ptr.get()));
                break;
            case Dtype::F64:
                fill(reinterpret_cast<double*>(ptr.get()),
                     reinterpret_cast<const double*>(cv.ptr.get()));
                break;
            case Dtype::I32:
                fill(reinterpret_cast<std::int32_t*>(ptr.get()),
                     reinterpret_cast<const std::int32_t*>(cv.ptr.get()));
                break;
            case Dtype::I64:
                fill(reinterpret_cast<std::int64_t*>(ptr.get()),
                     reinterpret_cast<const std::int64_t*>(cv.ptr.get()));
                break;
            default:
                ErrorBuilder("cpu_backend::diag").not_implemented("dtype not supported");
            }
            return Storage{CpuStorage{ptr, nb, dt}};
        }

        const std::int64_t Mv = v_shape[0], Nv = v_shape[1];
        const std::int64_t r0 = (k >= 0) ? 0 : -k;
        const std::int64_t c0 = (k >= 0) ? k : 0;
        const std::int64_t Lv = std::min(Mv - r0, Nv - c0);
        const std::int64_t out_len = std::max<std::int64_t>(Lv, 0);
        out_shape = {out_len};
        std::size_t nb = static_cast<std::size_t>(out_len) * elem;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto extract = [&](auto* dst, const auto* src) {
            using T = std::remove_pointer_t<decltype(dst)>;
            for (std::int64_t i = 0; i < out_len; ++i)
                dst[i] = static_cast<T>(src[(r0 + i) * Nv + (c0 + i)]);
        };
        switch (dt) {
        case Dtype::F32:
            extract(reinterpret_cast<float*>(ptr.get()),
                    reinterpret_cast<const float*>(cv.ptr.get()));
            break;
        case Dtype::F64:
            extract(reinterpret_cast<double*>(ptr.get()),
                    reinterpret_cast<const double*>(cv.ptr.get()));
            break;
        case Dtype::I32:
            extract(reinterpret_cast<std::int32_t*>(ptr.get()),
                    reinterpret_cast<const std::int32_t*>(cv.ptr.get()));
            break;
        case Dtype::I64:
            extract(reinterpret_cast<std::int64_t*>(ptr.get()),
                    reinterpret_cast<const std::int64_t*>(cv.ptr.get()));
            break;
        default:
            ErrorBuilder("cpu_backend::diag").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) override {
        const auto& ca = std::get<CpuStorage>(input);
        if (shape.size() < 2)
            ErrorBuilder("cpu_backend::tri").fail("input must have ndim >= 2");
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const std::int64_t M = shape[shape.size() - 2];
        const std::int64_t N = shape[shape.size() - 1];
        std::size_t batch = 1;
        for (std::size_t d = 0; d < shape.size() - 2; ++d)
            batch *= static_cast<std::size_t>(shape[d]);
        const std::size_t plane = static_cast<std::size_t>(M * N);
        auto run = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* src = reinterpret_cast<const T*>(ca.ptr.get());
            T* dst = reinterpret_cast<T*>(ptr.get());
            for (std::size_t b = 0; b < batch; ++b)
                for (std::int64_t i = 0; i < M; ++i)
                    for (std::int64_t j = 0; j < N; ++j) {
                        const std::size_t f = b * plane + static_cast<std::size_t>(i * N + j);
                        dst[f] = (upper ? (j - i >= k) : (j - i <= k)) ? src[f] : T{};
                    }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else
            ErrorBuilder("cpu_backend::tri").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage floordiv(const Storage& a, const Storage& b, const Shape& shape, Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * sizeof(std::int64_t);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto* dst = reinterpret_cast<std::int64_t*>(ptr.get());
        auto run = [&](const auto* p, const auto* q) {
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = static_cast<std::int64_t>(
                    std::floor(static_cast<double>(p[i]) / static_cast<double>(q[i])));
        };
        switch (dt) {
        case Dtype::F32:
            run(reinterpret_cast<const float*>(ca.ptr.get()),
                reinterpret_cast<const float*>(cb.ptr.get()));
            break;
        case Dtype::F64:
            run(reinterpret_cast<const double*>(ca.ptr.get()),
                reinterpret_cast<const double*>(cb.ptr.get()));
            break;
        case Dtype::I32:
            run(reinterpret_cast<const std::int32_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int32_t*>(cb.ptr.get()));
            break;
        case Dtype::I64:
            run(reinterpret_cast<const std::int64_t*>(ca.ptr.get()),
                reinterpret_cast<const std::int64_t*>(cb.ptr.get()));
            break;
        default:
            ErrorBuilder("cpu_backend::floordiv").not_implemented("dtype not supported");
        }
        return Storage{CpuStorage{ptr, nb, Dtype::I64}};
    }

    Storage inner(const Storage& a,
                  const Storage& b,
                  const Shape& a_shape,
                  const Shape& b_shape,
                  const Shape& out_shape,
                  Dtype dt) override {
        const auto& ca = std::get<CpuStorage>(a);
        const auto& cb = std::get<CpuStorage>(b);
        const std::int64_t K = a_shape.back();
        const std::size_t pa = shape_numel(Shape(a_shape.begin(), a_shape.end() - 1));
        const std::size_t pb = shape_numel(Shape(b_shape.begin(), b_shape.end() - 1));
        std::size_t n = shape_numel(out_shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        auto run = [&](auto* op, const auto* ap, const auto* bp) {
            using T = std::remove_pointer_t<decltype(op)>;
            for (std::size_t i = 0; i < pa; ++i)
                for (std::size_t j = 0; j < pb; ++j) {
                    T s{};
                    for (std::int64_t k = 0; k < K; ++k)
                        s = s + ap[i * K + k] * bp[j * K + k];
                    op[i * pb + j] = s;
                }
        };
        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(ca.ptr.get()),
                reinterpret_cast<const float*>(cb.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(ca.ptr.get()),
                reinterpret_cast<const double*>(cb.ptr.get()));
        else
            ErrorBuilder("cpu_backend::inner").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    CpuStorage permute_cpu(const CpuStorage& src,
                           const Shape& src_shape,
                           const std::vector<int>& perm,
                           Dtype dt) override {
        const std::size_t nd = src_shape.size();
        Shape dst_shape;
        for (int p : perm)
            dst_shape.push_back(src_shape[static_cast<std::size_t>(p)]);

        const std::size_t total = shape_numel(dst_shape);
        const std::size_t elem = dtype_size(dt);
        std::size_t nb = total * elem;
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);

        std::vector<std::size_t> src_stride(nd, 1);
        for (std::ptrdiff_t d = (std::ptrdiff_t)nd - 2; d >= 0; --d)
            src_stride[static_cast<std::size_t>(d)] =
                src_stride[static_cast<std::size_t>(d) + 1] *
                static_cast<std::size_t>(src_shape[static_cast<std::size_t>(d) + 1]);

        std::vector<std::int64_t> coord(nd, 0);
        for (std::size_t f = 0; f < total; ++f) {
            std::size_t src_flat = 0;
            for (std::size_t d = 0; d < nd; ++d)
                src_flat += static_cast<std::size_t>(coord[d]) *
                            src_stride[static_cast<std::size_t>(perm[d])];
            std::memcpy(ptr.get() + f * elem, src.ptr.get() + src_flat * elem, elem);

            for (std::ptrdiff_t d = (std::ptrdiff_t)nd - 1; d >= 0; --d) {
                if (++coord[static_cast<std::size_t>(d)] < dst_shape[static_cast<std::size_t>(d)])
                    break;
                coord[static_cast<std::size_t>(d)] = 0;
            }
        }
        return CpuStorage{ptr, nb, dt};
    }

    Storage tensordot(const Storage&,
                      const Storage&,
                      const Shape&,
                      const Shape&,
                      const Shape&,
                      const std::vector<int>&,
                      const std::vector<int>&,
                      Dtype) override {
        ErrorBuilder("cpu_backend::tensordot").not_implemented("not routed through backend");
    }

    Storage where_op(const Storage& cond,
                     const Storage& x,
                     const Storage& y,
                     const Shape& shape,
                     Dtype dt) override {
        const auto& cc = std::get<CpuStorage>(cond);
        const auto& cx = std::get<CpuStorage>(x);
        const auto& cy = std::get<CpuStorage>(y);
        std::size_t n = shape_numel(shape);
        std::size_t nb = n * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        const auto* c = reinterpret_cast<const std::uint8_t*>(cc.ptr.get());
        auto run = [&](auto* dst, const auto* xp, const auto* yp) {
            for (std::size_t i = 0; i < n; ++i)
                dst[i] = c[i] ? xp[i] : yp[i];
        };
        if (dt == Dtype::F32)
            run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(cx.ptr.get()),
                reinterpret_cast<const float*>(cy.ptr.get()));
        else if (dt == Dtype::F64)
            run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(cx.ptr.get()),
                reinterpret_cast<const double*>(cy.ptr.get()));
        else
            ErrorBuilder("cpu_backend::where_op").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage reduce_broadcast(const Storage& grad,
                             const Shape& input_shape,
                             const Shape& output_shape,
                             Dtype dt) override {
        const std::size_t nout = output_shape.size();
        const std::size_t nin = input_shape.size();
        Shape padded(nout, 1);
        std::copy(input_shape.begin(), input_shape.end(), padded.begin() + (nout - nin));
        const auto& gc = std::get<CpuStorage>(grad);
        std::size_t in_numel = shape_numel(input_shape);
        std::size_t nb = in_numel * dtype_size(dt);
        auto ptr = allocate_aligned_bytes(nb, Device::CPU);
        std::memset(ptr.get(), 0, nb);

        std::vector<std::size_t> in_str(nout, 0);
        std::size_t s = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 1; d >= 0; --d) {
            in_str[d] = (padded[d] == 1) ? 0 : s;
            s *= static_cast<std::size_t>(padded[d]);
        }
        const std::size_t out_numel = shape_numel(output_shape);
        auto run = [&](auto type_tag) {
            using T = decltype(type_tag);
            const T* gp = reinterpret_cast<const T*>(gc.ptr.get());
            T* dp = reinterpret_cast<T*>(ptr.get());
            std::vector<std::size_t> coord(nout, 0);
            for (std::size_t f = 0; f < out_numel; ++f) {
                std::size_t in_flat = 0;
                for (std::size_t d = 0; d < nout; ++d)
                    in_flat += coord[d] * in_str[d];
                dp[in_flat] += gp[f];
                for (std::ptrdiff_t d = (std::ptrdiff_t)nout - 1; d >= 0; --d) {
                    if (++coord[d] < static_cast<std::size_t>(output_shape[d]))
                        break;
                    coord[d] = 0;
                }
            }
        };
        if (dt == Dtype::F32)
            run(float{});
        else if (dt == Dtype::F64)
            run(double{});
        else if (dt == Dtype::I32)
            run(std::int32_t{});
        else if (dt == Dtype::I64)
            run(std::int64_t{});
        else
            ErrorBuilder("cpu_backend::reduce_broadcast").not_implemented("dtype not supported");
        return Storage{CpuStorage{ptr, nb, dt}};
    }

    Storage histogram_forward(const Storage& input,
                              const Shape& input_shape,
                              Dtype input_dtype,
                              double lo,
                              double hi,
                              std::int64_t bins,
                              bool density) override {
        const auto& cs = std::get<CpuStorage>(input);
        const std::size_t n = shape_numel(input_shape);
        const double step = (hi - lo) / static_cast<double>(bins);
        CpuStorage counts = alloc_cpu(static_cast<std::size_t>(bins), Dtype::F64);
        auto* dst = reinterpret_cast<double*>(counts.ptr.get());
        std::memset(dst, 0, counts.nbytes);
        auto read_val = [&](std::size_t i) -> double {
            switch (input_dtype) {
            case Dtype::F32:
                return static_cast<double>(reinterpret_cast<const float*>(cs.ptr.get())[i]);
            case Dtype::F64:
                return reinterpret_cast<const double*>(cs.ptr.get())[i];
            case Dtype::I32:
                return static_cast<double>(reinterpret_cast<const std::int32_t*>(cs.ptr.get())[i]);
            case Dtype::I64:
                return static_cast<double>(reinterpret_cast<const std::int64_t*>(cs.ptr.get())[i]);
            default:
                ErrorBuilder("cpu::histogram_forward").not_implemented("dtype");
                return 0.0;
            }
        };
        for (std::size_t i = 0; i < n; ++i) {
            const double v = read_val(i);
            if (v < lo || v > hi)
                continue;
            std::int64_t bin = static_cast<std::int64_t>((v - lo) / step);
            if (bin >= bins)
                bin = bins - 1;
            dst[bin] += 1.0;
        }
        if (density) {
            for (std::int64_t i = 0; i < bins; ++i)
                dst[i] /= (static_cast<double>(n) * step);
        }
        return Storage{std::move(counts)};
    }

    CpuStorage nonzero_forward(const Storage& input,
                               const Shape& input_shape,
                               Dtype input_dtype,
                               std::size_t& numel_out) override {
        const auto& cs = std::get<CpuStorage>(input);
        const std::size_t n = shape_numel(input_shape);
        const std::size_t ndim = input_shape.size();
        std::vector<bool> mask(n, false);
        auto check_nonzero = [&](const auto* p) {
            for (std::size_t i = 0; i < n; ++i)
                mask[i] = static_cast<double>(p[i]) != 0.0;
        };
        switch (input_dtype) {
        case Dtype::F32:
            check_nonzero(reinterpret_cast<const float*>(cs.ptr.get()));
            break;
        case Dtype::F64:
            check_nonzero(reinterpret_cast<const double*>(cs.ptr.get()));
            break;
        case Dtype::I32:
            check_nonzero(reinterpret_cast<const std::int32_t*>(cs.ptr.get()));
            break;
        case Dtype::I64:
            check_nonzero(reinterpret_cast<const std::int64_t*>(cs.ptr.get()));
            break;
        case Dtype::Bool:
            check_nonzero(reinterpret_cast<const std::uint8_t*>(cs.ptr.get()));
            break;
        default:
            ErrorBuilder("cpu::nonzero_forward").not_implemented("dtype not supported");
        }
        std::size_t count = 0;
        for (auto m : mask)
            if (m)
                ++count;
        numel_out = count;
        Shape out_shape{static_cast<std::int64_t>(count), static_cast<std::int64_t>(ndim)};
        CpuStorage out = alloc_cpu(count * ndim, Dtype::I64);
        auto* dst = reinterpret_cast<std::int64_t*>(out.ptr.get());
        Stride stride(ndim);
        if (ndim > 0) {
            stride.back() = 1;
            for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(ndim) - 2; d >= 0; --d)
                stride[static_cast<std::size_t>(d)] = stride[static_cast<std::size_t>(d) + 1] *
                                                      input_shape[static_cast<std::size_t>(d) + 1];
        }
        std::size_t row = 0;
        for (std::size_t flat = 0; flat < n; ++flat) {
            if (!mask[flat])
                continue;
            std::size_t rem = flat;
            for (std::size_t d = 0; d < ndim; ++d) {
                const std::int64_t coord =
                    static_cast<std::int64_t>(rem / static_cast<std::size_t>(stride[d]));
                rem %= static_cast<std::size_t>(stride[d]);
                dst[row * ndim + d] = coord;
            }
            ++row;
        }
        return out;
    }

    std::vector<Storage> lstm_forward(const Storage& input,
                                      const Storage& h0,
                                      const Storage& c0,
                                      const std::vector<Storage>& weights,
                                      const LstmOpts& opts,
                                      const Shape& out_shape,
                                      Dtype dt) override {
#ifdef __APPLE__
        if (dt == Dtype::F32 && !opts.bidirectional && opts.num_layers == 1 && opts.has_bias &&
            weights.size() >= 4) {
            const int T = opts.seq_len;
            const int B = opts.batch_size;
            const int I = opts.input_size;
            const int H = opts.hidden_size;
            const int fH = 4 * H;

            const auto& x_cpu = std::get<CpuStorage>(input);
            const auto& h0_cpu = std::get<CpuStorage>(h0);
            const auto& c0_cpu = std::get<CpuStorage>(c0);
            const auto& wih = std::get<CpuStorage>(weights[0]);
            const auto& whh = std::get<CpuStorage>(weights[1]);
            const auto& bih = std::get<CpuStorage>(weights[2]);
            const auto& bhh = std::get<CpuStorage>(weights[3]);

            CpuStorage out_cpu = alloc_cpu(static_cast<std::size_t>(T) * B * H, dt);
            CpuStorage hn_cpu = alloc_cpu(static_cast<std::size_t>(B) * H, dt);
            CpuStorage cn_cpu = alloc_cpu(static_cast<std::size_t>(B) * H, dt);

            CpuStorage bias_fused = alloc_cpu(static_cast<std::size_t>(fH), dt);
            {
                const float* bp = reinterpret_cast<const float*>(bih.ptr.get());
                const float* bq = reinterpret_cast<const float*>(bhh.ptr.get());
                float* dst = reinterpret_cast<float*>(bias_fused.ptr.get());
                for (int k = 0; k < fH; ++k)
                    dst[k] = bp[k] + bq[k];
            }

            float* wih_p = reinterpret_cast<float*>(wih.ptr.get());
            float* whh_p = reinterpret_cast<float*>(whh.ptr.get());
            float* bias_p = reinterpret_cast<float*>(bias_fused.ptr.get());

            auto make_gate = [&](int g) -> BNNSLSTMGateDescriptor {
                BNNSLSTMGateDescriptor gd = {};

                gd.iw_desc[0].layout = BNNSDataLayoutRowMajorMatrix;
                gd.iw_desc[0].size[0] = static_cast<std::size_t>(I);
                gd.iw_desc[0].size[1] = static_cast<std::size_t>(H);
                gd.iw_desc[0].data_type = BNNSDataTypeFloat32;
                gd.iw_desc[0].data = wih_p + g * H * I;

                gd.hw_desc.layout = BNNSDataLayoutRowMajorMatrix;
                gd.hw_desc.size[0] = static_cast<std::size_t>(H);
                gd.hw_desc.size[1] = static_cast<std::size_t>(H);
                gd.hw_desc.data_type = BNNSDataTypeFloat32;
                gd.hw_desc.data = whh_p + g * H * H;

                gd.b_desc.layout = BNNSDataLayoutVector;
                gd.b_desc.size[0] = static_cast<std::size_t>(H);
                gd.b_desc.data_type = BNNSDataTypeFloat32;
                gd.b_desc.data = bias_p + g * H;

                return gd;
            };

            BNNSNDArrayDescriptor seq_in = {};
            seq_in.layout = BNNSDataLayoutSNE;
            seq_in.size[0] = static_cast<std::size_t>(I);
            seq_in.size[1] = static_cast<std::size_t>(B);
            seq_in.size[2] = static_cast<std::size_t>(T);
            seq_in.data_type = BNNSDataTypeFloat32;
            seq_in.data = const_cast<float*>(reinterpret_cast<const float*>(x_cpu.ptr.get()));

            BNNSNDArrayDescriptor seq_out = {};
            seq_out.layout = BNNSDataLayoutSNE;
            seq_out.size[0] = static_cast<std::size_t>(H);
            seq_out.size[1] = static_cast<std::size_t>(B);
            seq_out.size[2] = static_cast<std::size_t>(T);
            seq_out.data_type = BNNSDataTypeFloat32;
            seq_out.data = reinterpret_cast<float*>(out_cpu.ptr.get());

            BNNSNDArrayDescriptor h_desc = {};
            h_desc.layout = BNNSDataLayoutRowMajorMatrix;
            h_desc.size[0] = static_cast<std::size_t>(H);
            h_desc.size[1] = static_cast<std::size_t>(B);
            h_desc.data_type = BNNSDataTypeFloat32;

            BNNSNDArrayDescriptor h0_desc = h_desc;
            h0_desc.data = const_cast<float*>(reinterpret_cast<const float*>(h0_cpu.ptr.get()));

            BNNSNDArrayDescriptor c0_desc = h_desc;
            c0_desc.data = const_cast<float*>(reinterpret_cast<const float*>(c0_cpu.ptr.get()));

            BNNSNDArrayDescriptor hn_desc = h_desc;
            hn_desc.data = reinterpret_cast<float*>(hn_cpu.ptr.get());

            BNNSNDArrayDescriptor cn_desc = h_desc;
            cn_desc.data = reinterpret_cast<float*>(cn_cpu.ptr.get());

            BNNSLayerParametersLSTM lstm_params = {};
            lstm_params.input_size = static_cast<std::size_t>(I);
            lstm_params.hidden_size = static_cast<std::size_t>(H);
            lstm_params.batch_size = static_cast<std::size_t>(B);
            lstm_params.num_layers = 1;
            lstm_params.seq_len = static_cast<std::size_t>(T);
            lstm_params.dropout = 0.0f;

            lstm_params.lstm_flags = BNNSLayerFlagsLSTMDefaultActivations;

            lstm_params.input_descriptor.data_desc = seq_in;
            lstm_params.input_descriptor.hidden_desc = h0_desc;
            lstm_params.input_descriptor.cell_state_desc = c0_desc;

            lstm_params.output_descriptor.data_desc = seq_out;
            lstm_params.output_descriptor.hidden_desc = hn_desc;
            lstm_params.output_descriptor.cell_state_desc = cn_desc;

            lstm_params.input_gate = make_gate(0);
            lstm_params.forget_gate = make_gate(1);
            lstm_params.candidate_gate = make_gate(2);
            lstm_params.output_gate = make_gate(3);

            BNNSActivation hidden_act = {};
            hidden_act.function = BNNSActivationFunctionTanh;
            lstm_params.hidden_activation = hidden_act;

            std::size_t cache_cap = BNNSComputeLSTMTrainingCacheCapacity(&lstm_params);
            std::vector<std::byte> cache_buf(cache_cap ? cache_cap : 1);
            int ret = BNNSDirectApplyLSTMBatchTrainingCaching(&lstm_params, nullptr,
                                                              cache_buf.data(), cache_cap);
            if (ret == 0) {
                return {Storage{std::move(out_cpu)}, Storage{std::move(hn_cpu)}, Storage {
                            std::move(cn_cpu)
                        }};
            }
        }
#endif
        return IBackend::lstm_forward(input, h0, c0, weights, opts, out_shape, dt);
    }

    std::vector<Storage> lstm_forward_train(const Storage& input,
                                            const Storage& h0,
                                            const Storage& c0,
                                            const std::vector<Storage>& weights,
                                            const LstmOpts& opts,
                                            Dtype dt) override {
        if (dt != Dtype::F32 || weights.size() < 4)
            return IBackend::lstm_forward_train(input, h0, c0, weights, opts, dt);

        const int T = opts.seq_len, B = opts.batch_size;
        const int I = opts.input_size, H = opts.hidden_size;
        const int fH = 4 * H;

        const auto& x_s = std::get<CpuStorage>(input);
        const auto& h0_s = std::get<CpuStorage>(h0);
        const auto& c0_s = std::get<CpuStorage>(c0);
        const auto& wih_s = std::get<CpuStorage>(weights[0]);
        const auto& whh_s = std::get<CpuStorage>(weights[1]);
        const auto& bih_s = std::get<CpuStorage>(weights[2]);
        const auto& bhh_s = std::get<CpuStorage>(weights[3]);

        const float* Xp = reinterpret_cast<const float*>(x_s.ptr.get());
        const float* h0p = reinterpret_cast<const float*>(h0_s.ptr.get());
        const float* c0p = reinterpret_cast<const float*>(c0_s.ptr.get());
        const float* Wih = reinterpret_cast<const float*>(wih_s.ptr.get());
        const float* Whh = reinterpret_cast<const float*>(whh_s.ptr.get());
        const float* Bih = reinterpret_cast<const float*>(bih_s.ptr.get());
        const float* Bhh = reinterpret_cast<const float*>(bhh_s.ptr.get());

        CpuStorage out_s = alloc_cpu(static_cast<std::size_t>(T) * B * H, dt);
        CpuStorage hn_s = alloc_cpu(static_cast<std::size_t>(B) * H, dt);
        CpuStorage cn_s = alloc_cpu(static_cast<std::size_t>(B) * H, dt);
        CpuStorage gates_s = alloc_cpu(static_cast<std::size_t>(T) * B * fH, dt);

        CpuStorage cells_s = alloc_cpu(static_cast<std::size_t>(T + 1) * B * H, dt);

        float* Yp = reinterpret_cast<float*>(out_s.ptr.get());
        float* Hnp = reinterpret_cast<float*>(hn_s.ptr.get());
        float* Cnp = reinterpret_cast<float*>(cn_s.ptr.get());
        float* Gates = reinterpret_cast<float*>(gates_s.ptr.get());
        float* Cells = reinterpret_cast<float*>(cells_s.ptr.get());

        std::memcpy(Cells, c0p, static_cast<std::size_t>(B) * H * sizeof(float));

        std::vector<float> bias_fused(static_cast<std::size_t>(fH));
        for (int k = 0; k < fH; ++k)
            bias_fused[static_cast<std::size_t>(k)] = Bih[k] + Bhh[k];

        std::vector<float> h_prev(static_cast<std::size_t>(B) * H);
        std::memcpy(h_prev.data(), h0p, static_cast<std::size_t>(B) * H * sizeof(float));

        std::vector<float> raw(static_cast<std::size_t>(B) * fH);

        for (int t = 0; t < T; ++t) {
            const float* xt = Xp + t * B * I;
            float* gt = Gates + t * B * fH;
            float* ct = Cells + (t + 1) * B * H;
            float* yt = Yp + t * B * H;
            const float* ct_prev = Cells + t * B * H;

            cpu::sgemm(false, true, B, fH, I, 1.0f, xt, I, Wih, I, 0.0f, raw.data(), fH);

            cpu::sgemm(false, true, B, fH, H, 1.0f, h_prev.data(), H, Whh, H, 1.0f, raw.data(), fH);

            for (int b = 0; b < B; ++b)
                for (int k = 0; k < fH; ++k)
                    raw[static_cast<std::size_t>(b * fH + k)] +=
                        bias_fused[static_cast<std::size_t>(k)];

            for (int b = 0; b < B; ++b) {
                float* rb = raw.data() + b * fH;
                float* gb = gt + b * fH;

                for (int k = 0; k < H; ++k)
                    gb[k] = 1.0f / (1.0f + std::exp(-rb[k]));
                for (int k = 0; k < H; ++k)
                    gb[H + k] = 1.0f / (1.0f + std::exp(-rb[H + k]));
                for (int k = 0; k < H; ++k)
                    gb[2 * H + k] = std::tanh(rb[2 * H + k]);
                for (int k = 0; k < H; ++k)
                    gb[3 * H + k] = 1.0f / (1.0f + std::exp(-rb[3 * H + k]));

                const float* cp = ct_prev + b * H;
                float* cnb = ct + b * H;
                float* ynb = yt + b * H;
                for (int k = 0; k < H; ++k)
                    cnb[k] = gb[H + k] * cp[k] + gb[k] * gb[2 * H + k];

                for (int k = 0; k < H; ++k)
                    ynb[k] = gb[3 * H + k] * std::tanh(cnb[k]);

                std::memcpy(h_prev.data() + b * H, ynb,
                            static_cast<std::size_t>(H) * sizeof(float));
            }
        }

        std::memcpy(Hnp, h_prev.data(), static_cast<std::size_t>(B) * H * sizeof(float));
        std::memcpy(Cnp, Cells + T * B * H, static_cast<std::size_t>(B) * H * sizeof(float));

        return {Storage{std::move(out_s)}, Storage{std::move(hn_s)}, Storage{std::move(cn_s)},
                Storage{std::move(gates_s)}, Storage{std::move(cells_s)}};
    }

    std::vector<Storage> lstm_backward(const Storage& grad_output,
                                       const Storage& grad_hn,
                                       const Storage& grad_cn,
                                       const Storage& input,
                                       const Storage& h0,
                                       const std::vector<Storage>& weights,
                                       const Storage& gates_all,
                                       const Storage& cells_all,
                                       const LstmOpts& opts,
                                       Dtype dt) override {
        if (dt != Dtype::F32 || weights.size() < 4)
            return IBackend::lstm_backward(grad_output, grad_hn, grad_cn, input, h0, weights,
                                           gates_all, cells_all, opts, dt);

        const int T = opts.seq_len, B = opts.batch_size;
        const int I = opts.input_size, H = opts.hidden_size;
        const int fH = 4 * H;

        const float* dY =
            reinterpret_cast<const float*>(std::get<CpuStorage>(grad_output).ptr.get());
        const float* dHn = reinterpret_cast<const float*>(std::get<CpuStorage>(grad_hn).ptr.get());
        const float* dCn = reinterpret_cast<const float*>(std::get<CpuStorage>(grad_cn).ptr.get());
        const float* Xp = reinterpret_cast<const float*>(std::get<CpuStorage>(input).ptr.get());
        const float* H0p = reinterpret_cast<const float*>(std::get<CpuStorage>(h0).ptr.get());
        const float* Wih =
            reinterpret_cast<const float*>(std::get<CpuStorage>(weights[0]).ptr.get());
        const float* Whh =
            reinterpret_cast<const float*>(std::get<CpuStorage>(weights[1]).ptr.get());
        const float* Gates =
            reinterpret_cast<const float*>(std::get<CpuStorage>(gates_all).ptr.get());
        const float* Cells =
            reinterpret_cast<const float*>(std::get<CpuStorage>(cells_all).ptr.get());

        CpuStorage dX_s = alloc_cpu(static_cast<std::size_t>(T) * B * I, dt);
        CpuStorage dH0_s = alloc_cpu(static_cast<std::size_t>(B) * H, dt);
        CpuStorage dC0_s = alloc_cpu(static_cast<std::size_t>(B) * H, dt);
        CpuStorage dWih_s = alloc_cpu(static_cast<std::size_t>(fH) * I, dt);
        CpuStorage dWhh_s = alloc_cpu(static_cast<std::size_t>(fH) * H, dt);
        CpuStorage dBih_s = alloc_cpu(static_cast<std::size_t>(fH), dt);
        CpuStorage dBhh_s = alloc_cpu(static_cast<std::size_t>(fH), dt);

        float* dXp = reinterpret_cast<float*>(dX_s.ptr.get());
        float* dH0p = reinterpret_cast<float*>(dH0_s.ptr.get());
        float* dC0p = reinterpret_cast<float*>(dC0_s.ptr.get());
        float* dWih = reinterpret_cast<float*>(dWih_s.ptr.get());
        float* dWhh = reinterpret_cast<float*>(dWhh_s.ptr.get());
        float* dBih = reinterpret_cast<float*>(dBih_s.ptr.get());
        float* dBhh = reinterpret_cast<float*>(dBhh_s.ptr.get());

        std::memset(dXp, 0, static_cast<std::size_t>(T) * B * I * sizeof(float));
        std::memset(dWih, 0, static_cast<std::size_t>(fH) * I * sizeof(float));
        std::memset(dWhh, 0, static_cast<std::size_t>(fH) * H * sizeof(float));
        std::memset(dBih, 0, static_cast<std::size_t>(fH) * sizeof(float));
        std::memset(dBhh, 0, static_cast<std::size_t>(fH) * sizeof(float));

        std::vector<float> dh_next(static_cast<std::size_t>(B) * H);
        std::vector<float> dc_next(static_cast<std::size_t>(B) * H);
        std::memcpy(dh_next.data(), dHn, static_cast<std::size_t>(B) * H * sizeof(float));
        std::memcpy(dc_next.data(), dCn, static_cast<std::size_t>(B) * H * sizeof(float));

        std::vector<float> d_gates(static_cast<std::size_t>(B) * fH);
        std::vector<float> h_prev_local(static_cast<std::size_t>(B) * H);

        for (int t = T - 1; t >= 0; --t) {
            const float* gt = Gates + t * B * fH;
            const float* ct = Cells + (t + 1) * B * H;
            const float* ct_prev = Cells + t * B * H;
            const float* xt = Xp + t * B * I;
            float* dxt = dXp + t * B * I;

            if (t == 0) {
                std::memcpy(h_prev_local.data(), H0p,
                            static_cast<std::size_t>(B) * H * sizeof(float));
            } else {
                const float* o_prev = Gates + (t - 1) * B * fH + 3 * H;
                const float* c_t_prev = Cells + t * B * H;
                for (int b = 0; b < B; ++b) {
                    for (int k = 0; k < H; ++k) {
                        h_prev_local[static_cast<std::size_t>(b * H + k)] =
                            o_prev[b * fH + k] * std::tanh(c_t_prev[b * H + k]);
                    }
                }
            }

            for (int b = 0; b < B; ++b) {
                const float* gb = gt + b * fH;
                const float* ctb = ct + b * H;
                const float* cpb = ct_prev + b * H;
                float* dg = d_gates.data() + b * fH;

                for (int k = 0; k < H; ++k) {
                    float dh_k =
                        dY[t * B * H + b * H + k] + dh_next[static_cast<std::size_t>(b * H + k)];
                    float tanh_c = std::tanh(ctb[k]);
                    float dc_k = dh_k * gb[3 * H + k] * (1.0f - tanh_c * tanh_c) +
                                 dc_next[static_cast<std::size_t>(b * H + k)];

                    dg[3 * H + k] = dh_k * tanh_c * gb[3 * H + k] * (1.0f - gb[3 * H + k]);

                    dg[H + k] = dc_k * cpb[k] * gb[H + k] * (1.0f - gb[H + k]);

                    dg[k] = dc_k * gb[2 * H + k] * gb[k] * (1.0f - gb[k]);

                    dg[2 * H + k] = dc_k * gb[k] * (1.0f - gb[2 * H + k] * gb[2 * H + k]);

                    dc_next[static_cast<std::size_t>(b * H + k)] = dc_k * gb[H + k];
                }
            }

            cpu::sgemm(false, false, B, I, fH, 1.0f, d_gates.data(), fH, Wih, I, 0.0f, dxt, I);

            cpu::sgemm(false, false, B, H, fH, 1.0f, d_gates.data(), fH, Whh, H, 0.0f,
                       dh_next.data(), H);

            cpu::sgemm(true, false, fH, I, B, 1.0f, d_gates.data(), fH, xt, I, 1.0f, dWih, I);

            cpu::sgemm(true, false, fH, H, B, 1.0f, d_gates.data(), fH, h_prev_local.data(), H,
                       1.0f, dWhh, H);

            for (int b = 0; b < B; ++b)
                for (int k = 0; k < fH; ++k) {
                    dBih[k] += d_gates[static_cast<std::size_t>(b * fH + k)];
                    dBhh[k] += d_gates[static_cast<std::size_t>(b * fH + k)];
                }
        }

        std::memcpy(dH0p, dh_next.data(), static_cast<std::size_t>(B) * H * sizeof(float));
        std::memcpy(dC0p, dc_next.data(), static_cast<std::size_t>(B) * H * sizeof(float));

        return {Storage{std::move(dX_s)},   Storage{std::move(dH0_s)},  Storage{std::move(dC0_s)},
                Storage{std::move(dWih_s)}, Storage{std::move(dWhh_s)}, Storage{std::move(dBih_s)},
                Storage{std::move(dBhh_s)}};
    }

    Storage fused_linear_relu_forward(const Storage& x,
                                      const Storage& w,
                                      const Storage& b,
                                      const Shape& out_shape,
                                      Dtype dt) override {
        if (dt != Dtype::F32)
            return IBackend::fused_linear_relu_forward(x, w, b, out_shape, dt);

        const std::size_t elem = dtype_size(dt);
        std::size_t M = 1;
        for (std::size_t i = 0; i + 1 < out_shape.size(); ++i)
            M *= static_cast<std::size_t>(out_shape[i]);
        const std::size_t N_out = static_cast<std::size_t>(out_shape.back());
        const std::size_t K = std::get<CpuStorage>(x).nbytes / (elem * (M > 0 ? M : 1));
        const Shape x_shape = {static_cast<std::int64_t>(M), static_cast<std::int64_t>(K)};
        const Shape weight_shape = {static_cast<std::int64_t>(N_out), static_cast<std::int64_t>(K)};

        Storage lin_out = linear(x, w, b, x_shape, weight_shape, out_shape, dt);

#ifdef __APPLE__
        {
            auto& cpu = std::get<CpuStorage>(lin_out);
            const std::size_t n = cpu.nbytes / sizeof(float);
            float* p = reinterpret_cast<float*>(cpu.ptr.get());
            cpu::vrelu_f32(p, p, n);
        }
#else
        {
            auto& cpu = std::get<CpuStorage>(lin_out);
            const std::size_t n = cpu.nbytes / sizeof(float);
            float* p = reinterpret_cast<float*>(cpu.ptr.get());
            for (std::size_t i = 0; i < n; ++i)
                if (p[i] < 0.f)
                    p[i] = 0.f;
        }
#endif
        return lin_out;
    }

    Storage fused_linear_gelu_forward(const Storage& x,
                                      const Storage& w,
                                      const Storage& b,
                                      const Shape& out_shape,
                                      Dtype dt) override {
        if (dt != Dtype::F32)
            return IBackend::fused_linear_gelu_forward(x, w, b, out_shape, dt);

        const std::size_t elem_g = dtype_size(dt);
        std::size_t M_g = 1;
        for (std::size_t i = 0; i + 1 < out_shape.size(); ++i)
            M_g *= static_cast<std::size_t>(out_shape[i]);
        const std::size_t N_g = static_cast<std::size_t>(out_shape.back());
        const std::size_t K_g = std::get<CpuStorage>(x).nbytes / (elem_g * (M_g > 0 ? M_g : 1));
        const Shape x_shape_g = {static_cast<std::int64_t>(M_g), static_cast<std::int64_t>(K_g)};
        const Shape weight_shape_g = {static_cast<std::int64_t>(N_g),
                                      static_cast<std::int64_t>(K_g)};

        Storage lin_out = linear(x, w, b, x_shape_g, weight_shape_g, out_shape, dt);

#ifdef __APPLE__
        {
            auto& cpu = std::get<CpuStorage>(lin_out);
            const int n = static_cast<int>(cpu.nbytes / sizeof(float));
            float* p = reinterpret_cast<float*>(cpu.ptr.get());

            std::vector<float> scratch(static_cast<std::size_t>(n));
            constexpr float kSqrt2OverPi = 0.7978845608f;
            constexpr float kCoeff = 0.044715f;

            for (int i = 0; i < n; ++i) {
                const float xi = p[i];
                scratch[static_cast<std::size_t>(i)] = kSqrt2OverPi * (xi + kCoeff * xi * xi * xi);
            }
            cpu::vtanh_f32(scratch.data(), scratch.data(), static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
                p[i] = 0.5f * p[i] * (1.f + scratch[static_cast<std::size_t>(i)]);
        }
#else
        {
            auto& cpu = std::get<CpuStorage>(lin_out);
            const std::size_t n = cpu.nbytes / sizeof(float);
            float* p = reinterpret_cast<float*>(cpu.ptr.get());
            constexpr float kSqrt2OverPi = 0.7978845608f;
            constexpr float kCoeff = 0.044715f;
            for (std::size_t i = 0; i < n; ++i) {
                const float xi = p[i];
                const float inner = kSqrt2OverPi * (xi + kCoeff * xi * xi * xi);
                p[i] = 0.5f * xi * (1.f + std::tanh(inner));
            }
        }
#endif
        return lin_out;
    }
};

}  // namespace backend
}  // namespace lucid

// Anonymous-namespace static registrar that calls Dispatcher::register_backend
// for Device::CPU at process startup, before any tensor code executes.
// BackendInit.cpp includes this header to trigger the registration.
namespace {
struct CpuBackendRegistrar {
    CpuBackendRegistrar() {
        lucid::backend::Dispatcher::register_backend(
            lucid::Device::CPU, std::make_unique<lucid::backend::CpuBackend>());
    }
} g_cpu_registrar;
}  // namespace
