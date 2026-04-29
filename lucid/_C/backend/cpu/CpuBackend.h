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
