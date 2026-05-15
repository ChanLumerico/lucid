// lucid/_C/core/TensorImpl.cpp
//
// Implementations of TensorImpl methods that are too large to inline.
// Key responsibilities handled here:
//
//   from_numpy  — converts a Python NumPy array to a TensorImpl, optionally
//                 uploading to GPU via the MLX bridge.
//   data/grad_as_python — materialises tensor data as a zero-copy (CPU) or
//                 synchronously-downloaded (GPU) NumPy array for Python.
//   copy_from   — typed in-place data copy with full variant dispatch.
//   make_view   — creates an aliasing TensorImpl with different geometry.

#include "TensorImpl.h"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "Allocator.h"
#include "Error.h"
#include "ErrorBuilder.h"

namespace lucid {

namespace {

// Overload-set helper for std::visit.  Deduces operator() from the provided
// lambdas using the standard "overloaded" pattern (C++17).  Each std::visit
// call below passes a set of lambdas, one per Storage variant.
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// Converts a NumPy dtype kind character and item size to the corresponding
// Lucid Dtype.  Unsigned integer kinds ('u') are rejected with a DtypeMismatch
// because the engine only supports signed integer arithmetic.
Dtype np_dtype_to_lucid(const py::dtype& d) {
    const char k = d.kind();
    const auto sz = static_cast<int>(d.itemsize());
    switch (k) {
    case 'b':
        return Dtype::Bool;
    case 'i':
        switch (sz) {
        case 1:
            return Dtype::I8;
        case 2:
            return Dtype::I16;
        case 4:
            return Dtype::I32;
        case 8:
            return Dtype::I64;
        }
        break;
    case 'u':
        throw DtypeMismatch("signed integer or float numpy dtype",
                            std::string("uint") + std::to_string(sz * 8),
                            "from_numpy: unsigned dtypes are not supported "
                            "(cast to int / float explicitly)");
    case 'f':
        switch (sz) {
        case 2:
            return Dtype::F16;
        case 4:
            return Dtype::F32;
        case 8:
            return Dtype::F64;
        }
        break;
    case 'c':
        if (sz == 8)
            return Dtype::C64;
        break;
    default:
        break;
    }
    throw DtypeMismatch("supported numpy dtype",
                        std::string("kind='") + k + "',itemsize=" + std::to_string(sz),
                        "from_numpy");
}

// Maps a Lucid Dtype to the corresponding NumPy dtype descriptor.
py::dtype lucid_dtype_to_np(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return py::dtype("bool");
    case Dtype::I8:
        return py::dtype("int8");
    case Dtype::I16:
        return py::dtype("int16");
    case Dtype::I32:
        return py::dtype("int32");
    case Dtype::I64:
        return py::dtype("int64");
    case Dtype::F16:
        return py::dtype("float16");
    case Dtype::F32:
        return py::dtype("float32");
    case Dtype::F64:
        return py::dtype("float64");
    case Dtype::C64:
        return py::dtype("complex64");
    }
    ErrorBuilder("lucid_dtype_to_np").fail("unknown Dtype");
}

// Creates a zero-copy NumPy array view backed by a CpuStorage buffer.
// A py::capsule is used as the NumPy "base" object so that the underlying
// shared_ptr keeps the allocation alive for the entire lifetime of the
// returned array — even if the originating TensorImpl is destroyed first.
//
// py::array requires a raw void* for the data pointer but needs to know who
// owns the memory; passing a capsule as the "base" object transfers that
// responsibility to the ref-counted shared_ptr copy inside the capsule.
py::object make_numpy_view(const CpuStorage& s, const Shape& shape, const Stride& stride) {
    // Heap-allocate a copy of the shared_ptr so the capsule's destructor can
    // call delete on the correct type.  The capsule holds a void* internally,
    // so we need a stable heap address for the shared_ptr object itself.
    auto* keepalive = new std::shared_ptr<std::byte[]>(s.ptr);
    py::capsule owner(keepalive,
                      [](void* p) { delete static_cast<std::shared_ptr<std::byte[]>*>(p); });

    std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
    std::vector<py::ssize_t> py_stride(stride.begin(), stride.end());

    return py::array(lucid_dtype_to_np(s.dtype), py_shape, py_stride,
                     static_cast<const void*>(s.ptr.get()), owner);
}

}  // namespace

// Stride is computed after all meta fields are set because contiguous_stride()
// needs both shape and elem_size; element size depends on dtype.
TensorImpl::TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad)
    : storage_(std::move(storage)) {
    meta_.shape = std::move(shape);
    meta_.dtype = dtype;
    meta_.device = device;
    // Derive the C-contiguous stride from the final shape and element size.
    meta_.stride = contiguous_stride(meta_.shape, dtype_size(dtype));
    if (requires_grad) {
        // emplace() constructs AutogradMeta in-place inside the optional,
        // avoiding an extra heap allocation versus make_optional.
        autograd_.emplace();
        autograd_->requires_grad = true;
    }
}

std::shared_ptr<TensorImpl>
TensorImpl::from_numpy(py::array arr, Device device, bool requires_grad) {
    // Request a C-contiguous view so that the subsequent memcpy can assume
    // row-major layout.  forcecast allows NumPy to create a temporary copy if
    // the source array is Fortran-order or non-contiguous, rather than failing.
    py::array_t<std::byte, py::array::c_style | py::array::forcecast> view =
        py::array_t<std::byte, py::array::c_style | py::array::forcecast>::ensure(arr);
    if (!arr) {
        ErrorBuilder("from_numpy").fail("input is not a numpy array");
    }

    Dtype dtype = np_dtype_to_lucid(arr.dtype());
    Shape shape;
    shape.reserve(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(static_cast<std::int64_t>(arr.shape(i)));
    }

    const std::size_t elem = dtype_size(dtype);
    const std::size_t total = shape_numel(shape) * elem;

    // Always land in a fresh aligned CPU buffer first, regardless of the
    // target device.  For GPU tensors the buffer is uploaded below via the
    // MLX bridge.  We never write into arr's buffer directly to avoid
    // inadvertently mutating the Python-side array.
    CpuStorage cpu;
    cpu.ptr = allocate_aligned_bytes(total);
    cpu.nbytes = total;
    cpu.dtype = dtype;

    // Obtain the C-contiguous view a second time (arr may have changed ownership
    // after the first ensure() call due to pybind11 temporaries).
    py::array contig = py::array::ensure(arr, py::array::c_style | py::array::forcecast);
    if (!contig) {
        ErrorBuilder("from_numpy").fail("failed to obtain C-contiguous view");
    }
    if (total > 0) {
        std::memcpy(cpu.ptr.get(), contig.data(), total);
    }

    if (device == Device::GPU) {
        // Upload the CPU-side copy to the MLX lazy graph and discard the
        // temporary CPU buffer; the GpuStorage retains the only live reference.
        return std::make_shared<TensorImpl>(Storage{gpu::upload_cpu_to_gpu(cpu, shape)},
                                            std::move(shape), dtype, device, requires_grad);
    }
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(shape), dtype, device,
                                        requires_grad);
}

std::size_t TensorImpl::numel() const {
    return meta_.numel();
}

std::size_t TensorImpl::nbytes() const {
    return numel() * dtype_size(meta_.dtype);
}

bool TensorImpl::is_contiguous() const {
    Stride expected = contiguous_stride(meta_.shape, dtype_size(meta_.dtype));
    return expected == meta_.stride;
}

// Dispatches on the active Storage variant to produce a NumPy array.
// GPU tensors are synchronously downloaded via the MLX bridge before wrapping;
// SharedStorage is exposed through its cpu_view() alias.
py::object TensorImpl::data_as_python() const {
    return std::visit(overloaded{
                          [&](const CpuStorage& s) -> py::object {
                              return make_numpy_view(s, meta_.shape, meta_.stride);
                          },
                          [&](const GpuStorage& g) -> py::object {
                              CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                              return make_numpy_view(cpu, meta_.shape, meta_.stride);
                          },

                          [&](const SharedStorage& sh) -> py::object {
                              CpuStorage v = sh.cpu_view();
                              return make_numpy_view(v, meta_.shape, meta_.stride);
                          },
                      },
                      storage_);
}

// Same variant dispatch as data_as_python() but applied to the optional
// gradient storage.  Returns py::none() when no gradient has been accumulated.
py::object TensorImpl::grad_as_python() const {
    if (!autograd_ || !autograd_->grad.has_value()) {
        return py::none();
    }
    return std::visit(overloaded{
                          [&](const CpuStorage& s) -> py::object {
                              return make_numpy_view(s, meta_.shape, meta_.stride);
                          },
                          [&](const GpuStorage& g) -> py::object {
                              CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                              return make_numpy_view(cpu, meta_.shape, meta_.stride);
                          },

                          [&](const SharedStorage& sh) -> py::object {
                              CpuStorage v = sh.cpu_view();
                              return make_numpy_view(v, meta_.shape, meta_.stride);
                          },
                      },
                      *autograd_->grad);
}

// ---------------------------------------------------------------------------
// NumPy-free Python interop: to_bytes / from_bytes / to_string
//
// These three implement everything ``import lucid`` needs for
// serialization and tensor printing without touching numpy.  Numpy is
// still used in the explicit bridge methods (data_as_python /
// from_numpy) but those are now strictly opt-in.
// ---------------------------------------------------------------------------

namespace {

// Walks a strided source buffer in row-major order, copying each element
// into a packed contiguous destination.  Used when the source storage's
// stride doesn't match the canonical contiguous stride (e.g. transposed
// or sliced views).
void walk_strided_to_contig(const std::byte* src,
                            std::byte* dst,
                            const Shape& shape,
                            const Stride& stride,
                            std::size_t depth,
                            std::size_t src_off,
                            std::size_t& dst_off,
                            std::size_t elem_size) {
    if (depth == shape.size()) {
        std::memcpy(dst + dst_off, src + src_off, elem_size);
        dst_off += elem_size;
        return;
    }
    for (std::int64_t i = 0; i < shape[depth]; ++i) {
        walk_strided_to_contig(src, dst, shape, stride, depth + 1,
                               src_off + static_cast<std::size_t>(i * stride[depth]), dst_off,
                               elem_size);
    }
}

// Materialises a contiguous byte snapshot of a CpuStorage view.  When the
// stride is already canonical we return a borrow of the underlying buffer
// (no copy); otherwise we walk the strides into a freshly allocated vector.
std::vector<std::byte> contig_snapshot_cpu(const CpuStorage& s,
                                           const Shape& shape,
                                           const Stride& stride,
                                           std::size_t storage_offset) {
    const std::size_t elem = dtype_size(s.dtype);
    const std::size_t total = shape_numel(shape) * elem;
    std::vector<std::byte> out(total);
    if (total == 0)
        return out;

    Stride contig = contiguous_stride(shape, elem);
    if (contig == stride && storage_offset == 0) {
        std::memcpy(out.data(), s.ptr.get(), total);
        return out;
    }
    std::size_t dst_off = 0;
    walk_strided_to_contig(s.ptr.get() + storage_offset, out.data(), shape, stride, 0, 0, dst_off,
                           elem);
    return out;
}

// Format a single scalar element at ``ptr`` of dtype ``dt`` into ``os``.
// Mirrors NumPy's default formatting closely enough for human reading:
//   * floats use %.<precision>g, with explicit "nan" / "inf" / "-inf"
//   * bools render as True / False
//   * complex64 renders as "(re+imj)" with Python-side conventions
//   * integers render with %lld / %ld via std::to_string
void format_element(const std::byte* ptr, Dtype dt, int precision, std::ostringstream& os) {
    auto fmt_float = [&](double v) {
        if (std::isnan(v)) {
            os << "nan";
            return;
        }
        if (std::isinf(v)) {
            os << (v < 0 ? "-inf" : "inf");
            return;
        }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
        std::string s(buf);
        // NumPy convention: a trailing dot is appended when the rendered
        // value would otherwise look like an integer (e.g. "1." instead
        // of "1") so the reader knows the dtype is floating-point.
        if (s.find_first_of(".eEnN") == std::string::npos) {
            s.push_back('.');
        }
        os << s;
    };

    switch (dt) {
    case Dtype::Bool: {
        const auto v = *reinterpret_cast<const std::uint8_t*>(ptr);
        os << (v ? "True" : "False");
        break;
    }
    case Dtype::I8:
        os << static_cast<int>(*reinterpret_cast<const std::int8_t*>(ptr));
        break;
    case Dtype::I16:
        os << *reinterpret_cast<const std::int16_t*>(ptr);
        break;
    case Dtype::I32:
        os << *reinterpret_cast<const std::int32_t*>(ptr);
        break;
    case Dtype::I64:
        os << *reinterpret_cast<const std::int64_t*>(ptr);
        break;
    case Dtype::F16: {
        // Half is stored as raw bits; cast through float for printing.  We
        // don't pull in __fp16 — manual IEEE-754 binary16 → float decode.
        const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(ptr);
        const std::uint32_t sign = (bits >> 15) & 0x1;
        const std::uint32_t exp = (bits >> 10) & 0x1f;
        const std::uint32_t mant = bits & 0x3ff;
        std::uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign << 31;
            } else {
                std::uint32_t e = 1;
                std::uint32_t m = mant;
                while ((m & 0x400) == 0) {
                    m <<= 1;
                    --e;
                }
                m &= 0x3ff;
                f = (sign << 31) | ((e + 112) << 23) | (m << 13);
            }
        } else if (exp == 31) {
            f = (sign << 31) | (0xff << 23) | (mant << 13);
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        fmt_float(static_cast<double>(out));
        break;
    }
    case Dtype::F32:
        fmt_float(static_cast<double>(*reinterpret_cast<const float*>(ptr)));
        break;
    case Dtype::F64:
        fmt_float(*reinterpret_cast<const double*>(ptr));
        break;
    case Dtype::C64: {
        // Stored as two contiguous f32: real, imag.
        const float re = *reinterpret_cast<const float*>(ptr);
        const float im = *reinterpret_cast<const float*>(ptr + 4);
        os << "(";
        fmt_float(static_cast<double>(re));
        if (im >= 0 || std::isnan(im))
            os << "+";
        fmt_float(static_cast<double>(im));
        os << "j)";
        break;
    }
    }
}

// Renders a contiguous (already-snapshot) buffer of shape × dtype as a
// NumPy-flavoured nested-bracket string.  ``threshold`` applies to total
// element count; ``edgeitems`` clips long axes once that ceiling is
// exceeded.
void render_nested(const std::byte* base,
                   const Shape& shape,
                   std::size_t depth,
                   std::size_t off,
                   const std::vector<std::int64_t>& contig_stride_bytes,
                   Dtype dt,
                   int precision,
                   bool truncate,
                   std::size_t edgeitems,
                   std::ostringstream& os) {
    if (depth == shape.size()) {
        format_element(base + off, dt, precision, os);
        return;
    }
    os << "[";
    const std::int64_t n = shape[depth];
    const std::int64_t step = contig_stride_bytes[depth];

    auto render_one = [&](std::int64_t i) {
        render_nested(base, shape, depth + 1, off + static_cast<std::size_t>(i * step),
                      contig_stride_bytes, dt, precision, truncate, edgeitems, os);
    };

    if (truncate && n > static_cast<std::int64_t>(edgeitems) * 2) {
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(edgeitems); ++i) {
            if (i > 0)
                os << ", ";
            render_one(i);
        }
        os << ", ..., ";
        for (std::int64_t i = n - static_cast<std::int64_t>(edgeitems); i < n; ++i) {
            render_one(i);
            if (i + 1 < n)
                os << ", ";
        }
    } else {
        for (std::int64_t i = 0; i < n; ++i) {
            if (i > 0)
                os << ", ";
            render_one(i);
        }
    }
    os << "]";
}

}  // namespace

py::bytes TensorImpl::to_bytes() const {
    const std::size_t total = nbytes();
    if (total == 0)
        return py::bytes();

    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> py::bytes {
                auto buf = contig_snapshot_cpu(s, meta_.shape, meta_.stride, offset_);
                return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
            },
            [&](const GpuStorage& g) -> py::bytes {
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                return py::bytes(reinterpret_cast<const char*>(cpu.ptr.get()), total);
            },
            [&](const SharedStorage& sh) -> py::bytes {
                CpuStorage v = sh.cpu_view();
                auto buf = contig_snapshot_cpu(v, meta_.shape, meta_.stride, offset_);
                return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
            },
        },
        storage_);
}

std::shared_ptr<TensorImpl> TensorImpl::from_bytes(
    py::bytes data, Shape shape, Dtype dtype, Device device, bool requires_grad) {
    const std::size_t elem = dtype_size(dtype);
    const std::size_t expected = shape_numel(shape) * elem;

    char* raw_ptr = nullptr;
    py::ssize_t raw_len = 0;
    if (PyBytes_AsStringAndSize(data.ptr(), &raw_ptr, &raw_len) != 0) {
        ErrorBuilder("from_bytes").fail("input is not a bytes object");
    }
    if (static_cast<std::size_t>(raw_len) != expected) {
        ErrorBuilder("from_bytes")
            .fail("byte length mismatch (got " + std::to_string(raw_len) + ", expected " +
                  std::to_string(expected) + ")");
    }

    CpuStorage cpu;
    cpu.ptr = allocate_aligned_bytes(expected);
    cpu.nbytes = expected;
    cpu.dtype = dtype;
    if (expected > 0) {
        std::memcpy(cpu.ptr.get(), raw_ptr, expected);
    }

    if (device == Device::GPU) {
        return std::make_shared<TensorImpl>(Storage{gpu::upload_cpu_to_gpu(cpu, shape)},
                                            std::move(shape), dtype, device, requires_grad);
    }
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(shape), dtype, device,
                                        requires_grad);
}

// Numpy-free cross-device copy used by ``Tensor.to(device=...)``.
// Mirrors the structure of ``to_bytes``+``from_bytes`` but skips the
// intermediate ``py::bytes`` round-trip — the contiguous-snapshot vector
// is reused directly as the new CpuStorage backing, so the data moves
// exactly once between source and destination storage.
std::shared_ptr<TensorImpl>
TensorImpl::transfer_to_device(Device target, bool requires_grad) const {
    auto build_cpu_from_view = [&](const CpuStorage& src) {
        auto snap = contig_snapshot_cpu(src, meta_.shape, meta_.stride, offset_);
        const std::size_t total = snap.size();
        CpuStorage out;
        out.ptr = allocate_aligned_bytes(total);
        out.nbytes = total;
        out.dtype = meta_.dtype;
        if (total > 0) {
            std::memcpy(out.ptr.get(), snap.data(), total);
        }
        return out;
    };

    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> std::shared_ptr<TensorImpl> {
                CpuStorage cpu = build_cpu_from_view(s);
                if (target == Device::GPU) {
                    return std::make_shared<TensorImpl>(
                        Storage{gpu::upload_cpu_to_gpu(cpu, meta_.shape)},
                        meta_.shape, meta_.dtype, target, requires_grad);
                }
                return std::make_shared<TensorImpl>(
                    Storage{std::move(cpu)}, meta_.shape, meta_.dtype, target, requires_grad);
            },
            [&](const GpuStorage& g) -> std::shared_ptr<TensorImpl> {
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                if (target == Device::GPU) {
                    // GPU → GPU: rare path; round-trip through CPU rather than
                    // adding an MLX-level array clone helper just for this.
                    return std::make_shared<TensorImpl>(
                        Storage{gpu::upload_cpu_to_gpu(cpu, meta_.shape)},
                        meta_.shape, meta_.dtype, target, requires_grad);
                }
                return std::make_shared<TensorImpl>(
                    Storage{std::move(cpu)}, meta_.shape, meta_.dtype, target, requires_grad);
            },
            [&](const SharedStorage& sh) -> std::shared_ptr<TensorImpl> {
                // _to.py routes is_metal_shared tensors through transfer_storage()
                // — that path is zero-copy relabel.  We provide this fallthrough
                // for safety: contiguous CPU snapshot then route as CPU storage.
                CpuStorage cpu = build_cpu_from_view(sh.cpu_view());
                if (target == Device::GPU) {
                    return std::make_shared<TensorImpl>(
                        Storage{gpu::upload_cpu_to_gpu(cpu, meta_.shape)},
                        meta_.shape, meta_.dtype, target, requires_grad);
                }
                return std::make_shared<TensorImpl>(
                    Storage{std::move(cpu)}, meta_.shape, meta_.dtype, target, requires_grad);
            },
        },
        storage_);
}

py::object TensorImpl::item() const {
    if (numel() != 1) {
        ErrorBuilder("item").fail("item() can only be called on a tensor with one element");
    }

    // Snapshot the single element to a contiguous CPU byte buffer.  This
    // dispatches through to_bytes() to avoid duplicating storage-variant
    // handling.
    py::bytes blob = to_bytes();
    char* raw = nullptr;
    py::ssize_t len = 0;
    PyBytes_AsStringAndSize(blob.ptr(), &raw, &len);

    switch (meta_.dtype) {
    case Dtype::F32: {
        float v;
        std::memcpy(&v, raw, sizeof(v));
        return py::float_(static_cast<double>(v));
    }
    case Dtype::F64: {
        double v;
        std::memcpy(&v, raw, sizeof(v));
        return py::float_(v);
    }
    case Dtype::F16: {
        // Reuse the same IEEE-754 binary16 → float decode used in
        // to_string()'s format_element(), kept inline so this stays a
        // self-contained engine helper.
        std::uint16_t bits;
        std::memcpy(&bits, raw, sizeof(bits));
        std::uint32_t sign = (bits >> 15) & 0x1u;
        std::uint32_t exp = (bits >> 10) & 0x1fu;
        std::uint32_t mant = bits & 0x3ffu;
        std::uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign << 31;
            } else {
                std::uint32_t e = 1;
                std::uint32_t m = mant;
                while ((m & 0x400u) == 0) {
                    m <<= 1;
                    --e;
                }
                m &= 0x3ffu;
                f = (sign << 31) | ((e + 112) << 23) | (m << 13);
            }
        } else if (exp == 31) {
            f = (sign << 31) | (0xffu << 23) | (mant << 13);
        } else {
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
        float out;
        std::memcpy(&out, &f, sizeof(out));
        return py::float_(static_cast<double>(out));
    }
    case Dtype::I64: {
        std::int64_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(v);
    }
    case Dtype::I32: {
        std::int32_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(v);
    }
    case Dtype::I16: {
        std::int16_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(static_cast<int>(v));
    }
    case Dtype::I8: {
        std::int8_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::int_(static_cast<int>(v));
    }
    case Dtype::Bool: {
        std::uint8_t v;
        std::memcpy(&v, raw, sizeof(v));
        return py::bool_(v != 0);
    }
    case Dtype::C64: {
        // Two contiguous f32: real, imag.
        float re, im;
        std::memcpy(&re, raw, sizeof(re));
        std::memcpy(&im, raw + sizeof(re), sizeof(im));
        return py::cast(std::complex<double>(static_cast<double>(re), static_cast<double>(im)));
    }
    }
    ErrorBuilder("item").fail("unsupported dtype");
    return py::none();
}

std::shared_ptr<TensorImpl> TensorImpl::grad_to_tensor() const {
    if (!autograd_)
        return nullptr;
    // Prefer the graph-mode gradient (set when backward(create_graph=True)).
    if (autograd_->grad_impl)
        return autograd_->grad_impl;
    if (!autograd_->grad.has_value())
        return nullptr;
    // Wrap the accumulated grad Storage as a fresh leaf TensorImpl with the
    // same shape/dtype/device as ``this``.  No data copy — the new impl
    // shares the underlying Storage variant.
    return std::make_shared<TensorImpl>(*autograd_->grad, meta_.shape, meta_.dtype, meta_.device,
                                        false);
}

std::string
TensorImpl::to_string(int precision, std::size_t threshold, std::size_t edgeitems) const {
    // Snapshot data on CPU regardless of device.  All formatting then
    // reads through a packed contiguous buffer with canonical stride.
    std::vector<std::byte> buf;
    std::visit(overloaded{
                   [&](const CpuStorage& s) {
                       buf = contig_snapshot_cpu(s, meta_.shape, meta_.stride, offset_);
                   },
                   [&](const GpuStorage& g) {
                       CpuStorage cpu = gpu::download_gpu_to_cpu(g, meta_.shape);
                       buf = contig_snapshot_cpu(
                           cpu, meta_.shape,
                           contiguous_stride(meta_.shape, dtype_size(meta_.dtype)), 0);
                   },
                   [&](const SharedStorage& sh) {
                       CpuStorage v = sh.cpu_view();
                       buf = contig_snapshot_cpu(v, meta_.shape, meta_.stride, offset_);
                   },
               },
               storage_);

    if (meta_.shape.empty()) {
        // 0-d scalar — render the single element with no brackets.
        std::ostringstream os;
        format_element(buf.data(), meta_.dtype, precision, os);
        return os.str();
    }

    // Compute byte-strides for the canonical contiguous layout we just
    // packed into ``buf``.
    const std::size_t elem = dtype_size(meta_.dtype);
    std::vector<std::int64_t> contig_stride_bytes(meta_.shape.size(), 0);
    if (!meta_.shape.empty()) {
        contig_stride_bytes.back() = static_cast<std::int64_t>(elem);
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(meta_.shape.size()) - 2; i >= 0; --i) {
            contig_stride_bytes[i] = contig_stride_bytes[i + 1] * meta_.shape[i + 1];
        }
    }

    const bool truncate = numel() > threshold;
    std::ostringstream os;
    render_nested(buf.data(), meta_.shape, 0, 0, contig_stride_bytes, meta_.dtype, precision,
                  truncate, edgeitems, os);
    return os.str();
}

// Copies element data from other into this tensor.
// Preconditions: same device, dtype, and shape (enforced via exceptions).
// Supported storage combinations: CPU←CPU, GPU←GPU, Shared←Shared,
// Shared←CPU, CPU←Shared.  Any other cross-variant combination throws a
// DeviceMismatch error.
void TensorImpl::copy_from(const TensorImpl& other) {
    if (other.device() != device()) {
        throw DeviceMismatch(std::string(device_name(meta_.device)),
                             std::string(device_name(other.device())), "copy_from");
    }
    if (other.dtype() != dtype()) {
        throw DtypeMismatch(std::string(dtype_name(meta_.dtype)),
                            std::string(dtype_name(other.dtype())), "copy_from");
    }
    if (other.shape() != shape()) {
        throw ShapeMismatch(meta_.shape, other.shape(), "copy_from");
    }

    std::visit(overloaded{
                   [&](CpuStorage& dst, const CpuStorage& src) {
                       if (dst.nbytes != src.nbytes) {
                           ErrorBuilder("copy_from").fail("nbytes mismatch");
                       }
                       if (dst.nbytes > 0) {
                           std::memcpy(dst.ptr.get(), src.ptr.get(), dst.nbytes);
                       }
                   },
                   [&](GpuStorage& dst, const GpuStorage& src) {
                       if (dst.nbytes != src.nbytes) {
                           ErrorBuilder("copy_from").fail("nbytes mismatch (GPU)");
                       }
                       if (!src.arr) {
                           ErrorBuilder("copy_from").fail("src GPU array is null");
                       }
                       // mlx::core::copy forces evaluation and produces a new
                       // independent array node; wrap_mlx_array converts it
                       // back into a GpuStorage with the correct dtype tag.
                       auto cloned = ::mlx::core::copy(*src.arr);
                       dst.arr = gpu::wrap_mlx_array(std::move(cloned), dst.dtype).arr;
                   },

                   [&](SharedStorage& dst, const SharedStorage& src) {
                       if (dst.nbytes != src.nbytes)
                           ErrorBuilder("copy_from").fail("nbytes mismatch (SharedStorage)");
                       if (dst.nbytes > 0)
                           std::memcpy(dst.cpu_ptr, src.cpu_ptr, dst.nbytes);
                   },
                   [&](SharedStorage& dst, const CpuStorage& src) {
                       if (dst.nbytes != src.nbytes)
                           ErrorBuilder("copy_from").fail("nbytes mismatch (Shared←CPU)");
                       if (dst.nbytes > 0)
                           std::memcpy(dst.cpu_ptr, src.ptr.get(), dst.nbytes);
                   },
                   [&](CpuStorage& dst, const SharedStorage& src) {
                       if (dst.nbytes != src.nbytes)
                           ErrorBuilder("copy_from").fail("nbytes mismatch (CPU←Shared)");
                       if (dst.nbytes > 0)
                           std::memcpy(dst.ptr.get(), src.cpu_ptr, dst.nbytes);
                   },
                   // Catch-all for cross-device variants that should never be
                   // reached given the device check above.
                   [&](auto&, auto&) {
                       throw DeviceMismatch(std::string(device_name(meta_.device)),
                                            std::string(device_name(other.device())),
                                            "copy_from (storage variant)");
                   },
               },
               storage_, other.storage());

    bump_version();
}

void TensorImpl::zero_grad() {
    if (autograd_) {
        autograd_->grad.reset();
        autograd_->grad_impl.reset();
    }
}

void TensorImpl::eval() const {
    if (meta_.device != Device::GPU)
        return;  // CPU: no-op
    const auto& gpu_st = std::get<GpuStorage>(storage_);
    if (gpu_st.arr)
        gpu_st.arr->eval();  // mlx::core::array::eval()
}

bool TensorImpl::storage_is_shared() const noexcept {
    // Only CpuStorage can be aliased through views; GPU arrays are never
    // directly shared at this level.
    if (const auto* cpu = std::get_if<CpuStorage>(&storage_))
        return cpu->ptr.use_count() > 1;
    return false;
}

// Creates a view TensorImpl that shares storage with base but has an
// independent shape, stride, and byte offset.  The view is not a leaf unless
// base is a leaf — it mirrors base's gradient-tracking status.
//
// Note: make_view is itself a non-differentiable metadata operation; there
// is no grad_fn attached to the view.  The caller must attach one separately
// if the view participates in a differentiable reshape or slice.
std::shared_ptr<TensorImpl> TensorImpl::make_view(const std::shared_ptr<TensorImpl>& base,
                                                  Shape shape,
                                                  Stride stride,
                                                  std::size_t offset_bytes) {
    // requires_grad=false here to avoid spuriously allocating AutogradMeta for
    // views of non-grad tensors; we set it manually below if base needs grads.
    auto view = std::make_shared<TensorImpl>(base->storage_, std::move(shape), base->meta_.dtype,
                                             base->meta_.device, false);
    view->meta_.stride = std::move(stride);
    // offset_ accumulates: a view of a view has the combined byte offset from
    // the original allocation's start.
    view->offset_ = base->offset_ + offset_bytes;

    if (base->requires_grad()) {
        view->set_requires_grad(true);
        view->set_leaf(base->is_leaf());
    }
    return view;
}

// Accumulate a graph-mode gradient into this leaf's grad_impl slot.
// First arrival: stores the incoming TensorImpl directly.
// Subsequent arrivals: add via add_op so the result remains differentiable.
void TensorImpl::accumulate_grad_impl(std::shared_ptr<TensorImpl> g) {
    ensure_autograd();
    if (!autograd_->grad_impl) {
        autograd_->grad_impl = std::move(g);
    } else {
        // Lazy-include to avoid a circular dependency at the header level.
        // add_op is defined in ops/bfunc/Add.h which is not included here;
        // declare it as an extern to keep TensorImpl.cpp independent of the ops layer.
        extern std::shared_ptr<TensorImpl> add_op(const std::shared_ptr<TensorImpl>&,
                                                  const std::shared_ptr<TensorImpl>&);
        autograd_->grad_impl = add_op(autograd_->grad_impl, g);
    }
}

}  // namespace lucid
