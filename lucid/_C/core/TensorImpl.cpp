#include "TensorImpl.h"

#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "Allocator.h"
#include "Exceptions.h"

namespace lucid {

namespace {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// numpy dtype char/kind -> lucid Dtype.
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
            // Reject unsigned NumPy dtypes — Lucid's Dtype enum is signed-only,
            // and silently reinterpreting `uint8`/`uint32` as `int8`/`int32`
            // changes values above the signed range and breaks round-trips.
            // Callers should explicitly cast their array to the matching
            // signed dtype (or to a float dtype) before constructing a Tensor.
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
    throw LucidError("lucid_dtype_to_np: unknown Dtype");
}

py::object make_numpy_view(const CpuStorage& s, const Shape& shape, const Stride& stride) {
    // Use the CpuStorage shared_ptr as the lifetime owner for the numpy array.
    // numpy holds a base object with a custom destructor that releases our ref.
    auto* keepalive = new std::shared_ptr<std::byte[]>(s.ptr);
    py::capsule owner(keepalive,
                      [](void* p) { delete static_cast<std::shared_ptr<std::byte[]>*>(p); });

    std::vector<py::ssize_t> py_shape(shape.begin(), shape.end());
    std::vector<py::ssize_t> py_stride(stride.begin(), stride.end());

    return py::array(lucid_dtype_to_np(s.dtype), py_shape, py_stride,
                     static_cast<const void*>(s.ptr.get()), owner);
}

}  // namespace

TensorImpl::TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad)
    : storage_(std::move(storage)),
      shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      requires_grad_(requires_grad) {
    stride_ = contiguous_stride(shape_, dtype_size(dtype_));
}

std::shared_ptr<TensorImpl> TensorImpl::from_numpy(py::array arr,
                                                   Device device,
                                                   bool requires_grad) {
    py::array_t<std::byte, py::array::c_style | py::array::forcecast> view =
        py::array_t<std::byte, py::array::c_style | py::array::forcecast>::ensure(arr);
    // Ensure() may return null on failure.
    if (!arr) {
        throw LucidError("from_numpy: input is not a numpy array");
    }

    Dtype dtype = np_dtype_to_lucid(arr.dtype());
    Shape shape;
    shape.reserve(arr.ndim());
    for (py::ssize_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(static_cast<std::int64_t>(arr.shape(i)));
    }

    const std::size_t elem = dtype_size(dtype);
    const std::size_t total = shape_numel(shape) * elem;

    CpuStorage cpu;
    cpu.ptr = allocate_aligned_bytes(total);
    cpu.nbytes = total;
    cpu.dtype = dtype;

    // Copy from numpy into aligned storage. Source must be C-contiguous of the
    // same dtype; ensure() above does the conversion if necessary.
    py::array contig = py::array::ensure(arr, py::array::c_style | py::array::forcecast);
    if (!contig) {
        throw LucidError("from_numpy: failed to obtain C-contiguous view");
    }
    if (total > 0) {
        std::memcpy(cpu.ptr.get(), contig.data(), total);
    }

    if (device == Device::GPU) {
        // Upload from the staging CpuStorage into an mlx::core::array. The
        // CpuStorage's shared_ptr is captured into the MLX array's deleter
        // so the host buffer outlives any zero-copy use MLX makes of it.
        auto gpu = gpu::upload_cpu_to_gpu(cpu, shape);
        return std::make_shared<TensorImpl>(Storage{std::move(gpu)}, std::move(shape), dtype,
                                            device, requires_grad);
    }
    return std::make_shared<TensorImpl>(Storage{std::move(cpu)}, std::move(shape), dtype, device,
                                        requires_grad);
}

std::size_t TensorImpl::numel() const {
    return shape_numel(shape_);
}

std::size_t TensorImpl::nbytes() const {
    return numel() * dtype_size(dtype_);
}

bool TensorImpl::is_contiguous() const {
    Stride expected = contiguous_stride(shape_, dtype_size(dtype_));
    return expected == stride_;
}

py::object TensorImpl::data_as_python() const {
    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> py::object { return make_numpy_view(s, shape_, stride_); },
            [&](const GpuStorage& g) -> py::object {
                // Eval + copy back to CPU so numpy gets a stable buffer.
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, shape_);
                return make_numpy_view(cpu, shape_, stride_);
            },
        },
        storage_);
}

py::object TensorImpl::grad_as_python() const {
    if (!grad_storage_.has_value()) {
        return py::none();
    }
    return std::visit(
        overloaded{
            [&](const CpuStorage& s) -> py::object { return make_numpy_view(s, shape_, stride_); },
            [&](const GpuStorage& g) -> py::object {
                CpuStorage cpu = gpu::download_gpu_to_cpu(g, shape_);
                return make_numpy_view(cpu, shape_, stride_);
            },
        },
        *grad_storage_);
}

void TensorImpl::copy_from(const TensorImpl& other) {
    if (other.device_ != device_) {
        throw DeviceMismatch(std::string(device_name(device_)),
                             std::string(device_name(other.device_)), "copy_from");
    }
    if (other.dtype_ != dtype_) {
        throw DtypeMismatch(std::string(dtype_name(dtype_)), std::string(dtype_name(other.dtype_)),
                            "copy_from");
    }
    if (other.shape_ != shape_) {
        throw ShapeMismatch(shape_, other.shape_, "copy_from");
    }

    std::visit(overloaded{
                   [&](CpuStorage& dst, const CpuStorage& src) {
                       if (dst.nbytes != src.nbytes) {
                           throw LucidError("copy_from: nbytes mismatch");
                       }
                       if (dst.nbytes > 0) {
                           std::memcpy(dst.ptr.get(), src.ptr.get(), dst.nbytes);
                       }
                   },
                   [&](GpuStorage& dst, const GpuStorage& src) {
                       if (dst.nbytes != src.nbytes) {
                           throw LucidError("copy_from: nbytes mismatch (GPU)");
                       }
                       // Re-point our shared_ptr at a new wrapper around a clone of
                       // the source array. We can't memcpy GPU buffers directly;
                       // MLX's `copy()` performs the device-side replication.
                       if (!src.arr) {
                           throw LucidError("copy_from: src GPU array is null");
                       }
                       auto cloned = ::mlx::core::copy(*src.arr);
                       dst.arr = gpu::wrap_mlx_array(std::move(cloned), dst.dtype).arr;
                   },
                   [&](auto&, auto&) {
                       throw DeviceMismatch(std::string(device_name(device_)),
                                            std::string(device_name(other.device_)),
                                            "copy_from (storage variant)");
                   },
               },
               storage_, other.storage_);

    version_ += 1;
}

void TensorImpl::zero_grad() {
    grad_storage_.reset();
}

}  // namespace lucid
