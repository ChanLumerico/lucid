#include "TensorImpl.h"

#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>

#include <mlx/ops.h>

#include "../backend/gpu/MlxBridge.h"
#include "Allocator.h"
#include "Error.h"
#include "ErrorBuilder.h"

namespace lucid {

namespace {

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

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

py::object make_numpy_view(const CpuStorage& s, const Shape& shape, const Stride& stride) {
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
    : storage_(std::move(storage)) {
    meta_.shape = std::move(shape);
    meta_.dtype = dtype;
    meta_.device = device;
    meta_.stride = contiguous_stride(meta_.shape, dtype_size(dtype));
    if (requires_grad) {
        autograd_.emplace();
        autograd_->requires_grad = true;
    }
}

std::shared_ptr<TensorImpl>
TensorImpl::from_numpy(py::array arr, Device device, bool requires_grad) {
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

    CpuStorage cpu;
    cpu.ptr = allocate_aligned_bytes(total);
    cpu.nbytes = total;
    cpu.dtype = dtype;

    py::array contig = py::array::ensure(arr, py::array::c_style | py::array::forcecast);
    if (!contig) {
        ErrorBuilder("from_numpy").fail("failed to obtain C-contiguous view");
    }
    if (total > 0) {
        std::memcpy(cpu.ptr.get(), contig.data(), total);
    }

    if (device == Device::GPU) {
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
    if (autograd_)
        autograd_->grad.reset();
}

bool TensorImpl::storage_is_shared() const noexcept {
    if (const auto* cpu = std::get_if<CpuStorage>(&storage_))
        return cpu->ptr.use_count() > 1;
    return false;
}

std::shared_ptr<TensorImpl> TensorImpl::make_view(const std::shared_ptr<TensorImpl>& base,
                                                  Shape shape,
                                                  Stride stride,
                                                  std::size_t offset_bytes) {
    auto view = std::make_shared<TensorImpl>(base->storage_, std::move(shape), base->meta_.dtype,
                                             base->meta_.device, false);
    view->meta_.stride = std::move(stride);
    view->offset_ = base->offset_ + offset_bytes;

    if (base->requires_grad()) {
        view->set_requires_grad(true);
        view->set_leaf(base->is_leaf());
    }
    return view;
}

}  // namespace lucid
