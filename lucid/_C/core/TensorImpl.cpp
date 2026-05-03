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
    if (autograd_)
        autograd_->grad.reset();
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

}  // namespace lucid
