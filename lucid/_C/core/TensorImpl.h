#pragma once

// =====================================================================
// Lucid C++ engine — TensorImpl (single source of truth for tensor data).
// =====================================================================
//
// Owns:           Storage (variant<CpuStorage, GpuStorage>), shape, stride,
//                 dtype, device, requires_grad flag, version counter, optional
//                 grad_storage_, optional grad_fn_ (autograd Node).
// Does not own:   Python references — `data_as_python()` returns a zero-copy
//                 numpy view that holds a Python capsule which keeps the
//                 underlying shared_ptr alive.
//
// Threading:      Not safe for concurrent writes. Concurrent reads are safe
//                 iff no other thread is mutating. See docs/concurrency.md.
//
// Layer:          core/. May include from core/ only.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "Shape.h"
#include "Storage.h"

namespace py = pybind11;

namespace lucid {

// Forward decl — concrete type lives in autograd/Node.h (Phase 2).
class Node;

class LUCID_API TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    Storage storage_;
    Shape shape_;
    Stride stride_;
    Dtype dtype_;
    Device device_;

    bool requires_grad_ = false;
    bool is_leaf_ = true;
    std::int64_t version_ = 0;

    // Phase 2 wires these.
    std::shared_ptr<Node> grad_fn_;
    std::optional<Storage> grad_storage_;

    TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad);

    static std::shared_ptr<TensorImpl> from_numpy(py::array arr, Device device, bool requires_grad);

    const Storage& storage() const noexcept { return storage_; }
    Storage& mutable_storage() noexcept { return storage_; }

    const Shape& shape() const noexcept { return shape_; }
    const Stride& stride() const noexcept { return stride_; }

    Dtype dtype() const noexcept { return dtype_; }
    Device device() const noexcept { return device_; }

    bool requires_grad() const noexcept { return requires_grad_; }
    bool is_leaf() const noexcept { return is_leaf_; }

    std::int64_t version() const noexcept { return version_; }

    const std::shared_ptr<Node>& grad_fn() const noexcept { return grad_fn_; }
    const std::optional<Storage>& grad_storage() const noexcept { return grad_storage_; }

    std::optional<Storage>& mutable_grad_storage() noexcept { return grad_storage_; }


    void set_requires_grad(bool requires_grad) noexcept { requires_grad_ = requires_grad; }
    void set_leaf(bool is_leaf) noexcept { is_leaf_ = is_leaf; }

    void set_grad_fn(std::shared_ptr<Node> grad_fn) noexcept { grad_fn_ = std::move(grad_fn); }
    void clear_grad_fn() noexcept { grad_fn_.reset(); }

    void set_grad_storage(Storage grad) { grad_storage_ = std::move(grad); }
    void bump_version() noexcept { version_ += 1; }

    std::size_t numel() const;
    std::size_t nbytes() const;
    bool is_contiguous() const;

    py::object data_as_python() const;
    py::object grad_as_python() const;

    void copy_from(const TensorImpl& other);

    void zero_grad();
};

}  // namespace lucid
