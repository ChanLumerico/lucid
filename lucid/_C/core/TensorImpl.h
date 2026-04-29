#pragma once

// =====================================================================
// Lucid C++ engine — TensorImpl (single source of truth for tensor data).
// =====================================================================
//
// Owns:           Storage (variant<CpuStorage, GpuStorage>), TensorMeta
//                 (shape/stride/dtype/device), optional AutogradMeta
//                 (requires_grad, grad_fn, grad, version).
// Does not own:   Python references — `data_as_python()` returns a zero-copy
//                 numpy view that holds a Python capsule which keeps the
//                 underlying shared_ptr alive.
//
// Fields are private. External code must go through the public accessor/
// mutator API below. The only classes granted friend access are TensorImpl
// itself (its own .cpp implementations) and a small set of autograd engine
// internals that must bypass the API for performance-critical paths.
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
#include "TensorMeta.h"

namespace py = pybind11;

namespace lucid {

/// Reference-counted tensor: owns storage, shape, dtype, device, and autograd metadata.
class LUCID_API TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    // ------------------------------------------------------------------ //
    // Construction
    // ------------------------------------------------------------------ //
    TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad);

    static std::shared_ptr<TensorImpl> from_numpy(py::array arr, Device device, bool requires_grad);

    // ------------------------------------------------------------------ //
    // Storage accessors
    // ------------------------------------------------------------------ //
    const Storage& storage() const noexcept { return storage_; }
    Storage& mutable_storage() noexcept { return storage_; }

    /// Byte offset of this view's data within the underlying buffer.
    std::size_t storage_offset() const noexcept { return offset_; }

    /// True when multiple TensorImpl objects share the same CPU buffer ptr.
    bool storage_is_shared() const noexcept;

    // Factory: create a metadata-only view sharing storage with `base`.
    // Ownership of the underlying bytes is shared via the CpuStorage shared_ptr
    // (CPU) or MLX lazy graph (GPU). `shape`, `stride`, `offset_bytes` describe
    // the view window within `base`'s buffer.
    static std::shared_ptr<TensorImpl> make_view(const std::shared_ptr<TensorImpl>& base,
                                                 Shape shape,
                                                 Stride stride,
                                                 std::size_t offset_bytes = 0);

    // ------------------------------------------------------------------ //
    // TensorMeta accessors (shape/stride/dtype/device)
    // ------------------------------------------------------------------ //
    const TensorMeta& meta() const noexcept { return meta_; }

    const Shape& shape() const noexcept { return meta_.shape; }
    const Stride& stride() const noexcept { return meta_.stride; }
    Dtype dtype() const noexcept { return meta_.dtype; }
    Device device() const noexcept { return meta_.device; }

    // ------------------------------------------------------------------ //
    // AutogradMeta accessors
    // ------------------------------------------------------------------ //
    bool requires_grad() const noexcept { return autograd_ ? autograd_->requires_grad : false; }
    bool is_leaf() const noexcept { return autograd_ ? autograd_->is_leaf : true; }
    std::int64_t version() const noexcept { return autograd_ ? autograd_->version : 0; }
    const std::shared_ptr<Node>& grad_fn() const noexcept {
        static const std::shared_ptr<Node> kNull;
        return autograd_ ? autograd_->grad_fn : kNull;
    }
    const std::optional<Storage>& grad_storage() const noexcept {
        static const std::optional<Storage> kEmpty;
        return autograd_ ? autograd_->grad : kEmpty;
    }
    std::optional<Storage>& mutable_grad_storage() noexcept {
        ensure_autograd();
        return autograd_->grad;
    }

    // ------------------------------------------------------------------ //
    // AutogradMeta mutators
    // ------------------------------------------------------------------ //
    // In-place ops may change dtype/device after construction (e.g. cast_).
    void set_dtype(Dtype dt) noexcept { meta_.dtype = dt; }
    void set_device(Device dv) noexcept { meta_.device = dv; }

    void set_requires_grad(bool v) noexcept {
        if (v || autograd_)
            ensure_autograd()->requires_grad = v;
    }
    void set_leaf(bool v) noexcept { ensure_autograd()->is_leaf = v; }
    void set_grad_fn(std::shared_ptr<Node> fn) noexcept {
        ensure_autograd()->grad_fn = std::move(fn);
    }
    void clear_grad_fn() noexcept {
        if (autograd_)
            autograd_->grad_fn.reset();
    }
    void set_grad_storage(Storage grad) { ensure_autograd()->grad = std::move(grad); }
    void bump_version() noexcept {
        if (autograd_)
            ++autograd_->version;
    }

    // ------------------------------------------------------------------ //
    // Derived metrics
    // ------------------------------------------------------------------ //
    std::size_t numel() const;
    std::size_t nbytes() const;
    bool is_contiguous() const;

    // ------------------------------------------------------------------ //
    // Python / NumPy interop
    // ------------------------------------------------------------------ //
    py::object data_as_python() const;
    py::object grad_as_python() const;

    // ------------------------------------------------------------------ //
    // Utility
    // ------------------------------------------------------------------ //
    void copy_from(const TensorImpl& other);
    void zero_grad();

private:
    Storage storage_;         // base buffer (may be shared with other views)
    std::size_t offset_ = 0;  // byte offset of this view into storage_
    TensorMeta meta_;
    std::optional<AutogradMeta> autograd_;

    // Lazily allocate AutogradMeta on first write; returns reference.
    AutogradMeta* ensure_autograd() {
        if (!autograd_)
            autograd_.emplace();
        return &*autograd_;
    }

    // Legacy field names — kept as aliases so TensorImpl.cpp (which was
    // written against the flat-field layout) compiles without edits.
    // These are private, so only TensorImpl.cpp sees them.
    Shape& shape_() noexcept { return meta_.shape; }
    Stride& stride_() noexcept { return meta_.stride; }
    Dtype& dtype_field() noexcept { return meta_.dtype; }
    Device& device_field() noexcept { return meta_.device; }
};

// TensorImplPtr is defined in fwd.h (std::shared_ptr<TensorImpl>).
// Include fwd.h if you only need the alias without the full definition.

}  // namespace lucid
