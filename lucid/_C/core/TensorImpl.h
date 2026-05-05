// lucid/_C/core/TensorImpl.h
//
// Central reference-counted tensor object.  TensorImpl combines three
// independent concerns into one heap-allocated object:
//
//   1. Storage — the backing memory (CpuStorage, GpuStorage, or
//      SharedStorage), potentially shared across multiple views.
//
//   2. Metadata — shape, stride, dtype, and device (TensorMeta), which can
//      differ between a base tensor and its views.
//
//   3. Autograd metadata — requires_grad flag, leaf status, version counter,
//      grad_fn pointer, and accumulated gradient (AutogradMeta).  The
//      AutogradMeta is stored in an std::optional so tensors that never
//      participate in differentiation pay no allocation cost.
//
// Ownership model:
//   TensorImpl is always managed through TensorImplPtr (shared_ptr<TensorImpl>).
//   Views created by make_view() share the same Storage as the base but own
//   an independent TensorMeta; the base tensor's lifetime is implicitly
//   extended by the shared Storage.
//
// Thread safety:
//   Individual field accessors are not thread-safe.  Concurrent writes to the
//   same TensorImpl require external synchronisation.  The VersionCounter
//   inside Storage uses atomic operations for cross-thread version reads, but
//   the autograd version field (AutogradMeta::version) is a plain int64 and
//   must only be mutated from one thread at a time.

#pragma once

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

// Reference-counted tensor implementation.  Python-level Tensor objects hold
// a TensorImplPtr; the C++ op layer receives and returns TensorImplPtr
// directly.  TensorImpl extends enable_shared_from_this so that member
// functions can safely produce additional shared_ptr references to *this.
class LUCID_API TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    // Constructs a TensorImpl from an already-allocated Storage.  The
    // C-contiguous stride is computed automatically from shape and dtype.
    // If requires_grad is true an AutogradMeta is immediately constructed.
    TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad);

    // Constructs a TensorImpl by copying the data from a NumPy array.
    // The data is always copied into a fresh CpuStorage buffer; if
    // device == Device::GPU the buffer is then uploaded to the MLX graph via
    // the GPU bridge.  Unsigned dtypes are rejected — the caller must cast
    // them to a signed or floating-point type first.
    static std::shared_ptr<TensorImpl> from_numpy(py::array arr, Device device, bool requires_grad);

    // ---------------------------------------------------------------------------
    // Storage accessors
    // ---------------------------------------------------------------------------

    const Storage& storage() const noexcept { return storage_; }
    Storage& mutable_storage() noexcept { return storage_; }

    // Byte offset from the start of the backing allocation to the first
    // element of this tensor.  Non-zero for views created with make_view().
    std::size_t storage_offset() const noexcept { return offset_; }

    // Returns true when the underlying CpuStorage's ptr is aliased by more
    // than one shared_ptr (i.e. at least one view of this tensor exists).
    // Always returns false for GPU and SharedStorage variants.
    bool storage_is_shared() const noexcept;

    // Creates a new TensorImpl that aliases base's Storage with a different
    // shape, stride, and optional byte offset.  The view inherits base's
    // requires_grad and is_leaf flags.  No data is copied.
    static std::shared_ptr<TensorImpl> make_view(const std::shared_ptr<TensorImpl>& base,
                                                 Shape shape,
                                                 Stride stride,
                                                 std::size_t offset_bytes = 0);

    // ---------------------------------------------------------------------------
    // Metadata accessors
    // ---------------------------------------------------------------------------

    const TensorMeta& meta() const noexcept { return meta_; }

    const Shape& shape() const noexcept { return meta_.shape; }
    const Stride& stride() const noexcept { return meta_.stride; }
    Dtype dtype() const noexcept { return meta_.dtype; }
    Device device() const noexcept { return meta_.device; }

    // ---------------------------------------------------------------------------
    // Autograd accessors
    // ---------------------------------------------------------------------------

    // Returns false when there is no AutogradMeta (common case for inference).
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

    // Returns a mutable reference to the gradient storage, constructing the
    // AutogradMeta on demand if it does not yet exist.
    std::optional<Storage>& mutable_grad_storage() noexcept {
        ensure_autograd();
        return autograd_->grad;
    }

    // ---------------------------------------------------------------------------
    // Mutation helpers
    // ---------------------------------------------------------------------------

    void set_dtype(Dtype dt) noexcept { meta_.dtype = dt; }
    void set_device(Device dv) noexcept { meta_.device = dv; }

    // Enables or disables gradient tracking, constructing AutogradMeta if
    // needed.  Setting to false does not destroy the AutogradMeta once it
    // exists — it only clears the flag.
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

    // Gradient as a full TensorImpl (set when backward was called with create_graph=true).
    // Returns nullptr when no graph-mode gradient has been accumulated.
    std::shared_ptr<TensorImpl> grad_as_impl() const noexcept {
        return autograd_ ? autograd_->grad_impl : nullptr;
    }
    void set_grad_impl(std::shared_ptr<TensorImpl> g) noexcept {
        ensure_autograd()->grad_impl = std::move(g);
    }
    // Accumulate a graph-mode gradient: first call sets it; subsequent calls add via add_op.
    void accumulate_grad_impl(std::shared_ptr<TensorImpl> g);

    // retain_grad: if true, Engine accumulates gradients into this tensor's grad
    // storage even when it is not a leaf, matching torch.Tensor.retain_grad().
    bool retains_grad() const noexcept {
        return autograd_ ? autograd_->retain_grad : false;
    }
    void set_retain_grad(bool v) noexcept { ensure_autograd()->retain_grad = v; }

    // Increments the autograd version counter; called by every in-place op.
    // No-op when there is no AutogradMeta (i.e. the tensor has never
    // participated in the autograd graph).
    void bump_version() noexcept {
        if (autograd_)
            ++autograd_->version;
    }

    // ---------------------------------------------------------------------------
    // Derived shape utilities
    // ---------------------------------------------------------------------------

    // Total number of elements (product of shape dimensions).
    std::size_t numel() const;

    // Total byte footprint of the tensor data (numel * dtype_size(dtype)).
    std::size_t nbytes() const;

    // True when the current stride equals the row-major contiguous stride for
    // this shape and dtype.
    bool is_contiguous() const;

    // ---------------------------------------------------------------------------
    // Python / NumPy interop
    // ---------------------------------------------------------------------------

    // Returns the tensor data as a NumPy array.  For GPU tensors the data is
    // synchronously downloaded to a temporary CPU buffer first.  The returned
    // array is a zero-copy view for CPU tensors (backed by a py::capsule that
    // keeps the shared_ptr alive).
    py::object data_as_python() const;

    // Returns the accumulated gradient as a NumPy array, or py::none() if no
    // gradient has been computed yet.  Same download logic as data_as_python().
    py::object grad_as_python() const;

    // ---------------------------------------------------------------------------
    // Data operations
    // ---------------------------------------------------------------------------

    // Copies data from other into this tensor.  other must have identical
    // device, dtype, and shape; mismatches throw the corresponding error type.
    // Supports CpuStorage↔CpuStorage, GpuStorage↔GpuStorage, and all
    // combinations involving SharedStorage.  Bumps the version counter on
    // success.
    void copy_from(const TensorImpl& other);

    // Clears the accumulated gradient (sets grad to std::nullopt).
    void zero_grad();

    // Forces evaluation of this tensor's lazy MLX computation graph.
    // GPU tensors: calls mlx::core::eval() on the underlying array.
    // CPU tensors: no-op.
    void eval() const;

private:
    Storage storage_;
    // Byte offset of the first element from the start of storage_.
    std::size_t offset_ = 0;
    TensorMeta meta_;
    // Lazily constructed; nullptr-equivalent (std::nullopt) for inference-only
    // tensors to avoid the overhead of the heap allocation.
    std::optional<AutogradMeta> autograd_;

    // Returns a pointer to the AutogradMeta, constructing it in-place if it
    // does not yet exist.  The returned pointer is stable (optional stores
    // in-place) as long as autograd_ is not reset.
    AutogradMeta* ensure_autograd() {
        if (!autograd_)
            autograd_.emplace();
        return &*autograd_;
    }

    // Private mutable references to the meta fields — used only within
    // TensorImpl's own methods; callers should use the public set_* functions.
    Shape& shape_() noexcept { return meta_.shape; }
    Stride& stride_() noexcept { return meta_.stride; }
    Dtype& dtype_field() noexcept { return meta_.dtype; }
    Device& device_field() noexcept { return meta_.device; }
};

}  // namespace lucid
