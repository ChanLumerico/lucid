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

    std::size_t numel() const;
    std::size_t nbytes() const;
    bool is_contiguous() const;

    py::object data_as_python() const;
    py::object grad_as_python() const;

    void copy_from(const TensorImpl& other);

    void zero_grad();
};

}  // namespace lucid
