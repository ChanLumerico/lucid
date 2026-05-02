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

class LUCID_API TensorImpl : public std::enable_shared_from_this<TensorImpl> {
public:
    TensorImpl(Storage storage, Shape shape, Dtype dtype, Device device, bool requires_grad);

    static std::shared_ptr<TensorImpl> from_numpy(py::array arr, Device device, bool requires_grad);

    const Storage& storage() const noexcept { return storage_; }
    Storage& mutable_storage() noexcept { return storage_; }

    std::size_t storage_offset() const noexcept { return offset_; }

    bool storage_is_shared() const noexcept;

    static std::shared_ptr<TensorImpl> make_view(const std::shared_ptr<TensorImpl>& base,
                                                 Shape shape,
                                                 Stride stride,
                                                 std::size_t offset_bytes = 0);

    const TensorMeta& meta() const noexcept { return meta_; }

    const Shape& shape() const noexcept { return meta_.shape; }
    const Stride& stride() const noexcept { return meta_.stride; }
    Dtype dtype() const noexcept { return meta_.dtype; }
    Device device() const noexcept { return meta_.device; }

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

    std::size_t numel() const;
    std::size_t nbytes() const;
    bool is_contiguous() const;

    py::object data_as_python() const;
    py::object grad_as_python() const;

    void copy_from(const TensorImpl& other);
    void zero_grad();

private:
    Storage storage_;
    std::size_t offset_ = 0;
    TensorMeta meta_;
    std::optional<AutogradMeta> autograd_;

    AutogradMeta* ensure_autograd() {
        if (!autograd_)
            autograd_.emplace();
        return &*autograd_;
    }

    Shape& shape_() noexcept { return meta_.shape; }
    Stride& stride_() noexcept { return meta_.stride; }
    Dtype& dtype_field() noexcept { return meta_.dtype; }
    Device& device_field() noexcept { return meta_.device; }
};

}  // namespace lucid
