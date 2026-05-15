// lucid/_C/optim/Optimizer.cpp
//
// Implementation of the abstract Optimizer base class. The two
// non-trivial methods here — step() and zero_grad() — encapsulate all
// bookkeeping that is common to every optimizer variant so that derived
// classes contain only their specific update mathematics.

#include "Optimizer.h"

#include <cstring>
#include <variant>

#include <mlx/array.h>
#include <mlx/ops.h>

#include "../core/Allocator.h"
#include "../core/ErrorBuilder.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace lucid {

// Wrap an optimizer state Storage as a TensorImpl that deep-copies the
// underlying buffer.  Used by state_buffers() so that a snapshot is
// independent of subsequent in-place updates.
std::shared_ptr<TensorImpl>
clone_state_storage(const Storage& src, const Shape& shape, Dtype dtype, Device device) {
    Storage dst;
    if (std::holds_alternative<CpuStorage>(src)) {
        const auto& s = std::get<CpuStorage>(src);
        CpuStorage cs;
        cs.dtype = s.dtype;
        cs.nbytes = s.nbytes;
        cs.ptr = allocate_aligned_bytes(s.nbytes, Device::CPU);
        if (s.nbytes > 0)
            std::memcpy(cs.ptr.get(), s.ptr.get(), s.nbytes);
        dst = std::move(cs);
    } else if (std::holds_alternative<GpuStorage>(src)) {
        const auto& s = std::get<GpuStorage>(src);
        // Force materialisation, then make an independent copy via MLX.
        s.arr->eval();
        auto copy = ::mlx::core::array(*s.arr);
        copy.eval();
        GpuStorage gs;
        gs.arr = std::make_shared<::mlx::core::array>(std::move(copy));
        dst = std::move(gs);
    } else {
        ErrorBuilder("clone_state_storage").fail("unsupported storage variant");
    }
    return std::make_shared<TensorImpl>(std::move(dst), shape, dtype, device, false);
}

// Copy ``src`` into ``dst`` in place — used by load_state_buffers().  Both
// must already share shape and dtype; only the buffer bytes are overwritten.
void overwrite_state_storage(Storage& dst, const Storage& src) {
    if (std::holds_alternative<CpuStorage>(dst) && std::holds_alternative<CpuStorage>(src)) {
        auto& d = std::get<CpuStorage>(dst);
        const auto& s = std::get<CpuStorage>(src);
        if (d.nbytes != s.nbytes)
            ErrorBuilder("load_state_buffers").fail("byte size mismatch");
        if (s.nbytes > 0)
            std::memcpy(d.ptr.get(), s.ptr.get(), s.nbytes);
        return;
    }
    if (std::holds_alternative<GpuStorage>(dst) && std::holds_alternative<GpuStorage>(src)) {
        auto& d = std::get<GpuStorage>(dst);
        const auto& s = std::get<GpuStorage>(src);
        d.arr = std::make_shared<::mlx::core::array>(*s.arr);
        d.arr->eval();
        return;
    }
    ErrorBuilder("load_state_buffers").fail("device mismatch between live and saved state");
}

// Drives one optimizer update across all registered parameters.
//
// The state_initialized_ vector is grown lazily to match params_ on
// the first step call (handles the case where params_ was extended
// after construction). A parameter is silently skipped if its pointer
// is null or if it has no gradient yet — this matches the reference framework
// convention where parameters without gradients are treated as
// non-trainable for the current step.
void Optimizer::step() {
    if (state_initialized_.size() != params_.size()) {
        state_initialized_.assign(params_.size(), false);
    }
    for (std::size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        if (!p)
            continue;
        const auto& grad = p->grad_storage();
        if (!grad.has_value())
            continue;
        if (!state_initialized_[i]) {
            init_state_slot(i, p);
            state_initialized_[i] = true;
        }
        update_one(i, p, *grad);

        // Bump the version so that any autograd nodes that captured this
        // parameter before the update detect an in-place modification.
        p->bump_version();
    }
}

// Clear accumulated gradients on all non-null parameters.
void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p)
            p->zero_grad();
    }
}

}  // namespace lucid
