// lucid/_C/optim/Optimizer.cpp
//
// Implementation of the abstract Optimizer base class. The two
// non-trivial methods here — step() and zero_grad() — encapsulate all
// bookkeeping that is common to every optimizer variant so that derived
// classes contain only their specific update mathematics.

#include "Optimizer.h"

#include <cstdint>
#include <cstring>
#include <variant>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>  // mlx::core::eval(std::vector<array>)

#include "../autograd/Helpers.h"
#include "../backend/gpu/MlxBridge.h"
#include "../core/Allocator.h"
#include "../core/ErrorBuilder.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "_OptimDetail.h"

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
        // Carry the storage's own dtype and byte count, as the CPU branch
        // does — left at their defaults a non-float state reads back as F32.
        GpuStorage gs;
        gs.dtype = s.dtype;
        gs.nbytes = s.nbytes;
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

namespace {

// Store ``value`` as one element of type ``T`` at ``dst``.
template <typename T>
void write_scalar(std::byte* dst, double value) {
    const T v = static_cast<T>(value);
    std::memcpy(dst, &v, sizeof(T));
}

// Load one element of type ``T`` from ``src`` as a double.
template <typename T>
double read_scalar(const std::byte* src) {
    T v;
    std::memcpy(&v, src, sizeof(T));
    return static_cast<double>(v);
}

}  // namespace

std::shared_ptr<TensorImpl> make_state_scalar(double value, Dtype dtype) {
    CpuStorage cs;
    cs.dtype = dtype;
    cs.nbytes = dtype_size(dtype);
    cs.ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
    switch (dtype) {
    case Dtype::F32:
        write_scalar<float>(cs.ptr.get(), value);
        break;
    case Dtype::F64:
        write_scalar<double>(cs.ptr.get(), value);
        break;
    case Dtype::I64:
        write_scalar<std::int64_t>(cs.ptr.get(), value);
        break;
    default:
        ErrorBuilder("make_state_scalar").not_implemented("dtype must be F32, F64 or I64");
    }
    return std::make_shared<TensorImpl>(Storage{std::move(cs)}, Shape{}, dtype, Device::CPU, false);
}

double read_state_scalar(const TensorImpl& t) {
    if (t.numel() != 1)
        ErrorBuilder("load_state_buffers").fail("a scalar state entry must hold one element");
    const Storage& st = t.storage();
    // A loader may rebuild the scalar on the parameter's device; bring a GPU
    // value to the host first.  The download is one element.
    CpuStorage host;
    const CpuStorage* cpu = std::get_if<CpuStorage>(&st);
    if (const auto* gs = std::get_if<GpuStorage>(&st)) {
        host = gpu::download_gpu_to_cpu(*gs, t.shape());
        cpu = &host;
    }
    if (cpu == nullptr)
        ErrorBuilder("load_state_buffers").fail("unsupported storage for a scalar state entry");
    const std::byte* src = cpu->ptr.get();
    switch (t.dtype()) {
    case Dtype::F32:
        return read_scalar<float>(src);
    case Dtype::F64:
        return read_scalar<double>(src);
    case Dtype::I32:
        return read_scalar<std::int32_t>(src);
    case Dtype::I64:
        return read_scalar<std::int64_t>(src);
    default:
        ErrorBuilder("load_state_buffers")
            .not_implemented("a scalar state entry must be F32, F64, I32 or I64");
    }
}

void Optimizer::sync_slot_vectors() {
    if (state_initialized_.size() != params_.size())
        state_initialized_.assign(params_.size(), false);
    if (steps_.size() != params_.size())
        steps_.assign(params_.size(), 0);
}

bool Optimizer::slot_has_state(std::size_t i) const {
    return i < state_initialized_.size() && state_initialized_[i] && i < params_.size() &&
           params_[i] != nullptr;
}

void Optimizer::ensure_state_slot(std::size_t i) {
    sync_slot_vectors();
    if (i >= params_.size() || !params_[i] || state_initialized_[i])
        return;
    init_state_slot(i, params_[i]);
    state_initialized_[i] = true;
}

void Optimizer::ensure_buffers(std::vector<Storage>& bufs) {
    if (bufs.size() < params_.size())
        bufs.resize(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (!slot_has_state(i) || optim_detail::holds_buffer(bufs[i]))
            continue;
        const auto& p = params_[i];
        bufs[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
    }
}

std::vector<std::shared_ptr<TensorImpl>>
Optimizer::clone_state_slots(const std::vector<Storage>& bufs) const {
    std::vector<std::shared_ptr<TensorImpl>> out(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (!slot_has_state(i) || i >= bufs.size())
            continue;
        const auto& p = params_[i];
        out[i] = clone_state_storage(bufs[i], p->shape(), p->dtype(), p->device());
    }
    return out;
}

std::vector<std::shared_ptr<TensorImpl>>
Optimizer::clone_held_slots(const std::vector<Storage>& bufs) const {
    std::vector<std::shared_ptr<TensorImpl>> out(params_.size());
    bool any = false;
    for (std::size_t i = 0; i < params_.size() && i < bufs.size(); ++i) {
        if (!slot_has_state(i) || !optim_detail::holds_buffer(bufs[i]))
            continue;
        const auto& p = params_[i];
        out[i] = clone_state_storage(bufs[i], p->shape(), p->dtype(), p->device());
        any = true;
    }
    if (!any)
        out.clear();
    return out;
}

void Optimizer::load_state_slots(std::vector<Storage>& bufs,
                                 const std::vector<std::shared_ptr<TensorImpl>>& saved) {
    for (std::size_t i = 0; i < saved.size() && i < params_.size(); ++i) {
        if (!saved[i] || !params_[i])
            continue;
        const auto& p = params_[i];
        const auto& s = saved[i];
        if (s->shape() != p->shape())
            ErrorBuilder("load_state_buffers").shape_mismatch(p->shape(), s->shape());
        if (s->dtype() != p->dtype())
            ErrorBuilder("load_state_buffers").dtype_mismatch(p->dtype(), s->dtype());
        if (s->device() != p->device())
            ErrorBuilder("load_state_buffers").device_mismatch(p->device(), s->device());
        ensure_state_slot(i);
        // A slot can hold state without this buffer: SGD makes its momentum
        // buffer at the first momentum step, not when the slot starts.
        if (bufs.size() < params_.size())
            bufs.resize(params_.size());
        if (!optim_detail::holds_buffer(bufs[i]))
            bufs[i] = make_zero_storage(p->shape(), p->dtype(), p->device());
        overwrite_state_storage(bufs[i], s->storage());
    }
}

std::vector<std::shared_ptr<TensorImpl>> Optimizer::clone_step_slots() const {
    std::vector<std::shared_ptr<TensorImpl>> out(params_.size());
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (slot_has_state(i) && i < steps_.size())
            out[i] = make_state_scalar(static_cast<double>(steps_[i]), Dtype::I64);
    }
    return out;
}

void Optimizer::load_step_slots(const std::vector<std::shared_ptr<TensorImpl>>& saved) {
    for (std::size_t i = 0; i < saved.size() && i < params_.size(); ++i) {
        if (!saved[i] || !params_[i])
            continue;
        ensure_state_slot(i);
        steps_[i] = static_cast<std::int64_t>(read_state_scalar(*saved[i]));
    }
}

void Optimizer::set_step_count(std::int64_t count) {
    sync_slot_vectors();
    for (std::size_t i = 0; i < params_.size(); ++i) {
        if (slot_has_state(i))
            steps_[i] = count;
    }
}

// Drives one optimizer update across all registered parameters.
//
// The per-slot vectors are grown lazily to match params_ on the first
// step call (handles the case where params_ was extended after
// construction). A parameter is silently skipped if its pointer is null
// or if it has no gradient yet — this matches the reference framework
// convention where parameters without gradients are treated as
// non-trainable for the current step, and a skipped slot's step counter
// does not advance.
void Optimizer::step() {
    sync_slot_vectors();
    // GPU param arrays updated this step, flushed in one batched eval below.
    std::vector<::mlx::core::array> to_flush;
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
        ++steps_[i];
        update_one(i, p, *grad);

        // Bump the version so that any autograd nodes that captured this
        // parameter before the update detect an in-place modification.
        p->bump_version();

        // Collect the (lazy) updated GPU param array for the batched eval.  The
        // CPU path mutates in place (no MLX graph) and is skipped.
        const Storage& st = p->storage();
        if (std::holds_alternative<GpuStorage>(st)) {
            const auto& gs = std::get<GpuStorage>(st);
            if (gs.arr)
                to_flush.push_back(*gs.arr);
        }
    }

    // Sever the lazy MLX graph: ``update_one`` writes each param back as an
    // UNEVALUATED array (param = subtract(param, lr*m/…)).  Without an eval here
    // every step composes its update on top of the prior step's still-lazy
    // graph, so the chain pins all prior steps' compute — an unbounded
    // active-memory / RSS leak on the GPU path that only sparse-``.item()``
    // loops happen to mask.  One batched eval per step flushes it (the param
    // transitively depends on this step's optimizer state, so m/v/momentum are
    // materialised too).  Near-zero marginal cost — the next forward forces this
    // compute anyway; eval just submits it now — and, unlike ``.item()``, no
    // host copy.
    if (!to_flush.empty())
        ::mlx::core::eval(to_flush);
}

// Clear accumulated gradients on all non-null parameters.
void Optimizer::zero_grad() {
    for (auto& p : params_) {
        if (p)
            p->zero_grad();
    }
}

}  // namespace lucid
