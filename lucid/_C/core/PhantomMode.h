// lucid/_C/core/PhantomMode.h
//
// Thread-local phantom mode: while it is on, every factory and op on the
// calling thread returns a phantom tensor — a real :class:`TensorImpl` whose
// storage is a :class:`PhantomStorage` — instead of computing anything.
//
// The one source of truth for "build this without allocating": the dispatcher
// routes every backend call to the phantom backend (:file:`PhantomBackend.h`)
// while the mode is on, the kernel bases skip their kernels, and the factories
// that fill a host buffer before they dispatch skip the fill.  Each output
// keeps the shape and dtype its op computes before it dispatches, so no shape
// rule is written twice.
//
// The flag is per thread, as :class:`GradMode`'s is — and so is its contract:
// a thread starts with the mode off, and an asyncio task shares its thread's.
//
// Header-only: the flag is an ``inline thread_local``, so the engine holds one
// per thread whichever translation unit reads it.

#pragma once

#include <atomic>
#include <string>
#include <string_view>

#include "Error.h"
#include "Shape.h"
#include "Storage.h"

namespace lucid {

namespace detail {
inline thread_local bool g_phantom_mode = false;
// How many threads have the mode on.  Every dispatch asks, and on Darwin a
// thread-local read is a call through its TLV descriptor; while no thread
// is in the mode — always, outside model construction — the answer is this
// one plain load.  A thread's own increment is visible to its own later
// reads, so the count never hides that thread's flag from itself.  A thread
// that ends with the mode on leaves the count raised: the others then read
// their own flags too, which costs the TLV call and never a wrong answer.
inline std::atomic<int> g_phantom_threads{0};
}  // namespace detail

// Thread-local switch for phantom tensors, modelled on :class:`GradMode`.
class PhantomMode {
public:
    // Whether factories and ops on the calling thread make phantom tensors.
    static bool is_enabled() noexcept {
        return detail::g_phantom_threads.load(std::memory_order_relaxed) != 0 &&
               detail::g_phantom_mode;
    }

    // Sets the flag for the calling thread only.  Prefer
    // :class:`PhantomModeGuard`, which restores the flag on unwind.
    static void set_enabled(bool value) noexcept {
        bool& mode = detail::g_phantom_mode;
        if (mode == value)
            return;
        mode = value;
        detail::g_phantom_threads.fetch_add(value ? 1 : -1, std::memory_order_relaxed);
    }
};

// RAII guard that sets phantom mode to ``enabled`` for its lifetime on the
// calling thread, and restores the flag it found.
class PhantomModeGuard {
public:
    explicit PhantomModeGuard(bool enabled) noexcept : prev_(PhantomMode::is_enabled()) {
        PhantomMode::set_enabled(enabled);
    }
    ~PhantomModeGuard() { PhantomMode::set_enabled(prev_); }

    PhantomModeGuard(const PhantomModeGuard&) = delete;
    PhantomModeGuard& operator=(const PhantomModeGuard&) = delete;

private:
    bool prev_;
};

// A phantom storage for ``shape`` elements of ``dt``, sized.
inline Storage phantom_storage(const Shape& shape, Dtype dt) {
    PhantomStorage p;
    p.nbytes = shape_numel(shape) * dtype_size(dt);
    p.dtype = dt;
    return Storage{std::move(p)};
}

// Whether an op given ``inputs`` makes a phantom result: the mode is on, or an
// input is a phantom (a phantom tensor used after its block is still one).
template <class... Impl>
bool phantom_dispatch(const Impl&... inputs) noexcept {
    return PhantomMode::is_enabled() || (inputs.is_phantom() || ...);
}

// The one refusal for a phantom tensor's values — a host read (``item``,
// ``tolist``, NumPy, bytes, printing), a data-dependent shape, and a write of
// a phantom into a real tensor alike: there are no values to give.
[[noreturn]] inline void refuse_phantom_read(std::string_view op) {
    throw LucidError("phantom tensor has no values: " + std::string(op));
}

}  // namespace lucid
