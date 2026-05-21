// lucid/_C/backend/Dispatcher.h
//
// Singleton router that maps a :class:`Device` tag to the concrete
// :class:`IBackend` instance that services it.
//
// On Apple Silicon, ``Device::CPU`` routes to :class:`CpuBackend` (Apple
// Accelerate — vDSP, vForce, BLAS, LAPACK) and ``Device::GPU`` routes to
// :class:`GpuBackend` (MLX / Metal).  ``BackendInit.cpp`` registers both
// backends at module-load time via the per-backend ``register_self()``
// helpers; after that, every op in :mod:`lucid._C.ops` calls
// :func:`Dispatcher::for_device` to obtain the right backend reference
// based on the tensor's device tag.
//
// Notes
// -----
// The dispatcher is a Meyers singleton — constructed on first use and
// destroyed at program exit.  Backends are owned by ``std::unique_ptr``
// so their lifetimes equal the lifetime of the process.
//
// See Also
// --------
// :class:`IBackend` — abstract interface implemented per device.
// :class:`CpuBackend` — Accelerate-backed CPU concrete backend.
// :class:`GpuBackend` — MLX-backed GPU concrete backend.

#pragma once

#include <array>
#include <memory>

#include "../core/Device.h"
#include "IBackend.h"

namespace lucid {
namespace backend {

// Routes compute operations to the backend registered for a given Device.
//
// Holds one :class:`IBackend` slot per :class:`Device` variant (currently
// CPU and GPU).  :func:`register_backend` transfers ownership via
// ``std::unique_ptr``; :func:`for_device` returns a non-owning reference
// valid for the lifetime of the process.  Every higher-level op in the
// Lucid engine fetches its backend through this class — the dispatcher is
// the single seam between device-agnostic op code and device-specific
// compute libraries.
//
// Notes
// -----
// Thread safety: :func:`register_backend` is called exactly once per
// device at process startup, before any worker threads exist.  After
// that, :func:`for_device` is read-only and safe to call concurrently
// from any thread.
//
// The internal storage is a fixed-size ``std::array`` indexed by the
// :class:`Device` enum value; ``kNumDevices`` must match the number of
// :class:`Device` variants (currently ``CPU=0``, ``GPU=1``).
//
// See Also
// --------
// :class:`IBackend` — abstract per-device interface.
// :func:`CpuBackend::register_self` — installs the CPU backend.
// :func:`GpuBackend::register_self` — installs the GPU backend.
class Dispatcher {
public:
    // Returns the backend registered for device ``d``.
    //
    // The returned reference is non-owning and remains valid for the
    // entire process lifetime; callers must not delete it.
    //
    // Parameters
    // ----------
    // d : Device
    //     Device tag selecting which backend to return (CPU or GPU).
    //
    // Returns
    // -------
    // IBackend&
    //     Reference to the singleton backend instance for ``d``.
    //
    // Notes
    // -----
    // Behaviour is undefined if no backend has been registered for ``d``
    // — :func:`register_backend` must have been called for that device
    // before any tensor compute occurs.  ``noexcept``; safe to call
    // concurrently after initialisation.
    //
    // See Also
    // --------
    // :func:`register_backend` — installer counterpart.
    static IBackend& for_device(Device d) noexcept {
        auto& self = instance();
        const auto idx = static_cast<std::size_t>(d);
        return *self.backends_[idx];
    }

    // Registers (or replaces) the backend for device ``d``.
    //
    // Transfers ownership of ``be`` into the dispatcher.  Any backend
    // previously stored at that slot is destroyed.
    //
    // Parameters
    // ----------
    // d : Device
    //     Device tag identifying which slot to populate.
    // be : std::unique_ptr<IBackend>
    //     Owning pointer to the concrete backend instance to install.
    //
    // Notes
    // -----
    // Called exactly once per :class:`Device` during process
    // initialisation (typically from :func:`CpuBackend::register_self`
    // and :func:`GpuBackend::register_self`, invoked by
    // ``BackendInit.cpp``).  Not thread-safe — must complete before any
    // worker threads start issuing :func:`for_device` calls.
    static void register_backend(Device d, std::unique_ptr<IBackend> be) {
        auto& self = instance();
        self.backends_[static_cast<std::size_t>(d)] = std::move(be);
    }

private:
    // Returns the Meyers-singleton dispatcher instance.
    //
    // Constructed on first call and destroyed at program exit per the
    // standard C++11 magic-statics guarantee.
    //
    // Returns
    // -------
    // Dispatcher&
    //     The process-wide dispatcher.
    static Dispatcher& instance() {
        static Dispatcher inst;
        return inst;
    }

    // Number of :class:`Device` variants the dispatcher can route to.
    //
    // Must be kept in sync with the :class:`Device` enum — currently
    // ``CPU=0`` and ``GPU=1``, so ``kNumDevices == 2``.
    static constexpr std::size_t kNumDevices = 2;

    // Per-device backend ownership table.
    //
    // Indexed by the integer value of :class:`Device`.  ``unique_ptr``
    // ensures backends are destroyed at program shutdown.
    std::array<std::unique_ptr<IBackend>, kNumDevices> backends_;
};

}  // namespace backend
}  // namespace lucid
