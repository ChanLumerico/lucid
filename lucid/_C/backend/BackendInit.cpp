// lucid/_C/backend/BackendInit.cpp
//
// Entry point that instantiates and registers both the CPU and GPU backends
// with the Dispatcher singleton.  This translation unit is pulled in by the
// pybind11 module initialiser (typically in bindings/Module.cpp) so that
// both backends are ready before any Python-side tensor operation executes.
// Including CpuBackend.h and GpuBackend.h here is intentional: the headers
// contain the full implementation (they are header-only implementations) and
// each defines a static register_self() that injects the backend into the
// Dispatcher via a unique_ptr.  The phantom backend (PhantomBackend.h) is
// built here too, on first use, and handed out by Dispatcher::for_device
// while phantom mode is on.

#include "Dispatcher.h"
#include "PhantomBackend.h"
#include "cpu/CpuBackend.h"
#include "gpu/GpuBackend.h"

namespace lucid::backend {

IBackend& Dispatcher::phantom_backend(Device d) noexcept {
    static PhantomBackend cpu{Device::CPU};
    static PhantomBackend gpu{Device::GPU};
    return d == Device::GPU ? static_cast<IBackend&>(gpu) : cpu;
}

}  // namespace lucid::backend
