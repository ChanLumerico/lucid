// lucid/_C/backend/BackendInit.cpp
//
// Entry point that instantiates and registers both the CPU and GPU backends
// with the Dispatcher singleton.  This translation unit is pulled in by the
// pybind11 module initialiser (in bindings/Module.cpp) so that both backends
// are ready before any Python-side tensor operation executes.  The phantom
// backend (PhantomBackend.h) is built here too, on first use, and handed out
// by Dispatcher::for_device while phantom mode is on.

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

namespace {
struct CpuBackendRegistrar {
    CpuBackendRegistrar() {
        lucid::backend::Dispatcher::register_backend(
            lucid::Device::CPU, std::make_unique<lucid::backend::CpuBackend>());
    }
} g_cpu_registrar;

struct GpuBackendRegistrar {
    GpuBackendRegistrar() {
        lucid::backend::Dispatcher::register_backend(
            lucid::Device::GPU, std::make_unique<lucid::backend::GpuBackend>());
    }
} g_gpu_registrar;
}  // namespace
