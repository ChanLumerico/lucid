// =====================================================================
// Lucid C++ engine — BackendInit.cpp
// =====================================================================
//
// Forces CpuBackend and GpuBackend to register with the Dispatcher.
// Including the backend headers here ensures their static-init registrars
// are instantiated in exactly one translation unit.
//
// This file must be compiled into the engine .so (it is listed in
// CMakeLists.txt / setup.py source list).

#include "cpu/CpuBackend.h"
#include "gpu/GpuBackend.h"

// The anonymous-namespace registrar structs in each header call
// Dispatcher::register_backend() at static init time:
//   CpuBackendRegistrar → registers CpuBackend for Device::CPU
//   GpuBackendRegistrar → registers GpuBackend for Device::GPU
//
// No code needed here — inclusion is sufficient.
