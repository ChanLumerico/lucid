// lucid/_C/backend/BackendInit.cpp
//
// Entry point that instantiates and registers both the CPU and GPU backends
// with the Dispatcher singleton.  This translation unit is pulled in by the
// pybind11 module initialiser (typically in bindings/Module.cpp) so that
// both backends are ready before any Python-side tensor operation executes.
// Including CpuBackend.h and GpuBackend.h here is intentional: the headers
// contain the full implementation (they are header-only implementations) and
// each defines a static register_self() that injects the backend into the
// Dispatcher via a unique_ptr.

#include "cpu/CpuBackend.h"
#include "gpu/GpuBackend.h"
