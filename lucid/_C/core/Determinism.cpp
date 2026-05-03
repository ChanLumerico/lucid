// lucid/_C/core/Determinism.cpp
//
// Global atomic determinism flag.  std::memory_order_relaxed is acceptable
// here because there is no ordering dependency between the flag write and any
// tensor computation — the flag is merely a gate that is read independently
// before each op dispatch.  A full fence is not required.

#include "Determinism.h"

#include <atomic>

namespace lucid {

namespace {
std::atomic<bool> g_deterministic{false};
}

bool Determinism::is_enabled() {
    return g_deterministic.load(std::memory_order_relaxed);
}

void Determinism::set_enabled(bool value) {
    g_deterministic.store(value, std::memory_order_relaxed);
}

}  // namespace lucid
