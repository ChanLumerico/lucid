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
