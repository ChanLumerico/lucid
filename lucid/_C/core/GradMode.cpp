#include "GradMode.h"

namespace lucid {

namespace {
thread_local bool g_grad_enabled = true;
}

bool GradMode::is_enabled() {
    return g_grad_enabled;
}

void GradMode::set_enabled(bool value) {
    g_grad_enabled = value;
}

NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) {
    g_grad_enabled = false;
}

NoGradGuard::~NoGradGuard() {
    g_grad_enabled = prev_;
}

}  // namespace lucid
