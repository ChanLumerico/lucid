// lucid/_C/core/GradMode.h
//
// Thread-local gradient-computation mode control.
//
// GradMode is checked by the op dispatch layer before any autograd-related
// bookkeeping (attaching grad_fn, saving inputs for backward).  When
// is_enabled() returns false, ops execute in pure-inference mode: no autograd
// graph is built, no inputs are saved, and no gradient metadata is propagated
// to outputs.
//
// NoGradGuard is the RAII equivalent of Python's a no-grad context context
// manager.  It saves the current mode on entry and restores it unconditionally
// on exit, even if the guarded scope throws.
//
// Thread safety: the gradient mode is per-thread (thread_local storage).
// Changing the mode in one thread does not affect other threads.

#pragma once

namespace lucid {

// Global (thread-local) gradient-tracking toggle.
class GradMode {
public:
    // Returns true if gradient computation is active on the current thread.
    // Defaults to true at thread startup.
    static bool is_enabled();

    // Enables or disables gradient computation on the current thread.
    // Prefer NoGradGuard over direct calls to set_enabled() in scoped contexts.
    static void set_enabled(bool value);
};

// RAII guard that disables gradient tracking for the lifetime of the object.
//
// Constructs with GradMode::is_enabled() saved as prev_; sets mode to false.
// Destructor unconditionally restores prev_, correctly nesting with any outer
// guards.
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_;
};

}  // namespace lucid
