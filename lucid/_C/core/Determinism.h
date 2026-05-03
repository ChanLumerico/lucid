// lucid/_C/core/Determinism.h
//
// Global determinism flag.  When enabled, any op whose OpSchema marks
// deterministic == false will throw a LucidError via SchemaGuard rather than
// silently producing non-reproducible results.
//
// Unlike GradMode and AmpPolicy — which are thread-local so that different
// Python threads can have independent settings — the determinism flag is
// process-global (backed by a std::atomic<bool>) because reproducibility is
// typically a process-wide concern (e.g. seeding for model training).
//
// Thread safety: is_enabled() and set_enabled() are individually atomic and
// safe to call from multiple threads concurrently.

#pragma once

namespace lucid {

// Process-wide determinism control.
class Determinism {
public:
    // Returns true if deterministic-mode is currently active.
    static bool is_enabled();

    // Enables or disables deterministic execution globally.  When true,
    // SchemaGuard will reject non-deterministic ops.
    static void set_enabled(bool value);
};

}  // namespace lucid
