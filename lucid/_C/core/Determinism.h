#pragma once

namespace lucid {

// Production framework contract: when the user opts in via
// `lucid.set_deterministic(True)`, every op that has a non-deterministic
// fast path must either fall back to a deterministic alternative or throw
// `NotImplementedError("op X has no deterministic implementation")`.
//
// This mirrors PyTorch's `torch.use_deterministic_algorithms(True)`.
// The flag is process-global (not thread-local) — determinism is a property
// of the user's *intent*, and crossing thread boundaries with different
// determinism modes would be confusing.
class Determinism {
public:
    static bool is_enabled();
    static void set_enabled(bool value);
};

}  // namespace lucid
