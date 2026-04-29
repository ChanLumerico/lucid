#pragma once

// =====================================================================
// Lucid C++ engine — IKernel: root kernel interface.
// =====================================================================
//
// Abstract base for every op kernel in the engine. Concrete family bases
// (BinaryKernel, UnaryKernel, ReduceKernel, NaryKernel, VariadicKernel)
// inherit from here; individual op classes inherit from those.
//
// Separating IKernel from the autograd Node lets the engine schedule,
// profile, and serialize ops without knowing the full autograd graph.
//
// Layer: kernel/. Depends on core/ only.

#include <string_view>
#include <vector>

#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {
namespace kernel {

/// Minimal compile-time policy bundle for a kernel.
/// Ops can specialize individual fields; defaults are conservative/safe.
struct KernelPolicy {
    bool saves_inputs = true;   ///< backward will reference forward inputs
    bool saves_output = false;  ///< backward will reference forward output
    bool has_gradient = true;   ///< op is differentiable
    bool deterministic = true;  ///< same inputs → same output (no RNG)
};

/// Abstract root interface. Every kernel is ultimately an IKernel.
/// Forward/backward signatures live on the family-specific base classes
/// (BinaryKernel, UnaryKernel, etc.) because they depend on input arity.
class IKernel {
public:
    virtual ~IKernel() = default;

    /// Op name, for error messages and profiling. Backed by schema_v1.name.
    virtual std::string_view name() const noexcept = 0;

    /// Backward pass: given incoming gradient, produce input gradients.
    /// Concrete implementation in each backward class's apply().
    virtual std::vector<Storage> apply(Storage grad_out) = 0;
};

}  // namespace kernel
}  // namespace lucid
