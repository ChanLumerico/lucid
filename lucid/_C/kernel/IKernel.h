// lucid/_C/kernel/IKernel.h
//
// Abstract interface that all op kernels expose to the framework. Every
// concrete kernel (unary, binary, reduce, variadic) inherits IKernel
// alongside its CRTP base so that kernel objects can be stored uniformly
// by ops/ dispatch tables and testing harnesses without knowing the
// concrete type.
//
// The two virtual methods form a split API: compute() is the forward
// entry point for generic dispatch code, and apply() is the backward
// entry point called by the autograd engine during reverse-mode
// differentiation. Concrete kernels typically override apply() via their
// CRTP base and delegate compute() to the static ::forward().

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {
namespace kernel {

// Compile-time policy flags for a kernel. Derived classes may override
// the individual constants (kSavesInput, kHasGradient, etc.) rather than
// instantiating this struct; KernelPolicy exists as a documentation aid.
struct KernelPolicy {
    bool saves_inputs = true;   // Whether forward() saves inputs for use in apply().
    bool saves_output = false;  // Whether forward() saves the output for use in apply().
    bool has_gradient = true;   // Whether this op participates in autograd.
    bool deterministic = true;  // Whether the op produces the same output for equal inputs.
};

// Pure-virtual base for all Lucid op kernels.
//
// Each concrete kernel class (e.g. AddOp, ReluOp) inherits this via one
// of the CRTP helpers (UnaryKernel, BinaryKernel, etc.) so it can be
// stored as an IKernel* and invoked polymorphically. The CRTP base
// provides correct implementations of both methods.
//
// Thread safety: IKernel instances are not thread-safe. The framework
// creates a separate instance per invocation and never shares them
// across threads.
class IKernel {
public:
    virtual ~IKernel() = default;

    // Return a short, stable name for this op (e.g., "add", "relu").
    // Used in error messages, profiling scopes, and schema lookups.
    virtual std::string_view name() const noexcept = 0;

    // Execute the forward pass for this kernel given a flat vector of
    // input Storages and return the single output Storage.
    // The default implementation throws because most kernels are invoked
    // via their type-safe static forward() rather than this polymorphic
    // entry point; override only when generic dispatch is needed.
    virtual Storage compute(const std::vector<Storage>&) {
        throw std::logic_error(std::string(name()) +
                               ": compute() not implemented — use the concrete static forward()");
    }

    // Execute the backward pass. grad_out is the gradient of the loss
    // with respect to this op's output; returns one gradient Storage
    // per input, in the same order as the forward inputs.
    // Called by the autograd engine during graph traversal.
    virtual std::vector<Storage> apply(Storage grad_out) = 0;
};

}  // namespace kernel
}  // namespace lucid
