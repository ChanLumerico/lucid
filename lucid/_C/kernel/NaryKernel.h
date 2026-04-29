#pragma once

// =====================================================================
// Lucid C++ engine — NaryKernel<Derived, N>: fixed N-input op base.
// =====================================================================
//
// For ops with a fixed number of inputs > 2 that don't fit BinaryKernel
// (e.g. Conv2d [x, weight, bias], Loss [input, target, weight], LayerNorm).
//
// Unlike BinaryKernel which enforces dtype/device match between ALL inputs,
// NaryKernel leaves input validation entirely to Derived::forward() —
// op-specific validation rules differ too much to centralize.
//
// Derived implements:
//   1. `static const OpSchema schema_v1;`
//   2. `static TensorImplPtr forward(inputs...)` — full op-specific forward
//   3. `std::vector<Storage> apply(Storage grad_out) override` — backward
//
// NaryKernel only contributes:
//   - AutogradNode<Derived, N> inheritance (saved metadata + version checks)
//   - validate_versions() called by the Engine before apply()
//
// Usage:
//   class ConvNdBackward : public NaryKernel<ConvNdBackward, 3> { ... };
//
// Layer: kernel/. Depends on kernel/AutogradNode.h, autograd/, core/.

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "../autograd/Node.h"
#include "../core/Storage.h"
#include "AutogradNode.h"

namespace lucid {
namespace kernel {

template <class Derived, std::size_t N>
class NaryKernel : public AutogradNode<Derived, N> {
public:
    // forward() is NOT defined here — each Derived writes its own because
    // parameter types differ (e.g. optional bias, stride, padding, ...).
    // Derived should call:
    //   this->input_shapes_  = ...;
    //   this->out_shape_     = ...;
    //   this->dtype_         = ...;
    //   this->device_        = ...;
    //   this->input_tensors_ = ...;
    //   this->saved_inputs_  = ...;  // if needed
    //   this->set_next_edges(...);
    //   this->set_saved_versions(...);

    // apply() is still purely virtual here; Derived overrides it.
    // (Inherited from Node via AutogradNode.)
};

}  // namespace kernel
}  // namespace lucid
