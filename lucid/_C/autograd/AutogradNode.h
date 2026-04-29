#pragma once

// =====================================================================
// Lucid C++ engine — AutogradNode: CRTP backward-node base.
// =====================================================================
//
// Phase 3: concrete implementation of the backward-node CRTP base,
// formerly known as FuncOp. Lives in autograd/ (rank 3) so it can
// extend Node (also rank 3) without layer violations.
//
// `autograd/FuncOp.h` re-exports this as the backward-compat alias:
//   template<class D, N> using FuncOp = AutogradNode<D, N>;
//
// `kernel/AutogradNode.h` also re-exports this so kernel/ headers
// can reference it without an upward include.
//
// Layer: autograd/. Depends on autograd/Node.h, core/.

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "Helpers.h"  // check_version_match
#include "Node.h"

namespace lucid {

class TensorImpl;  // forward decl

/// CRTP backward-node base. All concrete backward classes inherit from this.
/// N_IN is the number of differentiable inputs saved for the backward pass.
template <class Derived, std::size_t N_IN>
class AutogradNode : public Node {
public:
    static constexpr std::size_t kNumInputs = N_IN;

    // ----------------------------------------------------------------
    // Name (satisfies informal IKernel interface)
    // ----------------------------------------------------------------
    std::string_view name() const noexcept { return Derived::schema_v1.name; }

    // ----------------------------------------------------------------
    // Version checking — Engine calls this before apply().
    // ----------------------------------------------------------------
    void validate_versions() override {
        for (std::size_t i = 0; i < N_IN; ++i) {
            ::lucid::check_version_match(input_tensors_[i],
                                         saved_versions_.size() > i ? saved_versions_[i] : 0,
                                         Derived::schema_v1.name, i);
        }
    }

    // ----------------------------------------------------------------
    // Saved state — populated by forward(), consumed by apply().
    // ----------------------------------------------------------------

    /// Weak refs to the live input tensors for validate_versions().
    std::array<std::weak_ptr<TensorImpl>, N_IN> input_tensors_;

    /// Logical input shapes (before broadcast/permute).
    std::array<Shape, N_IN> input_shapes_;

    /// Output shape of the forward pass.
    Shape out_shape_;

    /// Dtype and device of the primary input.
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    /// Saved input Storage (populated when Derived::kSavesInputs == true).
    std::array<Storage, N_IN> saved_inputs_;

    /// Saved output Storage (populated when Derived::kSavesOutput == true).
    Storage saved_output_;
};

}  // namespace lucid
