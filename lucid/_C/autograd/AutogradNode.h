// lucid/_C/autograd/AutogradNode.h
//
// Provides AutogradNode<Derived, N_IN>, the CRTP base class used by every
// concrete backward node in the ops/ layer.  It extends Node with fixed-size
// arrays for saved input/output Storage, weak pointers to the original
// TensorImpl objects needed for version checking, and the shape/dtype/device
// metadata required to reconstruct gradient tensors.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "Helpers.h"
#include "Node.h"

namespace lucid {

class TensorImpl;

// CRTP helper that supplies boilerplate for concrete autograd backward nodes.
//
// Template parameters:
//   Derived  — the concrete node class; must expose a static constexpr member
//              `schema_v1` with at least a `.name` string_view field used for
//              error messages and debugging.
//   N_IN     — the number of input tensors to the corresponding forward op.
//              Determines the size of saved_inputs_, input_shapes_, and
//              input_tensors_.
//
// Usage pattern:
//   1. The forward op builder constructs an instance, fills input_tensors_,
//      input_shapes_, saved_inputs_, saved_output_, dtype_, and device_, then
//      calls set_next_edges() and set_saved_versions() before attaching the
//      node to the output TensorImpl via set_grad_fn().
//   2. During backward the Engine calls validate_versions() then apply().
//      Derived implements apply() and uses saved_inputs_ / saved_output_ for
//      any forward activations it needs.
//   3. After apply() returns, the Engine calls release_saved() which resets
//      all Storage slots to empty CpuStorage{} so that referenced memory can
//      be freed promptly.
//
// Thread safety: none.  All access is assumed to occur on the backward thread.
template <class Derived, std::size_t N_IN>
class AutogradNode : public Node {
public:
    // Number of forward inputs — mirrors the template parameter for callers
    // that need it as a value.
    static constexpr std::size_t kNumInputs = N_IN;

    // Human-readable name of the operation, taken from Derived::schema_v1.
    std::string_view name() const noexcept { return Derived::schema_v1.name; }

    // Expose input weak_ptrs so Engine can handle retain_grad on non-leaf tensors.
    std::vector<std::weak_ptr<TensorImpl>> retainable_inputs() const override {
        std::vector<std::weak_ptr<TensorImpl>> result;
        result.reserve(N_IN);
        for (std::size_t i = 0; i < N_IN; ++i)
            result.push_back(input_tensors_[i]);
        return result;
    }

    // Check that every input tensor's live version counter still matches the
    // version saved at forward time.  Throws VersionMismatch if any input has
    // been mutated in-place since the forward pass.
    void validate_versions() override {
        for (std::size_t i = 0; i < N_IN; ++i) {
            ::lucid::check_version_match(input_tensors_[i],
                                         saved_versions_.size() > i ? saved_versions_[i] : 0,
                                         Derived::schema_v1.name, i);
        }
    }

    // Drop all references to saved data so that memory can be reclaimed
    // immediately after this node has been executed by the Engine.
    // Each saved input and the saved output are reset to a default-constructed
    // (zero-byte) CpuStorage.  The weak_ptr array input_tensors_ is cleared
    // so that no TensorImpl is kept alive by the backward graph.
    void release_saved() override {
        for (auto& s : saved_inputs_)
            s = Storage{CpuStorage{}};
        saved_output_ = Storage{CpuStorage{}};
        input_tensors_ = {};
        for (auto& p : saved_impl_inputs_)
            p.reset();
        saved_impl_output_.reset();
    }

    // Weak references to the original input TensorImpl objects, used only by
    // validate_versions().  Weak to avoid extending the lifetime of inputs
    // beyond what the user code already holds.
    std::array<std::weak_ptr<TensorImpl>, N_IN> input_tensors_;

    // Shapes of the N_IN forward inputs, saved so that gradient broadcasts
    // and reductions can reconstruct the correct output shape without needing
    // the live TensorImpl.
    std::array<Shape, N_IN> input_shapes_;

    // Shape of the forward op's output, used to allocate or validate the
    // incoming gradient in apply().
    Shape out_shape_;

    // Dtype and device of the forward computation.  Gradient tensors are
    // created with the same dtype/device as the forward tensors.
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    // Copies of the forward input Storage objects that apply() needs during
    // backward (e.g. the input activations for an element-wise multiply).
    // Populated by the forward op builder; freed by release_saved().
    std::array<Storage, N_IN> saved_inputs_;

    // Copy of the forward output Storage, saved when the backward formula
    // requires the output activation (e.g. sigmoid, softmax).
    // May be left as a default CpuStorage{} when not needed.
    Storage saved_output_;

    // Strong references to the original input TensorImpl objects.
    // Set alongside saved_inputs_ so that apply_for_graph() can use
    // the full TensorImpl (with grad_fn) when create_graph=true.
    std::array<std::shared_ptr<TensorImpl>, N_IN> saved_impl_inputs_;

    // Strong reference to the forward output TensorImpl, used by ops whose
    // graph-mode backward needs the forward output value (e.g. Sigmoid).
    std::shared_ptr<TensorImpl> saved_impl_output_;
};

}  // namespace lucid
