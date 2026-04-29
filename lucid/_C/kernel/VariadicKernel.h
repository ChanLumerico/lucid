#pragma once

// =====================================================================
// Lucid C++ engine — VariadicKernel<Derived>: variable-input op base.
// =====================================================================
//
// For ops whose input count is not known at compile time: concat, stack,
// einsum, scatter, etc.
//
// Unlike AutogradNode<D, N> (fixed N), VariadicKernel stores saved metadata
// in std::vector instead of std::array.
//
// Derived implements:
//   1. `static const OpSchema schema_v1;`
//   2. `static TensorImplPtr forward(const std::vector<TensorImplPtr>& inputs, ...)`
//   3. `std::vector<Storage> apply(Storage grad_out) override`
//
// VariadicKernel provides:
//   - validate_versions() over all saved inputs
//   - std::vector<Shape> input_shapes_v_   (variable-length shapes)
//   - std::vector<Storage> saved_inputs_v_ (variable-length saved inputs)
//   - std::vector<weak_ptr<TensorImpl>> input_tensors_v_
//
// Usage:
//   class ConcatBackward : public VariadicKernel<ConcatBackward> { ... };
//
// Layer: kernel/. Depends on autograd/, core/.

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include "../autograd/Helpers.h"  // check_version_match
#include "../autograd/Node.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"

namespace lucid {

class TensorImpl;

namespace kernel {

template <class Derived>
class VariadicKernel : public Node {
public:
    // ----------------------------------------------------------------
    // IKernel interface
    // ----------------------------------------------------------------
    std::string_view name() const noexcept { return Derived::schema_v1.name; }

    // ----------------------------------------------------------------
    // Version checking — Engine calls this before apply().
    // ----------------------------------------------------------------
    void validate_versions() override {
        for (std::size_t i = 0; i < input_tensors_v_.size(); ++i) {
            ::lucid::check_version_match(input_tensors_v_[i],
                                         saved_versions_.size() > i ? saved_versions_[i] : 0,
                                         Derived::schema_v1.name, i);
        }
    }

    // ----------------------------------------------------------------
    // Saved state — populated by forward(), consumed by apply().
    // ----------------------------------------------------------------

    std::vector<std::weak_ptr<TensorImpl>> input_tensors_v_;
    std::vector<Shape> input_shapes_v_;
    Shape out_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;
    std::vector<Storage> saved_inputs_v_;
    Storage saved_output_;
};

}  // namespace kernel
}  // namespace lucid
