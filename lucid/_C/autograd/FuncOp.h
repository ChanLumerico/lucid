#pragma once

// =====================================================================
// Lucid C++ engine — FuncOp CRTP base.
// =====================================================================
//
// Common backward-node infrastructure shared by all op family bases. CRTP so
// derived classes can declare `static cpu_kernel(...)` / `grad_formula(...)`
// without virtual dispatch overhead.
//
// FuncOp itself does NOT implement `forward` — that lives on each family
// base (BinaryOp, UnaryOp, ReduceOp, ...) because the validation/dispatch
// boilerplate differs. FuncOp just provides:
//   - a place to stash saved input metadata (shapes, dtypes, device)
//   - the standard `apply(grad)` skeleton (still virtual, overridden by
//     family base)
//
// Layer: autograd/. Depends on core/.

#include <array>
#include <cstddef>
#include <memory>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/Storage.h"
#include "Helpers.h"  // check_version_match
#include "Node.h"

namespace lucid {

class TensorImpl;  // forward decl — full type only needed in .cpp callers

template <class Derived, std::size_t N_IN>
class FuncOp : public Node {
public:
    static constexpr std::size_t kNumInputs = N_IN;

    // Validates each saved input's version against the live tensor. Engine
    // calls this immediately before `apply`. Throws VersionMismatch if an
    // in-place op mutated an input between forward and backward. Items with
    // expired weak_ptr (intermediate tensors freed) silently no-op.
    void validate_versions() override {
        for (std::size_t i = 0; i < N_IN; ++i) {
            ::lucid::check_version_match(
                input_tensors_[i],
                saved_versions_.size() > i ? saved_versions_[i] : 0,
                Derived::schema_v1.name, i);
        }
    }

    // Each forward populates these so `validate_versions` can check the live
    // tensors at backward time.
    std::array<std::weak_ptr<TensorImpl>, N_IN> input_tensors_;

    // Always saved — every family base populates these in `forward`. Public
    // because shape-op factories are free functions that need to construct
    // backward nodes; CRTP bases of compute-op families set them from their
    // own static `forward` and could keep them protected, but the public
    // surface is the simpler invariant. (`apply` reads them; `forward`
    // writes them.)
    std::array<Shape, N_IN> input_shapes_;
    Shape out_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    // Optionally saved — only if Derived::kSavesInputs is true (default).
    std::array<Storage, N_IN> saved_inputs_;

    // Optionally saved output — used by ops whose backward references the
    // output value (e.g. exp's grad = output * grad_out, sqrt's grad =
    // 0.5 * grad_out / output). Populated when Derived::kSavesOutput is true.
    Storage saved_output_;
};

}  // namespace lucid
