#pragma once

// =====================================================================
// Lucid C++ engine — NaryKernel<Derived, N>: fixed N-input op base.
// =====================================================================
//
// For ops with a fixed number of inputs > 2 that don't fit BinaryKernel
// (e.g. Conv2d [x, weight, bias], Loss [input, target, weight], LayerNorm).
//
// Phase 3.4: adds `wire_autograd()` static helper so Derived::forward()
// only needs to write validation + compute. The autograd wiring block
// (~20 lines in every op) collapses to one call.
//
// Pattern in Derived::forward():
//   // ... validate, compute out ...
//   NaryKernel<Derived, N>::wire_autograd({x, w, b}, out);
//   return out;
//
// Layer: kernel/. Depends on autograd/, core/.

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/AutogradNode.h"
#include "../autograd/Node.h"
#include "../core/GradMode.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "BinaryKernel.h"  // detail::ensure_grad_fn

namespace lucid {
namespace kernel {

template <class Derived, std::size_t N>
class NaryKernel : public AutogradNode<Derived, N> {
public:
    // ----------------------------------------------------------------
    // Phase 3.4: autograd wiring helper
    // ----------------------------------------------------------------

    // ----------------------------------------------------------------
    // Primary overload: accepts a pre-created (and pre-populated) bwd node.
    //
    // Use this for ops with extra backward fields (stride_, reduction_, etc.):
    //
    //   auto bwd = std::make_shared<XxxBackward>();
    //   bwd->stride_ = stride;        // op-specific extras
    //   bwd->reduction_ = reduction;
    //   NaryKernel<XxxBackward,N>::wire_autograd(std::move(bwd), {x,w,b}, out);
    //   return out;
    // ----------------------------------------------------------------

    /// Wire the autograd graph using a pre-created backward node.
    /// Populates standard fields (input_shapes_, out_shape_, dtype_, device_,
    /// input_tensors_, saved_inputs_, edges, versions) and calls set_grad_fn.
    ///
    /// @returns true when wired; false when skipped (no requires_grad).
    static bool wire_autograd(std::shared_ptr<Derived> bwd,
                              const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        if (!GradMode::is_enabled())
            return false;

        bool any_grad = false;
        for (const auto& inp : inputs)
            any_grad |= (inp && inp->requires_grad());
        if (!any_grad)
            return false;

        // Primary dtype/device from first non-null input.
        for (const auto& inp : inputs) {
            if (inp) {
                bwd->dtype_ = inp->dtype();
                bwd->device_ = inp->device();
                break;
            }
        }
        bwd->out_shape_ = out->shape();

        std::vector<Edge> edges;
        std::vector<std::int64_t> versions;
        edges.reserve(N);
        versions.reserve(N);

        for (std::size_t i = 0; i < N; ++i) {
            const auto& inp = inputs[i];
            bwd->input_shapes_[i] = inp ? inp->shape() : Shape{};
            bwd->input_tensors_[i] = inp;
            if (save_ins && inp)
                bwd->saved_inputs_[i] = inp->storage();
            edges.emplace_back(detail::ensure_grad_fn(inp), /*input_nr=*/0);
            versions.push_back(inp ? inp->version() : 0);
        }

        bwd->set_next_edges(std::move(edges));
        bwd->set_saved_versions(std::move(versions));

        out->set_grad_fn(std::move(bwd));
        out->set_leaf(false);
        out->set_requires_grad(true);
        return true;
    }

    // ----------------------------------------------------------------
    // Convenience overload: no extra backward fields needed.
    //
    // Use this for simple ops (Linear, Matmul, …) where the backward
    // class has no custom fields beyond the AutogradNode base fields:
    //
    //   NaryKernel<XxxBackward, N>::wire_autograd({x, w, b}, out);
    //   return out;
    // ----------------------------------------------------------------

    /// Wire the autograd graph, creating the backward node internally.
    static bool wire_autograd(const std::array<TensorImplPtr, N>& inputs,
                              const TensorImplPtr& out,
                              bool save_ins = true) {
        return wire_autograd(std::make_shared<Derived>(), inputs, out, save_ins);
    }
};

}  // namespace kernel
}  // namespace lucid
