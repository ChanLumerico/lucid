// lucid/_C/autograd/Engine.h
//
// Declares Engine, the entry point for reverse-mode automatic differentiation.
// Callers invoke Engine::backward() with the scalar root tensor to start a
// backward pass; the engine handles graph traversal, gradient propagation, and
// cleanup internally.

#pragma once

#include <memory>

#include "../api.h"
#include "../core/TensorImpl.h"

namespace lucid {

// Executes the backward pass for a computation graph rooted at a TensorImpl.
//
// Engine is a stateless class that exposes a single static entry point.
// It is not instantiated; all work is performed inside backward().
class LUCID_API Engine {
public:
    // Run reverse-mode automatic differentiation from root.
    //
    // root       — the output tensor to differentiate.  Must have a grad_fn
    //              unless it is itself a leaf that requires a gradient.
    // grad_seed  — the initial gradient to feed into root's grad_fn.  If
    //              empty (default), a ones-tensor of root's shape/dtype/device
    //              is used, which corresponds to the common case of
    //              differentiating a scalar loss.
    // retain_graph — when false (default), each node's release_saved() is
    //              called after it executes and grad_fn is cleared from root,
    //              making a second backward() call impossible.  Pass true to
    //              preserve the graph for multiple backward calls.
    //
    // Throws if root is null, if input grads / edges counts mismatch on any
    // node (in non-trivial cases), or if validate_versions() detects an
    // in-place mutation.
    static void backward(const std::shared_ptr<TensorImpl>& root,
                         Storage grad_seed = Storage{CpuStorage{}},
                         bool retain_graph = false);
};

}  // namespace lucid
