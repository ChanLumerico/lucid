#pragma once

// =====================================================================
// Lucid C++ engine — backward Engine.
// =====================================================================
//
// Single entry point: `Engine::backward(root, grad_seed, retain_graph)`.
// Phase 2: DFS topological sort on the autograd graph rooted at
// `root.grad_fn_`, then reverse traversal calling each Node's apply().
// Phase 6 (JIT) will add a second entry path that replays a precompiled IR
// instead of walking the dynamic graph.
//
// Threading:  single-threaded per root (PyTorch-style). Concurrent calls on
//             disjoint graphs are safe.
// Layer:      autograd/.

#include <memory>

#include "../api.h"
#include "../core/TensorImpl.h"

namespace lucid {

/// Autograd backward engine — single-threaded DFS graph traversal.
class LUCID_API Engine {
public:
    // Run backward starting from `root`. If `grad_seed` is empty, an
    // ones_like(root) seed is generated automatically — matching the implicit
    // `loss.backward()` behavior. retain_graph mirrors PyTorch semantics.
    /// Run backward from root; accumulate gradients into leaf tensors.
    static void backward(const std::shared_ptr<TensorImpl>& root,
                         Storage grad_seed = Storage{CpuStorage{}},
                         bool retain_graph = false);
};

}  // namespace lucid
