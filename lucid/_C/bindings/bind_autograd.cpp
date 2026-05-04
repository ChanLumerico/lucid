// lucid/_C/bindings/bind_autograd.cpp
//
// Exposes the autograd graph and backward engine to Python.  Registers:
//   - Node / AccumulateGrad — the two graph node types visible from Python
//   - engine_backward() — triggers reverse-mode AD from a scalar root
//   - register_custom_function() — wires up lucid.autograd.Function so Python
//     subclasses can define custom forward/backward passes (delegates to
//     CustomFunction.h for the heavy lifting)
//   - _run_fusion_pass() — testing helper that runs the op-fusion pass on the
//     backward graph and returns the number of fusions detected

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/CustomFunction.h"
#include "../autograd/Engine.h"
#include "../autograd/FusionPass.h"
#include "../autograd/Node.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

// Registers the backward engine, autograd node types, and the custom-function
// registration helper.
void register_autograd(py::module_& m) {
    // Node is the abstract base for all backward-graph vertices.  It is held
    // as shared_ptr<Node> because backward-graph edges are also shared_ptrs;
    // Python may hold a reference longer than the graph itself.
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def_property_readonly("sequence_nr", &Node::sequence_nr)
        .def("__repr__", [](const Node& n) {
            return "<lucid.Node seq=" + std::to_string(n.sequence_nr()) + ">";
        });

    // AccumulateGrad is the leaf-variable sink node; Python code can detect it
    // via isinstance(tensor.grad_fn, engine.AccumulateGrad).
    py::class_<AccumulateGrad, Node, std::shared_ptr<AccumulateGrad>>(m, "AccumulateGrad");

    // engine_backward seeds the backward pass with a ones_like gradient so
    // scalar loss tensors do not need an explicit gradient argument.  The
    // CpuStorage{} seed is an empty storage; Engine::backward constructs the
    // actual ones tensor internally.
    m.def(
        "engine_backward",
        [](std::shared_ptr<TensorImpl> root, bool retain_graph, bool create_graph) {
            Engine::backward(root, Storage{CpuStorage{}}, retain_graph, create_graph);
        },
        py::arg("root"), py::arg("retain_graph") = false, py::arg("create_graph") = false,
        "Run backward starting at `root` with an implicit ones_like seed. "
        "If create_graph is True, the backward pass itself is tracked in the "
        "autograd graph, enabling second-order derivatives (MAML, Hessians, etc.).");

    // register_custom_function installs the Python-side CustomFunction class
    // and the _register_python_backward_node() hook used by lucid.autograd.Function.
    lucid::register_custom_function(m);

    // _run_fusion_pass is exposed for unit tests and profiling; in production
    // it is called automatically inside Engine::backward before the BFS
    // traversal.  Returns 0 if root has no grad_fn (leaf or detached tensor).
    m.def(
        "_run_fusion_pass",
        [](std::shared_ptr<TensorImpl> root) -> int {
            if (!root || !root->grad_fn())
                return 0;
            return lucid::run_fusion_pass(root->grad_fn().get());
        },
        py::arg("root"),
        "Run the op-fusion pass on the backward graph rooted at `root`. "
        "Returns the number of fusion patterns detected. "
        "Called automatically by Engine::backward(); exposed here for testing.");
}

}  // namespace lucid::bindings
