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

void register_autograd(py::module_& m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def_property_readonly("sequence_nr", &Node::sequence_nr)
        .def("__repr__", [](const Node& n) {
            return "<lucid.Node seq=" + std::to_string(n.sequence_nr()) + ">";
        });

    py::class_<AccumulateGrad, Node, std::shared_ptr<AccumulateGrad>>(m, "AccumulateGrad");

    m.def(
        "engine_backward",
        [](std::shared_ptr<TensorImpl> root, bool retain_graph) {
            Engine::backward(root, Storage{CpuStorage{}}, retain_graph);
        },
        py::arg("root"), py::arg("retain_graph") = false,
        "Run backward starting at `root` with an implicit ones_like seed.");

    lucid::register_custom_function(m);

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
