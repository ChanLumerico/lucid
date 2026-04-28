// =====================================================================
// Lucid C++ engine — pybind11 module entrypoint.
// =====================================================================
//
// Each `register_*` function lives in its own `bind_*.cpp` and exposes one
// subsystem. Order matters: `register_errors` runs first so the exception
// translator is registered before any binding can throw during init.
//
// Op-level binding surface mirrors `lucid/_func/{g,b,u}func.py`,
// `lucid/_utils/`, and `lucid/linalg/`:
//
//   register_gfunc  ↔ lucid/_func/gfunc.py    (zeros/ones/eye/...)
//   register_bfunc  ↔ lucid/_func/bfunc.py    (add/sub/.../matmul/dot/...)
//   register_ufunc  ↔ lucid/_func/ufunc.py    (exp/sin/sum/var/transpose/...)
//   register_utils  ↔ lucid/_utils/           (reshape/stack/where/...)
//   register_linalg ↔ lucid/linalg/           (inv/qr/svd/...)
//   register_nn     ↔ lucid/nn/               (conv, norm, pool, ...)
//   register_random ↔ lucid/random/           (rand/randn/...)
//   register_optim  ↔ lucid/optim/            (SGD/Adam/Schedulers/...)

#include <pybind11/pybind11.h>

#include "../version.h"

namespace py = pybind11;

namespace lucid::bindings {
void register_errors(py::module_& m);
void register_core(py::module_& m);
void register_tensor_impl(py::module_& m);
void register_autograd(py::module_& m);
void register_amp(py::module_& m);
void register_profiler(py::module_& m);
void register_op_registry(py::module_& m);
void register_optim(py::module_& m);
void register_nn(py::module_& m);
void register_random(py::module_& m);
void register_gfunc(py::module_& m);
void register_bfunc(py::module_& m);
void register_ufunc(py::module_& m);
void register_utils(py::module_& m);
void register_linalg(py::module_& m);
void register_einops(py::module_& m);
}  // namespace lucid::bindings

PYBIND11_MODULE(engine, m) {
    m.doc() = "Lucid C++ engine (production rebuild).";

    // Version surface — ABI consumers query these.
    m.attr("__version__") = LUCID_VERSION_STRING;
    m.attr("VERSION_MAJOR") = LUCID_VERSION_MAJOR;
    m.attr("VERSION_MINOR") = LUCID_VERSION_MINOR;
    m.attr("VERSION_PATCH") = LUCID_VERSION_PATCH;
    m.attr("ABI_VERSION") = LUCID_ABI_VERSION;

    // Errors first so the translator is in place before any other binding
    // can throw during initialization.
    lucid::bindings::register_errors(m);
    lucid::bindings::register_core(m);
    lucid::bindings::register_tensor_impl(m);
    lucid::bindings::register_amp(m);
    lucid::bindings::register_profiler(m);
    lucid::bindings::register_op_registry(m);
    lucid::bindings::register_autograd(m);
    auto nn = m.def_submodule("nn", "Neural-network ops (linear, conv, norm, pool, ...).");
    lucid::bindings::register_nn(nn);
    lucid::bindings::register_random(m);
    lucid::bindings::register_optim(m);
    // Op surface — mirrors lucid/_func/, lucid/_utils/, lucid/linalg/.
    lucid::bindings::register_gfunc(m);
    lucid::bindings::register_bfunc(m);
    lucid::bindings::register_ufunc(m);
    lucid::bindings::register_utils(m);
    auto linalg = m.def_submodule("linalg", "Linear-algebra ops.");
    lucid::bindings::register_linalg(linalg);
    auto einops = m.def_submodule("einops", "einops-style rearrange/reduce/repeat/einsum.");
    lucid::bindings::register_einops(einops);
}
