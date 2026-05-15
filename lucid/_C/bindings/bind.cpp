// lucid/_C/bindings/bind.cpp
//
// Top-level entry point for the `lucid._C.engine` Python extension module.
// This file owns the single PYBIND11_MODULE(engine, m) definition and is
// responsible for:
//   1. Exporting the ABI / version constants (VERSION_MAJOR, VERSION_MINOR,
//      VERSION_PATCH, ABI_VERSION, __version__).
//   2. Calling every subsystem register_xxx() function in dependency order so
//      that base types (errors, enums, TensorImpl) are visible before the ops
//      that use them.
//   3. Creating the `nn`, `linalg`, and `einops` sub-modules.
//   4. Defining the two fused-kernel helpers (_fused_linear_relu,
//      _fused_linear_gelu) and the custom Metal kernel launcher
//      (_run_metal_kernel) that are too tightly coupled to the Dispatcher to
//      live in a separate file.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../backend/Dispatcher.h"
#include "../core/TensorImpl.h"
#include "../core/fwd.h"
#include "../version.h"

namespace py = pybind11;

// Forward declarations of every subsystem binder.  Each is defined in its own
// bind_xxx.cpp translation unit and linked into the shared library.
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
void register_composite(py::module_& m);
void register_linalg(py::module_& m);
void register_einops(py::module_& m);
void register_fft(py::module_& m);
void register_complex(py::module_& m);
}  // namespace lucid::bindings

PYBIND11_MODULE(engine, m) {
    m.doc() = "Lucid C++ engine (production rebuild).";

    // Version constants are read by lucid/__init__.py to enforce ABI
    // compatibility between the Python wheel and the compiled extension.
    m.attr("__version__") = LUCID_VERSION_STRING;
    m.attr("VERSION_MAJOR") = LUCID_VERSION_MAJOR;
    m.attr("VERSION_MINOR") = LUCID_VERSION_MINOR;
    m.attr("VERSION_PATCH") = LUCID_VERSION_PATCH;
    m.attr("ABI_VERSION") = LUCID_ABI_VERSION;

    // Registration order matters: error translators must be installed first so
    // that exceptions thrown during subsequent registrations are mapped
    // correctly.  Core enums (Device, Dtype) must precede TensorImpl which
    // depends on them.
    lucid::bindings::register_errors(m);
    lucid::bindings::register_core(m);
    lucid::bindings::register_tensor_impl(m);
    lucid::bindings::register_amp(m);
    lucid::bindings::register_profiler(m);
    lucid::bindings::register_op_registry(m);
    lucid::bindings::register_autograd(m);
    // nn ops live in a dedicated sub-module (lucid._C.engine.nn) to avoid
    // polluting the top-level namespace.
    auto nn = m.def_submodule("nn", "Neural-network ops (linear, conv, norm, pool, ...).");
    lucid::bindings::register_nn(nn);
    lucid::bindings::register_random(m);
    lucid::bindings::register_optim(m);

    lucid::bindings::register_gfunc(m);
    lucid::bindings::register_bfunc(m);
    lucid::bindings::register_ufunc(m);
    lucid::bindings::register_utils(m);
    // Composite ops layered on top of the primitives — must come after
    // the primitive registrations so its forwards/aliases see them.
    lucid::bindings::register_composite(m);
    // linalg and einops each get their own sub-module so Python can import
    // them as lucid.linalg / lucid.einops without name collisions.
    auto linalg = m.def_submodule("linalg", "Linear-algebra ops.");
    lucid::bindings::register_linalg(linalg);
    auto einops = m.def_submodule("einops", "einops-style rearrange/reduce/repeat/einsum.");
    lucid::bindings::register_einops(einops);
    auto fft = m.def_submodule("fft", "Discrete Fourier transform ops (mlx::core::fft wrappers).");
    lucid::bindings::register_fft(fft);

    // Complex viewing ops (real / imag / complex / conj) live at the
    // top level — they're general-purpose, not under any sub-module.
    lucid::bindings::register_complex(m);

    // Fused kernel bindings that directly call IBackend dispatch and therefore
    // cannot be separated into their own translation units without exposing
    // internal Dispatcher headers to bind_nn.cpp.
    m.def(
        "_fused_linear_relu",
        // The lambda validates shapes and delegates to the backend; the
        // output TensorImpl is created without requires_grad because the
        // fused path has no autograd support (use linear + relu separately
        // if gradients are needed).
        [](const lucid::TensorImplPtr& x, const lucid::TensorImplPtr& w,
           const lucid::TensorImplPtr& b) -> lucid::TensorImplPtr {
            if (!x || !w || !b)
                throw std::invalid_argument("_fused_linear_relu: null input");

            const auto& xs = x->shape();
            const auto& ws = w->shape();
            if (ws.size() < 2)
                throw std::invalid_argument("_fused_linear_relu: weight must be >= 2-D");
            lucid::Shape out_shape(xs.begin(), xs.end() - 1);
            out_shape.push_back(ws[0]);

            auto& be = lucid::backend::Dispatcher::for_device(x->device());
            lucid::Storage out = be.fused_linear_relu_forward(x->storage(), w->storage(),
                                                              b->storage(), out_shape, x->dtype());
            return std::make_shared<lucid::TensorImpl>(std::move(out), out_shape, x->dtype(),
                                                       x->device(), false);
        },
        py::arg("x"), py::arg("weight"), py::arg("bias"),
        "Fused linear + ReLU: y = max(0, x @ W.T + b).  "
        "CPU: BLAS SGEMM + vDSP vrelu (single cache pass).");

    m.def(
        "_fused_linear_gelu",
        // Same structure as _fused_linear_relu; activation switched to GELU
        // approximated via vForce vtanh on the CPU backend.
        [](const lucid::TensorImplPtr& x, const lucid::TensorImplPtr& w,
           const lucid::TensorImplPtr& b) -> lucid::TensorImplPtr {
            if (!x || !w || !b)
                throw std::invalid_argument("_fused_linear_gelu: null input");

            const auto& xs = x->shape();
            const auto& ws = w->shape();
            if (ws.size() < 2)
                throw std::invalid_argument("_fused_linear_gelu: weight must be >= 2-D");
            lucid::Shape out_shape(xs.begin(), xs.end() - 1);
            out_shape.push_back(ws[0]);

            auto& be = lucid::backend::Dispatcher::for_device(x->device());
            lucid::Storage out = be.fused_linear_gelu_forward(x->storage(), w->storage(),
                                                              b->storage(), out_shape, x->dtype());
            return std::make_shared<lucid::TensorImpl>(std::move(out), out_shape, x->dtype(),
                                                       x->device(), false);
        },
        py::arg("x"), py::arg("weight"), py::arg("bias"),
        "Fused linear + GELU: y = GELU(x @ W.T + b).  "
        "CPU: BLAS SGEMM + vForce vtanh.");

    // Custom Metal kernel launcher.  The dtype string is converted to a
    // Dtype enum here rather than in C++ generic code so that Python callers
    // can pass plain strings ("f32", "f16", etc.).  The output TensorImpl
    // always lives on Device::GPU and is created without autograd.
    m.def(
        "_run_metal_kernel",
        [](const std::string& kernel_source, const std::string& function_name,
           const std::vector<lucid::TensorImplPtr>& inputs,
           const std::vector<std::int64_t>& output_shape_list, const std::string& dtype_str,
           const std::array<std::size_t, 3>& grid,
           const std::array<std::size_t, 3>& threads) -> lucid::TensorImplPtr {
            lucid::Dtype dt = lucid::Dtype::F32;
            if (dtype_str == "f16")
                dt = lucid::Dtype::F16;
            else if (dtype_str == "f32")
                dt = lucid::Dtype::F32;
            else if (dtype_str == "f64")
                dt = lucid::Dtype::F64;
            else if (dtype_str == "i32")
                dt = lucid::Dtype::I32;
            else if (dtype_str == "i64")
                dt = lucid::Dtype::I64;

            lucid::Shape out_shape(output_shape_list.begin(), output_shape_list.end());

            std::vector<lucid::Storage> in_storages;
            in_storages.reserve(inputs.size());
            for (const auto& t : inputs) {
                if (!t)
                    throw std::invalid_argument("_run_metal_kernel: null input tensor");
                in_storages.push_back(t->storage());
            }

            auto& be = lucid::backend::Dispatcher::for_device(lucid::Device::GPU);
            lucid::Storage out = be.run_custom_metal_kernel(
                kernel_source, function_name, in_storages, out_shape, dt, grid, threads);

            // MetalKernelRunner returns SharedStorage, which lives in MTLResourceStorageModeShared
            // memory — directly accessible from both the GPU and CPU without a copy.  We label
            // the TensorImpl as Device::CPU so that the CPU backend handles subsequent ops
            // (contiguous, numpy, etc.) via SharedStorage::cpu_view().  Users who need the
            // result as a true GPU (MLX) tensor should call .to(device='metal') afterward.
            return std::make_shared<lucid::TensorImpl>(std::move(out), out_shape, dt,
                                                       lucid::Device::CPU, false);
        },
        py::arg("kernel_source"), py::arg("function_name"), py::arg("inputs"),
        py::arg("output_shape"), py::arg("dtype") = "f32", py::arg("grid"), py::arg("threads"),
        "Execute a custom Metal Shading Language (MSL) compute kernel on the GPU.\n"
        "Inputs are bound as read-only buffers; the output buffer is appended last.");
}
