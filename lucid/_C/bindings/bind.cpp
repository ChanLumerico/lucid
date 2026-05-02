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
#include <pybind11/stl.h>

#include "../version.h"
#include "../backend/Dispatcher.h"
#include "../core/fwd.h"
#include "../core/TensorImpl.h"

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

    // ---- Phase 19: Fused ops ------------------------------------------------
    // Expose _fused_linear_relu and _fused_linear_gelu so that
    // lucid/nn/functional.py can call them without knowing about IBackend.
    m.def("_fused_linear_relu",
          [](const lucid::TensorImplPtr& x,
             const lucid::TensorImplPtr& w,
             const lucid::TensorImplPtr& b) -> lucid::TensorImplPtr {
              if (!x || !w || !b)
                  throw std::invalid_argument("_fused_linear_relu: null input");

              // Compute (M, N) output shape from x(..., K) × w(N, K)^T.
              const auto& xs = x->shape();
              const auto& ws = w->shape();
              if (ws.size() < 2)
                  throw std::invalid_argument("_fused_linear_relu: weight must be >= 2-D");
              lucid::Shape out_shape(xs.begin(), xs.end() - 1);
              out_shape.push_back(ws[0]);

              auto& be = lucid::backend::Dispatcher::for_device(x->device());
              lucid::Storage out = be.fused_linear_relu_forward(
                  x->storage(), w->storage(), b->storage(),
                  out_shape, x->dtype());
              return std::make_shared<lucid::TensorImpl>(
                  std::move(out), out_shape, x->dtype(), x->device(),
                  /*requires_grad=*/false);
          },
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          "Fused linear + ReLU: y = max(0, x @ W.T + b).  "
          "CPU: BLAS SGEMM + vDSP vrelu (single cache pass).");

    m.def("_fused_linear_gelu",
          [](const lucid::TensorImplPtr& x,
             const lucid::TensorImplPtr& w,
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
              lucid::Storage out = be.fused_linear_gelu_forward(
                  x->storage(), w->storage(), b->storage(),
                  out_shape, x->dtype());
              return std::make_shared<lucid::TensorImpl>(
                  std::move(out), out_shape, x->dtype(), x->device(),
                  /*requires_grad=*/false);
          },
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          "Fused linear + GELU: y = GELU(x @ W.T + b).  "
          "CPU: BLAS SGEMM + vForce vtanh.");

    // ---- Phase 18: Metal Shader Escape Hatch --------------------------------
    // Expose `_run_metal_kernel` so Python code can dispatch a custom MSL
    // compute kernel on the GPU backend.
    //
    // Example (Python):
    //   src = """kernel void scale(device float* x [[buffer(0)]],
    //                               device float* y [[buffer(1)]],
    //                               uint  i [[thread_position_in_grid]]) {
    //               y[i] = x[i] * 2.0f; }"""
    //   result = engine._run_metal_kernel(
    //       src, "scale", [input_tensor],
    //       list(input_tensor.shape), "f32",
    //       [numel,1,1], [256,1,1])
    //
    m.def("_run_metal_kernel",
          [](const std::string&              kernel_source,
             const std::string&              function_name,
             const std::vector<lucid::TensorImplPtr>& inputs,
             const std::vector<std::int64_t>& output_shape_list,
             const std::string&              dtype_str,
             const std::array<std::size_t,3>& grid,
             const std::array<std::size_t,3>& threads) -> lucid::TensorImplPtr {

              // Resolve dtype string → Dtype.
              lucid::Dtype dt = lucid::Dtype::F32;
              if      (dtype_str == "f16")  dt = lucid::Dtype::F16;
              else if (dtype_str == "f32")  dt = lucid::Dtype::F32;
              else if (dtype_str == "f64")  dt = lucid::Dtype::F64;
              else if (dtype_str == "i32")  dt = lucid::Dtype::I32;
              else if (dtype_str == "i64")  dt = lucid::Dtype::I64;

              lucid::Shape out_shape(output_shape_list.begin(), output_shape_list.end());

              // Collect input Storages.
              std::vector<lucid::Storage> in_storages;
              in_storages.reserve(inputs.size());
              for (const auto& t : inputs) {
                  if (!t) throw std::invalid_argument("_run_metal_kernel: null input tensor");
                  in_storages.push_back(t->storage());
              }

              // Dispatch through GpuBackend.
              auto& be = lucid::backend::Dispatcher::for_device(lucid::Device::GPU);
              lucid::Storage out = be.run_custom_metal_kernel(
                  kernel_source, function_name, in_storages, out_shape, dt, grid, threads);

              return std::make_shared<lucid::TensorImpl>(
                  std::move(out), out_shape, dt, lucid::Device::GPU, /*requires_grad=*/false);
          },
          py::arg("kernel_source"),
          py::arg("function_name"),
          py::arg("inputs"),
          py::arg("output_shape"),
          py::arg("dtype")   = "f32",
          py::arg("grid"),
          py::arg("threads"),
          "Execute a custom Metal Shading Language (MSL) compute kernel on the GPU.\n"
          "Inputs are bound as read-only buffers; the output buffer is appended last.");
}
