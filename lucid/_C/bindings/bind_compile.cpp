// lucid/_C/bindings/bind_compile.cpp
//
// Phase 1.1 Python bindings for :mod:`lucid.compile`.
//
// Exposes:
//   - TensorMeta / OpNode / TraceGraph (read-only views of the recorded IR).
//   - Tracer (the thread-local op recorder).
//   - set_current_tracer() / current_tracer() — install or detach the
//     process-wide Tracer pointer that OpScopeFull's hook reads on each
//     op entry.
//
// The shapes here are intentionally minimal — Phase 1.2 layers MpsBuilder /
// ExecutableCache on top using the same Tracer.graph() output, and the JSON
// trace dump invoked by LUCID_COMPILE_DEBUG=1 consumes the read-only fields
// surfaced below.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

#include "../compile/CompiledExecutable.h"
#include "../compile/ExecutableCache.h"
#include "../compile/MpsBuilder.h"
#include "../version.h"
#include "../compile/OpEmitters/OpEmitter.h"
#include "../compile/Tracer.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {
// Python-side wrapper that owns the raw :class:`CompiledExecutable*`
// allocated by :func:`compile_trace` and releases it via
// :func:`destroy_executable` on destruction.
class PyCompiledExecutable {
public:
    explicit PyCompiledExecutable(lucid::compile::CompiledExecutable* exe, bool owns = true)
        : exe_(exe), owns_(owns) {}
    ~PyCompiledExecutable() {
        if (owns_)
            lucid::compile::destroy_executable(exe_);
    }

    PyCompiledExecutable(const PyCompiledExecutable&) = delete;
    PyCompiledExecutable& operator=(const PyCompiledExecutable&) = delete;

    lucid::compile::CompiledExecutable* raw() const { return exe_; }

    std::size_t num_inputs() const {
        return lucid::compile::executable_num_inputs(exe_);
    }
    std::size_t num_outputs() const {
        return lucid::compile::executable_num_outputs(exe_);
    }
    std::vector<lucid::compile::TensorId> input_ids() const {
        return lucid::compile::executable_input_ids(exe_);
    }
    std::vector<lucid::compile::TensorId> output_ids() const {
        return lucid::compile::executable_output_ids(exe_);
    }
    std::vector<lucid::compile::TensorId> grad_output_ids() const {
        return lucid::compile::executable_grad_output_ids(exe_);
    }

private:
    lucid::compile::CompiledExecutable* exe_;
    bool owns_;
};

}  // namespace

// Registers the compile-time tracing surface as a sub-module of `_C.engine`.
//
// Mirrors :func:`register_profiler` — the Python wrapper holds the only
// owning reference to each :class:`Tracer`; the engine retains a raw
// pointer installed via :func:`set_current_tracer`.
void register_compile(py::module_& m) {
    using lucid::compile::OpNode;
    using lucid::compile::TensorMeta;
    using lucid::compile::TraceGraph;
    using lucid::compile::Tracer;

    // TensorMeta is a plain data record describing one tensor slot in the
    // trace.  All fields are read-only from Python.
    py::class_<TensorMeta>(m, "TensorMeta")
        .def_readonly("id", &TensorMeta::id)
        .def_readonly("shape", &TensorMeta::shape)
        .def_readonly("dtype", &TensorMeta::dtype)
        .def_readonly("device", &TensorMeta::device)
        .def("__repr__", [](const TensorMeta& t) {
            return "TensorMeta(id=" + std::to_string(t.id) + ")";
        });

    // OpNode mirrors the C++ aggregate; Python sees inputs as a list of
    // ints, outputs as a list of TensorMeta, and attrs as a dict whose
    // values are int / list[int] / float / bool / str (pybind11's
    // variant caster handles the conversion).
    py::class_<OpNode>(m, "OpNode")
        .def_readonly("name", &OpNode::name)
        .def_readonly("inputs", &OpNode::inputs)
        .def_readonly("outputs", &OpNode::outputs)
        .def_readonly("attrs", &OpNode::attrs)
        .def("__repr__", [](const OpNode& n) {
            return "OpNode(name='" + n.name +
                   "', outputs=" + std::to_string(n.outputs.size()) +
                   ", attrs=" + std::to_string(n.attrs.size()) + ")";
        });

    // TraceGraph exposes the recorded op list; ``next_id`` is internal
    // book-keeping but useful for debug dumps.
    py::class_<TraceGraph>(m, "TraceGraph")
        .def_readonly("ops", &TraceGraph::ops)
        .def_readonly("next_id", &TraceGraph::next_id)
        .def("__len__", [](const TraceGraph& g) { return g.ops.size(); });

    // Tracer is owned by Python; the engine keeps only a raw pointer
    // installed via set_current_tracer.  Python must not delete the Tracer
    // while it is the active tracer (keep-alive is the caller's job —
    // the `_tracing()` context manager enforces this).
    py::class_<Tracer>(m, "Tracer")
        .def(py::init<>())
        .def_property_readonly("graph", &Tracer::graph,
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "external_feeds",
            [](const Tracer& t) {
                // Convert id → TensorImplPtr map into a Python dict.
                py::dict d;
                for (const auto& [tid, impl] : t.external_feeds())
                    d[py::cast(tid)] = impl;
                return d;
            },
            "id → TensorImpl map for inputs not produced by any traced "
            "op (model parameters + user inputs).")
        .def(
            "lookup_id",
            [](const Tracer& t, const lucid::TensorImplPtr& impl) -> py::object {
                const lucid::compile::TensorId id = t.lookup_id(impl.get());
                if (id < 0) return py::none();
                return py::cast(id);
            },
            py::arg("impl"),
            "Reverse-lookup the trace TensorId for an observed TensorImpl. "
            "Returns None if ``impl`` is not in the trace.");

    // set_current_tracer accepts None to detach the active tracer.  The
    // engine holds a raw pointer, so the Python caller must keep the
    // Tracer alive until detached.
    m.def("set_current_tracer", &lucid::compile::set_current_tracer,
          py::arg("tracer").none(true));
    // current_tracer returns a borrowed reference; Python must not delete it.
    m.def("current_tracer", &lucid::compile::current_tracer,
          py::return_value_policy::reference);

    // ── Phase 1.2 step 1: compile / run / cache surface ──────────────────

    // Opaque executable handle.  Python owns one PyCompiledExecutable per
    // compile_trace() return; destruction releases the MPSGraphExecutable
    // via the wrapper's destructor.
    py::class_<PyCompiledExecutable, std::shared_ptr<PyCompiledExecutable>>(
        m, "CompiledExecutable")
        .def_property_readonly("num_inputs", &PyCompiledExecutable::num_inputs)
        .def_property_readonly("num_outputs", &PyCompiledExecutable::num_outputs)
        .def_property_readonly("input_ids", &PyCompiledExecutable::input_ids,
                               "Feed-order TensorId list — run_executable() "
                               "expects inputs in this order.")
        .def_property_readonly("output_ids", &PyCompiledExecutable::output_ids,
                               "Forward-output TensorId list.  run_executable() "
                               "returns these first, followed by grad outputs.")
        .def_property_readonly("grad_output_ids", &PyCompiledExecutable::grad_output_ids,
                               "Per-parameter gradient TensorId list (Phase 1.3 "
                               "backward path).  Empty for forward-only executables. "
                               "run_executable() returns one gradient tensor per "
                               "id here, immediately after the forward outputs.");

    // Compile a TraceGraph into an MPSGraph executable.
    // Returns None on any abort condition (unsupported op, mixed-device,
    // dtype error, …) — the caller falls back to eager dispatch.
    m.def(
        "compile_trace",
        [](const lucid::compile::TraceGraph& graph,
           const py::dict& external_feeds_py) -> py::object {
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_trace(graph, feeds, &err);
            if (exe == nullptr)
                return py::none();
            return py::cast(std::make_shared<PyCompiledExecutable>(exe));
        },
        py::arg("graph"), py::arg("external_feeds"),
        "Lower a TraceGraph to a compiled MPSGraph executable; "
        "returns None when any op in the trace is unsupported.");

    // Run a compiled executable with feed-order input tensors. Returns a
    // list of fresh GPU-backed TensorImpls in the executable's target
    // order.  Forward declaration of run_executable lives in
    // CompiledExecutable.mm.
    m.def(
        "run_executable",
        [](const std::shared_ptr<PyCompiledExecutable>& wrapper,
           const std::vector<lucid::TensorImplPtr>& inputs)
            -> std::vector<lucid::TensorImplPtr> {
            if (!wrapper)
                throw std::invalid_argument("run_executable: null wrapper");
            return lucid::compile::run_executable(wrapper->raw(), inputs);
        },
        py::arg("executable"), py::arg("inputs"),
        "Execute a CompiledExecutable; inputs in input_ids order, "
        "outputs in output_ids order.");

    m.def(
        "run_executable_inplace",
        [](const std::shared_ptr<PyCompiledExecutable>& wrapper,
           const std::vector<lucid::TensorImplPtr>& inputs,
           const std::vector<lucid::TensorImplPtr>& output_targets) -> void {
            if (!wrapper)
                throw std::invalid_argument(
                    "run_executable_inplace: null wrapper");
            lucid::compile::run_executable_inplace(
                wrapper->raw(), inputs, output_targets);
        },
        py::arg("executable"), py::arg("inputs"), py::arg("output_targets"),
        "Execute a CompiledExecutable, writing each output directly into "
        "the corresponding pre-allocated tensor in ``output_targets`` "
        "instead of returning fresh tensors.  Used by the compiled-"
        "optimizer step path — avoids the per-output allocation + copy_ "
        "round-trip when the targets are parameters / state buffers "
        "that already live on the GPU.");

    // Session cache surface — `find` / `insert` are internal (Phase 1.4 will
    // wrap them inside CompiledModule); for Phase 1.2 we expose only size +
    // clear so tests can observe / reset cache state.
    // Diagnostics: report whether a given op name has an emitter
    // registered, and whether that emitter is real-emit or a stub.
    // Used by ``tools/`` and the docstring coverage tracker — NOT
    // a public end-user API.
    m.def(
        "emitter_registered",
        [](const std::string& name) -> bool {
            return lucid::compile::find_emitter(name) != nullptr;
        },
        py::arg("name"),
        "Return True iff an emitter is registered for ``name``.  A "
        "registered emitter may still return nullptr at emit time "
        "(stub semantics) — this helper only distinguishes "
        "'registered but possibly a stub' from 'no emitter at all'.");

    m.def("session_cache_size",
          [] { return lucid::compile::ExecutableCache::session().size(); });
    m.def("session_cache_clear",
          [] { lucid::compile::ExecutableCache::session().clear(); });

    // Phase 1.2 secondary acceptance gate: cache-aware compile.  On the
    // first call with a given structural signature the trace is compiled
    // and inserted into the session-global cache; subsequent calls with
    // a matching CacheKey return the same executable (borrowed wrapper,
    // owns=false — cache owns the raw pointer).
    m.def(
        "compile_or_cached",
        [](const lucid::compile::TraceGraph& graph,
           const py::dict& external_feeds_py,
           bool dynamic_batch,
           const std::vector<lucid::compile::TensorId>& param_ids,
           const std::vector<lucid::compile::TensorId>& explicit_outputs) -> py::object {
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }

            if (dynamic_batch) {
                // Phase 1.6 dynamic path: bypass the session cache.
                // ``make_cache_key`` derives from concrete trace shapes
                // which would mint a fresh entry per batch size — the
                // very thing dynamic mode is meant to avoid.  The
                // caller (CompiledModule) holds its own
                // dynamic-batch-aware Python cache.
                std::string err;
                lucid::compile::CompiledExecutable* exe =
                    lucid::compile::compile_trace(
                        graph, feeds, &err, /*dynamic_batch=*/true, param_ids,
                        explicit_outputs);
                if (exe == nullptr) {
                    if (!err.empty())
                        throw std::runtime_error(err);
                    return py::none();
                }
                return py::cast(std::make_shared<PyCompiledExecutable>(exe, /*owns=*/true));
            }

            auto& cache = lucid::compile::ExecutableCache::session();
            lucid::compile::CacheKey key = lucid::compile::make_cache_key(graph);

            if (auto* hit = cache.find(key)) {
                return py::cast(std::make_shared<PyCompiledExecutable>(hit, /*owns=*/false));
            }

            // ``LUCID_COMPILE_DISK_CACHE=1`` opt-in: before falling
            // through to a fresh compile, try to load a previously-
            // serialised executable from
            // ``$LUCID_COMPILE_DISK_CACHE_DIR`` (default
            // ``$TMPDIR/lucid_mpsgraph_cache``).  Filename is keyed by
            // the cache key's hash + the engine ABI so an SDK or
            // engine version bump invalidates stale caches without
            // explicit user action.  Misses (file absent, format
            // version mismatch, parse error) fall through to compile
            // + save.
            static const bool disk_cache_enabled = []() {
                const char* s = std::getenv("LUCID_COMPILE_DISK_CACHE");
                return s && std::string(s) == "1";
            }();
            std::string disk_path;
            if (disk_cache_enabled) {
                const char* dir_env = std::getenv("LUCID_COMPILE_DISK_CACHE_DIR");
                std::string dir;
                if (dir_env && *dir_env) {
                    dir = dir_env;
                } else {
                    const char* tmp = std::getenv("TMPDIR");
                    dir = std::string(tmp ? tmp : "/tmp") +
                          "lucid_mpsgraph_cache";
                }
                // Ensure trailing slash + create directory if missing.
                if (!dir.empty() && dir.back() != '/') dir.push_back('/');
                // ``mkdir -p`` via POSIX ``mkdir`` (one level is enough
                // because $TMPDIR / user-provided dir already exists).
                (void)::mkdir(dir.c_str(), 0755);
                lucid::compile::CacheKeyHash h;
                std::size_t hashv = h(key);
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                              "%slucid_abi%d_%016zx",
                              dir.c_str(),
                              static_cast<int>(LUCID_ABI_VERSION),
                              hashv);
                disk_path = buf;

                std::string load_err;
                lucid::compile::CompiledExecutable* disk_hit =
                    lucid::compile::load_executable(disk_path, &load_err);
                if (disk_hit != nullptr) {
                    cache.insert(key, disk_hit);
                    auto* hit = cache.find(key);
                    return py::cast(std::make_shared<PyCompiledExecutable>(hit, /*owns=*/false));
                }
            }

            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_trace(graph, feeds, &err,
                                              /*dynamic_batch=*/false, param_ids,
                                              explicit_outputs);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            cache.insert(key, exe);
            auto* hit = cache.find(key);  // borrowed pointer from the cache
            if (disk_cache_enabled && !disk_path.empty())
                (void)lucid::compile::save_executable(hit, disk_path);
            return py::cast(std::make_shared<PyCompiledExecutable>(hit, /*owns=*/false));
        },
        py::arg("graph"), py::arg("external_feeds"),
        py::arg("dynamic_batch") = false,
        py::arg("param_ids") = std::vector<lucid::compile::TensorId>{},
        py::arg("explicit_outputs") = std::vector<lucid::compile::TensorId>{},
        "Cache-aware compile: on a cache miss the trace is compiled and "
        "inserted into the session-global ExecutableCache; on a hit the "
        "cached executable is returned (Python wrapper is borrowed — the "
        "cache retains ownership).  With ``dynamic_batch=True`` (Phase 1.6) "
        "the session cache is bypassed (caller manages a dynamic-aware "
        "Python cache) and ``param_ids`` declares which feed ids are "
        "parameters whose first dim must stay static.  ``explicit_outputs`` "
        "(when non-empty) overrides the auto-detect graph-output heuristic "
        "with exactly the listed ids — used by CompiledModule to mask "
        "Python-discarded intermediates from the executable's output list.");

    // Phase 1.10 AOT export: expose the previously-internal
    // ``save_executable`` / ``load_executable`` C++ APIs as the
    // Python-facing user surface.  Stable user-facing pickle / AOT
    // serialisation now flows through these.
    m.def(
        "save_executable",
        [](const std::shared_ptr<PyCompiledExecutable>& wrapper,
           const std::string& path) -> bool {
            if (!wrapper)
                throw std::invalid_argument("save_executable: null wrapper");
            return lucid::compile::save_executable(wrapper->raw(), path);
        },
        py::arg("executable"), py::arg("path"),
        "Serialise the executable to ``<path>.mpsgraphpackage`` (Apple-native "
        "MPSGraphExecutable archive, macOS 14+) and ``<path>.meta`` (Lucid I/O "
        "plan).  Returns ``True`` on success.  Both files together constitute "
        "one saved compile artifact — ``load_executable`` expects them paired.  "
        "Same on-disk format as the internal ``LUCID_COMPILE_DISK_CACHE`` "
        "path.  Throws on null wrapper; returns ``False`` on I/O / "
        "serialisation failure.");

    m.def(
        "load_executable",
        [](const std::string& path) -> py::object {
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::load_executable(path, &err);
            if (exe == nullptr) {
                if (!err.empty()) throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(std::make_shared<PyCompiledExecutable>(exe, /*owns=*/true));
        },
        py::arg("path"),
        "Reload a CompiledExecutable previously saved via "
        "``save_executable(executable, path)``.  Expects both "
        "``<path>.mpsgraphpackage`` and ``<path>.meta`` to exist.  Returns "
        "the executable on success, ``None`` on missing files / format "
        "mismatch; throws on a corrupted .meta sidecar.  ABI-version mismatch "
        "(SDK or engine bump) is rejected at load time — callers must "
        "recompile from the source trace.");

    // Phase 1.3: forward + backward in one executable.
    m.def(
        "compile_trace_with_backward",
        [](const lucid::compile::TraceGraph& graph,
           const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id,
           const std::vector<lucid::compile::TensorId>& param_ids,
           bool dynamic_batch) -> py::object {
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_trace_with_backward(
                    graph, feeds, loss_id, param_ids, &err, dynamic_batch);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(std::make_shared<PyCompiledExecutable>(exe, /*owns=*/true));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"),
        py::arg("param_ids"), py::arg("dynamic_batch") = false,
        "Lower a TraceGraph + (loss_id, param_ids) into a single MPSGraph "
        "executable that computes the forward (loss) and the auto-derived "
        "gradients of loss w.r.t. each parameter.  Returns None on any "
        "unsupported op (eager fallback).  Output of run_executable: "
        "[loss_tensor, grad_for_param_ids[0], grad_for_param_ids[1], ...].  "
        "With ``dynamic_batch=True`` (Phase 1.6) the leading dim of every "
        "non-parameter feed becomes a symbolic placeholder (-1), so one "
        "executable handles variable batch size.");

    // Phase 1.7: forward + backward + optimizer update in one executable.
    // Used by lucid.compile.fused_step(model, loss_fn, optimizer).
    py::enum_<lucid::compile::OptimizerSpec::Kind>(m, "OptimizerKind")
        .value("SGD", lucid::compile::OptimizerSpec::Kind::SGD)
        .value("ADAM", lucid::compile::OptimizerSpec::Kind::ADAM)
        .value("ADAMW", lucid::compile::OptimizerSpec::Kind::ADAMW);

    py::class_<lucid::compile::OptimizerSpec>(m, "OptimizerSpec")
        .def(py::init<>())
        .def_readwrite("kind", &lucid::compile::OptimizerSpec::kind)
        .def_readwrite("lr", &lucid::compile::OptimizerSpec::lr)
        .def_readwrite("momentum", &lucid::compile::OptimizerSpec::momentum)
        .def_readwrite("dampening", &lucid::compile::OptimizerSpec::dampening)
        .def_readwrite("weight_decay",
                       &lucid::compile::OptimizerSpec::weight_decay)
        .def_readwrite("nesterov", &lucid::compile::OptimizerSpec::nesterov)
        .def_readwrite("beta1", &lucid::compile::OptimizerSpec::beta1)
        .def_readwrite("beta2", &lucid::compile::OptimizerSpec::beta2)
        .def_readwrite("eps", &lucid::compile::OptimizerSpec::eps);

    m.def(
        "compile_fused_training_step",
        [](const lucid::compile::TraceGraph& graph,
           const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id,
           const std::vector<lucid::compile::TensorId>& param_ids,
           const lucid::compile::OptimizerSpec& opt_spec,
           const std::vector<std::vector<lucid::compile::TensorId>>&
               state_buf_ids_per_param,
           const std::vector<lucid::compile::TensorId>& scalar_input_ids)
            -> py::object {
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr>
                feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_fused_training_step(
                    graph, feeds, loss_id, param_ids, opt_spec,
                    state_buf_ids_per_param, scalar_input_ids, &err);
            if (exe == nullptr) {
                if (!err.empty()) throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(exe, /*owns=*/true));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"),
        py::arg("param_ids"), py::arg("opt_spec"),
        py::arg("state_buf_ids_per_param"), py::arg("scalar_input_ids"),
        "Lower (forward + backward + optimizer.step) into a single MPSGraph "
        "executable.  Outputs (in order): [loss, new_param_0, new_param_1, "
        "..., new_state_0_0, new_state_0_1, ..., new_state_N_0, ...].  Used "
        "by lucid.compile.fused_step.");

    // Phase 1.8: generic fused step — the optimizer update is captured
    // in the same TraceGraph (via ghost-grad placeholders), so any
    // optimizer expressible as Lucid tensor ops works without C++ math
    // duplication.
    m.def(
        "compile_generic_fused_step",
        [](const lucid::compile::TraceGraph& graph,
           const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id,
           const std::vector<lucid::compile::TensorId>& param_ids,
           const std::vector<lucid::compile::TensorId>& ghost_grad_ids,
           const std::vector<lucid::compile::TensorId>& output_target_ids)
            -> py::object {
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr>
                feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            // Note (2026-05-25): a disk-cache variant of this path was
            // prototyped — see git log for the experiment — but
            // measured slower than recompile on the fused_step case
            // (warm cache hit 129 ms vs fresh compile 132 ms; first
            // cold hit 828 ms while the OS file cache warmed).  The
            // forward-only ``compile_or_cached`` disk cache still
            // wins (6× cold-start) because forward MPSGraph packages
            // are smaller and serialise faster; fused_step packages
            // include the autograd-derived backward chain which
            // appears expensive to deserialise.  Decision: in-memory
            // session cache only for fused_step until the package
            // format improves.
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_generic_fused_step(
                    graph, feeds, loss_id, param_ids, ghost_grad_ids,
                    output_target_ids, &err);
            if (exe == nullptr) {
                if (!err.empty()) throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(exe, /*owns=*/true));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"),
        py::arg("param_ids"), py::arg("ghost_grad_ids"),
        py::arg("output_target_ids"),
        "Generic fused-step compile.  The TraceGraph contains both the "
        "forward+loss path and the optimizer update.  ``ghost_grad_ids`` "
        "are placeholder tensor ids that the Python wrapper reserved for "
        "the optimizer's grad inputs; this function derives those grads "
        "via MPSGraph autograd after emitting the forward, then binds "
        "each ghost grad to its derived gradient before emitting the "
        "remaining (optimizer) ops.  ``output_target_ids`` is the flat "
        "ordered list of new_param / new_state ids that become the "
        "executable's targets (loss is always target 0).");

    // Phase 1.9: stateful-variables variant.  Designated feeds become
    // ``variableWithData:`` (initialized from the Lucid Tensor's
    // current MTLBuffer contents at compile time); the matching
    // output_target_ids become ``assignVariable:`` operations + a
    // ``readVariable:`` tensor that flushes the post-assign value
    // into the Lucid Tensor's existing MTLBuffer on each call.
    m.def(
        "compile_generic_fused_step_with_vars",
        [](const lucid::compile::TraceGraph& graph,
           const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id,
           const std::vector<lucid::compile::TensorId>& param_ids,
           const std::vector<lucid::compile::TensorId>& ghost_grad_ids,
           const std::vector<lucid::compile::TensorId>& output_target_ids,
           const std::vector<std::pair<lucid::compile::TensorId,
                                       lucid::compile::TensorId>>& variable_pairs)
            -> py::object {
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr>
                feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_generic_fused_step_with_vars(
                    graph, feeds, loss_id, param_ids, ghost_grad_ids,
                    output_target_ids, variable_pairs, &err);
            if (exe == nullptr) {
                if (!err.empty()) throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(exe, /*owns=*/true));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"),
        py::arg("param_ids"), py::arg("ghost_grad_ids"),
        py::arg("output_target_ids"), py::arg("variable_pairs"),
        "Stateful-variables variant of compile_generic_fused_step.  "
        "``variable_pairs`` is a list of ``(feed_id, write_id)`` tuples: "
        "the feed becomes a persistent MPSGraph variable initialised "
        "from the Lucid Tensor's current buffer contents, and the "
        "matching write_id is bound to an ``assignVariable:`` + "
        "``readVariable:`` pair so subsequent calls update the variable "
        "internally and flush the new value back into the Lucid Tensor's "
        "existing MTLBuffer (no per-call newBufferWithLength).  Empty "
        "variable_pairs is equivalent to compile_generic_fused_step.");
}

namespace {
// No-op symbols pulled from this TU keep each emitter family's
// file-scope registrar alive against aggressive dead-code stripping
// when the bindings are linked into the engine MODULE.  Each emitter
// is otherwise self-contained (only side effect is the static
// registrar in its .mm), and `--gc-sections` + ``-fvisibility=hidden``
// would otherwise drop the entire .mm.
void touch_emitters() {
    (void)lucid::compile::find_emitter("linear");
    (void)lucid::compile::find_emitter("relu");
    (void)lucid::compile::find_emitter("sigmoid");
    (void)lucid::compile::find_emitter("tanh");
    (void)lucid::compile::find_emitter("add");
    (void)lucid::compile::find_emitter("sub");
    (void)lucid::compile::find_emitter("mul");
    (void)lucid::compile::find_emitter("div");
    (void)lucid::compile::find_emitter("view");
    (void)lucid::compile::find_emitter("reshape");
    (void)lucid::compile::find_emitter("squeeze");
    (void)lucid::compile::find_emitter("unsqueeze");
    (void)lucid::compile::find_emitter("flatten");
    (void)lucid::compile::find_emitter("contiguous");
    (void)lucid::compile::find_emitter("permute");
    (void)lucid::compile::find_emitter("sum");
    (void)lucid::compile::find_emitter("mean");
    (void)lucid::compile::find_emitter("neg");
    (void)lucid::compile::find_emitter("abs");
    (void)lucid::compile::find_emitter("exp");
    (void)lucid::compile::find_emitter("log");
    (void)lucid::compile::find_emitter("sqrt");
    (void)lucid::compile::find_emitter("square");
    (void)lucid::compile::find_emitter("reciprocal");
    (void)lucid::compile::find_emitter("rsqrt");
    (void)lucid::compile::find_emitter("softmax");
    (void)lucid::compile::find_emitter("log_softmax");
    (void)lucid::compile::find_emitter("sin");
    (void)lucid::compile::find_emitter("cos");
    (void)lucid::compile::find_emitter("tan");
    (void)lucid::compile::find_emitter("sinh");
    (void)lucid::compile::find_emitter("cosh");
    (void)lucid::compile::find_emitter("silu");
    (void)lucid::compile::find_emitter("gelu");
    (void)lucid::compile::find_emitter("gelu_exact");
    (void)lucid::compile::find_emitter("pow");
    (void)lucid::compile::find_emitter("matmul");
    (void)lucid::compile::find_emitter("max_pool2d");
    (void)lucid::compile::find_emitter("avg_pool2d");
    (void)lucid::compile::find_emitter("layer_norm");
    (void)lucid::compile::find_emitter("rms_norm");
    (void)lucid::compile::find_emitter("conv2d");
    (void)lucid::compile::find_emitter("batch_norm_eval");
    (void)lucid::compile::find_emitter("max");
    (void)lucid::compile::find_emitter("min");
    (void)lucid::compile::find_emitter("prod");
    (void)lucid::compile::find_emitter("concatenate");
    (void)lucid::compile::find_emitter("batch_norm");
    (void)lucid::compile::find_emitter("batch_norm1d");
    (void)lucid::compile::find_emitter("batch_norm3d");
    (void)lucid::compile::find_emitter("group_norm");
    (void)lucid::compile::find_emitter("full");
    (void)lucid::compile::find_emitter("astype");
    (void)lucid::compile::find_emitter("gather");
    (void)lucid::compile::find_emitter("equal");
    (void)lucid::compile::find_emitter("not_equal");
    (void)lucid::compile::find_emitter("greater");
    (void)lucid::compile::find_emitter("greater_equal");
    (void)lucid::compile::find_emitter("less");
    (void)lucid::compile::find_emitter("less_equal");
    // R1 chunk 1
    (void)lucid::compile::find_emitter("arccos");
    (void)lucid::compile::find_emitter("arcsin");
    (void)lucid::compile::find_emitter("arctan");
    (void)lucid::compile::find_emitter("ceil");
    (void)lucid::compile::find_emitter("floor");
    (void)lucid::compile::find_emitter("round");
    (void)lucid::compile::find_emitter("cube");
    (void)lucid::compile::find_emitter("cube_root");
    (void)lucid::compile::find_emitter("log2");
    (void)lucid::compile::find_emitter("sign");
    (void)lucid::compile::find_emitter("erf");
    (void)lucid::compile::find_emitter("clip");
    (void)lucid::compile::find_emitter("elu");
    (void)lucid::compile::find_emitter("leaky_relu");
    (void)lucid::compile::find_emitter("selu");
    (void)lucid::compile::find_emitter("mish");
    (void)lucid::compile::find_emitter("softplus");
    (void)lucid::compile::find_emitter("hard_sigmoid");
    (void)lucid::compile::find_emitter("hard_swish");
    (void)lucid::compile::find_emitter("relu6");
    (void)lucid::compile::find_emitter("maximum");
    (void)lucid::compile::find_emitter("minimum");
    (void)lucid::compile::find_emitter("floordiv");
    (void)lucid::compile::find_emitter("nextafter");
    (void)lucid::compile::find_emitter("pow_scalar");
    (void)lucid::compile::find_emitter("rpow_scalar");
    (void)lucid::compile::find_emitter("invert");
    (void)lucid::compile::find_emitter("var");
    // R1 chunk 2
    (void)lucid::compile::find_emitter("where");
    (void)lucid::compile::find_emitter("masked_fill");
    (void)lucid::compile::find_emitter("flip");
    (void)lucid::compile::find_emitter("roll");
    (void)lucid::compile::find_emitter("broadcast_to");
    (void)lucid::compile::find_emitter("pad");
    (void)lucid::compile::find_emitter("tile");
    (void)lucid::compile::find_emitter("repeat");
    (void)lucid::compile::find_emitter("stack");
    // R2 chunk 1 — loss family
    (void)lucid::compile::find_emitter("mse_loss");
    (void)lucid::compile::find_emitter("huber_loss");
    (void)lucid::compile::find_emitter("bce_loss");
    (void)lucid::compile::find_emitter("bce_with_logits");
    (void)lucid::compile::find_emitter("embedding");
    (void)lucid::compile::find_emitter("nll_loss");
    (void)lucid::compile::find_emitter("cross_entropy_loss");
    // R3 cumulative
    (void)lucid::compile::find_emitter("cumsum");
    (void)lucid::compile::find_emitter("cumprod");
    (void)lucid::compile::find_emitter("cummax");
    (void)lucid::compile::find_emitter("cummin");
    // MoreOps — sort family + linalg long tail + stubs
    (void)lucid::compile::find_emitter("sort");
    (void)lucid::compile::find_emitter("argsort");
    (void)lucid::compile::find_emitter("argmax");
    (void)lucid::compile::find_emitter("argmin");
    (void)lucid::compile::find_emitter("all");
    (void)lucid::compile::find_emitter("any");
    (void)lucid::compile::find_emitter("conj");
    (void)lucid::compile::find_emitter("real");
    (void)lucid::compile::find_emitter("imag");
    (void)lucid::compile::find_emitter("trace");
    (void)lucid::compile::find_emitter("inner");
    (void)lucid::compile::find_emitter("outer");
    (void)lucid::compile::find_emitter("dot");
    (void)lucid::compile::find_emitter("tril");
    (void)lucid::compile::find_emitter("triu");
    (void)lucid::compile::find_emitter("diagonal");
    (void)lucid::compile::find_emitter("nan_to_num");
    (void)lucid::compile::find_emitter("one_hot");
    (void)lucid::compile::find_emitter("scaled_dot_product_attention");
    (void)lucid::compile::find_emitter("lp_normalize");
    (void)lucid::compile::find_emitter("global_response_norm");
    (void)lucid::compile::find_emitter("scatter_add");
    (void)lucid::compile::find_emitter("scatter_amax");
    (void)lucid::compile::find_emitter("scatter_amin");
    (void)lucid::compile::find_emitter("scatter_prod");
    // Tier 1
    (void)lucid::compile::find_emitter("norm");
    (void)lucid::compile::find_emitter("matrix_power");
    (void)lucid::compile::find_emitter("det");
    (void)lucid::compile::find_emitter("inv");
    (void)lucid::compile::find_emitter("bilinear_layer");
    (void)lucid::compile::find_emitter("rotary_pos_embedding");
    (void)lucid::compile::find_emitter("tensordot");
    // Tier 2
    (void)lucid::compile::find_emitter("conv1d");
    (void)lucid::compile::find_emitter("conv3d");
    (void)lucid::compile::find_emitter("conv_transpose1d");
    (void)lucid::compile::find_emitter("conv_transpose2d");
    (void)lucid::compile::find_emitter("max_pool1d");
    (void)lucid::compile::find_emitter("avg_pool1d");
    // Tier 3
    (void)lucid::compile::find_emitter("interpolate_bilinear");
    (void)lucid::compile::find_emitter("interpolate_nearest_2d");
    // Tier 4 partial
    (void)lucid::compile::find_emitter("unfold");
    (void)lucid::compile::find_emitter("fold");
    (void)lucid::compile::find_emitter("affine_grid");
    (void)lucid::compile::find_emitter("unfold_dim");
}
[[maybe_unused]] auto kEmitterAnchor = (touch_emitters(), 0);
}  // namespace

}  // namespace lucid::bindings
