// lucid/_C/compile/VjpEmitters/VjpEmitter.mm
//
// Registry + BackwardContext + BackwardWalker implementation.
// MPSGraph-typed since :func:`BackwardContext::accumulate_grad` and
// :func:`BackwardContext::unreduce` emit MPSGraph ops directly.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <utility>

#include "../../core/Dtype.h"
#include "VjpEmitter.h"
#include "_VjpHelpers.h"

namespace lucid::compile {

// ────────────────────────────────────────────────────────────────────
// Registry
// ────────────────────────────────────────────────────────────────────

namespace {

struct VjpRegistry {
    std::mutex mu;
    std::unordered_map<std::string, std::unique_ptr<VjpEmitter>> map;
};

VjpRegistry& vjp_registry() {
    static VjpRegistry* instance = new VjpRegistry();
    return *instance;
}

// Trace ops that have no meaningful gradient (factories, integer-only
// shape ops, eager-fallback stubs).  The walker treats them as
// grad-sinks: it does not require a VJP emitter for them and does not
// propagate grads through their inputs.
const std::unordered_set<std::string>& no_grad_ops() {
    static const std::unordered_set<std::string> s{
        // factories — no inputs, just produce constants
        "zeros",        "ones",       "full",       "arange",
        "linspace",     "eye",        "empty",      "randn",
        "rand",         "randint",    "uniform",    "normal",
        // shape-only / integer ops that don't need gradients
        "cast_i8",      "cast_i16",   "cast_i32",   "cast_i64",
        "cast_u8",      "cast_b8",
        // boolean comparison ops — produce bool tensors that are
        // never on the backward differentiation path; treat as
        // grad-sinks so the walker doesn't error out looking for a
        // VJP that wouldn't make sense anyway.
        "equal",        "not_equal",  "less",       "less_equal",
        "greater",      "greater_equal", "logical_and", "logical_or",
        "logical_not",  "logical_xor", "isnan",      "isinf",
        "isfinite",
        // argmax / argmin produce integer indices — also grad-sinks.
        "argmax",       "argmin",
    };
    return s;
}

}  // namespace

void register_vjp_emitter(std::unique_ptr<VjpEmitter> vjp) {
    if (!vjp) return;
    auto& r = vjp_registry();
    std::lock_guard<std::mutex> lk(r.mu);
    std::string key{vjp->op_name()};
    r.map[std::move(key)] = std::move(vjp);
}

VjpEmitter* find_vjp_emitter(std::string_view op_name) {
    auto& r = vjp_registry();
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.map.find(std::string(op_name));
    return it == r.map.end() ? nullptr : it->second.get();
}

// ────────────────────────────────────────────────────────────────────
// Env-var gates
// ────────────────────────────────────────────────────────────────────

namespace {
bool env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    if (*v == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
}
}  // namespace

bool use_manual_vjp() {
    // Production default = ON.  Manual VJP runs first for every op
    // we have an emitter for; coverage gaps soft-fall through to
    // ``gradientForPrimaryTensor:`` (the legacy path) so behaviour
    // stays at-least-as-good as the pre-manual-VJP baseline.
    //
    // Explicit opt-out: ``LUCID_MANUAL_VJP=0``.  Set this when you
    // need bit-exact reproduction of a prior eager-autograd baseline
    // or are debugging a manual-VJP regression.
    const char* v = std::getenv("LUCID_MANUAL_VJP");
    if (v == nullptr) return true;       // default ON
    return env_truthy("LUCID_MANUAL_VJP");
}

bool use_manual_vjp_require() {
    return env_truthy("LUCID_MANUAL_VJP_REQUIRE");
}

bool use_manual_vjp_debug() {
    return env_truthy("LUCID_MANUAL_VJP_DEBUG");
}

VjpRegistration vjp_registration_status(const std::string& op_name) {
    {
        auto& r = vjp_registry();
        std::lock_guard<std::mutex> lk(r.mu);
        if (r.map.find(op_name) != r.map.end())
            return VjpRegistration::Registered;
    }
    const auto& sinks = no_grad_ops();
    if (sinks.find(op_name) != sinks.end())
        return VjpRegistration::GradSink;
    return VjpRegistration::Missing;
}

// ────────────────────────────────────────────────────────────────────
// Error formatting helpers — shared by the walker's failure paths so
// debug output stays consistent with the propagated error string.
// ────────────────────────────────────────────────────────────────────

namespace {

std::string format_shape(const std::vector<std::int64_t>& shape) {
    if (shape.empty()) return "()";
    std::ostringstream os;
    os << '(';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i + 1 < shape.size()) os << ", ";
    }
    os << ')';
    return os.str();
}

std::string format_node_signature(const TraceGraph& trace, const OpNode& node) {
    std::ostringstream os;
    os << "op='" << node.name << "'";
    os << " inputs=[";
    for (std::size_t i = 0; i < node.inputs.size(); ++i) {
        const TensorId tid = node.inputs[i];
        os << format_shape(shape_of_tid(trace, tid))
           << ':' << std::string(dtype_name(dtype_of_tid(trace, tid)));
        if (i + 1 < node.inputs.size()) os << ", ";
    }
    os << "] outputs=[";
    for (std::size_t i = 0; i < node.outputs.size(); ++i) {
        os << format_shape(node.outputs[i].shape)
           << ':' << std::string(dtype_name(node.outputs[i].dtype));
        if (i + 1 < node.outputs.size()) os << ", ";
    }
    os << "]";
    return os.str();
}

void debug_log_failure(const std::string& reason, const std::string& signature) {
    if (!use_manual_vjp_debug()) return;
    std::cerr << "[lucid.compile manual_vjp] " << reason << '\n'
              << "[lucid.compile manual_vjp]   " << signature << '\n'
              << "[lucid.compile manual_vjp]   verdict: "
              << (use_manual_vjp_require() ? "HARD-FAIL (LUCID_MANUAL_VJP_REQUIRE=1)"
                                           : "soft-fallback to gradientForPrimaryTensor:")
              << std::endl;
}

}  // namespace

ManualVjpStatus try_manual_vjp_grads(void* graph_void,
                                     BuilderContext& fwd,
                                     const TraceGraph& trace,
                                     TensorId loss_id,
                                     const std::vector<TensorId>& param_ids,
                                     std::vector<void*>& out_grads,
                                     std::string* error_msg) {
    if (!use_manual_vjp())
        return ManualVjpStatus::Disabled;
    BackwardWalker walker{graph_void, fwd, trace};
    std::string vjp_err;
    if (walker.compute_grads(loss_id, param_ids, out_grads, &vjp_err))
        return ManualVjpStatus::Success;
    if (error_msg)
        *error_msg = std::move(vjp_err);
    return use_manual_vjp_require() ? ManualVjpStatus::HardFail
                                    : ManualVjpStatus::FellBack;
}

// ────────────────────────────────────────────────────────────────────
// BackwardContext
// ────────────────────────────────────────────────────────────────────

BackwardContext::BackwardContext(void* graph_void, BuilderContext& fwd)
    : graph_(graph_void), fwd_(fwd) {}

void* BackwardContext::forward(TensorId tid) const {
    return fwd_.resolve(tid);
}

void* BackwardContext::resolve_grad(TensorId tid) const {
    auto it = grad_map_.find(tid);
    return it == grad_map_.end() ? nullptr : it->second;
}

void BackwardContext::accumulate_grad(TensorId tid, void* grad) {
    if (grad == nullptr) return;
    auto it = grad_map_.find(tid);
    if (it == grad_map_.end()) {
        grad_map_[tid] = grad;
        return;
    }
    // Sum existing contribution + new one.
    MPSGraph* g = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* a = (__bridge MPSGraphTensor*)it->second;
    MPSGraphTensor* b = (__bridge MPSGraphTensor*)grad;
    // Mixed-dtype reconciliation: under autocast, gradients from
    // different paths in the backward walk may have different dtypes
    // (e.g. residual + main path).  Cast b to match a's dtype so the
    // add succeeds — MPSGraph's additionWithPrimaryTensor: requires
    // matching dtypes and would otherwise fail with
    // ``'mps.add' op requires the same element type for all operands
    // and results``.  We pick a's dtype (the existing accumulator)
    // arbitrarily; either choice works as long as we're consistent.
    if (a != nil && b != nil && a.dataType != b.dataType) {
        b = [g castTensor:b toType:a.dataType name:@"grad_accum_cast"];
    }
    MPSGraphTensor* s =
        [g additionWithPrimaryTensor:a secondaryTensor:b name:@"grad_accum"];
    it->second = (__bridge void*)s;
}

void* BackwardContext::unreduce(void* grad,
                                const std::vector<std::int64_t>& target_shape,
                                const std::vector<std::int64_t>& broadcast_shape) {
    if (grad == nullptr) return nullptr;
    MPSGraph* g = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* t = (__bridge MPSGraphTensor*)grad;
    return (__bridge void*)unreduce_impl(g, t, broadcast_shape, target_shape);
}

// ────────────────────────────────────────────────────────────────────
// BackwardWalker
// ────────────────────────────────────────────────────────────────────

BackwardWalker::BackwardWalker(void* graph_void,
                               BuilderContext& fwd,
                               const TraceGraph& trace)
    : graph_(graph_void), trace_(trace), bctx_(graph_void, fwd) {}

bool BackwardWalker::compute_grads(TensorId loss_id,
                                   const std::vector<TensorId>& param_ids,
                                   std::vector<void*>& out_grads,
                                   std::string* error_msg) {
    auto set_err = [&](std::string msg) {
        if (error_msg) *error_msg = std::move(msg);
    };

    // 1) Seed the loss gradient: ∂loss/∂loss = ones(loss_shape).
    void* loss_void = bctx_.forward(loss_id);
    if (loss_void == nullptr) {
        set_err("manual_vjp: loss id " + std::to_string(loss_id) +
                " has no forward binding");
        return false;
    }
    auto loss_shape = shape_of_tid(trace_, loss_id);
    auto loss_dt = dtype_of_tid(trace_, loss_id);
    MPSGraph* graph = (__bridge MPSGraph*)graph_;
    MPSGraphTensor* loss_t = (__bridge MPSGraphTensor*)loss_void;
    // Fallback when the trace doesn't carry an explicit shape (rare):
    // use the live MPSGraph shape.
    if (loss_shape.empty())
        loss_shape = shape_of_mps(loss_t);
    MPSGraphTensor* loss_grad = ones_like_loss(graph, loss_shape, loss_dt);
    bctx_.accumulate_grad(loss_id, (__bridge void*)loss_grad);

    // 2) Reverse iterate ops.  Each iteration: pull grads on the op's
    //    outputs from bctx_, dispatch to the VJP, which writes input
    //    grads via :func:`accumulate_grad`.
    const auto& ops = trace_.ops;
    const auto& sink = no_grad_ops();
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        const OpNode& node = *it;

        // Collect grad demand on outputs.
        std::vector<void*> grad_outs;
        grad_outs.reserve(node.outputs.size());
        bool any_demand = false;
        for (const auto& om : node.outputs) {
            void* go = bctx_.resolve_grad(om.id);
            grad_outs.push_back(go);
            if (go != nullptr) any_demand = true;
        }
        if (!any_demand) continue;  // no path from loss touches this op

        // Grad-sink ops (factories, etc.) don't propagate.
        if (sink.find(node.name) != sink.end())
            continue;

        VjpEmitter* vjp = find_vjp_emitter(node.name);
        if (vjp == nullptr) {
            std::string sig = format_node_signature(trace_, node);
            std::string msg = "manual_vjp: no VJP emitter for op '" +
                              node.name + "' (" + sig + ")";
            debug_log_failure("no VJP emitter for op '" + node.name + "'", sig);
            set_err(std::move(msg));
            return false;
        }
        if (!vjp->emit(bctx_, node, grad_outs)) {
            std::string sig = format_node_signature(trace_, node);
            std::string msg = "manual_vjp: VJP for op '" + node.name +
                              "' returned false (" + sig + ")";
            debug_log_failure("VJP for op '" + node.name + "' returned false",
                              sig);
            set_err(std::move(msg));
            return false;
        }
    }

    // 3) Collect output grads for the requested params.
    out_grads.clear();
    out_grads.reserve(param_ids.size());
    for (TensorId pid : param_ids) {
        void* g = bctx_.resolve_grad(pid);
        if (g == nullptr) {
            set_err("manual_vjp: no gradient produced for param id " +
                    std::to_string(pid) +
                    " — most likely the param isn't on the loss path");
            return false;
        }
        out_grads.push_back(g);
    }
    return true;
}

}  // namespace lucid::compile
