// lucid/_C/autograd/TensorHooks.cpp
//
// The hook slots, the two helpers every traversal runs them through (one for
// Storage gradients, one for the tensors of a create_graph pass), and their
// registration.

#include "TensorHooks.h"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "AccumulateGrad.h"
#include "CustomFunction.h"
#include "Helpers.h"
#include "ModuleHookNode.h"

namespace lucid {

namespace {

// Grad mode set to ``on`` for a scope, then put back.
class GradModeScope {
public:
    explicit GradModeScope(bool on) : prev_(GradMode::is_enabled()) { GradMode::set_enabled(on); }
    ~GradModeScope() { GradMode::set_enabled(prev_); }

    GradModeScope(const GradModeScope&) = delete;
    GradModeScope& operator=(const GradModeScope&) = delete;

private:
    bool prev_;
};

constexpr const char* kHookResult = "a hook's returned gradient";

// Call the slot's runner on ``grad`` with grad mode ``graph`` — on for a
// create_graph pass, so what a hook computes joins the graph, off otherwise.
TensorImplPtr call_runner(const TensorHookSlot& slot, const TensorImplPtr& grad, bool graph) {
    py::gil_scoped_acquire gil;
    // A reference of its own: the hook may let go of the slot.
    const py::object runner = slot.runner;
    py::object result;
    {
        const GradModeScope mode(graph);
        result = runner(grad);
    }
    return tensor_from_python(result, "a tensor hook's result");
}

Device storage_device(const Storage& s) {
    return storage_is_gpu(s) ? Device::GPU : Device::CPU;
}

// The one place an eager gradient goes through a runner.
Storage run_hooks(const TensorHookSlot& slot, Storage grad, const Shape& shape) {
    const Dtype dtype = storage_dtype(grad);
    const Device device = storage_device(grad);
    auto given = std::make_shared<TensorImpl>(std::move(grad), shape, dtype, device, false);
    const TensorImplPtr returned = call_runner(slot, given, /*graph=*/false);
    Storage out;
    if (!returned || returned == given) {
        // ``None``: the gradient as the hook left it, an in-place write into
        // the tensor it was handed included.
        out = given->storage();
    } else {
        const ErrorBuilder err("Tensor.register_hook");
        check_grad_kind(err, kHookResult, dtype, device, shape, *returned);
        out = grad_slot_buffer(err, *returned);
    }
    return own_grad_copy(out);
}

// The one place a create_graph gradient goes through a runner.
TensorImplPtr run_hooks_for_graph(const TensorHookSlot& slot, TensorImplPtr grad) {
    const TensorImplPtr returned = call_runner(slot, grad, /*graph=*/true);
    if (!returned || returned == grad)
        return grad;
    check_grad_kind(ErrorBuilder("Tensor.register_hook"), kHookResult, grad->dtype(),
                    grad->device(), grad->shape(), *returned);
    return returned;
}

// The tensor that asked ``producer``'s output ``output_nr`` to keep its
// gradient, if it is alive and still sits there.  An in-place op moves a
// tensor to a new slot and registers it there; the old slot's entry then
// stops matching.
TensorImplPtr
retaining_tensor(const TensorHookSlot& slot, const Node& producer, std::uint32_t output_nr) {
    TensorImplPtr t = slot.retain.lock();
    if (!t || !t->retains_grad() || t->is_leaf() || t->grad_fn().get() != &producer ||
        t->grad_output_nr() != output_nr)
        return nullptr;
    return t;
}

}  // namespace

// ── Slots ────────────────────────────────────────────────────────────────────

TensorHookSlot::~TensorHookSlot() {
    clear();
}

void TensorHookSlot::clear() noexcept {
    retain.reset();
    release_py_object(runner);
}

std::shared_ptr<TensorHookSlot> NodeHooks::find(std::uint32_t output_nr) const {
    return output_nr < slots_.size() ? slots_[output_nr] : nullptr;
}

std::shared_ptr<TensorHookSlot> NodeHooks::ensure(std::uint32_t output_nr) {
    if (output_nr >= slots_.size())
        slots_.resize(static_cast<std::size_t>(output_nr) + 1);
    auto& slot = slots_[output_nr];
    if (!slot)
        slot = std::make_shared<TensorHookSlot>();
    return slot;
}

// ── Running ──────────────────────────────────────────────────────────────────

Storage run_slot_hooks(Node& producer, std::uint32_t output_nr, Storage grad) {
    const NodeHooks* hooks = producer.tensor_hooks();
    if (hooks == nullptr)
        return grad;
    // Held for the whole call: the hook may release or grow the node's slots.
    const std::shared_ptr<TensorHookSlot> slot = hooks->find(output_nr);
    if (!slot)
        return grad;
    if (slot->runner)
        grad = run_hooks(*slot, std::move(grad), slot->shape);
    if (const TensorImplPtr t = retaining_tensor(*slot, producer, output_nr)) {
        if (t->grad_as_impl()) {
            auto g_impl = std::make_shared<TensorImpl>(own_grad_copy(grad), t->shape(), t->dtype(),
                                                       t->device(), false);
            t->accumulate_grad_impl(std::move(g_impl));
        } else {
            // A buffer of its own: what flows on is added into in place.
            auto& kept = t->mutable_grad_storage();
            if (!kept.has_value())
                kept = own_grad_copy(grad);
            else
                accumulate_into(*kept, grad);
        }
    }
    return grad;
}

TensorImplPtr
run_slot_hooks_for_graph(Node& producer, std::uint32_t output_nr, TensorImplPtr grad) {
    const NodeHooks* hooks = producer.tensor_hooks();
    if (hooks == nullptr || !grad)
        return grad;
    const std::shared_ptr<TensorHookSlot> slot = hooks->find(output_nr);
    if (!slot)
        return grad;
    if (slot->runner)
        grad = run_hooks_for_graph(*slot, std::move(grad));
    // Summed by add_op, so the retained gradient carries its graph as a
    // leaf's does.
    if (const TensorImplPtr t = retaining_tensor(*slot, producer, output_nr))
        t->accumulate_grad_impl(grad);
    return grad;
}

Storage run_leaf_hooks(const TensorImpl& leaf, Storage grad) {
    const std::shared_ptr<TensorHookSlot> slot = leaf.leaf_hooks();
    if (!slot || !slot->runner)
        return grad;
    return run_hooks(*slot, std::move(grad), leaf.shape());
}

TensorImplPtr run_leaf_hooks_for_graph(const TensorImpl& leaf, TensorImplPtr grad) {
    const std::shared_ptr<TensorHookSlot> slot = leaf.leaf_hooks();
    if (!slot || !slot->runner || !grad)
        return grad;
    return run_hooks_for_graph(*slot, std::move(grad));
}

// ── Registering ──────────────────────────────────────────────────────────────

py::object tensor_hook_runner(const TensorImplPtr& t, const py::object& make) {
    const ErrorBuilder err("Tensor.register_hook");
    if (!t)
        err.invalid_argument("expected a tensor");
    if (!t->requires_grad())
        err.fail("cannot register a hook on a tensor that does not require grad");
    std::shared_ptr<TensorHookSlot> slot;
    if (t->is_leaf()) {
        auto& own = t->mutable_leaf_hooks();
        if (!own)
            own = std::make_shared<TensorHookSlot>();
        slot = own;
    } else {
        if (!t->grad_fn())
            err.fail("a non-leaf tensor without a grad_fn has no gradient to hook");
        slot = t->grad_fn()->ensure_tensor_hooks().ensure(t->grad_output_nr());
    }
    if (!slot->runner) {
        py::object runner = make();
        if (runner.is_none())
            err.invalid_argument("the runner factory returned None");
        slot->runner = std::move(runner);
        slot->shape = t->shape();
    }
    return slot->runner;
}

bool has_tensor_hooks(const TensorImpl& t) {
    if (t.is_leaf()) {
        const auto& slot = t.leaf_hooks();
        return slot && slot->runner;
    }
    const auto& fn = t.grad_fn();
    if (!fn || fn->tensor_hooks() == nullptr)
        return false;
    const auto slot = fn->tensor_hooks()->find(t.grad_output_nr());
    return slot && slot->runner;
}

void retain_grad(const TensorImplPtr& t) {
    if (!t)
        return;
    t->set_retain_grad(true);
    if (t->is_leaf() || !t->grad_fn())
        return;
    const auto slot = t->grad_fn()->ensure_tensor_hooks().ensure(t->grad_output_nr());
    slot->retain = t;
    slot->shape = t->shape();
}

// ── Gradients handed back by Python ──────────────────────────────────────────

TensorImplPtr tensor_from_python(py::handle obj, const std::string& what) {
    if (obj.is_none())
        return nullptr;
    if (py::isinstance<TensorImpl>(obj))
        return obj.cast<TensorImplPtr>();
    if (py::hasattr(obj, "impl")) {
        const py::object impl = obj.attr("impl");
        if (py::isinstance<TensorImpl>(impl))
            return impl.cast<TensorImplPtr>();
    }
    throw py::type_error(what + " must be a tensor or None, got " +
                         std::string(py::str(py::type::of(obj).attr("__name__"))));
}

void check_grad_kind(const ErrorBuilder& err,
                     const std::string& what,
                     Dtype dtype,
                     Device device,
                     const Shape& shape,
                     const TensorImpl& g) {
    if (g.dtype() != dtype)
        err.dtype_mismatch(dtype, g.dtype(), what + " must have the tensor's dtype");
    if (g.device() != device)
        err.device_mismatch(device, g.device(), what + " must be on the tensor's device");
    if (g.shape() != shape)
        err.shape_mismatch(shape, g.shape(), what + " must have the tensor's shape");
}

Storage grad_slot_buffer(const ErrorBuilder& err, const TensorImpl& g) {
    if (!storage_is_cpu(g.raw_storage()) && (g.storage_offset() != 0 || !g.is_contiguous()))
        err.not_implemented("a Metal gradient that is a strided view of another tensor's array — "
                            "pass a contiguous copy");
    return g.storage();
}

// ── Shared ───────────────────────────────────────────────────────────────────

Storage own_grad_copy(const Storage& s) {
    if (!storage_is_cpu(s))
        return s;
    const Dtype dt = storage_dtype(s);
    return clone_storage(s, storage_nbytes(s) / dtype_size(dt), dt, Device::CPU);
}

Storage present_gradient(Storage s) {
    auto* cpu = std::get_if<CpuStorage>(&s);
    if (cpu == nullptr || cpu->ptr || cpu->nbytes != 0)
        return s;
    static const std::shared_ptr<std::byte[]> kPlaceholder(new std::byte[1]{});
    cpu->ptr = kPlaceholder;
    return s;
}

// Every write of a whole gradient lands here or in a hook's replacement:
// ``.grad =``, clip_grad and GradScaler writing back.  The slot is a bare
// Storage — no dtype, shape or device of its own — and ``.grad``, the
// optimizers and the next backward read it as ``self``'s.  A buffer of any
// other kind was read as if it were one: float32 bits taken for float16
// values (``[1, 2, 4]`` came back ``[0, 1.875, 1.875]``), three elements read
// out of a buffer of two, a Metal array labelled CPU.  The reference refuses
// each of these, in this order, and so does this, before the slot changes.
Storage assignable_grad(const TensorImpl& self, const TensorImpl& g) {
    const ErrorBuilder err("Tensor.grad");
    if (&g == &self)
        err.fail("a tensor cannot be assigned as its own gradient");
    check_grad_kind(err, "an assigned gradient", self.dtype(), self.device(), self.shape(), g);
    return grad_slot_buffer(err, g);
}

// ── Python reference lifetime & cyclic garbage collection ────────────────────

namespace {

// During CPython's ``subtract_refs`` pass of ``gc.collect()``, ``tp_traverse``
// is invoked with ``arg == self`` on each tracked object in the collected
// generation.  Multiple live ``TensorImpl`` (or ``Node``) wrappers in the same
// generation can reach the same shared C++ ``Node`` / ``ModuleBackwardHookState``
// / ``TensorHookSlot``, whereas each C++ ``py::object`` member only owns a
// single ``Py_INCREF`` on its underlying ``PyObject*``.  Visiting the same C++
// ``py::object`` slot more than once during a single ``subtract_refs`` pass
// would subtract from ``gc_refs`` multiple times and underflow it in Python
// 3.14 (where ``_gc_prev`` is unsigned), making unreachable cycles appear
// reachable.
//
// We therefore deduplicate by ``const py::object*`` slot address across a
// single ``subtract_refs`` pass (``arg == self``), and reset the pass state as
// soon as ``move_unreachable`` (``arg != self``), ``tp_clear``, a new
// ``subtract_refs`` pass (seeing the same ``self`` again), or any C++ slot
// release runs.
struct GcDecrefPassState {
    std::unordered_set<PyObject*> seen_selves;
    std::unordered_set<const py::object*> seen_slots;

    void reset() noexcept {
        seen_selves.clear();
        seen_slots.clear();
    }
};

GcDecrefPassState& gc_decref_state() {
    static GcDecrefPassState state;
    return state;
}

void reset_gc_decref_pass() noexcept {
    gc_decref_state().reset();
}

template <typename T>
T* try_get_cpp_ptr(PyObject* self) noexcept {
    if (self == nullptr)
        return nullptr;
    auto* tinfo = py::detail::get_type_info(typeid(T));
    if (tinfo == nullptr)
        return nullptr;
    PyTypeObject* type = Py_TYPE(self);
    if (type != tinfo->type && !PyType_IsSubtype(type, tinfo->type))
        return nullptr;
    auto* inst = reinterpret_cast<py::detail::instance*>(self);
    auto vh = inst->get_value_and_holder(tinfo, /*throw_if_missing=*/false);
    if (!vh || !vh.holder_constructed())
        return nullptr;
    return vh.value_ptr<T>();
}

struct AutogradGcVisitor {
    visitproc visit;
    void* arg;
    bool is_decref_pass;
    std::unordered_set<const TensorImpl*> visited_tensors;
    std::unordered_set<const Node*> visited_nodes;
    std::unordered_set<const ModuleBackwardHookState*> visited_states;
    std::unordered_set<const TensorHookSlot*> visited_slots;
    std::unordered_set<const FunctionCtx*> visited_ctxs;
    std::vector<const TensorImpl*> tensor_stack;
    std::vector<const Node*> node_stack;

    AutogradGcVisitor(PyObject* self, visitproc v, void* a)
        : visit(v), arg(a), is_decref_pass(a != nullptr && a == static_cast<void*>(self)) {
        auto& state = gc_decref_state();
        if (is_decref_pass) {
            if (!state.seen_selves.insert(self).second) {
                state.reset();
                state.seen_selves.insert(self);
            }
        } else {
            state.reset();
        }
    }

    int visit_slot(const py::object& obj) {
        PyObject* ptr = obj.ptr();
        if (ptr == nullptr)
            return 0;
        if (is_decref_pass) {
            auto& state = gc_decref_state();
            if (!state.seen_slots.insert(&obj).second)
                return 0;
        }
        Py_VISIT(ptr);
        return 0;
    }

    void push_tensor(const TensorImpl* t) {
        if (t != nullptr && visited_tensors.insert(t).second)
            tensor_stack.push_back(t);
    }

    void push_node(const Node* n) {
        if (n != nullptr && visited_nodes.insert(n).second)
            node_stack.push_back(n);
    }

    int traverse_hook_slot(const TensorHookSlot* slot) {
        if (slot == nullptr || !visited_slots.insert(slot).second)
            return 0;
        return visit_slot(slot->runner);
    }

    int traverse_module_state(const ModuleBackwardHookState* state) {
        if (state == nullptr || !visited_states.insert(state).second)
            return 0;
        if (int r = visit_slot(state->pre_runner))
            return r;
        if (int r = visit_slot(state->full_runner))
            return r;
        for (const auto& g : state->grad_inputs)
            push_tensor(g.get());
        for (const auto& g : state->grad_outputs)
            push_tensor(g.get());
        return 0;
    }

    int traverse_function_ctx(const FunctionCtx* ctx) {
        if (ctx == nullptr || !visited_ctxs.insert(ctx).second)
            return 0;
        for (const auto& t : ctx->saved_tensors())
            push_tensor(t.get());
        for (const auto& kv : ctx->extras()) {
            if (int r = visit_slot(kv.second))
                return r;
        }
        return 0;
    }

    int step_tensor(const TensorImpl* t) {
        if (const auto& slot = t->leaf_hooks()) {
            if (int r = traverse_hook_slot(slot.get()))
                return r;
        }
        if (const auto& gi = t->grad_as_impl())
            push_tensor(gi.get());
        if (const auto& fn = t->grad_fn())
            push_node(fn.get());
        return 0;
    }

    int step_node_hooks(const NodeHooks* hooks) {
        if (hooks == nullptr)
            return 0;
        for (const auto& slot : hooks->slots()) {
            if (int r = traverse_hook_slot(slot.get()))
                return r;
        }
        return 0;
    }

    int step_python_node(const PythonBackwardNode* py_node) {
        if (int r = visit_slot(py_node->py_ctx))
            return r;
        if (int r = visit_slot(py_node->py_backward_fn))
            return r;
        if (const auto* ctx = try_get_cpp_ptr<FunctionCtx>(py_node->py_ctx.ptr()))
            return traverse_function_ctx(ctx);
        return 0;
    }

    int step_node_kind(const Node* n) {
        if (const auto* out_hook = dynamic_cast<const ModuleOutputHookNode*>(n))
            return traverse_module_state(out_hook->state().get());
        if (const auto* in_hook = dynamic_cast<const ModuleInputHookNode*>(n))
            return traverse_module_state(in_hook->state().get());
        if (const auto* py_node = dynamic_cast<const PythonBackwardNode*>(n))
            return step_python_node(py_node);
        if (const auto* acc = dynamic_cast<const AccumulateGrad*>(n)) {
            if (const TensorImplPtr leaf = acc->leaf().lock())
                push_tensor(leaf.get());
        }
        return 0;
    }

    int step_node(const Node* n) {
        if (int r = step_node_hooks(n->tensor_hooks()))
            return r;
        if (int r = step_node_kind(n))
            return r;
        for (const auto& edge : n->next_edges()) {
            if (edge.node)
                push_node(edge.node.get());
        }
        return 0;
    }

    int drain() {
        while (!tensor_stack.empty() || !node_stack.empty()) {
            while (!tensor_stack.empty()) {
                const TensorImpl* t = tensor_stack.back();
                tensor_stack.pop_back();
                if (int r = step_tensor(t))
                    return r;
            }
            while (!node_stack.empty()) {
                const Node* n = node_stack.back();
                node_stack.pop_back();
                if (int r = step_node(n))
                    return r;
            }
        }
        return 0;
    }
};

extern "C" int tensor_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    AutogradGcVisitor visitor(self, visit, arg);
    if (const auto* t = try_get_cpp_ptr<TensorImpl>(self)) {
        visitor.push_tensor(t);
        if (int r = visitor.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int tensor_tp_clear(PyObject* self) {
    reset_gc_decref_pass();
    if (auto* t = try_get_cpp_ptr<TensorImpl>(self)) {
        std::shared_ptr<Node> fn = t->grad_fn();
        std::shared_ptr<TensorImpl> gi = t->grad_as_impl();
        std::shared_ptr<TensorHookSlot> lh = t->leaf_hooks();
        t->clear_grad_fn();
        t->set_grad_impl(nullptr);
        t->mutable_leaf_hooks().reset();
        if (lh)
            lh->clear();
    }
    return 0;
}

extern "C" int node_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    AutogradGcVisitor visitor(self, visit, arg);
    if (const auto* n = try_get_cpp_ptr<Node>(self)) {
        visitor.push_node(n);
        if (int r = visitor.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int node_tp_clear(PyObject* self) {
    reset_gc_decref_pass();
    if (auto* n = try_get_cpp_ptr<Node>(self)) {
        n->release_tensor_hooks();
        n->release_saved();
    }
    return 0;
}

extern "C" int function_ctx_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    AutogradGcVisitor visitor(self, visit, arg);
    if (const auto* ctx = try_get_cpp_ptr<FunctionCtx>(self)) {
        if (int r = visitor.traverse_function_ctx(ctx))
            return r;
        if (int r = visitor.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int function_ctx_tp_clear(PyObject* self) {
    reset_gc_decref_pass();
    if (auto* ctx = try_get_cpp_ptr<FunctionCtx>(self))
        ctx->clear();
    return 0;
}

extern "C" int module_hook_state_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    AutogradGcVisitor visitor(self, visit, arg);
    if (const auto* state = try_get_cpp_ptr<ModuleBackwardHookState>(self)) {
        if (int r = visitor.traverse_module_state(state))
            return r;
        if (int r = visitor.drain())
            return r;
    }
    Py_VISIT(Py_TYPE(self));
    return 0;
}

extern "C" int module_hook_state_tp_clear(PyObject* self) {
    reset_gc_decref_pass();
    if (auto* state = try_get_cpp_ptr<ModuleBackwardHookState>(self))
        state->clear();
    return 0;
}

void install_gc_hooks(PyHeapTypeObject* heap_type, traverseproc traverse, inquiry clear) {
    auto* type = &heap_type->ht_type;
    type->tp_flags |= Py_TPFLAGS_HAVE_GC;
    type->tp_traverse = traverse;
    type->tp_clear = clear;
}

}  // namespace

void release_py_object(py::object& obj) noexcept {
    PyObject* ptr = obj.release().ptr();
    if (ptr == nullptr)
        return;
    // At interpreter shutdown there is nobody to hand the reference back to.
    if (!Py_IsInitialized() || Py_IsFinalizing())
        return;
    const PyGILState_STATE state = PyGILState_Ensure();
    reset_gc_decref_pass();
    Py_DECREF(ptr);
    PyGILState_Release(state);
}

void enable_tensor_autograd_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, tensor_tp_traverse, tensor_tp_clear);
}

void enable_node_autograd_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, node_tp_traverse, node_tp_clear);
}

void enable_function_ctx_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, function_ctx_tp_traverse, function_ctx_tp_clear);
}

void enable_module_hook_state_gc(PyHeapTypeObject* heap_type) {
    install_gc_hooks(heap_type, module_hook_state_tp_traverse, module_hook_state_tp_clear);
}

}  // namespace lucid
