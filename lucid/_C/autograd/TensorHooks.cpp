// lucid/_C/autograd/TensorHooks.cpp
//
// The hook slots, the two helpers every traversal runs them through (one for
// Storage gradients, one for the tensors of a create_graph pass), and their
// registration.

#include "TensorHooks.h"

#include <string>
#include <utility>

#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "Helpers.h"

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

// ``g``'s dtype, device and shape against the gradient it stands for, in the
// order — and with the errors — ``Tensor.grad =`` checks them.
void check_kind(const ErrorBuilder& err,
                const char* what,
                Dtype dtype,
                Device device,
                const Shape& shape,
                const TensorImpl& g) {
    const std::string subject(what);
    if (g.dtype() != dtype)
        err.dtype_mismatch(dtype, g.dtype(), subject + " must have the tensor's dtype");
    if (g.device() != device)
        err.device_mismatch(device, g.device(), subject + " must be on the tensor's device");
    if (g.shape() != shape)
        err.shape_mismatch(shape, g.shape(), subject + " must have the tensor's shape");
}

// ``g``'s values as a buffer read from its first byte in row-major order —
// the form a gradient slot holds.  A CPU view answers storage() with a packed
// copy of the elements it reads; a Metal tensor's array is its elements,
// unless it is a window at an offset or with strides into another tensor's
// array, which only the private ``_make_view`` builds and which has no such
// buffer to hand over.
Storage slot_buffer(const ErrorBuilder& err, const TensorImpl& g) {
    if (!storage_is_cpu(g.raw_storage()) && (g.storage_offset() != 0 || !g.is_contiguous()))
        err.not_implemented("a Metal gradient that is a strided view of another tensor's array — "
                            "pass a contiguous copy");
    return g.storage();
}

constexpr const char* kHookResult = "a hook's returned gradient";

// The tensor a runner handed back, or null for ``None``.
TensorImplPtr hook_result(const py::object& result) {
    if (result.is_none())
        return nullptr;
    if (py::isinstance<TensorImpl>(result))
        return result.cast<TensorImplPtr>();
    if (py::hasattr(result, "impl")) {
        py::object impl = result.attr("impl");
        if (py::isinstance<TensorImpl>(impl))
            return impl.cast<TensorImplPtr>();
    }
    throw py::type_error("a tensor hook must return a tensor or None, got " +
                         std::string(py::str(py::type::of(result).attr("__name__"))));
}

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
    return hook_result(result);
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
        check_kind(err, kHookResult, dtype, device, shape, *returned);
        out = slot_buffer(err, *returned);
    }
    return own_grad_copy(out);
}

// The one place a create_graph gradient goes through a runner.
TensorImplPtr run_hooks_for_graph(const TensorHookSlot& slot, TensorImplPtr grad) {
    const TensorImplPtr returned = call_runner(slot, grad, /*graph=*/true);
    if (!returned || returned == grad)
        return grad;
    check_kind(ErrorBuilder("Tensor.register_hook"), kHookResult, grad->dtype(), grad->device(),
               grad->shape(), *returned);
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
    PyObject* obj = runner.release().ptr();
    if (obj == nullptr)
        return;
    // At interpreter shutdown there is nobody to hand the reference back to.
    if (!Py_IsInitialized() || Py_IsFinalizing())
        return;
    const PyGILState_STATE state = PyGILState_Ensure();
    Py_DECREF(obj);
    PyGILState_Release(state);
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
        // A buffer of its own: what flows on is added into in place.
        auto& kept = t->mutable_grad_storage();
        if (!kept.has_value())
            kept = own_grad_copy(grad);
        else
            accumulate_into(*kept, grad);
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

// ── Shared ───────────────────────────────────────────────────────────────────

Storage own_grad_copy(const Storage& s) {
    if (!storage_is_cpu(s))
        return s;
    const Dtype dt = storage_dtype(s);
    return clone_storage(s, storage_nbytes(s) / dtype_size(dt), dt, Device::CPU);
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
    check_kind(err, "an assigned gradient", self.dtype(), self.device(), self.shape(), g);
    return slot_buffer(err, g);
}

}  // namespace lucid
