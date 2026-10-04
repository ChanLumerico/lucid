// lucid/_C/autograd/ModuleHookNode.h
//
// Backward hook barriers used by nn.Module. They are only inserted when a
// module has backward hooks, keeping the normal autograd path untouched.

#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/TensorImpl.h"
#include "GraphBarrier.h"
#include "Node.h"

namespace py = pybind11;

namespace lucid {

// Minimal tensor metadata snapshot retained by the hook barriers.
//
// The barrier nodes capture only ``Shape`` / ``Dtype`` / ``Device`` at wrap
// time so that incoming ``Storage`` gradients can later be rehydrated into a
// proper :class:`TensorImpl` when crossing into Python user code (which
// expects Tensor objects, not raw buffers).  Keeping the data tensor itself
// out of the state avoids keeping the forward outputs alive any longer than
// the normal autograd path would.
//
// Attributes
// ----------
// shape : Shape
//     Logical shape of the represented tensor.
// dtype : Dtype, default Dtype::F32
//     Element type for the eventual reconstructed tensor.
// device : Device, default Device::CPU
//     Stream / placement of the tensor when handed back to Python.
struct ModuleHookTensorMeta {
    Shape shape;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;
};

// Shared state shuttled between the input-side and output-side hook barriers
// of a single :class:`nn.Module` invocation.
//
// A ``Module`` with registered backward hooks is wrapped at forward time with
// two barriers — :class:`ModuleOutputHookNode` on each output edge and
// :class:`ModuleInputHookNode` on each gradient-receiving input edge.  Both
// barriers hold a ``shared_ptr`` to the same ``ModuleBackwardHookState`` so
// that the gradients arriving at the outputs (from downstream nodes) are
// available to the input-side hook callbacks, and any replacement gradients
// returned by the Python full-backward hook flow back through the inputs.
//
// Notes
// -----
// The two ``py::object`` runners are the Python-side dispatchers that fan
// each barrier event out to every registered hook on the module.  They are
// held as opaque ``py::object`` (the GIL is acquired only when they are
// actually invoked inside ``apply_barrier``).
//
// Attributes
// ----------
// pre_runner : py::object
//     Python callable that runs the *pre*-backward hooks.  Receives the
//     output-gradient tuple, may return a tuple replacing it.
// full_runner : py::object
//     Python callable that runs the *full*-backward hooks.  Receives both
//     the input-gradient tuple and the output-gradient tuple, may return a
//     tuple replacing the input gradients.
// n_inputs : std::size_t
//     Number of original Python-side input positions on the module call
//     (some may have ``requires_grad=False`` and therefore no edge).
// n_outputs : std::size_t
//     Number of outputs the module produced.
// pre_hooks_ran : bool
//     One-shot guard so the pre-backward hooks fire exactly once per
//     backward pass even when the barrier is visited from multiple edges.
// full_hooks_ran : bool
//     Same guard for the full-backward hook batch.
// pass : std::uint64_t
//     The :class:`BackwardPass` whose gradients the state holds.
// input_arg_indices : std::vector<std::uint32_t>
//     Maps each edge slot in :attr:`grad_inputs` back to its original
//     positional argument index in the Python signature.
// input_metas : std::vector<ModuleHookTensorMeta>
//     Metadata for each gradient-receiving input edge, parallel to
//     :attr:`grad_inputs`.
// grad_inputs : std::vector<TensorImplPtr>
//     Per-edge gradient reaching each input, summed over every use the
//     module body made of it.  Null means no gradient arrived for that slot.
// output_metas : std::vector<ModuleHookTensorMeta>
//     Metadata for every output position (``size() == n_outputs``).
// grad_outputs : std::vector<TensorImplPtr>
//     Gradient reaching each output position, summed over every consumer
//     of that output.
// output_edge_indices : std::vector<std::uint32_t>
//     For each edge of the output barrier, the output-position index it
//     corresponds to in :attr:`grad_outputs`.
//
// A slot sums what reaches it, as the reference's input buffer does: a
// module whose output feeds two consumers, or whose body reads an input
// twice, receives one gradient per use, and the hooks — and everything
// upstream — must see their sum.  Overwriting kept only the last one.
//
// The gradients are held as tensors in both modes: under ``create_graph``
// they carry their graph, and the sums are graph ops.
class ModuleBackwardHookState : public std::enable_shared_from_this<ModuleBackwardHookState> {
public:
    // Construct fresh hook state for a module forward call.
    //
    // Parameters
    // ----------
    // n_inputs : std::size_t
    //     Number of positional inputs in the Python module call.
    // pre_runner : py::object
    //     Python callable that will fan pre-backward hooks at the output
    //     barrier.
    // full_runner : py::object
    //     Python callable that will fan full-backward hooks at the input
    //     barrier.
    ModuleBackwardHookState(std::size_t n_inputs, py::object pre_runner, py::object full_runner);

    // Discard what an earlier backward pass left behind, if the running pass
    // is a different one.  Called before every read or write of the slots.
    void enter_pass();

    // Forget every gradient and hook guard of the current pass — once the
    // module's backward is complete, so the next pass starts empty and fires
    // the hooks again.
    void reset();

    py::object pre_runner;
    py::object full_runner;
    std::size_t n_inputs = 0;
    std::size_t n_outputs = 0;
    bool pre_hooks_ran = false;
    bool full_hooks_ran = false;
    std::uint64_t pass = 0;

    std::vector<std::uint32_t> input_arg_indices;
    std::vector<ModuleHookTensorMeta> input_metas;
    std::vector<TensorImplPtr> grad_inputs;

    std::vector<ModuleHookTensorMeta> output_metas;
    std::vector<TensorImplPtr> grad_outputs;
    std::vector<std::uint32_t> output_edge_indices;
};

// Barrier autograd node inserted on each output of a hooked :class:`nn.Module`.
//
// Plays two roles during backward:
//
// 1. Receives output gradients from downstream nodes through
//    :func:`accumulate_barrier_grad`, parking them into the shared
//    :class:`ModuleBackwardHookState` rather than producing input gradients
//    on the fly.  This lets the engine reach the barrier from every
//    consumer before any hook fires.
// 2. When :func:`apply_barrier` runs, invokes the user-registered
//    pre-backward hooks (if any), giving them an opportunity to replace
//    the output gradients before they continue propagating back through
//    the module body.
//
// Notes
// -----
// Marked as a barrier (``is_barrier() == true``), which the engine uses to
// switch from the fast ``apply()`` path to the staged barrier protocol, and
// a :class:`GraphBarrier`, so ``create_graph`` delivers to it slot by slot
// too.  The node does not modify gradients unless a registered hook
// explicitly returns a replacement tuple.
//
// See Also
// --------
// :class:`ModuleInputHookNode` : sibling barrier on the input side.
// :class:`ModuleBackwardHookState` : shared state container.
class ModuleOutputHookNode : public Node, public GraphBarrier {
public:
    // Construct an output-side barrier bound to shared hook state.
    //
    // Parameters
    // ----------
    // state : std::shared_ptr<ModuleBackwardHookState>
    //     Shared state container that also backs the matching input
    //     barrier.
    explicit ModuleOutputHookNode(std::shared_ptr<ModuleBackwardHookState> state);

    // Convenience entry point used when the engine reaches this node via
    // the non-barrier fast path: forwards to :func:`accumulate_barrier_grad`
    // and then immediately runs :func:`apply_barrier`.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming output gradient for slot 0.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One ``Storage`` per outgoing edge, in ``output_edge_indices`` order.
    std::vector<Storage> apply(Storage grad_out) override;

    // Identifies this node as a staged barrier so the engine routes
    // gradients through the barrier protocol instead of the direct path.
    bool is_barrier() const noexcept override { return true; }
    // A hook's ``None``, or a slot no gradient reached, is an empty storage.
    bool empty_grad_is_none() const noexcept override { return true; }

    // Add an incoming gradient to what its output position holds.
    //
    // Parameters
    // ----------
    // input_nr : std::uint32_t
    //     Output position the gradient belongs to.  Out-of-range values
    //     are silently ignored to keep the engine robust against
    //     pruned graphs.
    // grad : Storage
    //     Incoming gradient buffer.  Never added into in place: it may be
    //     the very buffer another holder still reads.
    void accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) override;

    // Fire the pre-backward hooks and emit one ``Storage`` per outgoing
    // edge.  Called by the engine exactly once after all incoming
    // gradients have been accumulated.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Gradients to push along each outgoing edge.  Empty slots become
    //     empty :class:`CpuStorage` placeholders.
    std::vector<Storage> apply_barrier() override;

    // :meth:`accumulate_barrier_grad` for a ``create_graph`` pass: the sum is
    // an ``add_op``, so it stays differentiable.
    void accumulate_barrier_grad_for_graph(std::uint32_t input_nr, TensorImplPtr grad) override;

    // :meth:`apply_barrier` for a ``create_graph`` pass: the hooks see the
    // gradients with their graph, and a null entry is an edge no gradient
    // reached.
    std::vector<TensorImplPtr> apply_barrier_for_graph() override;

    // Graph label — ``"ModuleOutputHook"`` — for debug printing and profiler
    // traces.  Overrides :func:`Node::node_name`.
    std::string node_name() const override { return "ModuleOutputHook"; }

private:
    // Fire the pre-backward hooks — and the full ones when the module has no
    // input that takes a gradient — once per pass.
    void run_hooks();

    std::shared_ptr<ModuleBackwardHookState> state_;
};

// Barrier autograd node inserted on each gradient-bearing input of a hooked
// :class:`nn.Module`.
//
// Mirrors :class:`ModuleOutputHookNode` but on the other side of the
// module: it collects the gradients flowing back into the module's inputs
// and fires the *full* backward hooks (which receive both ``grad_input`` and
// ``grad_output`` tuples) before forwarding the (possibly replaced)
// gradients to the upstream graph.
//
// Notes
// -----
// Hooks may return ``None`` (keep current gradients) or a tuple matching
// the positional input signature with ``None`` entries for slots that
// should remain untouched.  Hooks may NOT change the dtype / shape of
// returned gradients — downstream nodes rely on the original metadata
// captured in :attr:`ModuleBackwardHookState::input_metas`.
//
// See Also
// --------
// :class:`ModuleOutputHookNode` : sibling barrier on the output side.
class ModuleInputHookNode : public Node, public GraphBarrier {
public:
    // Construct an input-side barrier bound to shared hook state.
    //
    // Parameters
    // ----------
    // state : std::shared_ptr<ModuleBackwardHookState>
    //     Shared state container that also backs the matching output
    //     barrier.
    explicit ModuleInputHookNode(std::shared_ptr<ModuleBackwardHookState> state);

    // Convenience non-barrier entry point — see
    // :func:`ModuleOutputHookNode::apply`.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Incoming gradient for input slot 0.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One ``Storage`` per outgoing edge.
    std::vector<Storage> apply(Storage grad_out) override;

    // Identifies this node as a staged barrier.
    bool is_barrier() const noexcept override { return true; }
    // A hook's ``None``, or a slot no gradient reached, is an empty storage.
    bool empty_grad_is_none() const noexcept override { return true; }

    // Add an incoming input gradient to what its slot holds.
    //
    // Parameters
    // ----------
    // input_nr : std::uint32_t
    //     Edge slot index (parallel to :attr:`ModuleBackwardHookState::grad_inputs`).
    //     Out-of-range values are ignored.
    // grad : Storage
    //     Incoming gradient buffer.  Never added into in place.
    void accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) override;

    // Fire the full-backward hooks and emit one gradient per outgoing edge.
    // The module's backward ends here, so the shared state is reset for the
    // next pass.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     One ``Storage`` per input edge in declaration order.  Hooks may
    //     have replaced individual entries via their return tuple; empty
    //     slots become empty :class:`CpuStorage` placeholders.
    std::vector<Storage> apply_barrier() override;

    // :meth:`accumulate_barrier_grad` for a ``create_graph`` pass.
    void accumulate_barrier_grad_for_graph(std::uint32_t input_nr, TensorImplPtr grad) override;

    // :meth:`apply_barrier` for a ``create_graph`` pass.
    std::vector<TensorImplPtr> apply_barrier_for_graph() override;

    // Graph label — ``"ModuleInputHook"`` — for debug printing and
    // profiler traces.  Overrides :func:`Node::node_name`.
    std::string node_name() const override { return "ModuleInputHook"; }

private:
    // Fire the full-backward hooks once per pass.
    void run_hooks();

    std::shared_ptr<ModuleBackwardHookState> state_;
};

// Register :class:`ModuleBackwardHookState` and the wrapper helpers
// (``_create_module_backward_hook_state``, ``_wrap_module_backward_inputs``,
// ``_wrap_module_backward_outputs``) on a pybind11 module.
//
// Parameters
// ----------
// m : py::module_&
//     Target pybind11 module — typically the ``engine`` extension module
//     loaded via ``from lucid._C import engine as _C_engine``.
void register_module_hook_nodes(py::module_& m);

}  // namespace lucid
