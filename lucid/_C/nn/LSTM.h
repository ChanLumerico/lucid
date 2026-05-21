// lucid/_C/nn/LSTM.h
//
// Autograd-aware single-layer LSTM.
//
// Inference path (no grad required): delegates to IBackend::lstm_forward.
// On CPU this routes through lstm_forward_train and discards the saved
// gates/cells; other backends may provide a dedicated inference kernel.
//
// Training path: delegates to IBackend::lstm_forward_train, which uses
// hand-rolled BLAS and returns two extra Storage tensors:
//   gates_all  – shape (T, B, 4H), the pre-activation gate values for all
//                time steps; gate order is [i, f, g, o].
//   cells_all  – shape (T+1, B, H), the cell states c_0 through c_T.
// These are consumed by the BPTT backward in IBackend::lstm_backward.
//
// LstmBackward inherits directly from Node (not FuncOp) because the 7-edge
// topology {input, h0, c0, wih, whh, bih, bhh} is built manually rather than
// through the generic NaryKernel machinery.  Only the output hidden sequence
// (res[0]) carries the grad_fn; hn and cn are detached.

#pragma once

#include <vector>

#include "../api.h"
#include "../autograd/Node.h"
#include "../backend/IBackend.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Storage.h"
#include "../core/fwd.h"

namespace lucid {

// Autograd node for the single-layer LSTM with BPTT backward.
//
// Implements the standard Hochreiter & Schmidhuber (1997) LSTM gating
// equations:
// $$
//   \begin{aligned}
//     i_t &= \sigma(W_{ii}\,x_t + b_{ii} + W_{hi}\,h_{t-1} + b_{hi}) \\
//     f_t &= \sigma(W_{if}\,x_t + b_{if} + W_{hf}\,h_{t-1} + b_{hf}) \\
//     g_t &= \tanh(W_{ig}\,x_t + b_{ig} + W_{hg}\,h_{t-1} + b_{hg}) \\
//     o_t &= \sigma(W_{io}\,x_t + b_{io} + W_{ho}\,h_{t-1} + b_{ho}) \\
//     c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
//     h_t &= o_t \odot \tanh(c_t)
//   \end{aligned}
// $$
// where $\sigma$ is the logistic sigmoid, $\odot$ is the element-wise
// (Hadamard) product, and the four gates $i, f, g, o$ are stored in
// that exact concatenation order along the leading axis of
// ``gates_all`` (i.e. ``gates_all[..., 0:H]`` is the input-gate
// pre-activation, ``gates_all[..., H:2H]`` the forget gate, and so on).
//
// Inherits directly from ``Node`` rather than ``FuncOp`` because the
// 7-edge topology ``{input, h0, c0, wih, whh, bih, bhh}`` is built by
// hand: each tensor that requires a gradient gets an explicit
// ``AccumulateGrad`` leaf wired in ``forward()``, and non-differentiable
// tensors (such as biases when ``has_bias=false``) get a null edge.
// Only the output sequence ``res[0]`` carries this node as its
// ``grad_fn``; ``hn`` and ``cn`` are detached so they can be reused as
// initial states for a subsequent stacked layer in pure Python without
// double-counting gradients.
//
// Math
// ----
// The BPTT backward unrolls $T$ time steps in reverse, accumulating the
// gradient contributions through both the cell-state path
// $\partial c_t / \partial c_{t-1} = f_t$ and the gate path
// $\partial c_t / \partial \{i_t, f_t, g_t\}$, then maps cell-state
// gradients back through the sigmoid / tanh derivatives to give
// per-time-step weight gradients $\partial \mathcal{L}/\partial W_{ih}$,
// $\partial \mathcal{L}/\partial W_{hh}$, $\partial \mathcal{L}/\partial b_{ih}$,
// $\partial \mathcal{L}/\partial b_{hh}$.  The full unrolled BLAS path
// lives in ``IBackend::lstm_backward``.
//
// Attributes
// ----------
// saved_input : Storage
//     Original input sequence shaped ``(T, B, input_size)`` retained for
//     the backward pass.
// saved_h0 : Storage
//     Initial hidden state shaped ``(1, B, H_rec)`` where ``H_rec`` is
//     ``proj_size`` if projection is enabled, otherwise ``hidden_size``.
// saved_weights : std::vector<Storage>
//     Four-element vector ``{wih, whh, bih, bhh}`` in that exact order.
//     When ``opts.has_bias == false`` the bias slots still appear but
//     are zero-filled stand-ins so the backward kernel's BLAS layout
//     is uniform.
// gates_all : Storage
//     All pre-activation gate values shaped ``(T, B, 4H)`` in
//     ``[i, f, g, o]`` order, written by the training-mode forward.
// cells_all : Storage
//     Cell states ``c_0, c_1, ..., c_T`` shaped ``(T+1, B, H)``;
//     ``cells_all[0]`` is the supplied ``c_0`` and
//     ``cells_all[T]`` is ``cn``.
// opts : backend::IBackend::LstmOpts
//     Bundle of structural parameters propagated from ``forward()`` to
//     ``apply()`` — ``input_size``, ``hidden_size``, ``seq_len``,
//     ``batch_size``, ``has_bias``, ``proj_size``.  ``num_layers`` and
//     ``bidirectional`` are always ``1`` and ``false`` here; multi-layer
//     and bidirectional configurations are composed in Python.
// dtype : Dtype
//     Element type for the gradient buffers allocated in ``apply()``.
//     Default: ``Dtype::F32``.
// device : Device
//     Backend device on which to allocate gradients.  Default: CPU.
//
// Notes
// -----
// **Backend dispatch.**  CPU runs Apple Accelerate-backed hand-rolled
// BLAS through ``lstm_forward_train`` / ``lstm_backward``.  GPU dispatches
// to the MLX backend.  The Python wrapper in
// :class:`lucid.nn.modules.rnn.LSTM` always invokes the engine
// one-layer-one-direction at a time and composes stacking,
// bidirectionality and inter-layer dropout itself, so this node only
// ever sees ``num_layers == 1`` and ``bidirectional == false``.
//
// **Projection variant (``lstm_proj``).**  When ``opts.proj_size > 0``
// the recurrent path applies an additional learnable projection
// $W_{hr} \in \mathbb{R}^{\text{proj\_size} \times H}$ to $h_t$, so the
// hidden state fed to the next step has dimension ``proj_size``.  The
// cell state ``c_t`` is left at dimension ``hidden_size``.  This matches
// the LSTMP variant used in many speech-recognition stacks and reduces
// recurrent compute when ``proj_size < hidden_size``.
//
// **Peephole connections are not supported** — Lucid's LSTM follows the
// vanilla 1997 formulation without direct cell-to-gate links.
//
// References
// ----------
// Hochreiter & Schmidhuber, "Long Short-Term Memory" (Neural Computation
// 1997).  Hochreiter, "Untersuchungen zu dynamischen neuronalen Netzen"
// (1991) for the original analysis of the vanishing-gradient problem
// that motivates the gating mechanism.
class LUCID_API LstmBackward : public Node {
public:
    Storage saved_input;                 // Original input sequence (T, B, input_size).
    Storage saved_h0;                    // Initial hidden state (1, B, H).
    std::vector<Storage> saved_weights;  // {wih, whh, bih, bhh}.

    Storage gates_all;  // All gate pre-activations, shape (T, B, 4H).
    Storage cells_all;  // All cell states (including c_0), shape (T+1, B, H).

    backend::IBackend::LstmOpts opts;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;

    // Return the node name for debug / graph visualisation.
    //
    // Returns
    // -------
    // std::string_view
    //     The literal ``"LstmBackward"``.
    std::string_view name() const noexcept { return "LstmBackward"; }

    // Run the BPTT backward and return per-edge gradients.
    //
    // Dispatches to ``IBackend::lstm_backward`` which unrolls $T$ time
    // steps in reverse using the saved ``gates_all`` and ``cells_all``
    // tensors.  Gradients for ``hn`` and ``cn`` at the sequence end are
    // synthesised as zero buffers (matching detached outputs), because
    // only the output sequence ``res[0]`` is connected in the autograd
    // graph for the single-layer case.
    //
    // Parameters
    // ----------
    // grad_out : Storage
    //     Upstream gradient with respect to the output sequence,
    //     shaped ``(T, B, H_out)`` where ``H_out == proj_size`` if
    //     projection is enabled, else ``hidden_size``.
    //
    // Returns
    // -------
    // std::vector<Storage>
    //     Seven Storage objects in edge order ``{dInput, dh0, dc0, dW_ih,
    //     dW_hh, db_ih, db_hh}``.  Bias slots are zero buffers when
    //     ``opts.has_bias == false``.
    std::vector<Storage> apply(Storage grad_out) override;

    // Run the LSTM forward pass and return ``{output, hn, cn}``.
    //
    // Decides between inference and training paths based on
    // ``GradMode::is_enabled()`` and whether any of ``input``, ``h0``,
    // ``c0`` or the elements of ``weights`` require a gradient.  On the
    // training path the backend's ``lstm_forward_train`` is invoked and
    // a fresh ``LstmBackward`` node is wired with ``saved_input``,
    // ``saved_h0``, ``saved_weights``, ``gates_all`` and ``cells_all``
    // populated for BPTT.  On the inference path the backend's lighter
    // ``lstm_forward`` may be used (some backends route this through
    // ``lstm_forward_train`` and discard the saved tensors).
    //
    // Parameters
    // ----------
    // input : TensorImplPtr
    //     Input sequence shaped ``(T, B, input_size)``.  The
    //     ``batch_first`` flag in ``opts`` is forwarded but the Python
    //     LSTM wrapper always passes sequence-first.
    // h0 : TensorImplPtr
    //     Initial hidden state shaped ``(1, B, H_rec)`` where
    //     ``H_rec == proj_size`` if projection is enabled, else
    //     ``hidden_size``.
    // c0 : TensorImplPtr
    //     Initial cell state shaped ``(1, B, hidden_size)``.
    // weights : std::vector<TensorImplPtr>
    //     Layer weights in order ``{W_ih, W_hh, b_ih, b_hh}`` (and a
    //     trailing ``W_hr`` when ``opts.proj_size > 0``).  Each gate
    //     weight is the vertically-stacked 4-gate matrix in
    //     ``[i, f, g, o]`` order — see the class docstring's Math
    //     section.
    // opts : backend::IBackend::LstmOpts
    //     Structural parameters: ``input_size``, ``hidden_size``,
    //     ``seq_len``, ``batch_size``, ``has_bias``, ``proj_size``.
    //     Always called with ``num_layers == 1`` and
    //     ``bidirectional == false``; stacking is composed in Python.
    //
    // Returns
    // -------
    // std::vector<TensorImplPtr>
    //     Three tensors ``{output, hn, cn}`` where ``output`` is shaped
    //     ``(T, B, H_out)``, ``hn`` is shaped ``(1, B, H_out)`` and
    //     ``cn`` is shaped ``(1, B, hidden_size)``.  Only ``output``
    //     carries an autograd edge when the training path is taken;
    //     ``hn`` and ``cn`` are returned detached.
    //
    // Raises
    // ------
    // ErrorBuilder
    //     If any of ``input``, ``h0``, ``c0`` or a weight tensor is null,
    //     or if a backend method is not implemented for the current
    //     device.
    static std::vector<TensorImplPtr> forward(const TensorImplPtr& input,
                                              const TensorImplPtr& h0,
                                              const TensorImplPtr& c0,
                                              const std::vector<TensorImplPtr>& weights,
                                              const backend::IBackend::LstmOpts& opts);
};

// Run a single-layer LSTM forward pass and return ``{output, hn, cn}``.
//
// Thin public entry point that delegates to ``LstmBackward::forward``
// and is bound into Python as ``lucid._C.engine.nn.lstm_forward``.  The
// Python LSTM module in :mod:`lucid.nn.modules.rnn` calls this once per
// layer per direction and composes multi-layer, bidirectional and
// inter-layer dropout behaviour itself.
//
// Math
// ----
// Per-time-step gating (Hochreiter & Schmidhuber 1997):
// $$
//   \begin{aligned}
//     i_t &= \sigma(W_{ii}\,x_t + W_{hi}\,h_{t-1} + b_i) \\
//     f_t &= \sigma(W_{if}\,x_t + W_{hf}\,h_{t-1} + b_f) \\
//     g_t &= \tanh(W_{ig}\,x_t + W_{hg}\,h_{t-1} + b_g) \\
//     o_t &= \sigma(W_{io}\,x_t + W_{ho}\,h_{t-1} + b_o) \\
//     c_t &= f_t \odot c_{t-1} + i_t \odot g_t,\qquad h_t = o_t \odot \tanh(c_t)
//   \end{aligned}
// $$
//
// Parameters
// ----------
// input : TensorImplPtr
//     Input sequence shaped ``(T, B, input_size)``.
// h0 : TensorImplPtr
//     Initial hidden state shaped ``(1, B, H_rec)`` (zero-filled by the
//     caller if no explicit initial state is supplied).
// c0 : TensorImplPtr
//     Initial cell state shaped ``(1, B, hidden_size)``.
// weights : std::vector<TensorImplPtr>
//     Layer weights ``{W_ih, W_hh, b_ih, b_hh}`` (plus optional
//     ``W_hr`` when projection is enabled).  Gate matrices are stacked
//     in ``[i, f, g, o]`` order along the leading axis.
// opts : backend::IBackend::LstmOpts
//     Shape and structural parameters; see ``LstmBackward::forward``.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     ``{output, hn, cn}`` — output sequence ``(T, B, H_out)``, final
//     hidden state ``(1, B, H_out)``, final cell state
//     ``(1, B, hidden_size)``.
//
// Notes
// -----
// Only ``output`` participates in autograd; ``hn`` and ``cn`` are
// returned as detached tensors so the Python wrapper can feed them as
// initial states to a stacked next-layer call without leaking gradient
// edges between layers.
//
// See Also
// --------
// :class:`lucid.nn.LSTM` — Python multi-layer / bidirectional /
//     projection wrapper that composes single-layer engine calls.
// :class:`lucid.nn.LSTMCell` — single-time-step variant.
//
// References
// ----------
// Hochreiter & Schmidhuber, "Long Short-Term Memory" (Neural Computation
// 1997).
LUCID_API std::vector<TensorImplPtr> lstm_op(const TensorImplPtr& input,
                                             const TensorImplPtr& h0,
                                             const TensorImplPtr& c0,
                                             const std::vector<TensorImplPtr>& weights,
                                             const backend::IBackend::LstmOpts& opts);

}  // namespace lucid
