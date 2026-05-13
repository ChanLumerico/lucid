"""
Recurrent modules: LSTM, GRU, RNN.
"""

import math
from collections import OrderedDict
from typing import TYPE_CHECKING, cast

from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty, zeros
import lucid as _lucid
import lucid.nn.init as init
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.linear import linear
from lucid.nn.functional.activations import tanh, relu, sigmoid
from lucid import stack

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cat_last(a: Tensor, b: Tensor) -> Tensor:
    """Concatenate two tensors along the last dimension."""
    return _lucid.cat([a, b], a.ndim - 1)


def _check_not_packed(x: object, cls_name: str) -> None:
    """Reject ``PackedSequence`` inputs with a clear error.

    ``lucid.nn.utils.rnn`` provides ``pack_padded_sequence`` /
    ``pad_packed_sequence`` for manual use, but the recurrent modules
    themselves do not yet integrate the packed path.  Surface this
    explicitly instead of crashing on a missing attribute.
    """
    from lucid.nn.utils.rnn import PackedSequence

    if isinstance(x, PackedSequence):
        raise NotImplementedError(
            f"{cls_name} does not yet support PackedSequence input. "
            "Pad and unpack manually using "
            "`lucid.nn.utils.rnn.pad_packed_sequence` / `pack_padded_sequence`."
        )


# Cell-internal parameter names (RNNCell/GRUCell/LSTMCell).  Used by the
# state_dict naming hooks on GRU and RNN to translate between the lucid
# cell layout (`cell_l0.weight_ih`) and the flat layout used by the
# reference framework and many external checkpoints (`weight_ih_l0`).
_CELL_PARAM_NAMES: tuple[str, ...] = ("weight_ih", "weight_hh", "bias_ih", "bias_hh")


class _CellNamingMixin:
    r"""Mixin providing ``state_dict`` v2 hooks with flat checkpoint keys.

    GRU and RNN store per-layer weights inside child cell sub-modules
    (e.g. ``cell_l0`` of type :class:`GRUCell`).  The natural
    ``state_dict`` key for that layout is ``cell_l0.weight_ih``, but
    most external checkpoints flatten these into ``weight_ih_l0``.
    These hooks write/read the flat keys directly while tolerating the
    older ``cell_l*`` layout for backwards compatibility with checkpoints
    saved by earlier Lucid versions.

    Notes
    -----
    This is a **private** base class.  External code should never
    instantiate it directly; it is only mixed into :class:`GRU` and
    :class:`RNN`.

    The mixin sets ``_version = 2`` so the serialisation machinery can
    distinguish flat-key checkpoints from legacy cell-submodule ones.
    Setting ``_state_dict_skip_recursion = True`` prevents the default
    recursive walker from also emitting ``cell_l*.weight_ih`` entries,
    which would cause duplicate / unexpected keys on load.

    See Also
    --------
    GRU, RNN
    """

    # version = 2 ⇒ flat state_dict keys; missing/v1 metadata ⇒ accept
    # both the flat and the old `cell_l*.weight_*` layout on load.
    _version: int = 2
    # The walker should not descend into the cell submodules — we have
    # already serialised their parameters under flattened keys.
    _state_dict_skip_recursion: bool = True

    # Declared here so mypy knows _CellNamingMixin subclasses always have
    # these attributes (they are set in each concrete __init__).
    num_layers: int
    bidirectional: bool
    _modules: OrderedDict[str, Module | None]

    def _save_to_state_dict(
        self,
        destination: dict[str, object],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        # Skip the default impl entirely — we don't want the cell submodules
        # to recurse with their own keys; the top-level walker handles
        # children, but these are meant to be flattened.  Mark the cell
        # submodules so the recursive walker does not visit them.
        for layer in range(self.num_layers):
            for direction in range(2 if self.bidirectional else 1):
                suffix: str = "_reverse" if direction == 1 else ""
                cell: Module = cast(Module, self._modules[f"cell_l{layer}{suffix}"])
                for pname in _CELL_PARAM_NAMES:
                    p = cell._parameters.get(pname)
                    if p is None:
                        continue
                    key: str = f"{prefix}{pname}_l{layer}{suffix}"
                    destination[key] = p if keep_vars else p.detach()

    def _load_from_state_dict(
        self,
        state_dict: dict[str, object],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        from lucid._C import engine as _C_engine
        from lucid._tensor.tensor import Tensor

        for layer in range(self.num_layers):
            for direction in range(2 if self.bidirectional else 1):
                suffix: str = "_reverse" if direction == 1 else ""
                cell: Module = cast(Module, self._modules[f"cell_l{layer}{suffix}"])
                for pname in _CELL_PARAM_NAMES:
                    p = cell._parameters.get(pname)
                    if p is None:
                        continue
                    flat_key: str = f"{prefix}{pname}_l{layer}{suffix}"
                    legacy_key: str = f"{prefix}cell_l{layer}{suffix}.{pname}"
                    src: Tensor | None = cast(Tensor | None, state_dict.get(flat_key))
                    if src is None:
                        src = cast(Tensor | None, state_dict.get(legacy_key))
                    if src is None:
                        missing_keys.append(flat_key)
                        continue
                    if tuple(src.shape) != tuple(p.shape):
                        error_msgs.append(
                            f"size mismatch for {flat_key}: "
                            f"expected {tuple(p.shape)}, got {tuple(src.shape)}"
                        )
                        continue
                    converted: Tensor = src.to(device=p.device, dtype=p.dtype)
                    new_impl: _C_engine.TensorImpl = _C_engine.contiguous(
                        converted._impl
                    ).clone_with_grad(p.requires_grad)
                    p._impl = new_impl
                    # Mark whichever key was actually consumed so the top
                    # level walker doesn't list it as unexpected.
                    # (`_collect_expected` knows our flat layout via
                    # `_enumerate_local_keys` below.)

    def _local_state_dict_keys(self, prefix: str) -> list[str]:
        """Override: tell the top-level driver our flat key set.

        Returns *both* the canonical flat keys (e.g.
        ``weight_ih_l0_reverse``) and the legacy cell-submodule keys
        (``cell_l0_reverse.weight_ih``).  Either form satisfies an
        ``unexpected_keys`` check, so an external checkpoint using the
        flat layout AND a legacy-Lucid checkpoint using the cell layout
        both load cleanly.
        """
        keys: list[str] = []
        for layer in range(self.num_layers):
            for direction in range(2 if self.bidirectional else 1):
                suffix: str = "_reverse" if direction == 1 else ""
                cell: Module = cast(Module, self._modules[f"cell_l{layer}{suffix}"])
                for pname in _CELL_PARAM_NAMES:
                    if cell._parameters.get(pname) is not None:
                        keys.append(f"{prefix}{pname}_l{layer}{suffix}")
                        keys.append(f"{prefix}cell_l{layer}{suffix}.{pname}")
        return keys


# ── LSTM ──────────────────────────────────────────────────────────────────────


class LSTM(Module):
    r"""Long Short-Term Memory (LSTM) recurrent layer.

    Applies a multi-layer LSTM over an input sequence.  At each time step
    :math:`t` the following gated update equations are evaluated:

    .. math::

        i_t &= \sigma(W_{ii}\,x_t + b_{ii} + W_{hi}\,h_{t-1} + b_{hi}) \\
        f_t &= \sigma(W_{if}\,x_t + b_{if} + W_{hf}\,h_{t-1} + b_{hf}) \\
        g_t &= \tanh(W_{ig}\,x_t + b_{ig} + W_{hg}\,h_{t-1} + b_{hg}) \\
        o_t &= \sigma(W_{io}\,x_t + b_{io} + W_{ho}\,h_{t-1} + b_{ho}) \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        h_t &= o_t \odot \tanh(c_t)

    where :math:`\sigma` is the sigmoid function, :math:`\odot` is the
    element-wise (Hadamard) product, :math:`x_t` is the input at step
    :math:`t`, and :math:`h_{t-1}`, :math:`c_{t-1}` are the hidden and
    cell states carried from the previous step.

    The four gates have the following roles:

    * **Input gate** :math:`i_t` — controls how much new information
      enters the cell state.
    * **Forget gate** :math:`f_t` — controls how much of the previous
      cell state is retained.
    * **Cell gate** :math:`g_t` — candidate new content to add to the
      cell state.
    * **Output gate** :math:`o_t` — controls what portion of the cell
      state is exposed as the hidden state :math:`h_t`.

    When ``num_layers > 1`` the hidden state output of layer :math:`\ell`
    becomes the input to layer :math:`\ell + 1`.  Inter-layer dropout
    (probability ``dropout``) is applied between every pair of adjacent
    layers during training.

    When ``bidirectional=True``, two independent LSTMs process the
    sequence in opposite directions.  Their output hidden states are
    concatenated along the feature dimension at each time step, so the
    output feature size is ``2 * hidden_size``.  The hidden/cell state
    tensors ``h_n`` and ``c_n`` pack both directions along their leading
    axis in the order ``[fwd_l0, rev_l0, fwd_l1, rev_l1, ...]``.

    Parameters
    ----------
    input_size : int
        Number of expected features in each input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t` (denoted
        :math:`H` below).
    num_layers : int, optional
        Number of stacked recurrent layers.  Default: ``1``.
    bias : bool, optional
        If ``False`` all bias terms are treated as zero (the parameter
        tensors still exist for API compatibility, but are set to zero
        and contribute nothing numerically).  Default: ``True``.
    batch_first : bool, optional
        If ``True`` the expected input/output shape is
        ``(N, L, input_size)`` / ``(N, L, H * D)`` rather than the
        default ``(L, N, input_size)`` / ``(L, N, H * D)``.
        Default: ``False``.
    dropout : float, optional
        Dropout probability applied to the outputs of every LSTM layer
        except the last.  ``0.0`` disables dropout.  Default: ``0.0``.
    bidirectional : bool, optional
        If ``True`` use a bidirectional LSTM; doubles the output feature
        size and the leading dimension of ``h_n`` / ``c_n``.
        Default: ``False``.
    proj_size : int, optional
        If ``> 0``, adds a learnable linear projection of size
        ``proj_size`` after each LSTM cell output, reducing the
        recurrent hidden dimension fed to the next time step.
        Default: ``0`` (disabled).
    device : DeviceLike, optional
        Device on which to allocate weight tensors.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih_l0 : Parameter, shape ``(4H, I)``
        Input–hidden weight matrix for layer 0, forward direction.
        Stacks the four gate weight matrices :math:`[W_{ii}; W_{if};
        W_{ig}; W_{io}]`.  Here :math:`I` is ``input_size`` and
        :math:`H` is ``hidden_size``.
    weight_hh_l0 : Parameter, shape ``(4H, H)``
        Hidden–hidden weight matrix for layer 0, forward direction.
        Stacks :math:`[W_{hi}; W_{hf}; W_{hg}; W_{ho}]`.
        When ``proj_size > 0`` the second dimension shrinks to
        ``proj_size`` (the recurrent state is the projected output).
    bias_ih_l0 : Parameter, shape ``(4H,)``
        Input–hidden bias for layer 0, forward direction.  Present only
        when ``bias=True``.
    bias_hh_l0 : Parameter, shape ``(4H,)``
        Hidden–hidden bias for layer 0, forward direction.  Present only
        when ``bias=True``.

    For layer ``k`` in the forward direction substitute ``l0`` →
    ``l{k}``; for the backward direction append ``_reverse``,
    e.g. ``weight_ih_l1_reverse``.

    Shape
    -----
    * **Input** ``x``: ``(L, N, input_size)`` or ``(N, L, input_size)``
      when ``batch_first=True``.  ``L`` = sequence length,
      ``N`` = batch size.
    * **h_0**, **c_0** (optional): ``(D * num_layers, N, H)`` where
      ``D = 2`` if bidirectional else ``1``.  When omitted they default
      to zero tensors.
    * **output**: ``(L, N, D * H)`` or ``(N, L, D * H)`` when
      ``batch_first=True``.
    * **h_n**: ``(D * num_layers, N, H)`` — hidden state after the last
      time step for every layer and direction.
    * **c_n**: ``(D * num_layers, N, H)`` — cell state after the last
      time step for every layer and direction.

    Notes
    -----
    All weight matrices are initialised with a uniform distribution
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})`, matching the
    initialisation convention of the reference framework.

    The C++ engine processes a single layer and a single direction per
    call.  Multi-layer and bidirectional configurations are composed
    entirely in Python: the module loops over layers and directions,
    applies inter-layer dropout, and concatenates bidirectional outputs.

    :meth:`flatten_parameters` is a no-op provided purely for API
    compatibility with codebases that call it before the forward pass.

    ``PackedSequence`` input is not yet supported.  Use
    :func:`lucid.nn.utils.rnn.pad_packed_sequence` to unpack manually
    before passing to this module.

    Examples
    --------
    Basic sequence encoding (batch-first convention):

    >>> import lucid, lucid.nn as nn
    >>> lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
    >>> x = lucid.randn(2, 5, 10)          # (N=2, L=5, I=10)
    >>> output, (h_n, c_n) = lstm(x)
    >>> output.shape, h_n.shape, c_n.shape
    ((2, 5, 20), (1, 2, 20), (1, 2, 20))

    Bidirectional, 2-layer LSTM with dropout:

    >>> lstm2 = nn.LSTM(
    ...     input_size=16, hidden_size=32,
    ...     num_layers=2, dropout=0.3,
    ...     bidirectional=True, batch_first=True,
    ... )
    >>> x2 = lucid.randn(4, 12, 16)        # (N=4, L=12, I=16)
    >>> out2, (h_n2, c_n2) = lstm2(x2)
    >>> out2.shape    # D*H = 2*32 = 64
    (4, 12, 64)
    >>> h_n2.shape    # D*num_layers = 4
    (4, 4, 32)

    See Also
    --------
    LSTMCell : Single time-step LSTM cell.
    GRU : Gated Recurrent Unit (simpler, no cell state).
    RNN : Vanilla Elman RNN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        if proj_size < 0:
            raise ValueError(f"proj_size must be >= 0, got {proj_size}")
        if proj_size >= hidden_size and proj_size > 0:
            raise ValueError(
                f"proj_size ({proj_size}) must be smaller than hidden_size "
                f"({hidden_size})"
            )
        # proj_size > 0 with multi-layer / bidirectional configurations is
        # composed in Python by looping single-layer engine calls — see
        # ``forward``.  The C++ engine itself still handles only the
        # single-layer single-direction case; the Python wrapper bridges
        # the gap.
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers: int = num_layers
        self.bias: bool = bias
        self.batch_first: bool = batch_first
        self.dropout_val: float = dropout
        self.bidirectional: bool = bidirectional
        self.proj_size: int = proj_size

        num_directions: int = 2 if bidirectional else 1
        gate_size: int = 4 * hidden_size
        # Recurrent dim feeding the next time step.  When proj_size > 0 the
        # projected output of each step is what re-enters the recurrence,
        # so W_hh's input axis and the per-layer hidden carry shrink to
        # proj_size.
        rec_size: int = proj_size if proj_size > 0 else hidden_size

        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix: str = "_reverse" if direction == 1 else ""
                layer_input: int = (
                    input_size if layer == 0 else rec_size * num_directions
                )
                self.register_parameter(
                    f"weight_ih_l{layer}{suffix}",
                    Parameter(
                        empty(gate_size, layer_input, dtype=dtype, device=device)
                    ),
                )
                self.register_parameter(
                    f"weight_hh_l{layer}{suffix}",
                    Parameter(empty(gate_size, rec_size, dtype=dtype, device=device)),
                )
                if bias:
                    self.register_parameter(
                        f"bias_ih_l{layer}{suffix}",
                        Parameter(empty(gate_size, dtype=dtype, device=device)),
                    )
                    self.register_parameter(
                        f"bias_hh_l{layer}{suffix}",
                        Parameter(empty(gate_size, dtype=dtype, device=device)),
                    )
                if proj_size > 0:
                    # Projection W_hr: (proj_size × hidden_size).
                    self.register_parameter(
                        f"weight_hr_l{layer}{suffix}",
                        Parameter(
                            empty(proj_size, hidden_size, dtype=dtype, device=device)
                        ),
                    )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def flatten_parameters(self) -> None:
        """No-op for API compatibility with reference recurrent modules.

        Some external codepaths call ``flatten_parameters()`` to coalesce
        weights for cuDNN; Lucid's BLAS / MLX path has no such concern,
        so this is a placeholder that lets such code run unchanged.
        """
        return None

    def _reverse_along_time(self, x: Tensor) -> Tensor:
        """Flip a sequence-major tensor along the time (axis-0) dimension.

        Implemented via ``gather`` so the backward path works correctly —
        the engine's ``Tensor.flip`` backward is currently broken.
        """
        # Reverse range [T-1, T-2, …, 0] built on the engine.
        T: int = int(x.shape[0])
        rev_1d: Tensor = _lucid.arange(
            T - 1, -1, -1, dtype=_lucid.int32, device=x.device
        )
        target_shape: list[int] = [1] * x.ndim
        target_shape[0] = T
        bcast_shape: list[int] = [int(s) for s in x.shape]
        idx: Tensor = (
            rev_1d.reshape(target_shape).broadcast_to(tuple(bcast_shape)).contiguous()
        )
        return _lucid.gather(x, idx, 0)

    def _run_single_layer_engine(
        self,
        layer_input: Tensor,
        h0_layer: Tensor,
        c0_layer: Tensor,
        layer: int,
        direction: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one ``lstm_forward`` engine call for a single layer / direction."""
        suffix: str = "_reverse" if direction == 1 else ""
        weights: list[object] = []
        p_wih = self._parameters[f"weight_ih_l{layer}{suffix}"]
        p_whh = self._parameters[f"weight_hh_l{layer}{suffix}"]
        assert p_wih is not None
        assert p_whh is not None
        weights.append(_unwrap(p_wih))
        weights.append(_unwrap(p_whh))
        gate_size: int = 4 * self.hidden_size
        if self.bias:
            p_bih = self._parameters[f"bias_ih_l{layer}{suffix}"]
            p_bhh = self._parameters[f"bias_hh_l{layer}{suffix}"]
            assert p_bih is not None
            assert p_bhh is not None
            weights.append(_unwrap(p_bih))
            weights.append(_unwrap(p_bhh))
        else:
            dev = _unwrap(layer_input).device
            # Engine zero buffer — same dtype as the tensors we'll concat with.
            zero_b = _C_engine.zeros([gate_size], _C_engine.F32, dev)
            weights.append(zero_b)
            weights.append(zero_b)
        if self.proj_size > 0:
            p_hr = self._parameters[f"weight_hr_l{layer}{suffix}"]
            assert p_hr is not None
            weights.append(_unwrap(p_hr))

        lstm_result = _C_engine.nn.lstm_forward(
            _unwrap(layer_input),
            _unwrap(h0_layer),
            _unwrap(c0_layer),
            cast(list[_C_engine.TensorImpl], weights),
            self.hidden_size,
            1,  # single-layer engine call
            False,  # batch_first handled by us
            False,  # bidirectional handled by us
            True,
            self.proj_size,
        )
        out_impl: _C_engine.TensorImpl = lstm_result[0]
        h_impl: _C_engine.TensorImpl = lstm_result[1]
        c_impl: _C_engine.TensorImpl = lstm_result[2]
        return _wrap(out_impl), _wrap(h_impl), _wrap(c_impl)

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        x: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Multi-layer × bidirectional forward.

        The C++ engine only handles a single layer in a single direction
        at a time, so this loops over layers and directions, applying
        inter-layer dropout and concatenating bidirectional outputs as
        the input to the next layer.
        """
        _check_not_packed(x, "LSTM")

        if self.batch_first:
            x = x.permute([1, 0, 2])


        B: int = int(x.shape[1])
        num_dirs: int = 2 if self.bidirectional else 1
        L: int = self.num_layers
        rec_size: int = self.proj_size if self.proj_size > 0 else self.hidden_size

        # Allocate / split the initial states.
        if hx is None:
            h0_full: Tensor = _lucid.zeros(
                L * num_dirs, B, rec_size, device=x.device, dtype=x.dtype
            )
            c0_full: Tensor = _lucid.zeros(
                L * num_dirs, B, self.hidden_size, device=x.device, dtype=x.dtype
            )
        else:
            h0_full, c0_full = hx

        h_n_layers: list[Tensor] = []
        c_n_layers: list[Tensor] = []

        layer_input: Tensor = x

        for layer in range(L):
            dir_outs: list[Tensor] = []
            for direction in range(num_dirs):
                idx: int = layer * num_dirs + direction
                # Slice (1, B, *) initial state for this layer/direction.
                h0_slice: Tensor = h0_full[idx : idx + 1]
                c0_slice: Tensor = c0_full[idx : idx + 1]

                inp: Tensor = (
                    self._reverse_along_time(layer_input)
                    if direction == 1
                    else layer_input
                )
                out, h_n, c_n = self._run_single_layer_engine(
                    inp, h0_slice, c0_slice, layer, direction
                )
                if direction == 1:
                    out = self._reverse_along_time(out)
                dir_outs.append(out)
                h_n_layers.append(h_n)
                c_n_layers.append(c_n)

            # Combine forward / reverse outputs along the feature axis.
            if num_dirs == 2:
                layer_input = _lucid.cat(dir_outs, 2)
            else:
                layer_input = dir_outs[0]

            # Inter-layer dropout (skip after the last layer).
            if self.dropout_val > 0.0 and layer < L - 1 and self.training:
                from lucid.nn.functional.dropout import dropout as _dropout

                layer_input = _dropout(layer_input, self.dropout_val, training=True)

        h_n_final: Tensor = _lucid.cat(h_n_layers, 0)
        c_n_final: Tensor = _lucid.cat(c_n_layers, 0)

        if self.batch_first:
            layer_input = layer_input.permute([1, 0, 2])

        return layer_input, (h_n_final, c_n_final)

    def extra_repr(self) -> str:
        s: str = (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"bias={self.bias}, batch_first={self.batch_first}, "
            f"dropout={self.dropout_val}, bidirectional={self.bidirectional}"
        )
        if self.proj_size > 0:
            s += f", proj_size={self.proj_size}"
        return s


# ── Pure-Python RNN cells ─────────────────────────────────────────────────────


class RNNCell(Module):
    r"""Single time-step Elman RNN cell.

    Computes one recurrent update for a single time step:

    .. math::

        h_t = \phi\!\left(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh}\right)

    where :math:`\phi` is either :math:`\tanh` (default) or
    :math:`\text{ReLU}`, selected by ``nonlinearity``.

    Use this cell directly when you need fine-grained control over the
    time loop (e.g. teacher forcing, attention, custom stopping
    criteria).  For standard sequence processing prefer :class:`RNN`.

    Parameters
    ----------
    input_size : int
        Number of features in the input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t`.
    bias : bool, optional
        If ``False``, no bias terms are used.  Default: ``True``.
    nonlinearity : {'tanh', 'relu'}, optional
        Activation function applied to the pre-activation.
        Default: ``'tanh'``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih : Parameter, shape ``(hidden_size, input_size)``
        Input–hidden weight matrix :math:`W_{ih}`.
    weight_hh : Parameter, shape ``(hidden_size, hidden_size)``
        Hidden–hidden weight matrix :math:`W_{hh}`.
    bias_ih : Parameter or None, shape ``(hidden_size,)``
        Input–hidden bias :math:`b_{ih}`.  ``None`` when ``bias=False``.
    bias_hh : Parameter or None, shape ``(hidden_size,)``
        Hidden–hidden bias :math:`b_{hh}`.  ``None`` when ``bias=False``.

    Shape
    -----
    * **x**: ``(N, input_size)`` — batch of input vectors.
    * **hx** (optional): ``(N, hidden_size)`` — initial hidden state.
      Defaults to zeros when ``None``.
    * **Output** ``h_t``: ``(N, hidden_size)``.

    Notes
    -----
    Weights are initialised from
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})` where
    :math:`H` = ``hidden_size``.

    For long sequences, vanilla RNNs suffer from the vanishing-gradient
    problem because gradients are multiplied by :math:`W_{hh}` at every
    step.  Prefer :class:`LSTMCell` or :class:`GRUCell` for sequences
    longer than ~20 steps.

    Examples
    --------
    Manual time loop with ``tanh`` nonlinearity:

    >>> import lucid, lucid.nn as nn
    >>> cell = nn.RNNCell(input_size=8, hidden_size=16)
    >>> x_seq = lucid.randn(5, 3, 8)   # (L=5, N=3, I=8)
    >>> h = lucid.zeros(3, 16)
    >>> for t in range(5):
    ...     h = cell(x_seq[t], h)      # (N=3, H=16)
    >>> h.shape
    (3, 16)

    Using ``relu`` nonlinearity (avoids gradient saturation at extremes):

    >>> cell_relu = nn.RNNCell(8, 16, nonlinearity='relu')
    >>> h2 = cell_relu(lucid.randn(4, 8))
    >>> h2.shape
    (4, 16)

    See Also
    --------
    RNN : Multi-layer, multi-step wrapper around :class:`RNNCell`.
    LSTMCell : Single-step LSTM cell (gated, better for long sequences).
    GRUCell : Single-step GRU cell.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.weight_ih = Parameter(
            empty(hidden_size, input_size, dtype=dtype, device=device)
        )
        self.weight_hh = Parameter(
            empty(hidden_size, hidden_size, dtype=dtype, device=device)
        )
        self.bias_ih: Parameter | None = (
            Parameter(empty(hidden_size, dtype=dtype, device=device)) if bias else None
        )
        self.bias_hh: Parameter | None = (
            Parameter(empty(hidden_size, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, x: Tensor, hx: Tensor | None = None) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if hx is None:
            hx = zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        pre = linear(x, self.weight_ih, self.bias_ih) + linear(
            hx, self.weight_hh, self.bias_hh
        )
        return tanh(pre) if self.nonlinearity == "tanh" else relu(pre)

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, nonlinearity={self.nonlinearity!r}"
        )


class LSTMCell(Module):
    r"""Single time-step Long Short-Term Memory (LSTM) cell.

    Computes one recurrent update using the full LSTM gating equations:

    .. math::

        \begin{aligned}
        i_t &= \sigma(W_{ii}\,x_t + b_{ii} + W_{hi}\,h_{t-1} + b_{hi})
              & &\text{(input gate)}\\
        f_t &= \sigma(W_{if}\,x_t + b_{if} + W_{hf}\,h_{t-1} + b_{hf})
              & &\text{(forget gate)}\\
        g_t &= \tanh(W_{ig}\,x_t + b_{ig} + W_{hg}\,h_{t-1} + b_{hg})
              & &\text{(cell gate)}\\
        o_t &= \sigma(W_{io}\,x_t + b_{io} + W_{ho}\,h_{t-1} + b_{ho})
              & &\text{(output gate)}\\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t\\
        h_t &= o_t \odot \tanh(c_t)
        \end{aligned}

    The four weight matrices for the gates are stored as a single
    vertically-stacked parameter of shape ``(4H, *)``, in gate order
    ``[i; f; g; o]`` (i.e. the first ``H`` rows correspond to the input
    gate, the next ``H`` to the forget gate, and so on).

    Parameters
    ----------
    input_size : int
        Number of features in the input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden/cell states :math:`h_t, c_t`.
    bias : bool, optional
        If ``False``, all bias terms are omitted.  Default: ``True``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih : Parameter, shape ``(4 * hidden_size, input_size)``
        Stacked input–hidden weight matrices
        :math:`[W_{ii}; W_{if}; W_{ig}; W_{io}]`.
    weight_hh : Parameter, shape ``(4 * hidden_size, hidden_size)``
        Stacked hidden–hidden weight matrices
        :math:`[W_{hi}; W_{hf}; W_{hg}; W_{ho}]`.
    bias_ih : Parameter or None, shape ``(4 * hidden_size,)``
        Stacked input–hidden biases.  ``None`` when ``bias=False``.
    bias_hh : Parameter or None, shape ``(4 * hidden_size,)``
        Stacked hidden–hidden biases.  ``None`` when ``bias=False``.

    Shape
    -----
    * **x**: ``(N, input_size)`` — batch of input vectors.
    * **hx** (optional): tuple ``(h_0, c_0)`` each of shape
      ``(N, hidden_size)``.  Defaults to zero tensors when ``None``.
    * **Output**: tuple ``(h_t, c_t)`` each of shape
      ``(N, hidden_size)``.

    Notes
    -----
    Weights are initialised from
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})`.

    The forget-gate bias is **not** initialised to 1 by default
    (unlike some implementations).  If you observe vanishing gradients
    on medium-length sequences, consider manually setting
    ``cell.bias_ih.data[H:2H] = 1.0`` after construction.

    Examples
    --------
    Single-step update, carrying state across a manual loop:

    >>> import lucid, lucid.nn as nn
    >>> cell = nn.LSTMCell(input_size=10, hidden_size=20)
    >>> h, c = lucid.zeros(3, 20), lucid.zeros(3, 20)
    >>> x_seq = lucid.randn(7, 3, 10)    # (L=7, N=3, I=10)
    >>> for t in range(7):
    ...     h, c = cell(x_seq[t], (h, c))
    >>> h.shape, c.shape
    ((3, 20), (3, 20))

    Without providing an explicit initial state (zeros used):

    >>> cell2 = nn.LSTMCell(4, 8)
    >>> h2, c2 = cell2(lucid.randn(5, 4))
    >>> h2.shape
    (5, 8)

    See Also
    --------
    LSTM : Multi-layer, multi-step LSTM module.
    RNNCell : Simpler single-step cell without gating.
    GRUCell : Single-step GRU cell (fewer gates, no separate cell state).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(
            empty(4 * hidden_size, input_size, dtype=dtype, device=device)
        )
        self.weight_hh = Parameter(
            empty(4 * hidden_size, hidden_size, dtype=dtype, device=device)
        )
        self.bias_ih: Parameter | None = (
            Parameter(empty(4 * hidden_size, dtype=dtype, device=device))
            if bias
            else None
        )
        self.bias_hh: Parameter | None = (
            Parameter(empty(4 * hidden_size, dtype=dtype, device=device))
            if bias
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        x: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if hx is None:
            batch = x.shape[0]
            h0 = zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
            c0 = zeros(batch, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h0, c0 = hx
        gates = linear(x, self.weight_ih, self.bias_ih) + linear(
            h0, self.weight_hh, self.bias_hh
        )
        hs = self.hidden_size
        i_gate = sigmoid(gates[:, :hs])
        f_gate = sigmoid(gates[:, hs : 2 * hs])
        g_gate = tanh(gates[:, 2 * hs : 3 * hs])
        o_gate = sigmoid(gates[:, 3 * hs :])
        c1 = f_gate * c0 + i_gate * g_gate
        h1 = o_gate * tanh(c1)
        return h1, c1

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}"


class GRUCell(Module):
    r"""Single time-step Gated Recurrent Unit (GRU) cell.

    Computes one recurrent update using the three-gate GRU equations:

    .. math::

        \begin{aligned}
        r_t &= \sigma(W_{ir}\,x_t + b_{ir} + W_{hr}\,h_{t-1} + b_{hr})
              & &\text{(reset gate)}\\
        z_t &= \sigma(W_{iz}\,x_t + b_{iz} + W_{hz}\,h_{t-1} + b_{hz})
              & &\text{(update gate)}\\
        n_t &= \tanh\!\left(W_{in}\,x_t + b_{in}
                + r_t \odot (W_{hn}\,h_{t-1} + b_{hn})\right)
              & &\text{(new gate)}\\
        h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot n_t
        \end{aligned}

    The **reset gate** :math:`r_t` controls how much of the previous
    hidden state leaks into the candidate :math:`n_t`; setting it near
    zero makes the cell ignore past context.  The **update gate**
    :math:`z_t` interpolates between the old hidden state and the
    candidate, allowing the cell to retain information over many steps
    without an explicit cell state.

    The three gate weight matrices are stacked into single parameters
    of shape ``(3H, *)`` in gate order ``[r; z; n]``.

    Parameters
    ----------
    input_size : int
        Number of features in the input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t`.
    bias : bool, optional
        If ``False``, no bias terms are used.  Default: ``True``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih : Parameter, shape ``(3 * hidden_size, input_size)``
        Stacked input–hidden weight matrices
        :math:`[W_{ir}; W_{iz}; W_{in}]`.
    weight_hh : Parameter, shape ``(3 * hidden_size, hidden_size)``
        Stacked hidden–hidden weight matrices
        :math:`[W_{hr}; W_{hz}; W_{hn}]`.
    bias_ih : Parameter or None, shape ``(3 * hidden_size,)``
        Stacked input–hidden biases :math:`[b_{ir}; b_{iz}; b_{in}]`.
        ``None`` when ``bias=False``.
    bias_hh : Parameter or None, shape ``(3 * hidden_size,)``
        Stacked hidden–hidden biases :math:`[b_{hr}; b_{hz}; b_{hn}]`.
        ``None`` when ``bias=False``.

    Shape
    -----
    * **x**: ``(N, input_size)`` — batch of input vectors.
    * **hx** (optional): ``(N, hidden_size)`` — initial hidden state.
      Defaults to zeros when ``None``.
    * **Output** ``h_t``: ``(N, hidden_size)``.

    Notes
    -----
    Weights are initialised from
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})`.

    The GRU has fewer parameters than the LSTM (no cell state, three
    gates instead of four) and often converges faster on shorter
    sequences while matching LSTM quality on many benchmarks.

    Examples
    --------
    Manual sequence loop:

    >>> import lucid, lucid.nn as nn
    >>> cell = nn.GRUCell(input_size=8, hidden_size=16)
    >>> x_seq = lucid.randn(6, 4, 8)    # (L=6, N=4, I=8)
    >>> h = lucid.zeros(4, 16)
    >>> for t in range(6):
    ...     h = cell(x_seq[t], h)
    >>> h.shape
    (4, 16)

    No explicit initial state (defaults to zeros):

    >>> cell2 = nn.GRUCell(4, 12)
    >>> h2 = cell2(lucid.randn(3, 4))
    >>> h2.shape
    (3, 12)

    See Also
    --------
    GRU : Multi-layer, multi-step GRU module.
    LSTMCell : Single-step LSTM cell with separate cell state.
    RNNCell : Vanilla single-step cell without gating.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(
            empty(3 * hidden_size, input_size, dtype=dtype, device=device)
        )
        self.weight_hh = Parameter(
            empty(3 * hidden_size, hidden_size, dtype=dtype, device=device)
        )
        self.bias_ih: Parameter | None = (
            Parameter(empty(3 * hidden_size, dtype=dtype, device=device))
            if bias
            else None
        )
        self.bias_hh: Parameter | None = (
            Parameter(empty(3 * hidden_size, dtype=dtype, device=device))
            if bias
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)

    def forward(self, x: Tensor, hx: Tensor | None = None) -> Tensor:  # type: ignore[override]  # narrower signature than Module.forward(*args) by design
        if hx is None:
            hx = zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        hs = self.hidden_size
        gates_x = linear(x, self.weight_ih, self.bias_ih)
        gates_h = linear(hx, self.weight_hh, self.bias_hh)
        r = sigmoid(gates_x[:, :hs] + gates_h[:, :hs])
        z = sigmoid(gates_x[:, hs : 2 * hs] + gates_h[:, hs : 2 * hs])
        n = tanh(gates_x[:, 2 * hs :] + r * gates_h[:, 2 * hs :])
        h1 = (1.0 - z) * n + z * hx
        return h1

    def extra_repr(self) -> str:
        return f"{self.input_size}, {self.hidden_size}"


# ── Multi-step GRU ────────────────────────────────────────────────────────────


class GRU(_CellNamingMixin, Module):  # type: ignore[misc]
    r"""Multi-layer Gated Recurrent Unit (GRU) recurrent layer.

    Applies a stack of GRU cells over an input sequence.  At each time
    step :math:`t` the following equations are evaluated (see
    :class:`GRUCell` for the full derivation):

    .. math::

        \begin{aligned}
        r_t &= \sigma(W_{ir}\,x_t + W_{hr}\,h_{t-1} + b_r) \\
        z_t &= \sigma(W_{iz}\,x_t + W_{hz}\,h_{t-1} + b_z) \\
        n_t &= \tanh\!\left(W_{in}\,x_t + b_{in}
                + r_t \odot (W_{hn}\,h_{t-1} + b_{hn})\right) \\
        h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot n_t
        \end{aligned}

    The output of layer :math:`\ell` is used as the input of layer
    :math:`\ell + 1`.  When ``bidirectional=True`` two GRUs process the
    sequence in opposite directions and their outputs are concatenated
    along the feature axis at every time step.

    Inter-layer dropout (probability ``dropout``) is applied between
    adjacent layers during training, but not after the final layer.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t` (denoted
        :math:`H` below).
    num_layers : int, optional
        Number of stacked GRU layers.  Default: ``1``.
    bias : bool, optional
        If ``False``, all bias parameters are omitted.  Default: ``True``.
    batch_first : bool, optional
        If ``True`` the input/output tensors have shape
        ``(N, L, *)`` instead of the default ``(L, N, *)``.
        Default: ``False``.
    dropout : float, optional
        Dropout probability applied after every layer except the last.
        ``0.0`` disables dropout.  Default: ``0.0``.
    bidirectional : bool, optional
        If ``True``, a bidirectional GRU is used; the output feature
        dimension becomes ``2 * hidden_size``.  Default: ``False``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Shape
    -----
    * **Input** ``x``: ``(L, N, input_size)`` or ``(N, L, input_size)``
      when ``batch_first=True``.
    * **h_0** (optional): ``(D * num_layers, N, H)`` where ``D = 2`` if
      bidirectional else ``1``.  Defaults to zeros.
    * **output**: ``(L, N, D * H)`` or ``(N, L, D * H)``.
    * **h_n**: ``(D * num_layers, N, H)`` — hidden state at the final
      time step for each layer and direction.

    Notes
    -----
    Internally this module stores one :class:`GRUCell` sub-module per
    layer per direction (named ``cell_l{layer}`` and
    ``cell_l{layer}_reverse``).  The :class:`_CellNamingMixin` flattens
    these into ``weight_ih_l{layer}`` etc. for checkpoint compatibility
    with the reference framework.

    :meth:`flatten_parameters` is a no-op retained for API
    compatibility.

    ``PackedSequence`` input is not yet supported.

    Examples
    --------
    Two-layer GRU, batch-first:

    >>> import lucid, lucid.nn as nn
    >>> gru = nn.GRU(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)       # (N=2, L=5, I=8)
    >>> out, h_n = gru(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))

    Bidirectional GRU:

    >>> gru_bi = nn.GRU(8, 16, bidirectional=True, batch_first=True)
    >>> x2 = lucid.randn(3, 10, 8)
    >>> out2, h_n2 = gru_bi(x2)
    >>> out2.shape    # D*H = 2*16 = 32
    (3, 10, 32)
    >>> h_n2.shape    # D*num_layers = 2*1 = 2
    (2, 3, 16)

    See Also
    --------
    GRUCell : Single time-step GRU cell.
    LSTM : Long Short-Term Memory (carries a separate cell state).
    RNN : Vanilla Elman RNN (no gating).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_val = dropout
        self.bidirectional = bidirectional

        num_dirs = 2 if bidirectional else 1
        for layer in range(num_layers):
            for d in range(num_dirs):
                suffix = "_reverse" if d == 1 else ""
                in_sz = input_size if layer == 0 else hidden_size * num_dirs
                self.add_module(
                    f"cell_l{layer}{suffix}",
                    GRUCell(in_sz, hidden_size, bias=bias, device=device, dtype=dtype),
                )

    def _cell(self, layer: int, reverse: bool = False) -> GRUCell:
        suffix: str = "_reverse" if reverse else ""
        return self._modules[f"cell_l{layer}{suffix}"]  # type: ignore[return-value]

    def flatten_parameters(self) -> None:
        """No-op for API compatibility (see :meth:`LSTM.flatten_parameters`)."""
        return None

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        x: Tensor,
        hx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        _check_not_packed(x, "GRU")
        if self.batch_first:
            x = x.permute([1, 0, 2])

        T: int = x.shape[0]
        B: int = x.shape[1]
        num_dirs: int = 2 if self.bidirectional else 1

        if hx is None:
            hx = zeros(
                self.num_layers * num_dirs,
                B,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        h_n: list[Tensor] = []
        inp = x

        for layer in range(self.num_layers):
            # ── forward direction ────────────────────────────────────────────
            cell_fwd: GRUCell = self._cell(layer, reverse=False)
            h_fwd: Tensor = hx[layer * num_dirs]  # (B, hidden)
            fwd_out: list[Tensor] = []
            for t in range(T):
                h_fwd = cast("Tensor", cell_fwd(inp[t], h_fwd))
                fwd_out.append(h_fwd)
            h_n.append(h_fwd)

            if self.bidirectional:
                # ── reverse direction ────────────────────────────────────────
                cell_rev = self._cell(layer, reverse=True)
                h_rev: Tensor = hx[layer * num_dirs + 1]
                rev_out: list[Tensor] = []
                for t in range(T - 1, -1, -1):
                    h_rev = cast("Tensor", cell_rev(inp[t], h_rev))
                    rev_out.append(h_rev)
                rev_out.reverse()
                h_n.append(h_rev)
                # Concat forward and backward at each time step
                layer_out = stack(
                    [_cat_last(fwd_out[t], rev_out[t]) for t in range(T)], 0
                )
            else:
                layer_out = stack(fwd_out, 0)

            inp = layer_out

        out = inp
        h_n_tensor = stack(h_n, 0)  # (D*num_layers, B, hidden)

        if self.batch_first:
            out = out.permute([1, 0, 2])

        return out, h_n_tensor

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, bidirectional={self.bidirectional}"
        )


# ── Multi-step RNN ────────────────────────────────────────────────────────────


class RNN(_CellNamingMixin, Module):  # type: ignore[misc]
    r"""Multi-layer Elman recurrent neural network (RNN).

    Applies a stack of Elman RNN cells over an input sequence.  At each
    time step :math:`t` the hidden state is updated by:

    .. math::

        h_t = \phi\!\left(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh}\right)

    where :math:`\phi` is either :math:`\tanh` (default) or
    :math:`\text{ReLU}`, controlled by ``nonlinearity``.

    The output of layer :math:`\ell` is fed as the input of layer
    :math:`\ell + 1`.  When ``bidirectional=True``, two RNNs process the
    sequence in opposite directions and their outputs are concatenated
    along the feature axis at each time step.

    Inter-layer dropout (probability ``dropout``) is applied between
    adjacent layers during training.

    .. warning::

        Vanilla RNNs are prone to the **vanishing gradient problem**:
        gradients are multiplied by :math:`W_{hh}` at every time step,
        causing them to shrink exponentially for sequences longer than
        ~10–20 steps.  For longer sequences, prefer :class:`LSTM` or
        :class:`GRU`, which use gating mechanisms to maintain gradient
        flow.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t` (denoted
        :math:`H` below).
    num_layers : int, optional
        Number of stacked RNN layers.  Default: ``1``.
    nonlinearity : {'tanh', 'relu'}, optional
        Activation function :math:`\phi`.  ``'tanh'`` is recommended for
        most use cases; ``'relu'`` can help when gradients vanish with
        ``tanh``.  Default: ``'tanh'``.
    bias : bool, optional
        If ``False``, all bias parameters are omitted.  Default: ``True``.
    batch_first : bool, optional
        If ``True``, input/output tensors are ``(N, L, *)`` instead of
        the default ``(L, N, *)``.  Default: ``False``.
    dropout : float, optional
        Dropout probability applied after each layer except the last.
        ``0.0`` disables dropout.  Default: ``0.0``.
    bidirectional : bool, optional
        If ``True``, use a bidirectional RNN; the output feature
        dimension becomes ``2 * hidden_size``.  Default: ``False``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Shape
    -----
    * **Input** ``x``: ``(L, N, input_size)`` or ``(N, L, input_size)``
      when ``batch_first=True``.
    * **h_0** (optional): ``(D * num_layers, N, H)`` where ``D = 2`` if
      bidirectional else ``1``.  Defaults to zeros.
    * **output**: ``(L, N, D * H)`` or ``(N, L, D * H)``.
    * **h_n**: ``(D * num_layers, N, H)`` — hidden state at the final
      time step for each layer and direction.

    Notes
    -----
    Internally this module stores one :class:`RNNCell` sub-module per
    layer per direction.  The :class:`_CellNamingMixin` flattens these
    into ``weight_ih_l{layer}`` etc. for checkpoint compatibility with
    the reference framework.

    :meth:`flatten_parameters` is a no-op kept for API compatibility.

    ``PackedSequence`` input is not yet supported.

    Examples
    --------
    Two-layer RNN, batch-first:

    >>> import lucid, lucid.nn as nn
    >>> rnn = nn.RNN(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)       # (N=2, L=5, I=8)
    >>> out, h_n = rnn(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))

    Bidirectional RNN with ReLU activation:

    >>> rnn_bi = nn.RNN(
    ...     8, 16, nonlinearity='relu',
    ...     bidirectional=True, batch_first=True,
    ... )
    >>> x2 = lucid.randn(3, 7, 8)
    >>> out2, h_n2 = rnn_bi(x2)
    >>> out2.shape    # D*H = 2*16 = 32
    (3, 7, 32)
    >>> h_n2.shape    # D*num_layers = 2*1 = 2
    (2, 3, 16)

    See Also
    --------
    RNNCell : Single time-step Elman cell.
    LSTM : Gated RNN with a separate cell state (better for long seqs).
    GRU : Gated RNN without a separate cell state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        num_dirs = 2 if bidirectional else 1
        for layer in range(num_layers):
            for d in range(num_dirs):
                suffix = "_reverse" if d == 1 else ""
                in_sz = input_size if layer == 0 else hidden_size * num_dirs
                self.add_module(
                    f"cell_l{layer}{suffix}",
                    RNNCell(
                        in_sz,
                        hidden_size,
                        bias=bias,
                        nonlinearity=nonlinearity,
                        device=device,
                        dtype=dtype,
                    ),
                )

    def _cell(self, layer: int, reverse: bool = False) -> RNNCell:
        suffix: str = "_reverse" if reverse else ""
        return self._modules[f"cell_l{layer}{suffix}"]  # type: ignore[return-value]

    def flatten_parameters(self) -> None:
        """No-op for API compatibility (see :meth:`LSTM.flatten_parameters`)."""
        return None

    def forward(  # type: ignore[override]  # narrower signature than Function/Module base by design
        self,
        x: Tensor,
        hx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        _check_not_packed(x, "RNN")
        if self.batch_first:
            x = x.permute([1, 0, 2])

        T: int = x.shape[0]
        B: int = x.shape[1]
        num_dirs: int = 2 if self.bidirectional else 1

        if hx is None:
            hx = zeros(
                self.num_layers * num_dirs,
                B,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )

        h_n: list[Tensor] = []
        inp = x

        for layer in range(self.num_layers):
            cell_fwd = self._cell(layer, reverse=False)
            h_fwd: Tensor = hx[layer * num_dirs]
            fwd_out: list[Tensor] = []
            for t in range(T):
                h_fwd = cast("Tensor", cell_fwd(inp[t], h_fwd))
                fwd_out.append(h_fwd)
            h_n.append(h_fwd)

            if self.bidirectional:
                cell_rev = self._cell(layer, reverse=True)
                h_rev: Tensor = hx[layer * num_dirs + 1]
                rev_out: list[Tensor] = []
                for t in range(T - 1, -1, -1):
                    h_rev = cast("Tensor", cell_rev(inp[t], h_rev))
                    rev_out.append(h_rev)
                rev_out.reverse()
                h_n.append(h_rev)
                layer_out = stack(
                    [_cat_last(fwd_out[t], rev_out[t]) for t in range(T)], 0
                )
            else:
                layer_out = stack(fwd_out, 0)

            inp = layer_out

        out = inp
        h_n_tensor = stack(h_n, 0)

        if self.batch_first:
            out = out.permute([1, 0, 2])

        return out, h_n_tensor

    def extra_repr(self) -> str:
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"nonlinearity={self.nonlinearity!r}, batch_first={self.batch_first}, "
            f"bidirectional={self.bidirectional}"
        )
