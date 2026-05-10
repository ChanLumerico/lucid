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
    """state_dict v2 hooks that expose flat ``weight_ih_l{L}{_reverse}`` keys.

    GRU / RNN currently store their per-layer weights inside child
    cell submodules (e.g. ``cell_l0`` of type ``GRUCell``).  The natural
    state_dict key for that layout is ``cell_l0.weight_ih`` — but most
    external checkpoints (and the reference framework itself) flatten
    these into ``weight_ih_l0``.  These overrides write/read the flat
    keys directly, while still tolerating the old ``cell_l*`` layout
    for backwards compatibility with checkpoints saved by earlier Lucid
    versions.
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
        destination: dict,
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
        state_dict: dict,
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
                    src: Tensor | None = state_dict.get(flat_key)
                    if src is None:
                        src = state_dict.get(legacy_key)
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
                    new_impl: _C_engine.TensorImpl = cast(
                        _C_engine.TensorImpl,
                        _C_engine.contiguous(converted._impl).clone_with_grad(
                            p.requires_grad
                        ),
                    )
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
    """Long Short-Term Memory recurrent layer.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input.
    hidden_size : int
        Number of features in the hidden state.
    num_layers : int, optional
        Number of recurrent layers (default: 1).
    bias : bool, optional
        If ``False``, the layer does not use bias weights (default: ``True``).
        Due to an engine limitation, a zero-valued bias is still passed to the
        engine; numerically equivalent to no bias, but bias tensors exist.
    batch_first : bool, optional
        If ``True``, input/output tensors are ``(batch, seq, feature)``
        instead of ``(seq, batch, feature)`` (default: ``False``).
    dropout : float, optional
        Dropout probability between LSTM layers (default: 0.0).
    bidirectional : bool, optional
        If ``True``, use a bidirectional LSTM (default: ``False``).

    Notes
    -----
    The output hidden state ``h_n`` has shape
    ``(D * num_layers, batch, hidden_size)`` where ``D = 2`` if bidirectional
    else ``D = 1``.  This matches reference convention.

    Examples
    --------
    >>> lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
    >>> x = lucid.randn(2, 5, 10)
    >>> output, (h_n, c_n) = lstm(x)
    >>> output.shape
    (2, 5, 20)
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
        T: int = int(x.shape[0])
        # Reverse range [T-1, T-2, …, 0] built on the engine.
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
        out_impl = cast(_C_engine.TensorImpl, lstm_result[0])
        h_impl = cast(_C_engine.TensorImpl, lstm_result[1])
        c_impl = cast(_C_engine.TensorImpl, lstm_result[2])
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

        T: int = int(x.shape[0])
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
    """Single-step Elman RNN cell."""

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
    """Single-step LSTM cell."""

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
    """Single-step GRU cell."""

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


class GRU(_CellNamingMixin, Module):
    """Multi-layer GRU.

    Returns ``(output, h_n)`` where ``h_n`` has shape
    ``(D * num_layers, batch, hidden_size)`` — matching reference convention.

    Parameters
    ----------
    input_size, hidden_size, num_layers, bias, batch_first, dropout,
    bidirectional : standard bidirectional sequence semantics.

    Examples
    --------
    >>> gru = nn.GRU(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)
    >>> out, h_n = gru(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))
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

        h_n: list[Tensor] = []  # type: ignore[name-defined]
        inp = x

        for layer in range(self.num_layers):
            # ── forward direction ────────────────────────────────────────────
            cell_fwd: GRUCell = self._cell(layer, reverse=False)
            h_fwd: Tensor = hx[layer * num_dirs]  # (B, hidden)
            fwd_out: list[Tensor] = []  # type: ignore[name-defined]
            for t in range(T):
                h_fwd = cast("Tensor", cell_fwd(inp[t], h_fwd))
                fwd_out.append(h_fwd)
            h_n.append(h_fwd)

            if self.bidirectional:
                # ── reverse direction ────────────────────────────────────────
                cell_rev = self._cell(layer, reverse=True)
                h_rev: Tensor = hx[layer * num_dirs + 1]
                rev_out: list[Tensor] = []  # type: ignore[name-defined]
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


class RNN(_CellNamingMixin, Module):
    """Multi-layer Elman RNN.

    Returns ``(output, h_n)`` where ``h_n`` has shape
    ``(D * num_layers, batch, hidden_size)`` — matching reference convention.

    Parameters
    ----------
    input_size, hidden_size, num_layers, nonlinearity, bias, batch_first,
    dropout, bidirectional : standard recurrent layer semantics.

    Examples
    --------
    >>> rnn = nn.RNN(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)
    >>> out, h_n = rnn(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))
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

        h_n: list[Tensor] = []  # type: ignore[name-defined]
        inp = x

        for layer in range(self.num_layers):
            cell_fwd = self._cell(layer, reverse=False)
            h_fwd: Tensor = hx[layer * num_dirs]
            fwd_out: list[Tensor] = []  # type: ignore[name-defined]
            for t in range(T):
                h_fwd = cast("Tensor", cell_fwd(inp[t], h_fwd))
                fwd_out.append(h_fwd)
            h_n.append(h_fwd)

            if self.bidirectional:
                cell_rev = self._cell(layer, reverse=True)
                h_rev: Tensor = hx[layer * num_dirs + 1]
                rev_out: list[Tensor] = []  # type: ignore[name-defined]
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
