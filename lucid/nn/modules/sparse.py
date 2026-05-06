"""
Sparse / embedding modules.
"""

import math
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid.nn.functional.sparse import embedding


class Embedding(Module):
    """Learnable embedding lookup table.

    Parameters
    ----------
    num_embeddings, embedding_dim : int
        Table dimensions.
    padding_idx : int | None
        If set, ``weight[padding_idx]`` is excluded from gradient updates
        and ``__init__`` zero-initialises that row.
    max_norm : float | None
        If set, every row whose ``norm_type``-norm exceeds ``max_norm`` is
        renormalised in place at every forward call.
    norm_type : float
        The p in ``L_p``-norm used by ``max_norm``.
    scale_grad_by_freq : bool
        Currently raised — index-frequency scaling has no engine path yet.
    sparse : bool
        Sparse gradients are not yet emitted; left here for API parity
        with the reference framework.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        if scale_grad_by_freq:
            raise NotImplementedError(
                "Embedding(scale_grad_by_freq=True) is not supported yet. "
                "Apply frequency weighting manually after backward()."
            )
        if padding_idx is not None and not (
            -num_embeddings <= padding_idx < num_embeddings
        ):
            raise ValueError(
                f"padding_idx must be within [-{num_embeddings}, "
                f"{num_embeddings}); got {padding_idx}"
            )
        self.num_embeddings: int = num_embeddings
        self.embedding_dim: int = embedding_dim
        # Normalise negative padding_idx for downstream comparisons.
        self.padding_idx: int | None = (
            padding_idx + num_embeddings
            if padding_idx is not None and padding_idx < 0
            else padding_idx
        )
        self.max_norm: float | None = max_norm
        self.norm_type: float = norm_type
        self.scale_grad_by_freq: bool = scale_grad_by_freq
        self.sparse: bool = sparse
        self.weight: Parameter = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)
        # Zero out the pad row on init — matches the reference framework
        # so untouched models do not leak random values through the pad slot.
        if self.padding_idx is not None:
            self._zero_pad_row()

    def _zero_pad_row(self) -> None:
        """Set ``weight[padding_idx]`` to zero in-place via numpy round-trip.

        Cheap because it only fires from ``__init__``; runtime forward
        does not need this.
        """
        import lucid as _lucid
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg

        arr = self.weight.numpy().copy()
        arr[self.padding_idx] = 0.0
        new_impl = _ce.TensorImpl(arr, _ce.Device.CPU, False)
        self.weight._impl = _iwg(new_impl, self.weight._impl.requires_grad)

    def _renorm_weight_inplace(self) -> None:
        """Apply ``max_norm`` rescaling to rows that exceed the cap."""
        import lucid as _lucid
        from lucid._C import engine as _ce
        from lucid._tensor.tensor import _impl_with_grad as _iwg
        import numpy as _np

        arr: _np.ndarray = self.weight.numpy().copy()
        # Compute per-row Lp-norm without an extra Lucid graph.
        if self.norm_type == 2.0:
            norms: _np.ndarray = _np.linalg.norm(arr, ord=2, axis=1)
        elif self.norm_type == 1.0:
            norms = _np.abs(arr).sum(axis=1)
        else:
            norms = (_np.abs(arr) ** self.norm_type).sum(axis=1) ** (
                1.0 / self.norm_type
            )
        scale: _np.ndarray = _np.minimum(self.max_norm / (norms + 1e-7), 1.0).astype(
            arr.dtype
        )
        arr = arr * scale[:, None]
        new_impl = _ce.TensorImpl(arr, _ce.Device.CPU, False)
        self.weight._impl = _iwg(new_impl, self.weight._impl.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        if self.max_norm is not None:
            self._renorm_weight_inplace()
        return embedding(x, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        s: str = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2.0:
            s += f", norm_type={self.norm_type}"
        return s


class EmbeddingBag(Module):
    """Embedding lookup with per-bag reduction (``'sum'`` / ``'mean'`` / ``'max'``).

    When *offsets* is ``None`` the input is expected to have shape ``(B, L)``
    and the reduction is applied over the length dimension *L*.
    When *offsets* is provided the input is a flat 1-D index tensor and
    *offsets* marks the start of each bag.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        mode: str = "mean",
        sparse: bool = False,
        padding_idx: int | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.padding_idx = padding_idx
        self.weight = Parameter(
            empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )
        init.normal_(self.weight)

    def forward(self, x: Tensor, offsets: Tensor | None = None) -> Tensor:
        from lucid.nn.functional.sampling import embedding_bag as _eb

        _mode_map = {"sum": "sum", "mean": "mean", "max": "max"}
        return _eb(
            x,
            self.weight,
            offsets=offsets,
            mode=_mode_map.get(self.mode, "mean"),
            padding_idx=self.padding_idx,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, "
            f"mode={self.mode!r}, padding_idx={self.padding_idx}"
        )
