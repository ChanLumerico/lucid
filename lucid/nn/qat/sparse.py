"""QAT ``Embedding`` — fake-quantizes the lookup table during training.

The table stays a trainable float parameter; each forward fake-quantizes it (STE)
so the embeddings learn to survive the eventual int8 storage.  ``convert`` bakes
the trained, fake-quantized table into a quantized
:class:`~lucid.nn.quantized.Embedding`.
"""

from typing import TYPE_CHECKING, cast, override

import lucid.nn as nn
import lucid.nn.functional as F

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization.qconfig import QConfig


class Embedding(nn.Embedding):
    r"""Quantization-aware embedding — trainable float table, fake-quant every forward.

    The training-time stand-in that :func:`lucid.quantization.prepare_qat` installs in
    place of a float :class:`~lucid.nn.Embedding`. It keeps a **trainable float lookup
    table** but fake-quantizes that table on every forward, so the learned embeddings
    experience the int8 rounding error while training and settle into values that survive
    the eventual int8 storage. This is what lets a QAT-trained embedding match its float
    accuracy after conversion, where a naively-quantized table would shift every token
    vector by an uncontrolled rounding step.

    **Only the table is fake-quantized.** Unlike the QAT :class:`~lucid.nn.qat.Linear` /
    conv layers, there is **no activation observer** here: the layer's output is a plain
    ``gather`` of rows that are *already* on the (fake-)quantized grid, so the output range
    is fully determined by the table and a second observer would be redundant. The module
    therefore carries a single ``weight_fake_quant`` and no ``activation_post_process``.

    **Straight-through estimator (STE).** Rounding a table entry is a step function whose
    derivative is zero almost everywhere; the fake-quant rounds in the forward pass but
    passes the gradient *straight through* to the float table in the backward pass, so the
    embeddings stay fully trainable. :func:`lucid.quantization.convert` then reads the
    trained table and the weight observer's final qparams and bakes them into an inference
    :class:`~lucid.nn.quantized.Embedding` whose rows are stored as int8 codes.

    On each forward the table is fake-quantized, then the rows for ``x`` are gathered:

    .. math::

        \operatorname{fake\_quant}(t) = \bigl(
            \operatorname{clamp}(\operatorname{round}(t / s) + z,\ q_{\min},\ q_{\max})
            - z\bigr)\, s,
        \qquad
        \frac{\partial\, \operatorname{fake\_quant}(t)}{\partial t} = 1

    .. math::

        y = \operatorname{fake\_quant}_{w}(W)[\, x \,]

    where :math:`s, z` are the scale / zero-point the weight ``FakeQuantize`` derives from
    its observer and :math:`q_{\min}, q_{\max}` are the grid bounds. The straight-through
    unit derivative keeps the table trainable despite the non-differentiable ``round``.

    Parameters
    ----------
    num_embeddings : int
        Number of rows in the lookup table (vocabulary size) — forwarded to the float
        :class:`~lucid.nn.Embedding` parent.
    embedding_dim : int
        Dimensionality of each embedding vector — forwarded to the float parent.
    padding_idx : int or None, optional
        If given, the embedding at this index stays fixed and does not contribute to the
        gradient; forwarded to the float parent.
    qconfig : QConfig, keyword-only
        Quantization recipe supplying the weight :class:`~lucid.quantization.FakeQuantize`
        applied to the table during training. Required — constructing the layer without
        one raises ``ValueError``.

    Attributes
    ----------
    weight_fake_quant : FakeQuantize
        The table observer + fake-quant built from ``qconfig.weight()``; rounds the float
        table each forward and tracks its range so ``convert`` can pick the int8 qparams.
        There is no ``activation_post_process`` — the gather output needs no separate grid.

    Notes
    -----
    - **Table-only fake-quant.** Only the weight table is fake-quantized; the layer has no
      activation observer because a gather of quantized rows is already on the grid.
    - **STE differentiability.** ``round`` is applied forward but the gradient passes
      through as the identity, so the float table trains normally.
    - **Both directions are wired for you.** :func:`lucid.quantization.prepare_qat` swaps
      the float embedding *in*; :func:`lucid.quantization.convert` folds this layer *out*
      into the matching :class:`~lucid.nn.quantized.Embedding`. Manual construction is
      rarely needed and requires an explicit keyword ``qconfig``.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> import lucid.quantization as Q
    >>> import lucid.nn.qat as nnqat
    >>> m = nn.Sequential(nn.Embedding(20, 8))
    >>> qat = Q.prepare_qat(m, Q.get_default_qat_qconfig_mapping())
    >>> isinstance(qat[0], nnqat.Embedding)
    True
    >>> _ = qat(lucid.tensor([[1, 5, 9]], dtype=lucid.int64))   # fake-quant + gather
    >>> qat.eval()
    >>> qc = Q.convert(qat)                                     # -> quantized.Embedding
    >>> type(qc[0]).__name__
    'Embedding'

    A ``qconfig`` is mandatory — constructing the layer directly without one raises:

    >>> nnqat.Embedding(20, 8)
    Traceback (most recent call last):
        ...
    ValueError: qat.Embedding requires a qconfig

    See Also
    --------
    lucid.nn.quantized.Embedding : The int8 inference embedding ``convert`` bakes into.
    lucid.nn.qat.Linear : The QAT layer that *does* carry an activation observer.
    lucid.quantization.prepare_qat : Swaps the float ``Embedding`` for this QAT layer.
    lucid.quantization.convert : Bakes the trained QAT table into the quantized embedding.
    """

    weight_fake_quant: FakeQuantize

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        *,
        qconfig: QConfig | None = None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)
        if qconfig is None:
            raise ValueError("qat.Embedding requires a qconfig")
        self.qconfig = qconfig
        self.weight_fake_quant = cast("FakeQuantize", qconfig.weight())

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]  # index → embeddings
        """Fake-quantize the table, then look up ``x``."""
        w_q = cast("Tensor", self.weight_fake_quant(self.weight))
        return F.embedding(x, w_q, self.padding_idx)

    @classmethod
    def from_float(cls, mod: nn.Module) -> "Embedding":
        """Build a QAT ``Embedding`` from a float one (shares the trained table)."""
        emb = cast("nn.Embedding", mod)
        qat = cls(
            emb.num_embeddings,
            emb.embedding_dim,
            padding_idx=emb.padding_idx,
            qconfig=cast("QConfig", mod.qconfig),
        )
        qat.weight = emb.weight
        return qat
