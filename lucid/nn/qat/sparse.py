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
    """Quantization-aware embedding (trainable float table, fake-quant).

    A trainable float :class:`~lucid.nn.Embedding` whose lookup table is fake-quantized
    on every forward through a straight-through estimator (STE), so the learned
    embeddings survive the eventual int8 storage.  Inserted by
    :func:`lucid.quantization.prepare_qat`; :func:`lucid.quantization.convert` bakes the
    trained, fake-quantized table into a quantized
    :class:`~lucid.nn.quantized.Embedding`.

    Parameters
    ----------
    num_embeddings : int
        Number of rows in the lookup table (vocabulary size).
    embedding_dim : int
        Dimensionality of each embedding vector.
    padding_idx : int or None, optional
        If given, the embedding at this index stays fixed and does not contribute to
        the gradient.
    qconfig : QConfig
        Quantization recipe (keyword-only) supplying the weight
        :class:`~lucid.quantization.FakeQuantize` applied to the table during training.

    Notes
    -----
    Only the table (weight) is fake-quantized — there is no activation observer, since
    the output is a gather of already-quantized rows.
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
