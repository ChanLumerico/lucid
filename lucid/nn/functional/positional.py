"""Positional encoding primitives — Lucid-specific extensions to ``nn.functional``.

PyTorch's ``nn.functional`` doesn't ship positional-encoding helpers (every
model file rebuilds them locally).  Lucid centralises the common patterns so
families consume a single canonical implementation:

    * Sinusoidal positional encoding (1D)  — Vaswani et al., 2017
    * Sinusoidal positional encoding (2D)  — DETR §A.4 (Carion et al., 2020)
    * Rotary position embedding (RoPE)     — Su et al., 2021

The functional API here is **stateless** — each call rebuilds the table on
the requested device.  For training-time hot paths prefer the module form
in :mod:`lucid.nn` which caches the table as a buffer.
"""

import math
from typing import TYPE_CHECKING

import lucid

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional embedding — 1D (Vaswani et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────


def sinusoidal_embedding(
    num_positions: int,
    embedding_dim: int,
    *,
    base: float = 10_000.0,
    device: str = "cpu",
) -> Tensor:
    r"""Build the 1-D sinusoidal positional encoding table from "Attention Is All You Need".

    Returns a fixed (non-learnable) lookup table that injects absolute
    position information into token embeddings without adding parameters.
    Successive frequencies form a geometric progression spanning
    wavelengths from :math:`2\pi` to :math:`2\pi \cdot \text{base}`, so the
    encoding's components vary at vastly different rates and uniquely
    identify each position even at large sequence lengths.

    Parameters
    ----------
    num_positions : int
        Number of distinct positions :math:`p \in [0, \text{num\_positions})`.
    embedding_dim : int
        Per-position embedding size :math:`d`.  Must be even — half the
        entries hold ``sin`` values and half hold ``cos`` values.
    base : float, optional
        Frequency base :math:`\theta_0`.  Vaswani et al. use ``10_000``;
        larger values give longer effective context windows at the cost of
        finer per-step discrimination.
    device : str, optional
        Target device (``"cpu"`` or ``"metal"``) for the resulting buffer.

    Returns
    -------
    Tensor
        ``(num_positions, embedding_dim)`` float tensor.

    Raises
    ------
    ValueError
        If ``embedding_dim`` is not even.

    Notes
    -----
    Equation (5) of Vaswani et al. (2017):

    .. math::

        \mathrm{PE}_{p,\,2i}   &= \sin\!\left(p / \text{base}^{2i / d}\right) \\
        \mathrm{PE}_{p,\,2i+1} &= \cos\!\left(p / \text{base}^{2i / d}\right)

    The table is pure (deterministic in its arguments) so callers can
    safely share the result across model instances when dimensions match.
    For training-time hot paths prefer the module form in
    :mod:`lucid.nn`, which caches the table as a buffer.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import sinusoidal_embedding
    >>> pe = sinusoidal_embedding(num_positions=128, embedding_dim=64)
    >>> pe.shape
    (128, 64)
    """
    if embedding_dim % 2 != 0:
        raise ValueError(
            f"sinusoidal_embedding requires an even embedding_dim, "
            f"got {embedding_dim}"
        )
    half = embedding_dim // 2

    # Build the (num_positions, embedding_dim) table in Python and ship once.
    rows: list[list[float]] = []
    for p in range(num_positions):
        row: list[float] = [0.0] * embedding_dim
        for i in range(half):
            omega = math.pow(base, -2.0 * i / embedding_dim)
            row[2 * i] = math.sin(p * omega)
            row[2 * i + 1] = math.cos(p * omega)
        rows.append(row)
    return lucid.tensor(rows, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Sinusoidal positional embedding — 2D (DETR §A.4 / Carion et al., 2020)
# ─────────────────────────────────────────────────────────────────────────────


def sinusoidal_embedding_2d(
    height: int,
    width: int,
    embedding_dim: int,
    *,
    base: float = 10_000.0,
    device: str = "cpu",
) -> Tensor:
    r"""Build the 2-D sinusoidal positional encoding from DETR (Carion et al., 2020).

    Extends the 1-D sinusoidal encoding to spatial feature maps by
    concatenating two independent encodings — one for the column index
    and one for the row index — each occupying half of the embedding
    dimension.  This gives a position-unique vector for every grid cell
    without learnable parameters, and is the encoding used by DETR
    (§A.4), DiT, and other 2-D image transformers.

    Parameters
    ----------
    height : int
        Feature-map height :math:`H`.
    width : int
        Feature-map width :math:`W`.
    embedding_dim : int
        Per-position embedding size :math:`d`.  Must be divisible by 4 —
        each axis contributes :math:`d/2` dimensions of paired
        ``sin`` / ``cos`` values, so the half itself must be even.
    base : float, optional
        Frequency base.  DETR uses ``10_000``.
    device : str, optional
        Target device (``"cpu"`` or ``"metal"``).

    Returns
    -------
    Tensor
        ``(height * width, embedding_dim)`` float tensor, ordered
        row-major (outer loop ``r ∈ [0, H)``, inner loop ``c ∈ [0, W)``).

    Raises
    ------
    ValueError
        If ``embedding_dim`` is not divisible by 4.

    Notes
    -----
    Layout per position :math:`(r, c)`:

    .. math::

        \text{out}[r \cdot W + c, \; :d/2] &= \text{PE}_\text{col}(c) \\
        \text{out}[r \cdot W + c, \; d/2:] &= \text{PE}_\text{row}(r)

    where each axis-table is the standard 1-D encoding at dimension
    :math:`d/2`.  Flatten the result into a sequence of length
    :math:`H \cdot W` and add it to flattened image features before the
    first transformer block.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import sinusoidal_embedding_2d
    >>> pe = sinusoidal_embedding_2d(height=16, width=16, embedding_dim=128)
    >>> pe.shape
    (256, 128)
    """
    if embedding_dim % 4 != 0:
        raise ValueError(
            f"sinusoidal_embedding_2d requires embedding_dim divisible "
            f"by 4, got {embedding_dim}"
        )
    half = embedding_dim // 2  # dims per spatial axis
    quarter = half // 2  # pairs per axis

    def _axis_table(n: int) -> list[list[float]]:
        rows: list[list[float]] = []
        for idx in range(n):
            row: list[float] = []
            for i in range(quarter):
                omega = math.pow(base, -2.0 * i / half)
                row.append(math.sin(idx * omega))
                row.append(math.cos(idx * omega))
            rows.append(row)
        return rows

    col_table = _axis_table(width)  # (W, half)
    row_table = _axis_table(height)  # (H, half)

    # Tile to (H, W, d) then flatten to (H*W, d).
    out: list[list[float]] = []
    for r in range(height):
        for c in range(width):
            out.append(col_table[c] + row_table[r])
    return lucid.tensor(out, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary position embedding (RoPE — Su et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────


def _rotate_half(x: Tensor) -> Tensor:
    """Split the last dim in half, negate the second half, swap.

    For ``x = [x_0, ..., x_{d-1}]`` returns
    ``[-x_{d/2}, ..., -x_{d-1}, x_0, ..., x_{d/2-1}]``.  This is the "half-
    rotation" form used by HuggingFace / Meta LLaMA — pairs ``x_i`` with
    ``x_{i + d/2}`` rather than ``x_{2i}`` with ``x_{2i+1}``, matching the
    layout precomputed in :class:`lucid.nn.RotaryEmbedding`'s cos / sin tables.
    """
    half = int(x.shape[-1]) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return lucid.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Apply Rotary Position Embedding (RoPE) to query and key tensors.

    Encodes absolute position by **rotating** each consecutive pair of
    features through an angle that grows linearly with the token index.
    Unlike additive sinusoidal encodings, RoPE acts multiplicatively
    inside attention — it endows the dot product :math:`q_m \cdot k_n`
    with an explicit dependence on the relative offset :math:`m - n`,
    giving the model translation-equivariance "for free".  This is the
    encoding used by LLaMA, GPT-NeoX, PaLM, and most modern open-weights
    LLMs.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape ``(*, seq_len, d_head)``.  Any leading
        broadcast-compatible dims (batch, head, ...) are allowed.
    k : Tensor
        Key tensor of shape ``(*, seq_len, d_head)``.
    cos : Tensor
        Precomputed cosine table of shape ``(max_pos, d_head)``.
        Typically constructed once by :class:`lucid.nn.RotaryEmbedding`.
    sin : Tensor
        Precomputed sine table of shape ``(max_pos, d_head)``.
    position_ids : Tensor, optional
        Integer positions of shape ``(seq_len,)`` to gather from ``cos``
        / ``sin``.  When ``None`` (default), the first ``seq_len`` rows
        of the tables are used — appropriate for left-to-right
        tokenisation.  Pass custom IDs for sequence packing, KV caching,
        or sliding-window attention.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(q_rotated, k_rotated)`` — same shapes as the inputs.

    Notes
    -----
    Mathematical form for each feature pair :math:`(x_i, x_{i+d/2})` at
    position :math:`p`:

    .. math::

        \begin{pmatrix} x'_i \\ x'_{i+d/2} \end{pmatrix} =
        \begin{pmatrix} \cos\theta_{p,i} & -\sin\theta_{p,i} \\
                        \sin\theta_{p,i} &  \cos\theta_{p,i} \end{pmatrix}
        \begin{pmatrix} x_i \\ x_{i+d/2} \end{pmatrix}

    Lucid uses the "half-rotation" pairing
    :math:`(x_i, x_{i+d/2})` — the HuggingFace / LLaMA layout — rather
    than the original paper's interleaved pairing :math:`(x_{2i}, x_{2i+1})`.
    The two are equivalent up to a permutation of dimensions, and the
    half-rotation form is friendlier to contiguous matmul kernels.

    Inside attention the rotated dot product satisfies
    :math:`q_m'^\top k_n' = q_m^\top R(m-n) \, k_n`, so it depends only on
    the relative offset — the property RoPE is designed to produce.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.functional import apply_rotary_emb
    >>> q = lucid.randn(2, 8, 16, 64)          # (B, H, S, d_head)
    >>> k = lucid.randn(2, 8, 16, 64)
    >>> cos = lucid.randn(64, 64)              # precomputed up to max_pos=64
    >>> sin = lucid.randn(64, 64)
    >>> q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    >>> q_rot.shape, k_rot.shape
    ((2, 8, 16, 64), (2, 8, 16, 64))
    """
    seq_len = int(q.shape[-2])

    if position_ids is None:
        cos_emb = cos[:seq_len]
        sin_emb = sin[:seq_len]
    else:
        cos_emb = cos[position_ids]
        sin_emb = sin[position_ids]

    while cos_emb.ndim < q.ndim:
        cos_emb = cos_emb.unsqueeze(0)
        sin_emb = sin_emb.unsqueeze(0)

    q_rot = q * cos_emb + _rotate_half(q) * sin_emb
    k_rot = k * cos_emb + _rotate_half(k) * sin_emb
    return q_rot, k_rot
