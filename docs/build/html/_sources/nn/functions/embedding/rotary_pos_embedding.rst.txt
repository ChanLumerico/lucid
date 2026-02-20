nn.functional.rotary_pos_embedding
==================================

.. autofunction:: lucid.nn.functional.rotary_pos_embedding

The `rotary_pos_embedding` function applies Rotary Position Embedding (RoPE)
to the last dimension of an input tensor. It rotates each even/odd channel pair
using position-dependent angles and preserves the original tensor shape.

Function Signature
------------------

.. code-block:: python

    def rotary_pos_embedding(
        input_: Tensor,
        position_ids: Tensor | None = None,
        interleaved: bool = True,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  Input tensor of shape `(..., seq_len, embed_dim)`.
  The last dimension `embed_dim` must be even.

- **position_ids** (*Tensor | None*, optional):
  Optional 1-D position tensor of shape `(seq_len,)`. If `None`, positions
  are generated as `0, 1, ..., seq_len - 1`.

- **interleaved** (*bool*, optional):
  If `True`, applies adjacent-pair rotation layout `(0,1), (2,3), ...`.
  If `False`, applies half-split layout between first and second half channels.

Returns
-------

- **Tensor**:
  Tensor with the same shape as `input_`, after rotary position embedding is applied.

RoPE Formulation
----------------

RoPE applies a position-dependent rotation to each query/key channel pair.
For position index :math:`m` and pair index :math:`i`:

.. math::

    \theta_i = 10000^{-2(i-1) / D}, \quad i \in \{1, \dots, D/2\}

.. math::

    \phi_{m,i} = m\theta_i

2D Rotation Block
-----------------

For a single 2D pair :math:`(u, v)`, RoPE uses:

.. math::

    \mathbf{R}(\phi)=
    \begin{bmatrix}
    \cos\phi & -\sin\phi \\
    \sin\phi & \cos\phi
    \end{bmatrix}

so the rotated pair is :math:`\mathbf{R}(\phi_{m,i})[u, v]^\top`.

Generalized :math:`D`-Dimensional Rotation
------------------------------------------

For even :math:`D`, split channels into :math:`D/2` independent 2D subspaces.
The full rotary matrix is block diagonal:

.. math::

    \mathbf{R}_{\Theta,m}^{D}
    =
    \bigoplus_{i=1}^{D/2}
    \mathbf{R}(\phi_{m,i})

equivalently:

.. math::

    \mathbf{R}_{\Theta,m}^{D}
    =
    \mathrm{diag}\!\left(
    \mathbf{R}(\phi_{m,1}),
    \mathbf{R}(\phi_{m,2}),
    \dots,
    \mathbf{R}(\phi_{m,D/2})
    \right)

How It Is Applied in Attention
------------------------------

RoPE is applied after linear projection and before attention score computation:

.. math::

    \mathbf{q}_m = \mathbf{R}_{\Theta,m}^{D}\,\mathbf{W}_q \mathbf{x}_m,\quad
    \mathbf{k}_n = \mathbf{R}_{\Theta,n}^{D}\,\mathbf{W}_k \mathbf{x}_n

Then:

.. math::

    \mathbf{q}_m^\top \mathbf{k}_n
    =
    \mathbf{x}_m^\top \mathbf{W}_q^\top
    \left(\mathbf{R}_{\Theta,m}^{D}\right)^\top
    \mathbf{R}_{\Theta,n}^{D}
    \mathbf{W}_k \mathbf{x}_n
    =
    \mathbf{x}_m^\top \mathbf{W}_q^\top
    \mathbf{R}_{\Theta,n-m}^{D}
    \mathbf{W}_k \mathbf{x}_n

with:

.. math::

    \mathbf{R}_{\Theta,n-m}^{D}
    =
    \left(\mathbf{R}_{\Theta,m}^{D}\right)^\top
    \mathbf{R}_{\Theta,n}^{D}

This is the key point: relative position :math:`(n-m)` appears directly in
the attention dot product, consistent with RoFormer Eq. (16).

Computationally Efficient Equivalent Form
-----------------------------------------

Instead of explicit block-matrix multiplication, implementation uses:

.. math::

    \mathrm{RoPE}(\mathbf{x})
    =
    \mathbf{x}\odot\cos\boldsymbol{\phi}
    +
    \mathrm{rotate_half}(\mathbf{x})\odot\sin\boldsymbol{\phi}

where :math:`\mathrm{rotate_half}(x)` swaps each pair as
:math:`(x_{2i}, x_{2i+1}) \mapsto (-x_{2i+1}, x_{2i})`.
This is equivalent to the efficient realization described in RoFormer Eq. (34).

Implementation Notes
--------------------

- The function expects the sequence axis at `-2` and embedding axis at `-1`.
- Raises `ValueError` when `embed_dim` is odd.
- RoPE is parameter-free and deterministic.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> import lucid.nn.functional as F
    >>> x = lucid.random.randn(2, 8, 64)  # (batch, seq_len, embed_dim)
    >>> y = F.rotary_pos_embedding(x)
    >>> print(y.shape)
    (2, 8, 64)

    >>> q = lucid.random.randn(2, 12, 8, 64)  # (batch, heads, seq_len, head_dim)
    >>> q_rope = F.rotary_pos_embedding(q)
    >>> print(q_rope.shape)
    (2, 12, 8, 64)
