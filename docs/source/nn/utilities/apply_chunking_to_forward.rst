nn.utils.apply_chunking_to_forward
==================================

.. autofunction:: lucid.nn.utils.apply_chunking_to_forward

Function Signature
------------------

.. code-block:: python

    def apply_chunking_to_forward(
        forward_fn: Callable[..., Tensor],
        chunk_size: int,
        chunk_dim: int,
        *input_tensors: Tensor,
    ) -> Tensor

Overview
--------

`apply_chunking_to_forward` splits input tensors into chunks along `chunk_dim`,
applies `forward_fn` independently to each chunk, and concatenates outputs back
along the same dimension.

When `forward_fn` is independent across `chunk_dim`, this yields the same
result as applying `forward_fn` to full tensors directly, while reducing peak
activation memory.

Parameters
----------

- **forward_fn** (*Callable[..., Tensor]*):
  Forward function to execute per chunk.

- **chunk_size** (*int*):
  Size of each chunk along `chunk_dim`.
  If `chunk_size == 0`, chunking is disabled and `forward_fn` is called once.

- **chunk_dim** (*int*):
  Dimension index used for chunking.

- **input_tensors** (*Tensor*):
  One or more tensors passed into `forward_fn`.
  All tensors must have the same size at `chunk_dim`.

Return Value
------------

- **Tensor**:
  Concatenated output tensor across all chunks.

Validation Rules
----------------

- `input_tensors` must be non-empty.
- `chunk_size` must be non-negative.
- `chunk_dim` must be a valid dimension for the inputs.
- All inputs must share the same length along `chunk_dim`.
- If chunking is enabled (`chunk_size > 0`), the size at `chunk_dim` must be
  divisible by `chunk_size`.

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn

    class TinyHead(nn.Module):
        def __init__(self, hidden_size: int, chunk_size: int = 64):
            super().__init__()
            self.decoder = nn.Linear(hidden_size, hidden_size)
            self.chunk_size = chunk_size
            self.seq_dim = 1

        def forward_chunk(self, hidden_states: lucid.Tensor) -> lucid.Tensor:
            return self.decoder(hidden_states)

        def forward(self, hidden_states: lucid.Tensor) -> lucid.Tensor:
            return nn.utils.apply_chunking_to_forward(
                self.forward_chunk,
                self.chunk_size,
                self.seq_dim,
                hidden_states,
            )
