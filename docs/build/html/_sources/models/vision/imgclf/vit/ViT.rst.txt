ViT
===

.. toctree::
    :maxdepth: 1
    :hidden:

    ViTConfig.rst
    vit_tiny.rst
    vit_small.rst
    vit_base.rst
    vit_large.rst
    vit_huge.rst

|transformer-badge| |vision-transformer-badge| 

.. autoclass:: lucid.models.ViT

The `ViT` class provides a full implementation of the Vision Transformer model,
including patch embedding, positional encoding, and the final classification
head. Model structure is defined through `ViTConfig`.

.. mermaid::
    :name: ViT

    %%{init: {"flowchart":{"curve":"monotoneX","nodeSpacing":50,"rankSpacing":50}} }%%
    flowchart LR
      linkStyle default stroke-width:2.0px
      subgraph sg_m0["<span style='font-size:20px;font-weight:700'>vit_base</span>"]
      style sg_m0 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
        m1["Conv2d<br/><span style='font-size:11px;color:#c53030;font-weight:400'>(1,3,224,224) → (1,768,14,14)</span>"];
        m2["Dropout"];
        subgraph sg_m3["TransformerEncoder"]
        style sg_m3 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
          subgraph sg_m4["layers"]
            direction TB;
          style sg_m4 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
            subgraph sg_m5["TransformerEncoderLayer x 12"]
              direction TB;
            style sg_m5 fill:#000000,fill-opacity:0.05,stroke:#000000,stroke-opacity:0.75,stroke-width:1px
              m5_in(["Input"]);
              m5_out(["Output"]);
      style m5_in fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
      style m5_out fill:#e2e8f0,stroke:#64748b,stroke-width:1px;
              m6["MultiHeadAttention<br/><span style='font-size:11px;color:#2f855a;font-weight:400'>(1,197,768)x3 → (1,197,768)</span>"];
              m7(["Linear x 2<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,197,768) → (1,197,3072)</span>"]);
              m8(["Dropout x 3"]);
              m9(["LayerNorm x 2"]);
            end
          end
        end
        m10["LayerNorm"];
        m11["Linear<br/><span style='font-size:11px;color:#2b6cb0;font-weight:400'>(1,768) → (1,1000)</span>"];
      end
      input["Input<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,3,224,224)</span>"];
      output["Output<br/><span style='font-size:11px;color:#a67c00;font-weight:400'>(1,1000)</span>"];
      style input fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style output fill:#fff3cd,stroke:#a67c00,stroke-width:1px;
      style m1 fill:#ffe8e8,stroke:#c53030,stroke-width:1px;
      style m2 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m6 fill:#f0fff4,stroke:#2f855a,stroke-width:1px;
      style m7 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      style m8 fill:#edf2f7,stroke:#4a5568,stroke-width:1px;
      style m9 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m10 fill:#e6fffa,stroke:#2c7a7b,stroke-width:1px;
      style m11 fill:#ebf8ff,stroke:#2b6cb0,stroke-width:1px;
      input --> m1;
      m1 --> m2;
      m10 --> m11;
      m11 --> output;
      m2 -.-> m6;
      m5_in -.-> m6;
      m5_out --> m10;
      m5_out -.-> m5_in;
      m6 -.-> m8;
      m7 -.-> m8;
      m8 -.-> m7;
      m8 --> m9;
      m9 -.-> m5_in;
      m9 --> m5_out;
      m9 -.-> m7;

Class Signature
---------------

.. code-block:: python

    class ViT(nn.Module):
        def __init__(self, config: ViTConfig) -> None

Parameters
----------
- **config** (*ViTConfig*):
  Configuration object describing the input shape, patch size, embedding width,
  transformer depth, attention heads, MLP width, and classifier size.

Architecture
------------

The ViT model follows the standard image transformer pipeline:

1. **Patch Embedding**:

   - The input image is split into non-overlapping patches by a strided convolution.
   - Each patch is projected into the transformer embedding space.

2. **Class Token and Position Embedding**:

   - A learnable class token is prepended to the patch sequence.
   - A learnable positional embedding is added to preserve patch order.

3. **Transformer Encoder**:

   - The sequence is processed by stacked encoder blocks with multi-head self-attention.
   - Each block mixes global context and feedforward projection layers.

4. **Classification Head**:

   - The final class token is normalized and projected to `num_classes`.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> config = models.ViTConfig(
    ...     image_size=224,
    ...     patch_size=16,
    ...     num_classes=1000,
    ...     embedding_dim=768,
    ...     depth=12,
    ...     num_heads=12,
    ...     mlp_dim=3072,
    ... )
    >>> vit = models.ViT(config)
    >>> print(vit)
    ViT()
