SENetConfig
===========

.. autoclass:: lucid.models.SENetConfig

`SENetConfig` stores the squeeze-and-excitation backbone choices used by
:class:`lucid.models.SENet`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class SENetConfig:
        block: Literal["se_basic", "bottleneck"]
        layers: tuple[int, int, int, int] | list[int]
        reduction: int = 16
        cardinality: int = 1
        base_width: int = 64
        num_classes: int = 1000
        in_channels: int = 3
        stem_width: int = 64
        stem_type: Literal["deep"] | None = None
        avg_down: bool = False
        channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
        block_args: dict[str, Any] = field(default_factory=dict)
