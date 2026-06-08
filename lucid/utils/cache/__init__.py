"""
lucid.utils.cache: key/value caches for incremental attention (generation).

Class names, method names, and argument names mirror the reference
``cache_utils`` module so that generation code ports across with only an
import change.  The eager default is :class:`DynamicCache`; encoder-decoder
models use :class:`EncoderDecoderCache`.  Canonical paths:
``lucid.utils.cache.Cache`` / ``DynamicCache`` / ``EncoderDecoderCache``.
"""

from lucid.utils.cache._base import Cache
from lucid.utils.cache._dynamic import DynamicCache
from lucid.utils.cache._encoder_decoder import EncoderDecoderCache

__all__ = [
    "Cache",
    "DynamicCache",
    "EncoderDecoderCache",
]
