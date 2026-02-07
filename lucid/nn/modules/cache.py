from abc import ABC, abstractmethod

import lucid
from lucid._tensor import Tensor


__all__ = ["KVCache", "DynamicKVCache", "StaticKVCache"]


class KVCache(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: list[Tensor | None]
        self.value_cache: list[Tensor | None]

    def _check_valid_layer_idx(self, layer_idx: int, dynamic: bool = False) -> None:
        if layer_idx < 0:
            raise ValueError(f"Invalid layer_idx '{layer_idx}'")
        if dynamic:
            return
        if layer_idx >= len(self.key_cache):
            raise ValueError(
                f"layer_idx '{layer_idx}' bigger "
                f"than cache length '{len(self.key_cache)}'"
            )

    def _check_key_value_shape(self, key: Tensor, value: Tensor) -> None:
        if key.shape != value.shape:
            raise ValueError(
                f"key and value has different shapes: {key.shape} != {value.shape}"
            )

    @abstractmethod
    def update(
        self,
        key: Tensor,
        value: Tensor,
        layer_idx: int,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...

    @abstractmethod
    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None: ...

    @abstractmethod
    def get_seq_length(self, layer_idx: int = 0) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...


class DynamicKVCache(KVCache):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: list[Tensor | None] = []
        self.value_cache: list[Tensor | None] = []

    def update(
        self,
        key: Tensor,
        value: Tensor,
        layer_idx: int,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._check_valid_layer_idx(layer_idx, dynamic=True)
        self._check_key_value_shape(key, value)

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        past_k = self.key_cache[layer_idx]
        past_v = self.value_cache[layer_idx]

        if past_k is None or past_v is None:
            new_k, new_v = key, value
        else:
            new_k = lucid.concatenate([past_k, key], axis=-2)
            new_v = lucid.concatenate([past_v, value], axis=-2)

        self.key_cache[layer_idx] = new_k
        self.value_cache[layer_idx] = new_v

        return new_k, new_v

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        if layer_idx >= len(self.key_cache):
            return None

        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        if k is None or v is None:
            return None
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._check_valid_layer_idx(layer_idx)
        item = self.get(layer_idx)
        if item is None:
            return 0

        k, _ = item
        return k.shape[-2]

    def reset(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()


class StaticKVCache(KVCache):
    def __init__(self, max_cache_len: int, num_layers: int) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self.num_layers = num_layers

        self.key_cache: list[Tensor | None] = [None] * num_layers
        self.value_cache: list[Tensor | None] = [None] * num_layers

        self._seq_lens: list[int] = [0] * num_layers

    def update(
        self,
        key: Tensor,
        value: Tensor,
        layer_idx: int,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._check_valid_layer_idx(layer_idx)
        self._check_key_value_shape(key, value)

        raise NotImplementedError

    def get(self, layer_idx: int) -> tuple[Tensor, Tensor] | None:
        self._check_valid_layer_idx(layer_idx)
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]

        if k is None or v is None:
            return None

        seq_len = self._seq_lens[layer_idx]
        return k[..., :seq_len, :], v[..., :seq_len, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._check_valid_layer_idx(layer_idx)
        return self._seq_lens[layer_idx]

    def reset(self) -> None:
        self.key_cache = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers
        self._seq_lens = [0] * self.num_layers
