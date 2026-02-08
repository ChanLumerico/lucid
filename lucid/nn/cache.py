from abc import ABC, abstractmethod
from typing import Any, override

import lucid
from lucid._tensor import Tensor


__all__ = ["Cache", "KVCache", "DynamicKVCache", "StaticKVCache", "EncoderDecoderCache"]


class Cache(ABC):
    @abstractmethod
    def reset(self) -> None: ...

    def reorder_cache(self, beam_idx: Tensor) -> None:
        self.batch_select_indices(beam_idx)

    @abstractmethod
    def batch_select_indices(self, indices: Tensor) -> None: ...

    @abstractmethod
    def batch_repeat_interleave(self, repeats: int) -> None: ...

    @abstractmethod
    def crop(self, max_length: int) -> None: ...

    def update(self, *args, **kwargs) -> Any: ...

    def get_max_cache_shape(self) -> int | None:
        return None


class KVCache(Cache):
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

    @override
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
    def _crop_layer(self, layer_idx: int, max_length: int) -> None: ...

    def _coerce_index_device(self, index: Tensor, device: str) -> Tensor:
        if index.device == device:
            return index
        if index.is_free:
            return index.to(device)
        raise lucid.DeviceMismatchError(to=device, from_=index.device)

    def batch_select_indices(self, indices: Tensor) -> None:
        if indices.ndim != 1:
            raise ValueError(
                "indices must be 1-D for cache selection "
                f"(got indices.ndim={indices.ndim})."
            )

        for layer_idx in range(len(self.key_cache)):
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
            if k is None or v is None:
                continue

            cur_indices = self._coerce_index_device(indices, k.device)
            self.key_cache[layer_idx] = k[cur_indices, ...]
            self.value_cache[layer_idx] = v[cur_indices, ...]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1 (got {repeats}).")
        if repeats == 1:
            return

        for layer_idx in range(len(self.key_cache)):
            k = self.key_cache[layer_idx]
            v = self.value_cache[layer_idx]
            if k is None or v is None:
                continue

            self.key_cache[layer_idx] = k.repeat(repeats, axis=0)
            self.value_cache[layer_idx] = v.repeat(repeats, axis=0)

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            raise ValueError(f"max_length must be >= 0 (got {max_length}).")

        for layer_idx in range(len(self.key_cache)):
            self._crop_layer(layer_idx, max_length)


class DynamicKVCache(KVCache):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: list[Tensor | None] = []
        self.value_cache: list[Tensor | None] = []
        self._seq_lens: list[int] = []

    def update(
        self,
        key: Tensor,
        value: Tensor,
        layer_idx: int,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._check_valid_layer_idx(layer_idx, dynamic=True)
        self._check_key_value_shape(key, value)

        if key.ndim < 2:
            raise ValueError(
                "KV cache expects key/value to have at least 2 dimensions "
                f"(got key.ndim={key.ndim})."
            )

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
            self._seq_lens.append(0)

        past_k = self.key_cache[layer_idx]
        past_v = self.value_cache[layer_idx]

        if cache_position is None:
            if past_k is None or past_v is None:
                new_k, new_v = key, value
            else:
                new_k = lucid.concatenate([past_k, key], axis=-2)
                new_v = lucid.concatenate([past_v, value], axis=-2)
        else:
            if cache_position.device != key.device:
                if cache_position.is_free:
                    cache_position = cache_position.to(key.device)
                else:
                    raise lucid.DeviceMismatchError(
                        to=key.device, from_=cache_position.device
                    )

            min_pos = int(lucid.min(cache_position).item())
            max_pos = int(lucid.max(cache_position).item())
            if min_pos < 0:
                raise ValueError(f"cache_position must be >= 0 (got {min_pos}).")

            target_len = max_pos + 1
            cur_len = self._seq_lens[layer_idx]

            if past_k is None or past_v is None:
                cache_shape = list(key.shape)
                cache_shape[-2] = target_len
                cache_shape_t = tuple(cache_shape)

                new_k = lucid.zeros(cache_shape_t, dtype=key.dtype, device=key.device)
                new_v = lucid.zeros(
                    cache_shape_t, dtype=value.dtype, device=value.device
                )
            else:
                expected_shape = past_k.shape[:-2] + past_k.shape[-1:]
                incoming_shape = key.shape[:-2] + key.shape[-1:]

                if expected_shape != incoming_shape:
                    raise ValueError(
                        "DynamicKVCache shape mismatch on non-seq dimensions: "
                        f"expected {expected_shape}, got {incoming_shape}"
                    )

                if cur_len >= target_len:
                    new_k, new_v = past_k, past_v
                else:
                    grow_shape = list(past_k.shape)
                    grow_shape[-2] = target_len - cur_len
                    grow_shape_t = tuple(grow_shape)

                    pad_k = lucid.zeros(
                        grow_shape_t, dtype=past_k.dtype, device=past_k.device
                    )
                    pad_v = lucid.zeros(
                        grow_shape_t, dtype=past_v.dtype, device=past_v.device
                    )
                    new_k = lucid.concatenate([past_k, pad_k], axis=-2)
                    new_v = lucid.concatenate([past_v, pad_v], axis=-2)

            if cache_position.ndim == 0:
                if key.shape[-2] != 1:
                    raise ValueError(
                        "0-d cache_position only supports single-token updates "
                        f"(got new_tokens={key.shape[-2]})."
                    )

                pos = int(cache_position.item())
                new_k[..., pos : pos + 1, :] = key
                new_v[..., pos : pos + 1, :] = value
                self._seq_lens[layer_idx] = max(cur_len, pos + 1)

            elif cache_position.ndim == 1:
                if cache_position.shape[0] != key.shape[-2]:
                    raise ValueError(
                        "cache_position length must match key/value seq_len "
                        f"(got {cache_position.shape[0]} vs {key.shape[-2]})."
                    )

                new_k[..., cache_position, :] = key
                new_v[..., cache_position, :] = value
                self._seq_lens[layer_idx] = max(cur_len, max_pos + 1)

            elif cache_position.ndim == 2:
                if key.ndim < 3:
                    raise ValueError(
                        "2-D cache_position requires key/value to have a batch axis "
                        f"(got key.ndim={key.ndim})."
                    )
                if cache_position.shape[0] != key.shape[0]:
                    raise ValueError(
                        "cache_position batch size must match key/value batch size "
                        f"(got {cache_position.shape[0]} vs {key.shape[0]})."
                    )
                if cache_position.shape[1] != key.shape[-2]:
                    raise ValueError(
                        "cache_position second dim must match key/value seq_len "
                        f"(got {cache_position.shape[1]} vs {key.shape[-2]})."
                    )

                for batch_idx in range(key.shape[0]):
                    pos = cache_position[batch_idx]
                    for token_idx in range(key.shape[-2]):
                        token_pos = int(pos[token_idx].item())
                        new_k[batch_idx, ..., token_pos : token_pos + 1, :] = key[
                            batch_idx, ..., token_idx : token_idx + 1, :
                        ]
                        new_v[batch_idx, ..., token_pos : token_pos + 1, :] = value[
                            batch_idx, ..., token_idx : token_idx + 1, :
                        ]
                self._seq_lens[layer_idx] = max(cur_len, max_pos + 1)

            else:
                raise ValueError(
                    "Only scalar, 1-D, or 2-D cache_position is supported "
                    f"(got cache_position.ndim={cache_position.ndim})."
                )
        if cache_position is None:
            self._seq_lens[layer_idx] = new_k.shape[-2]

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
        seq_len = self._seq_lens[layer_idx]
        return k[..., :seq_len, :], v[..., :seq_len, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._check_valid_layer_idx(layer_idx, dynamic=True)
        return self._seq_lens[layer_idx]

    def reset(self) -> None:
        self.key_cache.clear()
        self.value_cache.clear()
        self._seq_lens.clear()

    def _crop_layer(self, layer_idx: int, max_length: int) -> None:
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        if k is None or v is None:
            return

        seq_len = k.shape[-2]
        if seq_len <= max_length:
            return

        if max_length == 0:
            self.key_cache[layer_idx] = k[..., :0, :]
            self.value_cache[layer_idx] = v[..., :0, :]
            self._seq_lens[layer_idx] = 0
            return

        self.key_cache[layer_idx] = k[..., -max_length:, :]
        self.value_cache[layer_idx] = v[..., -max_length:, :]
        self._seq_lens[layer_idx] = max_length


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

        if key.ndim < 2:
            raise ValueError(
                "KV cache expects key/value to have at least 2 dimensions "
                f"(got key.ndim={key.ndim})."
            )

        new_tokens = key.shape[-2]
        if new_tokens < 1:
            raise ValueError(f"key/value seq_len must be >= 1 (got {new_tokens}).")

        cache_shape = list(key.shape)
        cache_shape[-2] = self.max_cache_len
        cache_shape_t = tuple(cache_shape)

        k_cache = self.key_cache[layer_idx]
        v_cache = self.value_cache[layer_idx]

        if k_cache is None or v_cache is None:
            k_cache = lucid.zeros(cache_shape_t, dtype=key.dtype, device=key.device)
            v_cache = lucid.zeros(cache_shape_t, dtype=value.dtype, device=value.device)

            self.key_cache[layer_idx] = k_cache
            self.value_cache[layer_idx] = v_cache

        else:
            if k_cache.shape != cache_shape_t or v_cache.shape != cache_shape_t:
                raise ValueError(
                    "StaticKVCache key/value cache shape mismatch: "
                    f"expected {cache_shape_t}, got {k_cache.shape} / {v_cache.shape}"
                )

        cur_len = self._seq_lens[layer_idx]

        if cache_position is None:
            start = cur_len
            end = start + new_tokens
            if end > self.max_cache_len:
                raise ValueError(
                    f"StaticKVCache exceeded max_cache_len={self.max_cache_len}: "
                    f"need end={end} (cur_len={cur_len}, new_tokens={new_tokens})."
                )

            k_cache[..., start:end, :] = key
            v_cache[..., start:end, :] = value
            self._seq_lens[layer_idx] = end

        else:
            if cache_position.device != key.device:
                if cache_position.is_free:
                    cache_position = cache_position.to(key.device)
                else:
                    raise lucid.DeviceMismatchError(
                        to=key.device, from_=cache_position.device
                    )

            min_pos = int(lucid.min(cache_position).item())
            max_pos = int(lucid.max(cache_position).item())

            if min_pos < 0:
                raise ValueError(f"cache_position must be >= 0 (got {min_pos}).")
            if max_pos >= self.max_cache_len:
                raise ValueError(
                    f"cache_position out of bounds for max_cache_len={self.max_cache_len}: "
                    f"max_pos={max_pos}."
                )

            if cache_position.ndim == 0:
                if new_tokens != 1:
                    raise ValueError(
                        "0-d cache_position only supports single-token updates "
                        f"(got new_tokens={new_tokens})."
                    )

                pos = int(cache_position.item())
                k_cache[..., pos : pos + 1, :] = key
                v_cache[..., pos : pos + 1, :] = value
                self._seq_lens[layer_idx] = max(cur_len, pos + 1)

            elif cache_position.ndim == 1:
                if cache_position.shape[0] != new_tokens:
                    raise ValueError(
                        "cache_position length must match key/value seq_len "
                        f"(got {cache_position.shape[0]} vs {new_tokens})."
                    )

                k_cache[..., cache_position, :] = key
                v_cache[..., cache_position, :] = value
                self._seq_lens[layer_idx] = max(cur_len, max_pos + 1)

            elif cache_position.ndim == 2:
                if key.ndim < 3:
                    raise ValueError(
                        "2-D cache_position requires key/value to have a batch axis "
                        f"(got key.ndim={key.ndim})."
                    )
                if cache_position.shape[0] != key.shape[0]:
                    raise ValueError(
                        "cache_position batch size must match key/value batch size "
                        f"(got {cache_position.shape[0]} vs {key.shape[0]})."
                    )
                if cache_position.shape[1] != new_tokens:
                    raise ValueError(
                        "cache_position second dim must match key/value seq_len "
                        f"(got {cache_position.shape[1]} vs {new_tokens})."
                    )

                for batch_idx in range(key.shape[0]):
                    pos = cache_position[batch_idx]
                    for token_idx in range(new_tokens):
                        token_pos = int(pos[token_idx].item())
                        k_cache[batch_idx, ..., token_pos : token_pos + 1, :] = key[
                            batch_idx, ..., token_idx : token_idx + 1, :
                        ]
                        v_cache[batch_idx, ..., token_pos : token_pos + 1, :] = value[
                            batch_idx, ..., token_idx : token_idx + 1, :
                        ]
                self._seq_lens[layer_idx] = max(cur_len, max_pos + 1)

            else:
                raise ValueError(
                    "Only scalar, 1-D, or 2-D cache_position is supported "
                    f"(got cache_position.ndim={cache_position.ndim})."
                )

        out = self.get(layer_idx)
        assert out is not None
        return out

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

    @override
    def get_max_cache_shape(self) -> int | None:
        return self.max_cache_len

    def reset(self) -> None:
        self.key_cache = [None] * self.num_layers
        self.value_cache = [None] * self.num_layers
        self._seq_lens = [0] * self.num_layers

    def _crop_layer(self, layer_idx: int, max_length: int) -> None:
        k = self.key_cache[layer_idx]
        v = self.value_cache[layer_idx]
        if k is None or v is None:
            return

        seq_len = self._seq_lens[layer_idx]
        if seq_len <= max_length:
            return

        if max_length == 0:
            self._seq_lens[layer_idx] = 0
            return

        start = seq_len - max_length
        tail_k = k[..., start:seq_len, :].detach()
        tail_v = v[..., start:seq_len, :].detach()

        k[..., :max_length, :] = tail_k
        v[..., :max_length, :] = tail_v

        self._seq_lens[layer_idx] = max_length


class EncoderDecoderCache(Cache):
    def __init__(
        self,
        self_attention_cache: KVCache | None = None,
        cross_attention_cache: KVCache | None = None,
    ) -> None:
        self.self_attention_cache = (
            self_attention_cache
            if self_attention_cache is not None
            else DynamicKVCache()
        )
        self.cross_attention_cache = (
            cross_attention_cache
            if cross_attention_cache is not None
            else DynamicKVCache()
        )
        self.is_updated: dict[int, bool] = {}

    @staticmethod
    def _select_cache(
        self_attention_cache: KVCache,
        cross_attention_cache: KVCache,
        is_cross_attention: bool,
    ) -> KVCache:
        return cross_attention_cache if is_cross_attention else self_attention_cache

    def update(
        self,
        key: Tensor,
        value: Tensor,
        layer_idx: int,
        cache_position: Tensor | None = None,
        is_cross_attention: bool = False,
    ) -> tuple[Tensor, Tensor]:
        target_cache = self._select_cache(
            self.self_attention_cache, self.cross_attention_cache, is_cross_attention
        )
        updated = target_cache.update(key, value, layer_idx, cache_position)
        if is_cross_attention:
            self.is_updated[layer_idx] = True
        return updated

    def get(
        self,
        layer_idx: int,
        is_cross_attention: bool = False,
    ) -> tuple[Tensor, Tensor] | None:
        target_cache = self._select_cache(
            self.self_attention_cache, self.cross_attention_cache, is_cross_attention
        )
        return target_cache.get(layer_idx)

    def get_seq_length(
        self, layer_idx: int = 0, is_cross_attention: bool = False
    ) -> int:
        target_cache = self._select_cache(
            self.self_attention_cache, self.cross_attention_cache, is_cross_attention
        )
        return target_cache.get_seq_length(layer_idx)

    def reset(self) -> None:
        self.self_attention_cache.reset()
        self.cross_attention_cache.reset()
        self.is_updated.clear()

    def batch_select_indices(self, indices: Tensor) -> None:
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def crop(self, max_length: int) -> None:
        self.self_attention_cache.crop(max_length)
        self.cross_attention_cache.crop(max_length)

    @override
    def get_max_cache_shape(self) -> int | None:
        return self.self_attention_cache.get_max_cache_shape()
