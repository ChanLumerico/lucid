from dataclasses import dataclass
from typing import Any

from lucid._jit.executor import CompiledPlan


@dataclass(frozen=True)
class CacheKey:
    shape_dtypes: tuple[tuple[tuple[int, ...], Any, str], ...]
    grad_enabled: bool
    training_mode: bool

    @classmethod
    def from_inputs(
        cls,
        inputs: tuple,
        grad_enabled: bool,
        training_mode: bool,
    ) -> CacheKey:
        shape_dtypes = tuple(
            (tuple(t.shape), t.dtype, t.device) for t in inputs if hasattr(t, "shape")
        )
        return cls(
            shape_dtypes=shape_dtypes,
            grad_enabled=grad_enabled,
            training_mode=training_mode,
        )


class PlanCache:
    def __init__(self, max_entries: int = 8) -> None:
        self._cache: dict[CacheKey, CompiledPlan] = {}
        self._max_entries = max_entries

    def get(self, key: CacheKey) -> CompiledPlan | None:
        return self._cache.get(key)

    def put(self, key: CacheKey, plan: CompiledPlan) -> None:
        if len(self._cache) >= self._max_entries:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = plan

    def invalidate(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
