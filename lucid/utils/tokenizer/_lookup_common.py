"""Internal helpers shared between the simple lookup-based
tokenizers (Char / Whitespace / Word / Regex).

Each of those tokenizers has the same persistence + special-token
loading boilerplate; factoring it here keeps the user-facing
modules focused on the algorithm.

Not part of the public API — names are underscore-prefixed at the
module level too (just to make it obvious in stack traces).
"""

import json
import os

from lucid.utils.tokenizer._base import SpecialTokens
from lucid.utils.tokenizer._bpe import _special_tokens_from_map


def _save_vocab_txt(vocab: dict[str, int], path: str) -> None:
    """Write a ``vocab.txt`` file: one token per line, line index = id.

    Same format BERT / WordPiece use; suitable for any lookup-style
    tokenizer.  Tokens are written in id-ascending order; any "holes"
    in the id range (rare) are filled with empty lines so id ↔ line
    mapping survives the round-trip.
    """
    if not vocab:
        with open(path, "w", encoding="utf-8") as f:
            pass
        return
    max_id = max(vocab.values())
    inv = {v: k for k, v in vocab.items()}
    with open(path, "w", encoding="utf-8") as f:
        for i in range(max_id + 1):
            f.write(inv.get(i, "") + "\n")


def _load_vocab_txt(path: str) -> dict[str, int]:
    """Inverse of :func:`_save_vocab_txt`.  Empty lines are skipped
    (the corresponding ids are simply absent from the returned dict)."""
    out: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            tok = line.rstrip("\n")
            if tok:
                out[tok] = i
    return out


def _load_special_tokens_map(
    directory: str,
) -> SpecialTokens | None:
    """Read ``special_tokens_map.json`` from ``directory`` if present."""
    sp_path = os.path.join(directory, "special_tokens_map.json")
    if not os.path.isfile(sp_path):
        return None
    with open(sp_path, encoding="utf-8") as f:
        sp = json.load(f)
    return _special_tokens_from_map(sp)
