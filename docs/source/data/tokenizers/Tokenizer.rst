data.tokenizers.Tokenizer
=========================

.. autoclass:: lucid.data.tokenizers.Tokenizer
   :members:
   :undoc-members:

The `Tokenizer` class defines a Hugging Face-style tokenizer interface for Lucid.
It is an abstract base class intended for custom tokenizer implementations in
`lucid.data.tokenizers`.

Overview
--------

The base interface separates tokenization and vocabulary conversion into explicit steps:

- `tokenize` converts raw text into a token list.
- `convert_tokens_to_ids` maps tokens to integer ids.
- `convert_ids_to_tokens` maps ids back to tokens.
- `convert_tokens_to_string` reconstructs text from tokens.

It also provides common utility methods such as:

- `encode` / `decode`
- `batch_encode` / `batch_decode`
- `build_inputs_with_special_tokens`
- `save_pretrained` / `from_pretrained` (abstract)

Class Signature
---------------

.. code-block:: python

    class Tokenizer(ABC):
        def __init__(
            self,
            unk_token: SpecialTokens | str = SpecialTokens.UNK,
            pad_token: SpecialTokens | str = SpecialTokens.PAD,
            bos_token: SpecialTokens | str | None = SpecialTokens.BOS,
            eos_token: SpecialTokens | str | None = SpecialTokens.EOS,
        ) -> None

Core Abstract Methods
---------------------

Subclasses must implement:

.. code-block:: python

    @property
    def vocab_size(self) -> int

    def tokenize(self, text: str) -> list[str]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]

    def convert_ids_to_tokens(self, ids: int | list[int]) -> str | list[str]

    def convert_tokens_to_string(self, tokens: list[str]) -> str

    def save_pretrained(self, save_directory: Path | str) -> list[str]

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs
    ) -> Tokenizer

Common Usage
------------

.. code-block:: python

    import lucid
    from lucid.data.tokenizers import Tokenizer

    class MyTokenizer(Tokenizer):
        ...

    tokenizer = MyTokenizer()
    ids = tokenizer.encode("hello world")
    text = tokenizer.decode(ids)

    # Tensor output for model input
    batch = tokenizer.batch_encode(
        ["hello world", "hi"],
        padding=True,
        return_tensor=True,
        device="cpu",
    )

.. note::

    - `encode(..., return_tensor=True)` returns `lucid.LongTensor`.
    - `batch_encode(..., return_tensor=True)` can return a 2D tensor only when
      sequence lengths are uniform or when `padding=True` is used.
    - `decode` accepts either `list[int]` or `lucid.LongTensor`.
