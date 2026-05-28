// lucid/_C/utils/tokenizer/Tokenizer.h
//
// Base interface for the C++ tokenizer family.  Mirrors the Python
// :class:`lucid.utils.tokenizer.Tokenizer` ABC so the
// :class:`*TokenizerFast` Python wrappers can hold a polymorphic
// pointer regardless of the underlying algorithm.
//
// The Fast tokenizers are designed for **encode/decode hot loops**:
// no per-call Python ↔ C++ marshalling beyond the input string and
// the output id vector.  Training is also exposed (in-place mutation
// of merge table / vocab) so users can call ``BPETokenizerFast.train``
// from Python directly.
//
// Lifetime: tokenizer instances are heap-allocated and owned by the
// Python wrapper (via ``std::unique_ptr`` returned from the factory
// + ``py::class_`` registration).
//
// Thread safety: encode/decode are const + reentrant; train mutates
// internal tables and is NOT thread-safe (caller responsibility).

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lucid::utils::tokenizer {

// 32-bit token id type — matches Hugging Face convention and is wide
// enough for every published vocab (largest is GPT-2 50257; even
// llama-3 (128 256) + BPE-trained variants fit comfortably).
using TokenId = std::int32_t;

// Container that every tokenizer's ``encode`` returns.  Wrapped as
// ``list[int]`` on the Python side via pybind's automatic conversion.
using IdSequence = std::vector<TokenId>;

// Special token registry.  All sub-tokenizers honour the same name
// scheme (``pad`` / ``unk`` / ``bos`` / ``eos`` / ``mask`` / ``sep`` /
// ``cls``) so :meth:`Tokenizer::__call__` can attach attention masks
// + special-token masks uniformly.  Custom special tokens (e.g. GPT-2's
// ``<|endoftext|>``) live in the ``extra`` map.
struct SpecialTokens {
    // Canonical optional slots — std::nullopt = "this token is not
    // defined for this tokenizer" (e.g. GPT-2 has no PAD by default).
    std::optional<TokenId> pad;
    std::optional<TokenId> unk;
    std::optional<TokenId> bos;
    std::optional<TokenId> eos;
    std::optional<TokenId> mask;
    std::optional<TokenId> sep;
    std::optional<TokenId> cls;
    // String-keyed extras — anything outside the canonical 7.  Stored
    // as ``(string, id)`` so the round-trip Python side can serialise
    // to ``tokenizer.json`` 's ``added_tokens`` block.
    std::unordered_map<std::string, TokenId> extra;
};

// Abstract base.  Each concrete algorithm (BPE / WordPiece / Unigram /
// ByteLevelBPE) subclasses it + overrides the four pure-virtual hooks.
// The Python Fast wrapper holds a ``std::unique_ptr<Tokenizer>`` and
// dispatches through the v-table.
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // ── Required overrides ─────────────────────────────────────────

    // Convert a string of input text into the corresponding token id
    // sequence.  Implementations should apply the algorithm-specific
    // pipeline: normalize → pre-tokenize → algorithm-specific encode.
    // Special tokens (BOS/EOS/CLS/SEP) are NOT auto-added — the
    // caller (the Python ``encode`` wrapper) decides via
    // ``add_special_tokens``.
    virtual IdSequence encode(const std::string& text) const = 0;

    // Convert a sequence of token ids back to text.  Special token
    // skipping is delegated to the Python wrapper (which knows the
    // ``skip_special_tokens`` flag); this raw method outputs every
    // token's surface form including specials.
    virtual std::string decode(const IdSequence& ids) const = 0;

    // Total vocabulary size — number of distinct ids the tokenizer
    // can emit (including special tokens).
    virtual std::size_t vocab_size() const = 0;

    // Algorithm name — used by ``save`` / ``from_pretrained`` to
    // route to the right loader.  Lower-case, e.g. ``"bpe"``,
    // ``"wordpiece"``, ``"unigram"``, ``"byte_bpe"``.
    virtual std::string algo() const = 0;

    // ── Optional overrides (default = noop / not-supported) ───────

    // In-place training from an iterable of text samples.  Default
    // throws ``std::runtime_error`` (algorithm doesn't support
    // training in C++).  BPE / WordPiece overrides should populate
    // the internal vocab + merge table from scratch.
    virtual void train(const std::vector<std::string>& corpus, std::size_t target_vocab_size);

    // Special token registry.  Default is empty.  Subclasses set
    // it during construction or via ``set_special_tokens``.
    virtual const SpecialTokens& special_tokens() const noexcept;
    virtual void set_special_tokens(SpecialTokens st);

    // Batched encode — default loops over ``encode``.  Subclasses
    // can override for parallelism / batched algorithm-specific
    // optimisations.
    virtual std::vector<IdSequence> encode_batch(const std::vector<std::string>& texts) const;

    // Batched decode — default loops over ``decode``.
    virtual std::vector<std::string> decode_batch(const std::vector<IdSequence>& batch) const;

    // Vocab introspection: map from token string → id.  Used by the
    // Python ``__call__`` for special-token-mask computation and by
    // ``save`` for serialisation.  Default empty; subclasses build it.
    virtual std::unordered_map<std::string, TokenId> get_vocab() const;

    // Inverse: id → token string.  Default scans ``get_vocab``;
    // subclasses with an O(1) reverse table should override.
    virtual std::string id_to_token(TokenId id) const;

protected:
    SpecialTokens special_;
};

}  // namespace lucid::utils::tokenizer
