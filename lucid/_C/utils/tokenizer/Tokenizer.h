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

// Shared UTF-8, whitespace, and reverse-vocab helpers used across
// LookupTokenizer, BPE, WordPiece, and Unigram.
namespace detail {

struct WordRunState {
    std::vector<std::string> symbols;
    std::uint64_t count = 0;
};

inline bool is_ascii_whitespace(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline std::size_t utf8_lead_len(unsigned char c0) noexcept {
    if (c0 < 0x80)
        return 1;
    if ((c0 >> 5) == 0b110)
        return 2;
    if ((c0 >> 4) == 0b1110)
        return 3;
    if ((c0 >> 3) == 0b11110)
        return 4;
    return 1;  // invalid lead — fall through as single byte
}

inline std::size_t utf8_codepoint_len(const std::string& s, std::size_t pos) noexcept {
    if (pos >= s.size())
        return 0;
    const std::size_t cp_len = utf8_lead_len(static_cast<unsigned char>(s[pos]));
    return (pos + cp_len > s.size()) ? (s.size() - pos) : cp_len;
}

template <typename EmitFn>
inline void for_each_ascii_word(const std::string& text, EmitFn&& emit) {
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && is_ascii_whitespace(text[i]))
            ++i;
        if (i >= text.size())
            break;
        const std::size_t start = i;
        while (i < text.size() && !is_ascii_whitespace(text[i]))
            ++i;
        emit(text.substr(start, i - start));
    }
}

inline std::vector<std::string> split_ascii_whitespace(const std::string& text) {
    std::vector<std::string> out;
    for_each_ascii_word(text, [&](std::string word) { out.push_back(std::move(word)); });
    return out;
}

inline std::unordered_map<std::string, std::uint64_t>
count_whitespace_words(const std::vector<std::string>& corpus) {
    std::unordered_map<std::string, std::uint64_t> word_freq;
    for (const auto& doc : corpus) {
        for_each_ascii_word(doc, [&](std::string word) { ++word_freq[std::move(word)]; });
    }
    return word_freq;
}

inline void rebuild_id_to_token(const std::unordered_map<std::string, TokenId>& vocab,
                                std::vector<std::string>& id_to_token) {
    TokenId max_id = -1;
    for (const auto& [tok, id] : vocab) {
        if (id > max_id)
            max_id = id;
    }
    id_to_token.assign(static_cast<std::size_t>(max_id + 1), std::string{});
    for (const auto& [tok, id] : vocab) {
        if (id >= 0 && static_cast<std::size_t>(id) < id_to_token.size())
            id_to_token[static_cast<std::size_t>(id)] = tok;
    }
}

inline std::string lookup_id_to_token(const std::vector<std::string>& id_to_token, TokenId id) {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token.size())
        return "";
    return id_to_token[static_cast<std::size_t>(id)];
}

inline std::string decode_joined_tokens(const IdSequence& ids,
                                        const std::vector<std::string>& id_to_token,
                                        bool space_separated) {
    std::string out;
    bool first = true;
    for (TokenId id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token.size())
            continue;
        if (space_separated && !first)
            out += ' ';
        out += id_to_token[static_cast<std::size_t>(id)];
        first = false;
    }
    return out;
}

inline std::optional<std::pair<std::string, std::string>>
pick_best_adjacent_pair(const std::vector<WordRunState>& words) {
    std::unordered_map<std::string, std::uint64_t> pair_freq;
    for (const auto& ws : words) {
        for (std::size_t k = 0; k + 1 < ws.symbols.size(); ++k) {
            std::string key = ws.symbols[k] + '\x01' + ws.symbols[k + 1];
            pair_freq[key] += ws.count;
        }
    }
    if (pair_freq.empty())
        return std::nullopt;
    auto best_it = pair_freq.begin();
    for (auto it = pair_freq.begin(); it != pair_freq.end(); ++it) {
        if (it->second > best_it->second ||
            (it->second == best_it->second && it->first < best_it->first)) {
            best_it = it;
        }
    }
    if (best_it->second < 2)
        return std::nullopt;
    const std::string& key = best_it->first;
    const auto sep = key.find('\x01');
    return std::make_pair(key.substr(0, sep), key.substr(sep + 1));
}

inline void apply_symbol_merge(std::vector<WordRunState>& words,
                               const std::string& left,
                               const std::string& right,
                               const std::string& merged) {
    for (auto& ws : words) {
        std::vector<std::string> next;
        next.reserve(ws.symbols.size());
        std::size_t k = 0;
        while (k < ws.symbols.size()) {
            if (k + 1 < ws.symbols.size() && ws.symbols[k] == left && ws.symbols[k + 1] == right) {
                next.push_back(merged);
                k += 2;
            } else {
                next.push_back(ws.symbols[k]);
                ++k;
            }
        }
        ws.symbols = std::move(next);
    }
}

}  // namespace detail

}  // namespace lucid::utils::tokenizer
