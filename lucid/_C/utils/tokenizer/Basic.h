// lucid/_C/utils/tokenizer/Basic.h
//
// "Basic" tokenizer family — every concrete class here boils down to
// **vocab lookup over a pre-tokenized sequence**.  The algorithms
// differ only in HOW the input string is chopped into chunks before
// lookup:
//
//   * :class:`ByteTokenizer`        — split into individual UTF-8 bytes
//                                     (vocab is fixed 256-entry).
//   * :class:`CharTokenizer`        — split into Unicode codepoints.
//   * :class:`WhitespaceTokenizer`  — split on whitespace.
//   * :class:`WordTokenizer`        — same as Whitespace + an UNK
//                                     fallback for OOV.
//   * :class:`RegexTokenizer`       — split via a user-supplied
//                                     regular expression (std::regex).
//
// All five derive from a tiny intermediate ``LookupTokenizer`` base
// that holds the shared vocab + reverse table + OOV/UNK handling.
// Concrete subclasses override only ``split_to_chunks_`` to express
// the pre-tokenization rule.
//
// Training behaviour: every tokenizer in this family builds its vocab
// from the corpus by collecting unique chunks (and assigning ids in
// insertion order).  Byte is special — its vocab is fixed at 256 so
// ``train`` is a no-op.

#pragma once

#include <cstdint>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "Tokenizer.h"

namespace lucid::utils::tokenizer {

// Shared base for every lookup-based tokenizer.  Holds the vocab +
// reverse table + UNK fallback + the standard encode/decode/train
// plumbing.  Concrete subclasses implement :meth:`split_to_chunks_`.
class LookupTokenizer : public Tokenizer {
public:
    LookupTokenizer();
    explicit LookupTokenizer(std::unordered_map<std::string, TokenId> vocab);

    // Tokenizer overrides — shared across all 5 concretes.
    IdSequence encode(const std::string& text) const override;
    std::string decode(const IdSequence& ids) const override;
    std::size_t vocab_size() const override { return vocab_.size(); }
    std::unordered_map<std::string, TokenId> get_vocab() const override {
        return vocab_;
    }
    std::string id_to_token(TokenId id) const override;

    // Default training: pre-tokenize via subclass's
    // ``split_to_chunks_`` + insert each unique chunk into vocab in
    // insertion order until ``target_vocab_size`` is reached.
    // ``ByteTokenizer`` overrides with a no-op (its vocab is fixed).
    void train(const std::vector<std::string>& corpus,
               std::size_t target_vocab_size) override;

protected:
    // Algorithm-specific pre-tokenization — chop ``text`` into the
    // list of chunks that the lookup step then turns into ids.
    virtual std::vector<std::string>
    split_to_chunks_(const std::string& text) const = 0;

    // Rebuild reverse table after vocab mutation.
    void rebuild_id_to_token_();

    std::unordered_map<std::string, TokenId> vocab_;
    std::vector<std::string> id_to_token_;
};

// ── ByteTokenizer ───────────────────────────────────────────────────
//
// Vocab is fixed at 256 — each UTF-8 byte maps to id == byte value.
// No OOV is possible (every input is a sequence of bytes).  Useful as
// a robust multilingual fallback (ByT5, byte-level LMs).
class ByteTokenizer final : public LookupTokenizer {
public:
    ByteTokenizer();
    std::string algo() const override { return "byte"; }
    // Fixed-vocab; training is a no-op (warns via the Python wrapper
    // if invoked — here we silently keep the vocab unchanged).
    void train(const std::vector<std::string>& /*corpus*/,
               std::size_t /*target_vocab_size*/) override {}

protected:
    std::vector<std::string> split_to_chunks_(
        const std::string& text) const override;
};

// ── CharTokenizer ───────────────────────────────────────────────────
//
// One token per Unicode codepoint.  Vocab is the set of distinct
// codepoints encountered during training (or supplied at construction
// time).
class CharTokenizer final : public LookupTokenizer {
public:
    CharTokenizer();
    explicit CharTokenizer(std::unordered_map<std::string, TokenId> vocab);
    std::string algo() const override { return "char"; }

protected:
    std::vector<std::string> split_to_chunks_(
        const std::string& text) const override;
};

// ── WhitespaceTokenizer ─────────────────────────────────────────────
//
// One token per whitespace-delimited word.  Whitespace itself is
// dropped from the vocab — decode joins tokens with a single space.
class WhitespaceTokenizer final : public LookupTokenizer {
public:
    WhitespaceTokenizer();
    explicit WhitespaceTokenizer(
        std::unordered_map<std::string, TokenId> vocab);
    std::string algo() const override { return "whitespace"; }
    std::string decode(const IdSequence& ids) const override;

protected:
    std::vector<std::string> split_to_chunks_(
        const std::string& text) const override;
};

// ── WordTokenizer ───────────────────────────────────────────────────
//
// Same chunking as Whitespace, but emits UNK for OOV instead of
// dropping (which Whitespace also does — the difference is purely
// semantic / API-clarity).  The Python wrapper enforces that
// ``special_tokens.unk`` is set.
class WordTokenizer final : public LookupTokenizer {
public:
    WordTokenizer();
    explicit WordTokenizer(std::unordered_map<std::string, TokenId> vocab);
    std::string algo() const override { return "word"; }
    std::string decode(const IdSequence& ids) const override;

protected:
    std::vector<std::string> split_to_chunks_(
        const std::string& text) const override;
};

// ── RegexTokenizer ──────────────────────────────────────────────────
//
// Pre-tokenization via a user-supplied ECMA-style regular expression.
// Every match is emitted as a chunk (non-matching spans are dropped).
// Pattern is captured by value at construction time so the
// ``std::regex`` instance is reusable across encode calls.
class RegexTokenizer final : public LookupTokenizer {
public:
    explicit RegexTokenizer(const std::string& pattern,
                             std::unordered_map<std::string, TokenId> vocab = {});
    std::string algo() const override { return "regex"; }
    std::string decode(const IdSequence& ids) const override;
    const std::string& pattern() const noexcept { return pattern_str_; }

protected:
    std::vector<std::string> split_to_chunks_(
        const std::string& text) const override;

private:
    std::string pattern_str_;
    std::regex pattern_;
};

}  // namespace lucid::utils::tokenizer
