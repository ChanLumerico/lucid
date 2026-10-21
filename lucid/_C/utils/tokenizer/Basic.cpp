// lucid/_C/utils/tokenizer/Basic.cpp
//
// Implementations for the 5 Basic-tier tokenizers.  See Basic.h for
// the API contract + the family overview.
//
// All concrete classes share the ``LookupTokenizer`` base — encode is
// always "split → vocab.find for each chunk → emit id (or UNK)" and
// decode is "id → token string concat".  Subclasses differ ONLY in
// the ``split_to_chunks_`` rule, so the per-class implementations
// below are intentionally short.

#include "Basic.h"

#include <algorithm>
#include <stdexcept>

namespace lucid::utils::tokenizer {

// ── LookupTokenizer (shared base) ───────────────────────────────────

LookupTokenizer::LookupTokenizer() = default;

LookupTokenizer::LookupTokenizer(std::unordered_map<std::string, TokenId> vocab)
    : vocab_(std::move(vocab)) {
    rebuild_id_to_token_();
}

// Re-derive the dense id → token reverse table from ``vocab_``.
// Invariant: must be called after any mutation of ``vocab_`` so that
// ``decode`` / ``id_to_token`` see consistent state.
void LookupTokenizer::rebuild_id_to_token_() {
    detail::rebuild_id_to_token(vocab_, id_to_token_);
}

std::string LookupTokenizer::id_to_token(TokenId id) const {
    return detail::lookup_id_to_token(id_to_token_, id);
}

// Shared encode hot path — pre-tokenize via the subclass's
// ``split_to_chunks_`` then map each chunk through the vocab.  Misses
// fall back to UNK if one is configured, otherwise are silently
// dropped (matches HF behaviour for vocab-less Whitespace).
IdSequence LookupTokenizer::encode(const std::string& text) const {
    auto chunks = split_to_chunks_(text);
    IdSequence ids;
    ids.reserve(chunks.size());
    for (const auto& chunk : chunks) {
        auto it = vocab_.find(chunk);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else if (special_.unk.has_value()) {
            ids.push_back(*special_.unk);
        }
        // else: silently drop (matches HF behaviour for vocab-less Whitespace)
    }
    return ids;
}

// Shared decode — concatenate each id's surface form.  Subclasses
// that need delimiter insertion (Whitespace / Word / Regex) override
// to emit spaces between tokens.
std::string LookupTokenizer::decode(const IdSequence& ids) const {
    return detail::decode_joined_tokens(ids, id_to_token_, /*space_separated=*/false);
}

// Shared training — walk the corpus through the subclass's
// pre-tokenizer and insert each new chunk in encounter order, capped
// at ``target_vocab_size``.  Special tokens are deliberately NOT
// auto-seeded here (they're configuration, not corpus-derived); the
// Python wrapper is responsible for pre-seeding or post-adding them.
// Hard Rule H13 (no unstructured jumps) is satisfied via a sentinel flag.
void LookupTokenizer::train(const std::vector<std::string>& corpus, std::size_t target_vocab_size) {
    // Wipe existing vocab + start fresh.  Preserve any special-token
    // ids the user has already configured (they live at the LOW ids).
    vocab_.clear();
    TokenId next_id = 0;

    // Reserve low ids for special tokens — they must appear in the
    // vocab so encode can resolve them later.  Walk the canonical
    // optional slots + extras in a deterministic order.
    auto add_special = [&](const std::optional<TokenId>& slot_id, const std::string& slot_name) {
        (void)slot_id;
        (void)slot_name;
        // Specials are NOT auto-added by train (they're config, not
        // corpus-derived); the user is expected to either pre-seed
        // the vocab with them OR add them via the Python wrapper's
        // ``add_special_tokens`` after training.
    };
    add_special(special_.pad, "pad");
    add_special(special_.unk, "unk");

    // Per Hard Rule H13 (no unstructured jumps), early-exit on vocab-full uses
    // a sentinel flag the outer loop checks each iteration.
    bool full = false;
    for (const auto& doc : corpus) {
        if (full)
            break;
        auto chunks = split_to_chunks_(doc);
        for (const auto& chunk : chunks) {
            if (vocab_.size() >= target_vocab_size) {
                full = true;
                break;
            }
            if (vocab_.find(chunk) == vocab_.end()) {
                vocab_[chunk] = next_id++;
            }
        }
    }
    rebuild_id_to_token_();
}

// ── ByteTokenizer ───────────────────────────────────────────────────

ByteTokenizer::ByteTokenizer() {
    // Fixed 256-byte vocab: id == byte value.  Token strings are the
    // single byte interpreted as a 1-char Latin-1 string (matches
    // ByT5 / ByteLevel-BPE conventions).
    for (int b = 0; b < 256; ++b) {
        std::string s(1, static_cast<char>(b));
        vocab_[s] = static_cast<TokenId>(b);
    }
    rebuild_id_to_token_();
}

// Split rule: one chunk per raw byte (no UTF-8 awareness — that's the
// whole point of byte-level tokenization).
std::vector<std::string> ByteTokenizer::split_to_chunks_(const std::string& text) const {
    std::vector<std::string> out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        out.emplace_back(1, static_cast<char>(c));
    }
    return out;
}

// ── CharTokenizer ───────────────────────────────────────────────────

CharTokenizer::CharTokenizer() = default;

CharTokenizer::CharTokenizer(std::unordered_map<std::string, TokenId> vocab)
    : LookupTokenizer(std::move(vocab)) {}

// Split rule: one chunk per UTF-8 codepoint.  Lead-byte inspection
// derives the codepoint length; invalid lead bytes are treated as
// single-byte chunks so the function never throws on malformed input.
std::vector<std::string> CharTokenizer::split_to_chunks_(const std::string& text) const {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < text.size()) {
        const std::size_t cp_len = detail::utf8_codepoint_len(text, i);
        out.emplace_back(text.substr(i, cp_len));
        i += cp_len;
    }
    return out;
}

// ── WhitespaceTokenizer ─────────────────────────────────────────────

WhitespaceTokenizer::WhitespaceTokenizer() = default;

WhitespaceTokenizer::WhitespaceTokenizer(std::unordered_map<std::string, TokenId> vocab)
    : LookupTokenizer(std::move(vocab)) {}

// Split rule: emit each whitespace-delimited run as a chunk; the
// whitespace itself is discarded (decode rebuilds it as single spaces).
std::vector<std::string> WhitespaceTokenizer::split_to_chunks_(const std::string& text) const {
    return detail::split_ascii_whitespace(text);
}

std::string WhitespaceTokenizer::decode(const IdSequence& ids) const {
    // Whitespace tokenizer joins tokens with a single space — the
    // original whitespace was destroyed during encode, so " " is the
    // canonical reconstruction.
    return detail::decode_joined_tokens(ids, id_to_token_, /*space_separated=*/true);
}

// ── WordTokenizer ───────────────────────────────────────────────────

WordTokenizer::WordTokenizer() = default;

WordTokenizer::WordTokenizer(std::unordered_map<std::string, TokenId> vocab)
    : LookupTokenizer(std::move(vocab)) {}

// Split rule: identical to Whitespace.  The semantic difference is
// purely encode-time — Word expects ``special_.unk`` to be configured
// so OOV words become UNK instead of being dropped.
std::vector<std::string> WordTokenizer::split_to_chunks_(const std::string& text) const {
    // Same split rule as Whitespace.  The semantic difference (UNK
    // fallback on OOV instead of drop) is enforced by the base
    // ``encode`` plus a configured ``special_.unk``.
    return detail::split_ascii_whitespace(text);
}

std::string WordTokenizer::decode(const IdSequence& ids) const {
    return detail::decode_joined_tokens(ids, id_to_token_, /*space_separated=*/true);
}

// ── RegexTokenizer ──────────────────────────────────────────────────

RegexTokenizer::RegexTokenizer(const std::string& pattern,
                               std::unordered_map<std::string, TokenId> vocab)
    : LookupTokenizer(std::move(vocab)),
      pattern_str_(pattern),
      pattern_(pattern, std::regex::ECMAScript) {}

std::vector<std::string> RegexTokenizer::split_to_chunks_(const std::string& text) const {
    std::vector<std::string> out;
    auto begin = std::sregex_iterator(text.begin(), text.end(), pattern_);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        out.emplace_back(it->str());
    }
    return out;
}

std::string RegexTokenizer::decode(const IdSequence& ids) const {
    // Regex tokenizer has no canonical reconstruction (the unmatched
    // delimiters were dropped during encode).  Join with single
    // space as a reasonable default — same as Whitespace/Word.
    return detail::decode_joined_tokens(ids, id_to_token_, /*space_separated=*/true);
}

}  // namespace lucid::utils::tokenizer
