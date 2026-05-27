// lucid/_C/utils/tokenizer/Tokenizer.cpp
//
// Defaults for the optional :class:`Tokenizer` overrides.  Concrete
// algorithm subclasses (BPE / WordPiece / Unigram) override the
// algorithm-specific bits; everything here is the reasonable
// "no-op / lookup-via-vocab" fallback.

#include "Tokenizer.h"

#include <stdexcept>

namespace lucid::utils::tokenizer {

// Default training stub.  Algorithms that don't implement on-line
// training in C++ inherit this fallback so calling ``train`` from
// Python surfaces a clear, actionable error instead of silently
// returning success with an empty vocab.
void Tokenizer::train(const std::vector<std::string>& /*corpus*/,
                     std::size_t /*target_vocab_size*/) {
    throw std::runtime_error(
        "Tokenizer::train: this algorithm doesn't support C++ training "
        "— use the Python reference implementation (XxxTokenizer, no "
        "Fast suffix) or train via the algorithm's specific API.");
}

// Trivial accessor — returns the stored special-token registry by
// const reference (noexcept; safe to call from any thread).
const SpecialTokens& Tokenizer::special_tokens() const noexcept {
    return special_;
}

// Bulk-replace the special-token registry.  Used by the Python
// wrapper after ``add_special_tokens`` / ``from_pretrained`` to
// install the canonical PAD/UNK/BOS/EOS slots.  Not thread-safe.
void Tokenizer::set_special_tokens(SpecialTokens st) {
    special_ = std::move(st);
}

// Default batched encode — sequential fallback that simply loops over
// ``encode``.  Algorithm-specific subclasses can override for genuine
// parallelism (e.g. shared pre-tokenizer state amortised across the
// batch); the default keeps the API consistent for any subclass.
std::vector<IdSequence>
Tokenizer::encode_batch(const std::vector<std::string>& texts) const {
    std::vector<IdSequence> out;
    out.reserve(texts.size());
    for (const auto& t : texts)
        out.push_back(encode(t));
    return out;
}

// Default batched decode — sequential mirror of ``encode_batch``.
std::vector<std::string>
Tokenizer::decode_batch(const std::vector<IdSequence>& batch) const {
    std::vector<std::string> out;
    out.reserve(batch.size());
    for (const auto& ids : batch)
        out.push_back(decode(ids));
    return out;
}

// Default vocab introspection — returns an empty map.  Concrete
// algorithms with a real vocab (BPE / WordPiece / Unigram / Lookup)
// override to expose their internal table.
std::unordered_map<std::string, TokenId> Tokenizer::get_vocab() const {
    return {};
}

// Default id → token lookup.  O(N) linear scan over ``get_vocab`` —
// acceptable for one-off introspection but subclasses that maintain
// an indexed reverse table should override for O(1) decode.
std::string Tokenizer::id_to_token(TokenId id) const {
    for (const auto& [tok, tid] : get_vocab())
        if (tid == id) return tok;
    return "";
}

}  // namespace lucid::utils::tokenizer
