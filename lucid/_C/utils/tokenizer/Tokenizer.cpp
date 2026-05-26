// lucid/_C/utils/tokenizer/Tokenizer.cpp
//
// Defaults for the optional :class:`Tokenizer` overrides.  Concrete
// algorithm subclasses (BPE / WordPiece / Unigram) override the
// algorithm-specific bits; everything here is the reasonable
// "no-op / lookup-via-vocab" fallback.

#include "Tokenizer.h"

#include <stdexcept>

namespace lucid::utils::tokenizer {

void Tokenizer::train(const std::vector<std::string>& /*corpus*/,
                     std::size_t /*target_vocab_size*/) {
    throw std::runtime_error(
        "Tokenizer::train: this algorithm doesn't support C++ training "
        "— use the Python reference implementation (XxxTokenizer, no "
        "Fast suffix) or train via the algorithm's specific API.");
}

const SpecialTokens& Tokenizer::special_tokens() const noexcept {
    return special_;
}

void Tokenizer::set_special_tokens(SpecialTokens st) {
    special_ = std::move(st);
}

std::vector<IdSequence>
Tokenizer::encode_batch(const std::vector<std::string>& texts) const {
    std::vector<IdSequence> out;
    out.reserve(texts.size());
    for (const auto& t : texts)
        out.push_back(encode(t));
    return out;
}

std::vector<std::string>
Tokenizer::decode_batch(const std::vector<IdSequence>& batch) const {
    std::vector<std::string> out;
    out.reserve(batch.size());
    for (const auto& ids : batch)
        out.push_back(decode(ids));
    return out;
}

std::unordered_map<std::string, TokenId> Tokenizer::get_vocab() const {
    return {};
}

std::string Tokenizer::id_to_token(TokenId id) const {
    // O(N) fallback — subclasses with an O(1) reverse table override.
    for (const auto& [tok, tid] : get_vocab())
        if (tid == id) return tok;
    return "";
}

}  // namespace lucid::utils::tokenizer
