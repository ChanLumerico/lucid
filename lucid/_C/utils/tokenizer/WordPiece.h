// lucid/_C/utils/tokenizer/WordPiece.h
//
// WordPiece tokenizer — the algorithm BERT/DistilBERT/ELECTRA use.
// First proposed by Schuster & Nakajima (2012) for Japanese/Korean
// speech recognition; popularised in NLP by Devlin et al. (2018).
//
// Encode algorithm (greedy longest-match)
// ---------------------------------------
// For each pre-tokenized word:
//   1. If the whole word is in vocab, emit its id.
//   2. Otherwise, walk left-to-right finding the longest prefix that
//      IS in vocab; emit its id and recurse on the remainder
//      (with the standard "##" continuation prefix).
//   3. If no valid prefix can be found at any position, emit UNK
//      for the entire word and stop.
//
// Format compat
// -------------
// On-disk vocab is HF-compatible ``vocab.txt`` — one token per line,
// id = line index.  Continuation pieces start with ``##`` (BERT
// convention).  Round-trips with any published BERT-family
// checkpoint without modification.
//
// Training
// --------
// Training is computationally heavier than BPE — at each iteration
// you score every candidate merge by the log-likelihood gain it
// provides under a unigram model, not just raw frequency.  The C++
// implementation here uses a simpler "BPE-like" greedy frequency
// training (matching HF's ``tokenizers`` crate ``WordPieceTrainer``
// behaviour, which is itself an approximation of the full likelihood
// approach).

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Tokenizer.h"

namespace lucid::utils::tokenizer {

// WordPiece tokenizer.  Owns a string→id vocab + dense reverse table.
// Continuation pieces are stored with the ``##`` prefix already baked
// into the vocab key (BERT convention) — encode/decode are responsible
// for adding/stripping the prefix at segment boundaries.  ``unk_id_``
// is cached so the encode hot path never has to hash the UNK string.
// Matches the Python ``lucid.utils.tokenizer.WordPieceTokenizer``.
class WordPiece final : public Tokenizer {
public:
    // Default-construct an empty model; populate via ``train`` or by
    // re-constructing with an explicit vocab from a loaded checkpoint.
    WordPiece();

    // Construct from a HF-compatible ``vocab.txt`` mapping
    // (token-string → id).  ``unk_token`` and ``continuing_prefix``
    // configure the algorithm's two key knobs (defaults match BERT).
    WordPiece(std::unordered_map<std::string, TokenId> vocab,
              std::string unk_token = "[UNK]",
              std::string continuing_prefix = "##",
              std::size_t max_chars_per_word = 100);

    // ── Tokenizer overrides ────────────────────────────────────────

    IdSequence encode(const std::string& text) const override;
    std::string decode(const IdSequence& ids) const override;
    std::size_t vocab_size() const override { return vocab_.size(); }
    std::string algo() const override { return "wordpiece"; }
    std::unordered_map<std::string, TokenId> get_vocab() const override {
        return vocab_;
    }
    std::string id_to_token(TokenId id) const override;

    // Train on a list of pre-tokenized words (the Python wrapper does
    // the normalization + word-splitting via BertNormalizer +
    // WhitespacePunctuationSplit first).  Builds a vocab of size up
    // to ``target_vocab_size`` by greedy frequency merging — close
    // to HF's ``WordPieceTrainer`` behaviour without the full
    // log-likelihood objective.
    void train(const std::vector<std::string>& corpus,
               std::size_t target_vocab_size) override;

    // ── WordPiece-specific accessors ───────────────────────────────

    const std::string& unk_token() const noexcept { return unk_token_; }
    const std::string& continuing_prefix() const noexcept {
        return continuing_prefix_;
    }

private:
    // Apply greedy longest-match to ONE pre-tokenized word.  Returns
    // the matched ids, or a single UNK id if the word cannot be
    // tokenized at any position.
    IdSequence encode_word_(const std::string& word) const;

    // Rebuild reverse table after vocab mutation.
    void rebuild_tables_();

    std::unordered_map<std::string, TokenId> vocab_;
    std::vector<std::string> id_to_token_;
    std::string unk_token_;
    std::string continuing_prefix_;
    std::size_t max_chars_per_word_;
    // Cached UNK id (or -1 if ``unk_token_`` is not in vocab).
    TokenId unk_id_ = -1;
};

}  // namespace lucid::utils::tokenizer
