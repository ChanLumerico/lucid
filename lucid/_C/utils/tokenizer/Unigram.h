// lucid/_C/utils/tokenizer/Unigram.h
//
// Unigram Language Model tokenizer — the SentencePiece flavour used
// by T5 / mBART / ALBERT / XLNet / LLaMA / Mistral.  Kudo (2018)
// "Subword Regularization" introduced the algorithm; the practical
// reference is the SentencePiece library.
//
// Algorithm summary
// -----------------
// Vocab is a set of *pieces* (subword strings), each with a
// probability ``p(piece)``.  At encode time, find the segmentation
// of the input string that maximises ``Σ log p(piece)`` — solved
// exactly by Viterbi forward-DP over piece boundaries.  At training
// time, fit the probabilities + prune the vocab via EM.
//
// Encode (Viterbi)
// ----------------
// For each input chunk:
//   1. Walk position i from 0 to N (one past last char).
//   2. dp[i] = max over all (j, piece) where word[j..i] == piece
//             of (dp[j] + log_prob(piece))
//   3. Trace back from dp[N] to recover the chosen segmentation.
//
// Tie-break: when two segmentations have equal log-prob (rare in
// practice but possible with uniform-probability vocabs), prefer
// the one with the longer rightmost piece (matches SentencePiece's
// behaviour exactly).
//
// Training (EM with vocab pruning)
// --------------------------------
// 1. Build seed vocab from all distinct substrings of length up to
//    ``max_piece_length`` appearing in the corpus.
// 2. Initialise probabilities = frequency / total.
// 3. Repeat:
//    a. E-step — for each word, run forward-backward to compute the
//       expected count of each piece.
//    b. M-step — update p(piece) = expected_count / Σ expected_count.
//    c. Prune the bottom ``shrink_factor`` fraction of pieces (by
//       probability), but always keep all single-character pieces
//       so every encode is guaranteed to terminate.
//    d. Stop when vocab_size ≤ target OR after ``num_iterations``.
//
// On-disk format
// --------------
// HF-style tokenizer.json carries ``model.vocab`` as
// ``[(piece_str, log_prob), ...]``.  SentencePiece's .model binary
// is a separate format (protobuf) — not parsed here; convert via
// SP's Python API + emit our tokenizer.json.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Tokenizer.h"

namespace lucid::utils::tokenizer {

// Unigram language model tokenizer.  Owns an ordered ``(piece, log_prob)``
// table — entry index doubles as the token id, so the order is part of
// the on-disk contract and must be preserved across save/load.  All
// encode paths are Viterbi-optimal; the UNK piece is always at a known
// id so feasibility of segmentation is guaranteed post-train.  Matches
// the Python ``lucid.utils.tokenizer.UnigramTokenizer`` wrapper exactly.
class Unigram final : public Tokenizer {
public:
    // Default-construct an empty model; populate via ``train`` or by
    // re-constructing with explicit pieces from a loaded checkpoint.
    Unigram();

    // Construct from explicit (piece, log_prob) pairs.  The order
    // determines token ids — entry index = id.  ``unk_token`` and
    // ``unk_log_prob`` configure the fallback piece for inputs that
    // can't be covered by any combination of vocab pieces.
    Unigram(std::vector<std::pair<std::string, double>> pieces,
            std::string unk_token = "<unk>",
            double unk_log_prob = -100.0);

    // ── Tokenizer overrides ────────────────────────────────────────

    IdSequence encode(const std::string& text) const override;
    std::string decode(const IdSequence& ids) const override;
    std::size_t vocab_size() const override { return pieces_.size(); }
    std::string algo() const override { return "unigram"; }
    std::unordered_map<std::string, TokenId> get_vocab() const override;
    std::string id_to_token(TokenId id) const override;

    // EM training.  See the file header for the algorithm; key knobs:
    //   * ``target_vocab_size`` — stop pruning when reached.
    //   * ``num_iterations`` — EM passes before each prune step.
    //   * ``shrink_factor`` — fraction of pieces removed per iteration
    //     (default 0.75 — matches SentencePiece "shrinking_factor").
    //   * ``max_piece_length`` — cap on seed substring length.
    //   * ``initial_vocab_multiplier`` — seed vocab is sized at
    //     ``multiplier × target_vocab_size`` before pruning starts.
    void train(const std::vector<std::string>& corpus, std::size_t target_vocab_size) override;

    // Extended train with all knobs exposed.
    void train_with_options(const std::vector<std::string>& corpus,
                            std::size_t target_vocab_size,
                            std::size_t num_iterations,
                            double shrink_factor,
                            std::size_t max_piece_length,
                            std::size_t initial_vocab_multiplier);

    // ── Unigram-specific accessors ─────────────────────────────────

    // Vocab as (piece, log_prob) pairs — exposed for save / inspection.
    const std::vector<std::pair<std::string, double>>& pieces() const noexcept { return pieces_; }

    const std::string& unk_token() const noexcept { return unk_token_; }
    double unk_log_prob() const noexcept { return unk_log_prob_; }

private:
    // Build the piece-string → id map + the longest-piece cache.
    void rebuild_tables_();

    // Viterbi encode for one chunk.
    IdSequence viterbi_encode_(const std::string& chunk) const;

    // Forward-backward to get expected piece counts for one chunk
    // under the current probability model.  Used by EM training.
    // Adds to ``expected_counts`` (indexed by piece id).
    void forward_backward_accumulate_(const std::string& chunk,
                                      std::uint64_t weight,
                                      std::vector<double>& expected_counts) const;

    std::vector<std::pair<std::string, double>> pieces_;
    std::unordered_map<std::string, TokenId> piece_to_id_;
    std::string unk_token_;
    double unk_log_prob_;
    TokenId unk_id_ = -1;
    // Cached max piece byte-length — bounds the Viterbi back-scan.
    std::size_t max_piece_bytes_ = 0;
};

}  // namespace lucid::utils::tokenizer
