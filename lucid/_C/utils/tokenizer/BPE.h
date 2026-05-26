// lucid/_C/utils/tokenizer/BPE.h
//
// Byte-Pair Encoding (Sennrich et al. 2016) — the classic algorithm
// used by every text model from GPT through LLaMA via the canonical
// "vocab.json + merges.txt" pair on disk.  This C++ implementation
// matches the Hugging Face ``tokenizers`` crate's BPE semantics
// exactly so any HF-published BPE checkpoint loads and round-trips
// here without surprises.
//
// Pre-tokenization is the caller's responsibility (the Python
// wrapper applies a pre-tokenizer + normalizer chain before
// passing chunks here).  This class operates on **already
// pre-tokenized strings** — each input ``encode`` call produces the
// id sequence for one chunk, and the Python side concatenates them.
//
// File formats supported via ``BPE::from_files``:
//   * ``vocab.json`` — JSON dict ``{token_str: id}``
//   * ``merges.txt`` — one merge per line ``"a b"`` (BPE rank = line #)
//
// Train surface: standard BPE training (count adjacent pair freqs +
// greedily merge the most frequent until target_vocab_size is reached).
// O(corpus_size · target_vocab_size / vocab_size_per_iter) — acceptable
// for vocab sizes up to ~100K on a single core; users wanting faster
// training should use the reference Python ``BPETokenizer.train`` with
// multiprocessing or wait for a future parallel BPE implementation.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "Tokenizer.h"

namespace lucid::utils::tokenizer {

// Internal merge-table type — maps an ordered pair ``(left, right)``
// of existing token ids to the merged token id + the merge's rank
// (lower rank = higher priority, applied first during encoding).
struct BPEMerge {
    TokenId result;
    std::uint32_t rank;
};

// Pair-key hash for the merge table.  Two int32 ids packed into a
// 64-bit hash combined via a fast splitmix-style mix.
struct PairHash {
    std::size_t operator()(const std::pair<TokenId, TokenId>& p) const noexcept {
        std::uint64_t h = (static_cast<std::uint64_t>(p.first) << 32) |
                          static_cast<std::uint64_t>(static_cast<std::uint32_t>(p.second));
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return static_cast<std::size_t>(h);
    }
};

class BPE final : public Tokenizer {
public:
    BPE();

    // Construct from explicit vocab + merge table.  ``vocab`` maps
    // token-string → id (must include every single-character token
    // that any merge could produce).  ``merges`` is the ordered list
    // of pairs to merge — index = rank.
    BPE(std::unordered_map<std::string, TokenId> vocab,
        std::vector<std::pair<std::string, std::string>> merges);

    // ── Tokenizer overrides ────────────────────────────────────────

    IdSequence encode(const std::string& text) const override;
    std::string decode(const IdSequence& ids) const override;
    std::size_t vocab_size() const override { return vocab_.size(); }
    std::string algo() const override { return "bpe"; }

    void train(const std::vector<std::string>& corpus,
               std::size_t target_vocab_size) override;

    std::unordered_map<std::string, TokenId> get_vocab() const override {
        return vocab_;
    }
    std::string id_to_token(TokenId id) const override;

    // ── BPE-specific accessors ─────────────────────────────────────

    // Ordered merge list (rank ascending) — exposed for serialisation
    // to ``merges.txt`` on the Python side.
    const std::vector<std::pair<std::string, std::string>>& merges() const
        noexcept { return merges_str_; }

protected:
    // Vocab + reverse table.  ``id_to_token_`` rebuilds whenever
    // ``vocab_`` changes (after construction + after train).
    std::unordered_map<std::string, TokenId> vocab_;
    std::vector<std::string> id_to_token_;

    // Compiled merge table: pair-of-ids → (result-id, rank).  Built
    // from ``merges_str_`` whenever ``train`` / construction completes.
    std::unordered_map<std::pair<TokenId, TokenId>, BPEMerge, PairHash>
        pair_to_merge_;

    // Original textual merges in order — kept for save/round-trip
    // and for ``merges()`` accessor.
    std::vector<std::pair<std::string, std::string>> merges_str_;

private:
    // Rebuild ``id_to_token_`` + ``pair_to_merge_`` from the current
    // ``vocab_`` + ``merges_str_``.  Called after construction and
    // after ``train``.
    void rebuild_tables_();

    // Apply the merge sequence to one pre-tokenized chunk.  Operates
    // on a working buffer of (id, next-pair-rank) pairs and repeatedly
    // picks the lowest-rank pair to merge.  O(N · log K) per chunk
    // where N = chunk length and K = #active merges (typically << M
    // because most merges' pairs never appear in the chunk).
    IdSequence encode_chunk_(const std::string& chunk) const;
};

}  // namespace lucid::utils::tokenizer
