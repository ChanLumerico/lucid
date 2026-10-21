// lucid/_C/utils/tokenizer/BPE.cpp
//
// BPE algorithm implementation.  See BPE.h for the API contract +
// the algorithm summary.  This file is the canonical encode / decode
// / train surface for every BPE-based checkpoint Lucid loads — the
// Python ``BPETokenizerFast`` wrapper holds a ``std::unique_ptr<BPE>``
// and forwards directly through the v-table.
//
// Implementation notes
// --------------------
// * Encode is a greedy lowest-rank-first merge loop with an O(N · M)
//   inner cost (N = chunk length, M = remaining merges).  Acceptable
//   for production: post-pre-tokenization chunks average ~10 ids.
// * Train uses simple whitespace pre-tokenization + greedy frequency
//   merging — the same algorithm as Sennrich 2016 (no log-likelihood
//   heuristic).  Callers wanting richer training should use the slow
//   Python ``BPETokenizer.train`` which exposes a full normaliser +
//   pre-tokeniser chain.
// * UTF-8 codepoints are honoured during chunk seeding so multi-byte
//   characters aren't split mid-encoding.

#include "BPE.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <utility>

namespace lucid::utils::tokenizer {

// Default constructor — empty vocab + merge table.  The Python
// ``from_pretrained`` factory populates state by calling ``train`` or
// by injecting via the explicit-vocab constructor below.
BPE::BPE() = default;

// Construct from explicit vocab + ordered merge list.  Both inputs
// are moved in; ``rebuild_tables_`` then compiles the textual merges
// into the id-indexed merge map used by the encode hot loop.
BPE::BPE(std::unordered_map<std::string, TokenId> vocab,
         std::vector<std::pair<std::string, std::string>> merges)
    : vocab_(std::move(vocab)), merges_str_(std::move(merges)) {
    rebuild_tables_();
}

// Re-derive every cached table from the canonical ``vocab_`` +
// ``merges_str_`` state.  Invariant: must be called after ANY
// mutation of either field (construction / train / external setters)
// — encode/decode read the cached tables directly and would otherwise
// see stale data.  Malformed merges (missing halves or missing
// merged form in the vocab) are silently skipped so partially-broken
// HF dumps still load.
void BPE::rebuild_tables_() {
    // Build reverse id→string table.
    detail::rebuild_id_to_token(vocab_, id_to_token_);

    // Build pair-id → merge table from the textual merges.  Each
    // merge requires both halves AND the merged string to exist
    // in vocab — drop any merges that don't (malformed file).
    pair_to_merge_.clear();
    pair_to_merge_.reserve(merges_str_.size());
    std::uint32_t rank = 0;
    for (const auto& [a, b] : merges_str_) {
        auto ia = vocab_.find(a);
        auto ib = vocab_.find(b);
        std::string merged = a + b;
        auto ic = vocab_.find(merged);
        if (ia == vocab_.end() || ib == vocab_.end() || ic == vocab_.end()) {
            ++rank;
            continue;  // skip malformed
        }
        pair_to_merge_.emplace(std::make_pair(ia->second, ib->second), BPEMerge{ic->second, rank});
        ++rank;
    }
}

// O(1) reverse lookup via the dense ``id_to_token_`` array.  Returns
// the empty string for out-of-range ids (matches HF behaviour rather
// than throwing — callers can detect by checking ``vocab_size()``).
std::string BPE::id_to_token(TokenId id) const {
    return detail::lookup_id_to_token(id_to_token_, id);
}

// Encode a single pre-tokenized chunk to its BPE id sequence.
// Two phases:
//   1. Seed: split the chunk into UTF-8 codepoints + look each up in
//      the vocab (UNK fallback if configured; silently dropped if not,
//      matching HF semantics).
//   2. Greedy merge: repeatedly pick the lowest-rank applicable pair
//      and merge in place.  Naive O(N · M) but the constant factor is
//      tiny because pre-tokenization keeps chunks short (~10 ids).
IdSequence BPE::encode_chunk_(const std::string& chunk) const {
    // Initial id sequence: one entry per UTF-8 codepoint (the Python
    // side has already done byte-level escaping for byte-level BPE; for
    // classical BPE every chunk is plain text).  We look up each char
    // in the vocab — if missing, fall back to UNK if available, else
    // skip with a warning (silent here; the Python wrapper has the
    // logging hook).
    IdSequence ids;
    ids.reserve(chunk.size());

    // Iterate UTF-8 codepoints by stepping on continuation bytes.
    std::size_t i = 0;
    while (i < chunk.size()) {
        const std::size_t cp_len = detail::utf8_codepoint_len(chunk, i);
        std::string cp = chunk.substr(i, cp_len);
        auto it = vocab_.find(cp);
        if (it != vocab_.end()) {
            ids.push_back(it->second);
        } else if (special_.unk.has_value()) {
            ids.push_back(*special_.unk);
        }
        // else: silently drop (matches HF BPE behaviour with no UNK)
        i += cp_len;
    }

    if (ids.size() < 2)
        return ids;

    // Greedy merge: at each step, find the pair with the lowest rank
    // (highest priority) that's currently in the sequence; merge it;
    // repeat until no more applicable pairs.
    //
    // Naive implementation: O(N · M) where M is the number of remaining
    // merge applications.  Sufficient for production BPE — average
    // sequence length is ~10 tokens post-pre-tokenization, so the
    // constant factor is tiny.
    while (true) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::size_t best_pos = ids.size();
        TokenId best_merged = -1;
        for (std::size_t k = 0; k + 1 < ids.size(); ++k) {
            auto it = pair_to_merge_.find({ids[k], ids[k + 1]});
            if (it == pair_to_merge_.end())
                continue;
            if (it->second.rank < best_rank) {
                best_rank = it->second.rank;
                best_pos = k;
                best_merged = it->second.result;
            }
        }
        if (best_pos == ids.size())
            break;  // no applicable merge
        // Apply the merge: replace ids[best_pos..best_pos+2) with the
        // merged id.  Erase + insert is O(N) per merge — fine for
        // typical chunk sizes.
        ids[best_pos] = best_merged;
        ids.erase(ids.begin() + best_pos + 1);
    }
    return ids;
}

// Public encode entry point.  Treats the entire input as one chunk;
// callers that need pre-tokenization (split on whitespace + punct,
// byte-level remap, etc.) handle that in the Python wrapper before
// invoking ``encode``.
IdSequence BPE::encode(const std::string& text) const {
    // The Python wrapper does pre-tokenization (split on whitespace +
    // punctuation per the configured pre-tokenizer) BEFORE calling
    // ``encode``; if the user reaches the C++ ``encode`` directly with
    // a multi-word string, we treat the whole thing as one chunk.
    // The natural Python ``BPETokenizerFast.encode`` loop applies the
    // pre-tokenizer + concatenates the per-chunk results, matching
    // the slow ``BPETokenizer`` behaviour.
    return encode_chunk_(text);
}

// Decode by concatenating each id's surface form.  Out-of-range ids
// are skipped silently; special-token suppression is the Python
// wrapper's responsibility (it knows the ``skip_special_tokens``
// flag and the byte-level un-mapping for byte-level BPE).
std::string BPE::decode(const IdSequence& ids) const {
    return detail::decode_joined_tokens(ids, id_to_token_, /*space_separated=*/false);
}

// ── Training ────────────────────────────────────────────────────────
//
// Classical BPE training: split each word in the corpus into
// characters; repeatedly find the most frequent adjacent pair across
// all words; merge it into a new token; record the merge.  Stop when
// vocab reaches ``target_vocab_size`` or no more pairs occur.

// Public training entry point.  Drives the four-phase classical BPE
// pipeline (word-count → seed-vocab → greedy-merge → rebuild-tables)
// in-place on ``vocab_`` / ``merges_str_``.  Throws on degenerate
// target sizes; otherwise always succeeds (possibly with a smaller
// vocab if the corpus runs out of distinct merges before the target
// is reached).  NOT thread-safe — mutates internal tables.
void BPE::train(const std::vector<std::string>& corpus, std::size_t target_vocab_size) {
    if (target_vocab_size < 1)
        throw std::runtime_error("BPE::train: target_vocab_size must be at least 1");

    // 1. Count word frequencies via whitespace pre-tokenization.
    //    (Real HF training has a more sophisticated pre-tokenizer
    //    chain; the C++ training path is intentionally simple —
    //    callers wanting richer training should use the Python
    //    ``BPETokenizer.train`` which exposes the full normalizer +
    //    pre-tokenizer surface.)
    auto word_freq = detail::count_whitespace_words(corpus);

    // 2. Seed the vocab with every distinct character (UTF-8 codepoint)
    //    that appears in any word.  Each char gets a fresh id.
    vocab_.clear();
    merges_str_.clear();
    TokenId next_id = 0;
    auto add_to_vocab = [&](const std::string& tok) {
        auto [it, inserted] = vocab_.try_emplace(tok, next_id);
        if (inserted)
            ++next_id;
        return it->second;
    };

    std::vector<detail::WordRunState> words;
    words.reserve(word_freq.size());
    for (const auto& [w, freq] : word_freq) {
        detail::WordRunState ws;
        ws.count = freq;
        // Split into UTF-8 codepoints + seed chars in vocab.
        std::size_t i = 0;
        while (i < w.size()) {
            const std::size_t cp_len = detail::utf8_codepoint_len(w, i);
            std::string cp = w.substr(i, cp_len);
            add_to_vocab(cp);
            ws.symbols.push_back(std::move(cp));
            i += cp_len;
        }
        words.push_back(std::move(ws));
    }

    // 3. Iteratively merge the most-frequent pair until we hit
    //    ``target_vocab_size``.
    while (vocab_.size() < target_vocab_size) {
        auto best_pair = detail::pick_best_adjacent_pair(words);
        if (!best_pair.has_value())
            break;
        const auto& [left, right] = *best_pair;
        std::string merged = left + right;
        merges_str_.emplace_back(left, right);
        add_to_vocab(merged);
        detail::apply_symbol_merge(words, left, right, merged);
    }

    rebuild_tables_();
}

}  // namespace lucid::utils::tokenizer
