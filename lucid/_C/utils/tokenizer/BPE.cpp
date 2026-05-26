// lucid/_C/utils/tokenizer/BPE.cpp
//
// BPE algorithm implementation.  See BPE.h for the API contract.

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

BPE::BPE() = default;

BPE::BPE(std::unordered_map<std::string, TokenId> vocab,
         std::vector<std::pair<std::string, std::string>> merges)
    : vocab_(std::move(vocab)), merges_str_(std::move(merges)) {
    rebuild_tables_();
}

void BPE::rebuild_tables_() {
    // Build reverse id→string table.
    TokenId max_id = -1;
    for (const auto& [tok, id] : vocab_)
        if (id > max_id) max_id = id;
    id_to_token_.assign(static_cast<std::size_t>(max_id + 1), std::string{});
    for (const auto& [tok, id] : vocab_) {
        if (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size())
            id_to_token_[static_cast<std::size_t>(id)] = tok;
    }

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
        pair_to_merge_.emplace(
            std::make_pair(ia->second, ib->second),
            BPEMerge{ic->second, rank});
        ++rank;
    }
}

std::string BPE::id_to_token(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
        return "";
    return id_to_token_[static_cast<std::size_t>(id)];
}

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
        unsigned char c0 = static_cast<unsigned char>(chunk[i]);
        std::size_t cp_len;
        if (c0 < 0x80) cp_len = 1;
        else if ((c0 >> 5) == 0b110) cp_len = 2;
        else if ((c0 >> 4) == 0b1110) cp_len = 3;
        else if ((c0 >> 3) == 0b11110) cp_len = 4;
        else cp_len = 1;  // invalid lead byte — treat as single char
        if (i + cp_len > chunk.size()) cp_len = chunk.size() - i;
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
            if (it == pair_to_merge_.end()) continue;
            if (it->second.rank < best_rank) {
                best_rank = it->second.rank;
                best_pos = k;
                best_merged = it->second.result;
            }
        }
        if (best_pos == ids.size()) break;  // no applicable merge
        // Apply the merge: replace ids[best_pos..best_pos+2) with the
        // merged id.  Erase + insert is O(N) per merge — fine for
        // typical chunk sizes.
        ids[best_pos] = best_merged;
        ids.erase(ids.begin() + best_pos + 1);
    }
    return ids;
}

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

std::string BPE::decode(const IdSequence& ids) const {
    std::string out;
    out.reserve(ids.size() * 2);
    for (TokenId id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
            continue;
        out += id_to_token_[static_cast<std::size_t>(id)];
    }
    return out;
}

// ── Training ────────────────────────────────────────────────────────
//
// Classical BPE training: split each word in the corpus into
// characters; repeatedly find the most frequent adjacent pair across
// all words; merge it into a new token; record the merge.  Stop when
// vocab reaches ``target_vocab_size`` or no more pairs occur.

namespace {

// Per-word state: a sequence of token strings + a frequency count.
// Words sharing the same character split share the count (we
// pre-aggregate by exact word match).
struct WordState {
    std::vector<std::string> symbols;
    std::uint64_t count;
};

}  // namespace

void BPE::train(const std::vector<std::string>& corpus,
                std::size_t target_vocab_size) {
    if (target_vocab_size < 1)
        throw std::runtime_error(
            "BPE::train: target_vocab_size must be at least 1");

    // 1. Count word frequencies via whitespace pre-tokenization.
    //    (Real HF training has a more sophisticated pre-tokenizer
    //    chain; the C++ training path is intentionally simple —
    //    callers wanting richer training should use the Python
    //    ``BPETokenizer.train`` which exposes the full normalizer +
    //    pre-tokenizer surface.)
    std::unordered_map<std::string, std::uint64_t> word_freq;
    for (const auto& doc : corpus) {
        std::size_t i = 0;
        while (i < doc.size()) {
            // skip whitespace
            while (i < doc.size() &&
                   (doc[i] == ' ' || doc[i] == '\t' || doc[i] == '\n' ||
                    doc[i] == '\r'))
                ++i;
            std::size_t start = i;
            while (i < doc.size() && !(doc[i] == ' ' || doc[i] == '\t' ||
                                       doc[i] == '\n' || doc[i] == '\r'))
                ++i;
            if (i > start)
                ++word_freq[doc.substr(start, i - start)];
        }
    }

    // 2. Seed the vocab with every distinct character (UTF-8 codepoint)
    //    that appears in any word.  Each char gets a fresh id.
    vocab_.clear();
    merges_str_.clear();
    TokenId next_id = 0;
    auto add_to_vocab = [&](const std::string& tok) {
        auto [it, inserted] = vocab_.try_emplace(tok, next_id);
        if (inserted) ++next_id;
        return it->second;
    };

    std::vector<WordState> words;
    words.reserve(word_freq.size());
    for (const auto& [w, freq] : word_freq) {
        WordState ws;
        ws.count = freq;
        // Split into UTF-8 codepoints + seed chars in vocab.
        std::size_t i = 0;
        while (i < w.size()) {
            unsigned char c0 = static_cast<unsigned char>(w[i]);
            std::size_t cp_len;
            if (c0 < 0x80) cp_len = 1;
            else if ((c0 >> 5) == 0b110) cp_len = 2;
            else if ((c0 >> 4) == 0b1110) cp_len = 3;
            else if ((c0 >> 3) == 0b11110) cp_len = 4;
            else cp_len = 1;
            if (i + cp_len > w.size()) cp_len = w.size() - i;
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
        // Count adjacent pair frequencies across all words.
        std::unordered_map<std::string, std::uint64_t> pair_freq;
        for (const auto& ws : words) {
            for (std::size_t k = 0; k + 1 < ws.symbols.size(); ++k) {
                // Key: "left\x01right" — \x01 is invalid in UTF-8 word
                // tokens, so it's a safe delimiter.
                std::string key = ws.symbols[k] + '\x01' + ws.symbols[k + 1];
                pair_freq[key] += ws.count;
            }
        }
        if (pair_freq.empty()) break;

        // Find the max-count pair (ties broken by lexicographic order
        // of the key for determinism).
        auto best_it = pair_freq.begin();
        for (auto it = pair_freq.begin(); it != pair_freq.end(); ++it) {
            if (it->second > best_it->second ||
                (it->second == best_it->second && it->first < best_it->first)) {
                best_it = it;
            }
        }
        if (best_it->second < 2) break;  // no point merging singletons

        // Split the key back into the two halves.
        std::string key = best_it->first;
        auto sep = key.find('\x01');
        std::string left = key.substr(0, sep);
        std::string right = key.substr(sep + 1);
        std::string merged = left + right;

        // Record merge + add to vocab.
        merges_str_.emplace_back(left, right);
        add_to_vocab(merged);

        // Apply the merge across all words (rewrite symbol sequences).
        for (auto& ws : words) {
            std::vector<std::string> next;
            next.reserve(ws.symbols.size());
            std::size_t k = 0;
            while (k < ws.symbols.size()) {
                if (k + 1 < ws.symbols.size() && ws.symbols[k] == left &&
                    ws.symbols[k + 1] == right) {
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

    rebuild_tables_();
}

}  // namespace lucid::utils::tokenizer
