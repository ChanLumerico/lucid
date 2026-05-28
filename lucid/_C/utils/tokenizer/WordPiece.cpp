// lucid/_C/utils/tokenizer/WordPiece.cpp
//
// WordPiece algorithm implementation.  See WordPiece.h for the API
// contract + algorithm summary.  This file is the canonical encode /
// decode / train surface for every WordPiece-based checkpoint Lucid
// loads — the Python ``WordPieceTokenizer`` wrapper holds a
// ``std::unique_ptr<WordPiece>`` and forwards directly through the
// v-table.
//
// Implementation notes
// --------------------
// * Encode is greedy longest-match per word with an O(W^2) inner
//   cost (W = word length in codepoints).  Acceptable because BERT
//   capped word length at 100; we honour the same cap via
//   ``max_chars_per_word_``.
// * Train uses simple whitespace pre-tokenization + greedy frequency
//   pair-merging — a BPE-style approximation of the full likelihood
//   objective, matching the HF ``tokenizers`` crate's
//   ``WordPieceTrainer``.
// * UTF-8 codepoints are honoured during both encode and train so
//   multi-byte characters are never split.
// * Continuation pieces carry the ``##`` prefix as a literal vocab
//   key — encode prepends it on non-leading segments and decode
//   strips it back off.

#include "WordPiece.h"

#include <algorithm>
#include <stdexcept>

namespace lucid::utils::tokenizer {

// Default constructor — empty vocab.  Populate via ``train`` or by
// re-constructing with an explicit vocab from ``vocab.txt``.
WordPiece::WordPiece() = default;

// Construct from explicit ``vocab.txt``-style mapping plus the BERT
// knobs (UNK token, ``##`` continuation prefix, max chars per word).
// ``rebuild_tables_`` then populates the dense reverse table and
// caches ``unk_id_`` so the encode hot path skips a hash lookup.
WordPiece::WordPiece(std::unordered_map<std::string, TokenId> vocab,
                     std::string unk_token,
                     std::string continuing_prefix,
                     std::size_t max_chars_per_word)
    : vocab_(std::move(vocab)),
      unk_token_(std::move(unk_token)),
      continuing_prefix_(std::move(continuing_prefix)),
      max_chars_per_word_(max_chars_per_word) {
    rebuild_tables_();
}

// Re-derive every cached table (``id_to_token_``, ``unk_id_``) from
// the canonical ``vocab_`` map.  Invariant: must be called after ANY
// mutation of ``vocab_`` — encode/decode read the cached tables
// directly and would otherwise see stale data.
void WordPiece::rebuild_tables_() {
    TokenId max_id = -1;
    for (const auto& [tok, id] : vocab_)
        if (id > max_id)
            max_id = id;
    id_to_token_.assign(static_cast<std::size_t>(max_id + 1), std::string{});
    for (const auto& [tok, id] : vocab_) {
        if (id >= 0 && static_cast<std::size_t>(id) < id_to_token_.size())
            id_to_token_[static_cast<std::size_t>(id)] = tok;
    }
    auto it = vocab_.find(unk_token_);
    unk_id_ = (it == vocab_.end()) ? -1 : it->second;
}

// O(1) reverse lookup via the dense ``id_to_token_`` array.  Empty
// string for out-of-range ids (matches the BERT-tokenizers convention
// rather than throwing).
std::string WordPiece::id_to_token(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
        return "";
    return id_to_token_[static_cast<std::size_t>(id)];
}

// Greedy longest-match WordPiece for ONE pre-tokenized word.  Three
// stages — (1) whole-word fast path, (2) BERT's max-length guard
// (default 100 chars → UNK), (3) left-to-right scan that picks the
// longest vocab-resident prefix at each position, applying the ``##``
// continuation prefix to every non-leading segment.  If no valid
// prefix exists at any position the whole word collapses to a single
// UNK (BERT semantics).  All segmentation seeks UTF-8 codepoint
// boundaries so multi-byte characters survive intact.
IdSequence WordPiece::encode_word_(const std::string& word) const {
    // 1. Whole-word match shortcut.
    if (auto it = vocab_.find(word); it != vocab_.end())
        return {it->second};
    // 2. Cap on length — BERT drops anything longer than 100 chars.
    if (word.size() > max_chars_per_word_) {
        if (unk_id_ >= 0)
            return {unk_id_};
        return {};
    }
    // 3. Greedy longest-match left-to-right.  Walk through UTF-8
    //    codepoint boundaries so we don't split a multi-byte char.
    //    Pre-compute the codepoint start offsets for O(1) seek.
    std::vector<std::size_t> cp_offsets;
    cp_offsets.reserve(word.size() + 1);
    std::size_t i = 0;
    while (i < word.size()) {
        cp_offsets.push_back(i);
        unsigned char c0 = static_cast<unsigned char>(word[i]);
        std::size_t cp_len;
        if (c0 < 0x80)
            cp_len = 1;
        else if ((c0 >> 5) == 0b110)
            cp_len = 2;
        else if ((c0 >> 4) == 0b1110)
            cp_len = 3;
        else if ((c0 >> 3) == 0b11110)
            cp_len = 4;
        else
            cp_len = 1;
        if (i + cp_len > word.size())
            cp_len = word.size() - i;
        i += cp_len;
    }
    cp_offsets.push_back(word.size());

    IdSequence ids;
    std::size_t cp_start = 0;  // index into cp_offsets
    while (cp_start + 1 < cp_offsets.size()) {
        // Try longest prefix first.
        TokenId match_id = -1;
        std::size_t match_cp_end = cp_start;
        for (std::size_t cp_end = cp_offsets.size() - 1; cp_end > cp_start; --cp_end) {
            std::string sub =
                word.substr(cp_offsets[cp_start], cp_offsets[cp_end] - cp_offsets[cp_start]);
            if (cp_start > 0)
                sub = continuing_prefix_ + sub;
            auto it = vocab_.find(sub);
            if (it != vocab_.end()) {
                match_id = it->second;
                match_cp_end = cp_end;
                break;
            }
        }
        if (match_id < 0) {
            // No valid prefix at this position → entire word is UNK.
            if (unk_id_ >= 0)
                return {unk_id_};
            return {};
        }
        ids.push_back(match_id);
        cp_start = match_cp_end;
    }
    return ids;
}

// Public encode entry point.  Splits on whitespace as a fallback when
// called directly; the Python wrapper normally pre-tokenizes via the
// BertNormalizer + WhitespacePunctuationSplit chain before invoking.
IdSequence WordPiece::encode(const std::string& text) const {
    // The Python wrapper splits the input into pre-tokenized words
    // before calling encode (via the configured WordPieceNormalizer
    // + WhitespacePunctuationSplit chain).  If ``encode`` is called
    // directly on a multi-word string, we fall back to a simple
    // whitespace split here.
    IdSequence out;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() &&
               (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r'))
            ++i;
        if (i >= text.size())
            break;
        std::size_t start = i;
        while (i < text.size() &&
               !(text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r'))
            ++i;
        auto word_ids = encode_word_(text.substr(start, i - start));
        out.insert(out.end(), word_ids.begin(), word_ids.end());
    }
    return out;
}

// Decode by stitching surface forms back together.  Continuation
// pieces (``##`` prefix) glue onto the previous token without a
// space; every non-continuation token gets a leading space (except
// the very first).  Out-of-range ids are skipped silently.
std::string WordPiece::decode(const IdSequence& ids) const {
    // Join tokens; strip leading "##" from continuation pieces and
    // emit a space before each non-continuation token.
    std::string out;
    bool first = true;
    for (TokenId id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= id_to_token_.size())
            continue;
        const std::string& tok = id_to_token_[static_cast<std::size_t>(id)];
        bool is_continuation = tok.size() >= continuing_prefix_.size() &&
                               tok.compare(0, continuing_prefix_.size(), continuing_prefix_) == 0;
        if (is_continuation) {
            out += tok.substr(continuing_prefix_.size());
        } else {
            if (!first)
                out += ' ';
            out += tok;
        }
        first = false;
    }
    return out;
}

// ── Training ────────────────────────────────────────────────────────
//
// Greedy frequency-based training (approximates HF's
// WordPieceTrainer).  Algorithm:
//
//   1. Build word frequency dict from corpus via whitespace split.
//   2. Seed vocab with every distinct char + continuation forms
//      ``##char`` for every non-leading char.
//   3. Iteratively pick the most-frequent (left, right) bigram
//      across all current symbol sequences + merge into a new token
//      (the merged token's surface form keeps ``##`` only if BOTH
//      halves were continuations or if the right half was — matches
//      BERT vocab layout).
//
// This is the same logic as BPE training but with the ``##``
// continuation suffix machinery; for our purposes that's sufficient
// to produce a BERT-compatible vocab.

namespace {

struct WordState {
    std::vector<std::string> symbols;
    std::uint64_t count;
};

}  // namespace

// Public training entry point.  Three-phase pipeline — (1) tally
// word frequencies via whitespace pre-tokenization, (2) seed vocab
// with every codepoint (plain + ``##`` continuation) and the UNK,
// (3) greedy bigram-frequency merging until ``target_vocab_size`` is
// reached or no further pair occurs more than once.  Merged surface
// forms strip the right half's ``##`` prefix (keeping the left's, if
// any) so the resulting vocab matches BERT layout byte-for-byte.
// NOT thread-safe — mutates ``vocab_`` / cached tables in place.
void WordPiece::train(const std::vector<std::string>& corpus, std::size_t target_vocab_size) {
    // 1. Word frequencies.
    std::unordered_map<std::string, std::uint64_t> word_freq;
    for (const auto& doc : corpus) {
        std::size_t i = 0;
        while (i < doc.size()) {
            while (i < doc.size() &&
                   (doc[i] == ' ' || doc[i] == '\t' || doc[i] == '\n' || doc[i] == '\r'))
                ++i;
            std::size_t start = i;
            while (i < doc.size() &&
                   !(doc[i] == ' ' || doc[i] == '\t' || doc[i] == '\n' || doc[i] == '\r'))
                ++i;
            if (i > start)
                ++word_freq[doc.substr(start, i - start)];
        }
    }
    // 2. Seed vocab with chars + continuation chars + the UNK token.
    vocab_.clear();
    TokenId next_id = 0;
    if (!unk_token_.empty()) {
        vocab_[unk_token_] = next_id++;
    }
    auto add_token = [&](const std::string& s) {
        auto [it, inserted] = vocab_.try_emplace(s, next_id);
        if (inserted)
            ++next_id;
        return it->second;
    };
    std::vector<WordState> words;
    words.reserve(word_freq.size());
    for (const auto& [w, freq] : word_freq) {
        WordState ws;
        ws.count = freq;
        std::size_t pos = 0;
        bool first = true;
        while (pos < w.size()) {
            unsigned char c0 = static_cast<unsigned char>(w[pos]);
            std::size_t cp_len;
            if (c0 < 0x80)
                cp_len = 1;
            else if ((c0 >> 5) == 0b110)
                cp_len = 2;
            else if ((c0 >> 4) == 0b1110)
                cp_len = 3;
            else if ((c0 >> 3) == 0b11110)
                cp_len = 4;
            else
                cp_len = 1;
            if (pos + cp_len > w.size())
                cp_len = w.size() - pos;
            std::string ch = w.substr(pos, cp_len);
            std::string tok = first ? ch : continuing_prefix_ + ch;
            add_token(tok);
            ws.symbols.push_back(std::move(tok));
            first = false;
            pos += cp_len;
        }
        words.push_back(std::move(ws));
    }
    // 3. Greedy pair merging.
    while (vocab_.size() < target_vocab_size) {
        std::unordered_map<std::string, std::uint64_t> pair_freq;
        for (const auto& ws : words) {
            for (std::size_t k = 0; k + 1 < ws.symbols.size(); ++k) {
                std::string key = ws.symbols[k] + '\x01' + ws.symbols[k + 1];
                pair_freq[key] += ws.count;
            }
        }
        if (pair_freq.empty())
            break;
        auto best_it = pair_freq.begin();
        for (auto it = pair_freq.begin(); it != pair_freq.end(); ++it) {
            if (it->second > best_it->second ||
                (it->second == best_it->second && it->first < best_it->first)) {
                best_it = it;
            }
        }
        if (best_it->second < 2)
            break;
        std::string key = best_it->first;
        auto sep = key.find('\x01');
        std::string left = key.substr(0, sep);
        std::string right = key.substr(sep + 1);
        // Form the merged surface.  Strip the right half's "##"
        // before concatenation (the merged token keeps "##" iff the
        // LEFT half had one).
        std::string right_core = right;
        if (right.size() >= continuing_prefix_.size() &&
            right.compare(0, continuing_prefix_.size(), continuing_prefix_) == 0) {
            right_core = right.substr(continuing_prefix_.size());
        }
        std::string merged = left + right_core;
        add_token(merged);
        // Rewrite words to apply the merge.
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
