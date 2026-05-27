// lucid/_C/utils/tokenizer/Unigram.cpp
//
// Unigram tokenizer implementation — see Unigram.h for the API + the
// algorithm summary.  Comments inline focus on the why of the
// numerical-stability and pruning choices.

#include "Unigram.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace lucid::utils::tokenizer {

namespace {

// Float ``log(0)`` sentinel — matches SentencePiece's convention.
// Pieces with this score are treated as "impossible" by Viterbi.
constexpr double kNegInf = -std::numeric_limits<double>::infinity();

// UTF-8 codepoint length given a lead byte.
inline std::size_t utf8_cp_len(unsigned char c0) noexcept {
    if (c0 < 0x80) return 1;
    if ((c0 >> 5) == 0b110) return 2;
    if ((c0 >> 4) == 0b1110) return 3;
    if ((c0 >> 3) == 0b11110) return 4;
    return 1;  // invalid lead — treat as single byte
}

// Build the per-byte "is codepoint start" mask used by both Viterbi
// and forward-backward to constrain piece boundaries to legitimate
// UTF-8 boundaries.  Without this constraint we'd carve pieces in
// the middle of multi-byte characters.
std::vector<bool> build_is_cp_start_(const std::string& s) {
    std::vector<bool> mask(s.size() + 1, false);
    std::size_t i = 0;
    while (i < s.size()) {
        mask[i] = true;
        i += utf8_cp_len(static_cast<unsigned char>(s[i]));
    }
    mask[s.size()] = true;
    return mask;
}

// Numerically-stable log-sum-exp for forward-backward.
inline double logsumexp_(double a, double b) {
    if (a == kNegInf) return b;
    if (b == kNegInf) return a;
    double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}

}  // namespace

Unigram::Unigram() = default;

Unigram::Unigram(std::vector<std::pair<std::string, double>> pieces,
                 std::string unk_token,
                 double unk_log_prob)
    : pieces_(std::move(pieces)),
      unk_token_(std::move(unk_token)),
      unk_log_prob_(unk_log_prob) {
    rebuild_tables_();
}

void Unigram::rebuild_tables_() {
    piece_to_id_.clear();
    piece_to_id_.reserve(pieces_.size());
    max_piece_bytes_ = 0;
    for (std::size_t i = 0; i < pieces_.size(); ++i) {
        piece_to_id_[pieces_[i].first] = static_cast<TokenId>(i);
        if (pieces_[i].first.size() > max_piece_bytes_)
            max_piece_bytes_ = pieces_[i].first.size();
    }
    auto it = piece_to_id_.find(unk_token_);
    unk_id_ = (it == piece_to_id_.end()) ? -1 : it->second;
}

std::unordered_map<std::string, TokenId> Unigram::get_vocab() const {
    return piece_to_id_;
}

std::string Unigram::id_to_token(TokenId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size()) return "";
    return pieces_[static_cast<std::size_t>(id)].first;
}

IdSequence Unigram::viterbi_encode_(const std::string& chunk) const {
    const std::size_t N = chunk.size();
    if (N == 0) return {};

    // ``dp[i]`` = max log-prob for the chunk prefix ending at byte i.
    // ``back[i]`` = (start byte of last piece, piece id) achieving dp[i].
    std::vector<double> dp(N + 1, kNegInf);
    std::vector<std::pair<std::size_t, TokenId>> back(
        N + 1, {static_cast<std::size_t>(-1), -1});
    dp[0] = 0.0;

    auto is_cp_start = build_is_cp_start_(chunk);

    for (std::size_t i = 1; i <= N; ++i) {
        if (!is_cp_start[i]) continue;  // can't end a piece mid-codepoint
        // Look back at every j < i where (j, i) covers a vocab piece.
        std::size_t j_min = (i > max_piece_bytes_) ? (i - max_piece_bytes_) : 0;
        for (std::size_t j = j_min; j < i; ++j) {
            if (!is_cp_start[j]) continue;
            if (dp[j] == kNegInf) continue;
            std::string sub = chunk.substr(j, i - j);
            auto it = piece_to_id_.find(sub);
            double score;
            TokenId pid;
            if (it != piece_to_id_.end()) {
                score = dp[j] + pieces_[static_cast<std::size_t>(it->second)].second;
                pid = it->second;
            } else if (i - j == utf8_cp_len(static_cast<unsigned char>(chunk[j]))) {
                // Single-codepoint fallback to UNK.
                score = dp[j] + unk_log_prob_;
                pid = unk_id_;
            } else {
                continue;
            }
            if (score > dp[i]) {
                dp[i] = score;
                back[i] = {j, pid};
            }
        }
    }

    // If dp[N] is still -inf, the chunk can't be encoded at all
    // (no UNK + no covering pieces).  Return empty rather than crash.
    if (dp[N] == kNegInf) return {};

    // Trace back.
    IdSequence ids;
    std::size_t i = N;
    while (i > 0) {
        auto [j, pid] = back[i];
        if (pid >= 0) ids.push_back(pid);
        i = j;
    }
    std::reverse(ids.begin(), ids.end());
    return ids;
}

IdSequence Unigram::encode(const std::string& text) const {
    // The Python wrapper does normalisation + pre-tokenisation BEFORE
    // calling encode.  When called directly on a multi-word string we
    // run Viterbi over the whole input (slower but correct).
    return viterbi_encode_(text);
}

std::string Unigram::decode(const IdSequence& ids) const {
    std::string out;
    for (TokenId id : ids) {
        if (id >= 0 && static_cast<std::size_t>(id) < pieces_.size())
            out += pieces_[static_cast<std::size_t>(id)].first;
    }
    return out;
}

void Unigram::forward_backward_accumulate_(
    const std::string& chunk,
    std::uint64_t weight,
    std::vector<double>& expected_counts) const {
    const std::size_t N = chunk.size();
    if (N == 0) return;

    auto is_cp_start = build_is_cp_start_(chunk);

    // Forward log-probs: alpha[i] = log Σ_paths P(path ending at i).
    std::vector<double> alpha(N + 1, kNegInf);
    alpha[0] = 0.0;
    for (std::size_t i = 1; i <= N; ++i) {
        if (!is_cp_start[i]) continue;
        std::size_t j_min = (i > max_piece_bytes_) ? (i - max_piece_bytes_) : 0;
        for (std::size_t j = j_min; j < i; ++j) {
            if (!is_cp_start[j]) continue;
            if (alpha[j] == kNegInf) continue;
            std::string sub = chunk.substr(j, i - j);
            auto it = piece_to_id_.find(sub);
            if (it == piece_to_id_.end()) continue;
            double score = alpha[j]
                + pieces_[static_cast<std::size_t>(it->second)].second;
            alpha[i] = logsumexp_(alpha[i], score);
        }
    }

    // Backward log-probs: beta[i] = log Σ_paths P(path from i to N).
    std::vector<double> beta(N + 1, kNegInf);
    beta[N] = 0.0;
    for (std::size_t i = N; i > 0;) {
        --i;
        if (!is_cp_start[i]) continue;
        std::size_t k_max = std::min(N, i + max_piece_bytes_);
        for (std::size_t k = i + 1; k <= k_max; ++k) {
            if (!is_cp_start[k]) continue;
            if (beta[k] == kNegInf) continue;
            std::string sub = chunk.substr(i, k - i);
            auto it = piece_to_id_.find(sub);
            if (it == piece_to_id_.end()) continue;
            double score = beta[k]
                + pieces_[static_cast<std::size_t>(it->second)].second;
            beta[i] = logsumexp_(beta[i], score);
        }
    }

    // Total likelihood Z = alpha[N] (= beta[0]).  If Z is -inf the
    // word is unreachable under current vocab; skip.
    double Z = alpha[N];
    if (Z == kNegInf) return;

    // Accumulate expected counts: for each (j, i) edge,
    //   E_count(piece) += w * exp(alpha[j] + log p(piece) + beta[i] - Z).
    for (std::size_t i = 1; i <= N; ++i) {
        if (!is_cp_start[i]) continue;
        std::size_t j_min = (i > max_piece_bytes_) ? (i - max_piece_bytes_) : 0;
        for (std::size_t j = j_min; j < i; ++j) {
            if (!is_cp_start[j]) continue;
            if (alpha[j] == kNegInf || beta[i] == kNegInf) continue;
            std::string sub = chunk.substr(j, i - j);
            auto it = piece_to_id_.find(sub);
            if (it == piece_to_id_.end()) continue;
            double log_lp = pieces_[static_cast<std::size_t>(it->second)].second;
            double posterior_log = alpha[j] + log_lp + beta[i] - Z;
            expected_counts[static_cast<std::size_t>(it->second)]
                += static_cast<double>(weight) * std::exp(posterior_log);
        }
    }
}

void Unigram::train(const std::vector<std::string>& corpus,
                    std::size_t target_vocab_size) {
    train_with_options(corpus, target_vocab_size,
                       /*num_iterations=*/2,
                       /*shrink_factor=*/0.75,
                       /*max_piece_length=*/16,
                       /*initial_vocab_multiplier=*/10);
}

void Unigram::train_with_options(
    const std::vector<std::string>& corpus,
    std::size_t target_vocab_size,
    std::size_t num_iterations,
    double shrink_factor,
    std::size_t max_piece_length,
    std::size_t initial_vocab_multiplier) {
    if (target_vocab_size < 2)
        throw std::runtime_error(
            "Unigram::train: target_vocab_size must be >= 2");
    if (shrink_factor <= 0.0 || shrink_factor >= 1.0)
        throw std::runtime_error(
            "Unigram::train: shrink_factor must be in (0, 1)");

    // 1. Word frequency dict (whitespace split — the Python wrapper
    //    is responsible for the heavy normalisation chain).
    std::unordered_map<std::string, std::uint64_t> word_freq;
    for (const auto& doc : corpus) {
        std::size_t i = 0;
        while (i < doc.size()) {
            while (i < doc.size() &&
                   (doc[i] == ' ' || doc[i] == '\t' || doc[i] == '\n' ||
                    doc[i] == '\r'))
                ++i;
            std::size_t start = i;
            while (i < doc.size() && !(doc[i] == ' ' || doc[i] == '\t' ||
                                       doc[i] == '\n' || doc[i] == '\r'))
                ++i;
            if (i > start) ++word_freq[doc.substr(start, i - start)];
        }
    }

    // 2. Seed vocab: every distinct substring of length up to
    //    ``max_piece_length`` bytes that lies on UTF-8 codepoint
    //    boundaries, weighted by the frequency of its parent words.
    std::unordered_map<std::string, std::uint64_t> piece_freq;
    for (const auto& [w, freq] : word_freq) {
        auto is_cp = build_is_cp_start_(w);
        for (std::size_t j = 0; j < w.size(); ++j) {
            if (!is_cp[j]) continue;
            std::size_t k_max = std::min(w.size(), j + max_piece_length);
            for (std::size_t k = j + 1; k <= k_max; ++k) {
                if (!is_cp[k]) continue;
                piece_freq[w.substr(j, k - j)] += freq;
            }
        }
    }

    // 3. Seed pieces — top ``initial_vocab_multiplier * target`` by
    //    frequency, but always include every single-codepoint piece
    //    (so Viterbi is always feasible).
    std::vector<std::pair<std::string, std::uint64_t>> seeds(
        piece_freq.begin(), piece_freq.end());
    // Mark single-codepoint pieces as "always keep" by giving them a
    // huge frequency bonus during the partial_sort that follows.
    auto is_single_cp = [](const std::string& s) {
        if (s.empty()) return false;
        return s.size() == utf8_cp_len(static_cast<unsigned char>(s[0]));
    };
    std::size_t initial_size = std::min(
        seeds.size(), initial_vocab_multiplier * target_vocab_size);
    std::partial_sort(
        seeds.begin(), seeds.begin() + initial_size, seeds.end(),
        [&](const auto& a, const auto& b) {
            // is_single_cp pieces sort first, then by frequency desc.
            bool sa = is_single_cp(a.first);
            bool sb = is_single_cp(b.first);
            if (sa != sb) return sa;
            return a.second > b.second;
        });
    seeds.resize(initial_size);

    // 4. Initialise probabilities from frequencies + UNK token.
    pieces_.clear();
    pieces_.reserve(seeds.size() + 1);
    // Reserve id 0 for UNK so it's always present.
    if (unk_token_.empty()) unk_token_ = "<unk>";
    pieces_.emplace_back(unk_token_, unk_log_prob_);

    std::uint64_t total = 0;
    for (const auto& [p, f] : seeds) {
        if (p == unk_token_) continue;
        total += f;
    }
    for (const auto& [p, f] : seeds) {
        if (p == unk_token_) continue;
        double prob = (total > 0) ? (static_cast<double>(f) / total) : 1.0 / seeds.size();
        // Floor to avoid log(0); SentencePiece uses 1e-10.
        if (prob < 1e-10) prob = 1e-10;
        pieces_.emplace_back(p, std::log(prob));
    }
    rebuild_tables_();

    // 5. EM iterations + pruning until target vocab reached.
    while (pieces_.size() > target_vocab_size) {
        // EM iterations to refit probabilities.
        for (std::size_t iter = 0; iter < num_iterations; ++iter) {
            std::vector<double> expected_counts(pieces_.size(), 0.0);
            for (const auto& [w, freq] : word_freq) {
                forward_backward_accumulate_(w, freq, expected_counts);
            }
            double total_count = std::accumulate(
                expected_counts.begin(), expected_counts.end(), 0.0);
            if (total_count <= 0.0) break;
            for (std::size_t i = 0; i < pieces_.size(); ++i) {
                if (static_cast<TokenId>(i) == unk_id_) continue;
                double prob = expected_counts[i] / total_count;
                if (prob < 1e-10) prob = 1e-10;
                pieces_[i].second = std::log(prob);
            }
        }
        // Prune ``shrink_factor`` of pieces by probability — but
        // always keep single-codepoint pieces + the UNK.
        std::vector<std::pair<double, std::size_t>> scored;
        scored.reserve(pieces_.size());
        for (std::size_t i = 0; i < pieces_.size(); ++i) {
            if (static_cast<TokenId>(i) == unk_id_) continue;
            scored.emplace_back(pieces_[i].second, i);
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first;  // descending log-prob
                  });
        std::size_t keep_n = std::max(
            target_vocab_size,
            static_cast<std::size_t>(scored.size() * (1.0 - shrink_factor)));
        keep_n = std::min(keep_n, scored.size());
        std::vector<std::pair<std::string, double>> new_pieces;
        new_pieces.reserve(keep_n + 1);
        new_pieces.emplace_back(unk_token_, unk_log_prob_);
        // Always-keep single-codepoint pieces.
        std::vector<bool> kept(pieces_.size(), false);
        kept[static_cast<std::size_t>(unk_id_)] = true;
        for (std::size_t i = 0; i < pieces_.size(); ++i) {
            if (i == static_cast<std::size_t>(unk_id_)) continue;
            if (is_single_cp(pieces_[i].first)) {
                new_pieces.push_back(pieces_[i]);
                kept[i] = true;
            }
        }
        // Then top-scored multi-codepoint pieces, in score order.
        for (const auto& [score, idx] : scored) {
            if (new_pieces.size() >= keep_n + 1) break;  // +1 for UNK
            if (kept[idx]) continue;
            new_pieces.push_back(pieces_[idx]);
            kept[idx] = true;
        }
        if (new_pieces.size() >= pieces_.size()) break;  // can't shrink
        pieces_ = std::move(new_pieces);
        rebuild_tables_();
    }
}

}  // namespace lucid::utils::tokenizer
