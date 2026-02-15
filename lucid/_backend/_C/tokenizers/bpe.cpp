#include "tokenizers.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace lucid::tokenizers::fast {
    
    BPETokenizer::BPETokenizer(
        std::optional<Vocab> vocab,
        std::optional<std::vector<Merge>> merges,

        std::optional<std::filesystem::path> vocab_file,
        std::optional<std::filesystem::path> merges_file,

        std::string unk_token,
        std::string pad_token,

        std::optional<std::string> bos_token,
        std::optional<std::string> eos_token,

        bool lowercase,
        bool clean_text,

        std::string end_of_word_suffix
    ) {
        if (vocab.has_value() && vocab_file.has_value()) {
            throw std::invalid_argument("Provide only one of 'vocab' or 'vocab_file'.");
        }
        if (merges.has_value() && merges_file.has_value()) {
            throw std::invalid_argument("Provide only one of 'merges' or 'merges_file'.");
        }

        set_special_tokens(
            std::move(unk_token),
            std::move(pad_token),
            std::move(bos_token),
            std::move(eos_token)
        );

        if (vocab_file.has_value()) vocab_ = load_vocab(*vocab_file);
        else if (vocab.has_value()) vocab_ = std::move(*vocab);
        else vocab_.clear();

        if (merges_file.has_value()) merges_ = load_merges(*merges_file);
        else if (merges.has_value()) merges_ = std::move(*merges);
        else merges_.clear();

        lowercase_ = lowercase;
        clean_text_ = clean_text;
        end_of_word_suffix_ = std::move(end_of_word_suffix);

        ensure_special_tokens();
        rebuild_merge_ranks();
        rebuild_id_to_tokens();
    }

    std::size_t BPETokenizer::vocab_size() const {
        return vocab_.size();
    }

    std::vector<std::string> BPETokenizer::tokenize(std::string_view text) const {
        std::string input(text);
        if (clean_text_) input = detail::clean_text(input);

        const auto basic_tokens = detail::basic_tokenize(input, lowercase_);
        std::vector<std::string> out;

        for (const auto& tok : basic_tokens) {
            const auto pieces = bpe_tokenize(tok);
            out.insert(out.end(), pieces.begin(), pieces.end());
        }
        return out;
    }

    int32_t BPETokenizer::convert_token_to_id(std::string_view token) const {
        auto it = vocab_.find(std::string(token));
        if (it != vocab_.end()) return it->second;
        return unk_id();
    }

    std::vector<int32_t> BPETokenizer::convert_tokens_to_ids(
        const std::vector<std::string>& tokens
    ) const {
        std::vector<int32_t> out;
        out.reserve(tokens.size());

        for (const auto& t : tokens) out.push_back(convert_token_to_id(t));
        return out;
    }

    std::string BPETokenizer::convert_id_to_token(int32_t id) const {
        return id_to_token_impl(id);
    }

    std::vector<std::string> BPETokenizer::convert_ids_to_tokens(
        const std::vector<int32_t>& ids
    ) const {
        std::vector<std::string> out;
        out.reserve(ids.size());

        for (const auto& id : ids) out.push_back(convert_id_to_token(id));
        return out;
    }

    std::string BPETokenizer::convert_tokens_to_string(
        const std::vector<std::string>& tokens
    ) const {
        std::string text;
        for (const auto& t: tokens) text += t;

        std::size_t pos = 0;
        while ((pos = text.find(end_of_word_suffix_, pos)) != std::string::npos) {
            text.replace(pos, end_of_word_suffix_.size(), " ");
            ++pos;
        }

        const std::vector<std::string> right_attach = {
            ".", ",", "!", "?", ";", ":", "%", ")", "]", "}"
        };
        for (const auto& p : right_attach) {
            const std::string needle = " " + p;
            std::size_t rpos = 0;

            while ((rpos = text.find(needle, rpos)) != std::string::npos) {
                text.replace(rpos, needle.size(), p);
                rpos += p.size();
            }
        }

        const std::vector<std::string> left_attach = {"(", "[", "{"};
        for (const auto& p : left_attach) {
            const std::string needle = p + " ";
            std::size_t lpos = 0;

            while ((lpos = text.find(needle, lpos)) != std::string::npos) {
                text.replace(lpos, needle.size(), p);
                lpos += p.size();
            }
        }

        text = std::regex_replace(text, std::regex("\\s+"), " ");
        if (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
            text.erase(text.begin());
        }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
            text.pop_back();
        }

        text = std::regex_replace(text, std::regex("([.!?])\\1+"), "$1");
        return text;
    }

    BPETokenizer& BPETokenizer::fit(
        const std::vector<std::string>& texts,
        std::size_t target_vocab_size,
        std::size_t min_frequency
    ) {
        if (min_frequency < 1) {
            throw std::invalid_argument("'min_frequency' must be >= 1.");
        }

        const auto special_tokens = all_special_tokens();
        if (target_vocab_size < special_tokens.size()) {
            throw std::invalid_argument(
                "'vocab_size' must be >= number of special tokens ("
                + std::to_string(special_tokens.size()) + ")."
            );
        }

        const auto word_freq = build_word_frequency(texts);
        if (word_freq.empty()) {
            throw std::invalid_argument("Cannot train BPE vocabulary from empty corpus.");
        }

        std::unordered_map<std::string, std::vector<std::string>> word_splits;
        std::unordered_map<std::string, bool> token_set;

        for (const auto& [word, freq] : word_freq) {
            (void)freq;
            if (word.empty()) continue;

            std::vector<std::string> symbols;
            symbols.reserve(word.size());
            for (char ch : word) symbols.emplace_back(1, ch);

            symbols.back() += end_of_word_suffix_;
            word_splits[word] = symbols;
            for (const auto& s : symbols) token_set[s] = true;
        }

        std::vector<Merge> learned_merges;
        const std::size_t token_target = target_vocab_size - special_tokens.size();

        while (token_set.size() < token_target) {
            std::unordered_map<Merge, std::size_t, detail::TokenPairHash> pair_counts;

            for (const auto& [word, split] : word_splits) {
                const auto wf_it = word_freq.find(word);
                if (wf_it == word_freq.end()) continue;

                const std::size_t freq = wf_it->second;
                for (std::size_t i = 0; i < split.size() - 1; ++i) {
                    pair_counts[{split[i], split[i + 1]}] += freq;
                }
            }
            if (pair_counts.empty()) break;

            bool has_best = false;
            Merge best_pair;
            std::size_t best_count = 0;

            for (const auto& [pair, count] : pair_counts) {
                if (
                    !has_best
                    || count > best_count
                    || (count == best_count && pair < best_pair)
                ) {
                    has_best = true;
                    best_pair = pair;
                    best_count = count;
                }
            }
            if (!has_best || best_count < min_frequency) break;

            learned_merges.push_back(best_pair);
            const std::string merged_token = best_pair.first + best_pair.second;
            token_set[merged_token] = true;

            for (auto& [word, split] : word_splits) {
                (void)word;
                split = merge_word_once(split, best_pair);
            }
        }

        std::vector<std::string> vocab_tokens;
        vocab_tokens.reserve(token_set.size());

        for (const auto& [tok, seen] : token_set) {
            if (seen) vocab_tokens.push_back(tok);
        }

        std::sort(vocab_tokens.begin(), vocab_tokens.end());
        if (vocab_tokens.size() > token_target) vocab_tokens.resize(token_target);

        std::vector<std::string> full_vocab;
        full_vocab.reserve(special_tokens.size() + vocab_tokens.size());
        full_vocab.insert(full_vocab.end(), special_tokens.begin(), special_tokens.end());
        full_vocab.insert(full_vocab.end(), vocab_tokens.begin(), vocab_tokens.end());

        vocab_.clear();
        for (std::size_t i = 0; i < full_vocab.size(); ++i) {
            vocab_[full_vocab[i]] = static_cast<int32_t>(i);
        }

        merges_ = std::move(learned_merges);
        rebuild_merge_ranks();
        rebuild_id_to_tokens();

        return *this;
    }

    BPETokenizer::Vocab BPETokenizer::load_vocab(
        const std::filesystem::path& vocab_file
    ) {
        std::ifstream fin(vocab_file);
        if (!fin.is_open()) {
            throw std::runtime_error(
                "Failed to open vocabulary file: " + vocab_file.string()
            );
        }
        Vocab vocab;
        std::string line;
        std::regex kv(R"kv(^\s*"((?:\\.|[^"\\])*)"\s*:\s*([0-9]+)\s*,?\s*$)kv");

        while (std::getline(fin, line)) {
            std::smatch m;
            if (!std::regex_match(line, m, kv)) continue;

            const std::string token = unescape_json_string(m[1].str());
            const int32_t id = static_cast<int32_t>(std::stoll(m[2].str()));
            vocab[token] = id;
        }

        if (vocab.empty()) {
            throw std::runtime_error(
                "Parsed empty vocabulary from file: " + vocab_file.string()
            );
        }
        return vocab;
    }

    std::vector<BPETokenizer::Merge> BPETokenizer::load_merges(
        const std::filesystem::path& merges_file
    ) {
        //    
    }

    void BPETokenizer::ensure_special_tokens() {
        //
    }

    void BPETokenizer::rebuild_id_to_tokens() {
        //
    }

    void BPETokenizer::rebuild_merge_ranks() {
        //
    }

    int32_t BPETokenizer::unk_id() const {
        //
    }

    std::string BPETokenizer::id_to_token_impl(int32_t id) const {
        //
    }

    std::vector<std::string> BPETokenizer::bpe_tokenize(
        const std::string& token
    ) const {
        //    
    }

    std::unordered_map<std::string, std::size_t> BPETokenizer::build_word_frequency(
        const std::vector<std::string>& texts
    ) const {
        //
    }

    std::vector<std::string> BPETokenizer::merge_word_once(
        const std::vector<std::string>& symbols,
        const Merge& pair
    ) {
        //
    }

    namespace {

        std::string unescape_json_string(const std::string& s) {
            std::string out;
            out.reserve(s.size());

            for (std::size_t i = 0; i < s.size(); ++i) {
                const char ch = s[i];
                if (ch != '\\') {
                    out.push_back(ch);
                    continue;
                }
                if (i + 1 >= s.size()) break;

                const char esc = s[++i];
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default:
                        out.push_back(esc);
                        break;
                }
            }
            return out;
        }
    }
}