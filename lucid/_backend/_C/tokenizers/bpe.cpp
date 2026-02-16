#include "tokenizers.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace lucid::tokenizers::fast {
    
    namespace {
        bool parse_hex4(const std::string& s, std::size_t pos, uint32_t& out) {
            if (pos + 4 > s.size()) return false;
            uint32_t value = 0;

            for (std::size_t i = 0; i < 4; ++i) {
                const char c = s[pos + i];
                value <<= 4;

                if (c >= '0' && c <= '9') value |= static_cast<uint32_t>(c - '0');
                else if (c >= 'a' && c <= 'f') value |= static_cast<uint32_t>(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') value |= static_cast<uint32_t>(c - 'A' + 10);
                else return false;
            }
            out = value;
            return true;
        }

        void append_utf8(std::string& out, uint32_t cp) {
            if (cp <= 0x7F) {
                out.push_back(static_cast<char>(cp));

            } else if (cp <= 0x7FF) {
                out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));

            } else if (cp <= 0xFFFF) {
                out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));

            } else {
                out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        }

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
                    case 'u': {
                        uint32_t cp1 = 0;
                        if (!parse_hex4(s, i + 1, cp1)) {
                            out.push_back('u');
                            break;
                        }
                        i += 4;
                        if (cp1 >= 0xD800 && cp1 <= 0xDBFF) {
                            if (
                                i + 6 < s.size()
                                && s[i + 1] == '\\'
                                && s[i + 2] == 'u'
                            ) {
                                uint32_t cp2 = 0;
                                if (parse_hex4(s, i + 3, cp2) && cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                                    const uint32_t hi = cp1 - 0xD800;
                                    const uint32_t lo = cp2 - 0xDC00;
                                    const uint32_t full = 0x10000 + ((hi << 10) | lo);
                                    append_utf8(out, full);
                                    i += 6;
                                    break;
                                }
                            }
                        }
                        append_utf8(out, cp1);
                        break;
                    }
                    default:
                        out.push_back(esc);
                        break;
                }
            }
            return out;
        }
    }
    
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

    std::vector<int32_t> BPETokenizer::encode_ids(
        std::string_view text,
        bool add_special_tokens
    ) const {
        const auto tokens = tokenize(text);
        std::vector<int32_t> ids;
        ids.reserve(tokens.size() + 2);

        if (add_special_tokens && bos_token_.has_value()) {
            ids.push_back(convert_token_to_id(*bos_token_));
        }
        for (const auto& t : tokens) {
            ids.push_back(convert_token_to_id(t));
        }
        if (add_special_tokens && eos_token_.has_value()) {
            ids.push_back(convert_token_to_id(*eos_token_));
        }
        return ids;
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

    const BPETokenizer::Vocab& BPETokenizer::vocab() const noexcept {
        return vocab_;
    }

    const std::vector<BPETokenizer::Merge>& BPETokenizer::merges() const noexcept {
        return merges_;
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
        const std::string content(
            (std::istreambuf_iterator<char>(fin)),
            std::istreambuf_iterator<char>()
        );
        if (content.empty()) {
            throw std::runtime_error(
                "Parsed empty vocabulary from file: " + vocab_file.string()
            );
        }

        Vocab vocab;
        const std::regex kv(R"kv("((?:\\.|[^"\\])*)"\s*:\s*([0-9]+))kv");
        for (
            std::sregex_iterator it(content.begin(), content.end(), kv), end;
            it != end;
            ++it
        ) {
            const auto& m = *it;
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
        std::ifstream fin(merges_file);
        if (!fin.is_open()) {
            throw std::runtime_error(
                "Failed to open merges file: " + merges_file.string()
            );
        }
        std::vector<Merge> merges;
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            if (!line.empty() && line[0] == '#') continue;

            std::istringstream iss(line);
            std::string left;
            std::string right;

            if (!(iss >> left >> right)) continue;
            merges.emplace_back(std::move(left), std::move(right));
        }
        return merges;
    }

    void BPETokenizer::ensure_special_tokens() {
        const auto specials = all_special_tokens();
        int32_t next_id = 0;
        for (const auto& [token, id] : vocab_) {
            (void)token;
            if (id >= next_id) next_id = id + 1;
        }
        for (const auto& token : specials) {
            if (vocab_.find(token) == vocab_.end()) {
                vocab_[token] = next_id++;
            }
        }
    }

    void BPETokenizer::rebuild_id_to_tokens() {
        int32_t max_id = -1;
        for (const auto& [token, id] : vocab_) {
            (void)token;
            if (id < 0) {
                throw std::runtime_error(
                    "Vocabulary indices must be non-negative integers."
                );
            }
            if (id > max_id) max_id = id;
        }

        if (max_id < 0) {
            ids_to_tokens_.clear();
            return;
        }

        ids_to_tokens_.assign(static_cast<std::size_t>(max_id) + 1, "");
        for (const auto& [token, id] : vocab_) {
            auto& slot = ids_to_tokens_[static_cast<std::size_t>(id)];
            if (!slot.empty()) {
                throw std::runtime_error(
                    "Duplicate token index detected: " + std::to_string(id)
                );
            }
            slot = token;
        }
        for (auto& token : ids_to_tokens_) {
            if (token.empty()) token = unk_token_;
        }
    }

    void BPETokenizer::rebuild_merge_ranks() {
        merge_ranks_.clear();
        merge_ranks_.reserve(merges_.size());
        for(std::size_t i = 0; i < merges_.size(); ++i){
            merge_ranks_[merges_[i]] = i;
        }
    }

    int32_t BPETokenizer::unk_id() const {
        const auto it = vocab_.find(unk_token_);
        if (it == vocab_.end()) {
            throw std::runtime_error(
                "Unknown token '" + unk_token_ + "' is not in vocabulary."
            );
        }
        return it->second;
    }

    std::string BPETokenizer::id_to_token_impl(int32_t id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= ids_to_tokens_.size()) {
            return unk_token_;
        }
        const auto& tok = ids_to_tokens_[static_cast<std::size_t>(id)];
        return tok.empty() ? unk_token_ : tok;
    }

    std::vector<std::string> BPETokenizer::bpe_tokenize(
        const std::string& token
    ) const {
        if (token.empty()) return {};

        std::vector<std::string> symbols;
        symbols.reserve(token.size());

        for (char ch : token) symbols.emplace_back(1, ch);
        symbols.back() += end_of_word_suffix_;

        while (symbols.size() > 1) {
            bool found = false;
            std::size_t best_rank = static_cast<std::size_t>(-1);
            Merge best_pair;

            for (std::size_t i = 0; i < symbols.size() - 1; ++i) {
                Merge p{symbols[i], symbols[i + 1]};
                auto it = merge_ranks_.find(p);
                if (it == merge_ranks_.end()) continue;
                
                if (!found || it->second < best_rank) {
                    found = true;
                    best_rank = it->second;
                    best_pair = std::move(p);
                }
            }

            if (!found) break;
            symbols = merge_word_once(symbols, best_pair);
        }
        return symbols;
    }

    std::unordered_map<std::string, std::size_t> BPETokenizer::build_word_frequency(
        const std::vector<std::string>& texts
    ) const {
        std::unordered_map<std::string, std::size_t> word_freq;
        for (const auto& raw_text : texts) {
            std::string text = raw_text;
            if (clean_text_) text = detail::clean_text(text);

            const auto tokens = detail::basic_tokenize(text, lowercase_);
            for (const auto& tok : tokens) {
                if (!tok.empty()) ++word_freq[tok];
            }
        }
        return word_freq;
    }

    std::vector<std::string> BPETokenizer::merge_word_once(
        const std::vector<std::string>& symbols,
        const Merge& pair
    ) {
        std::vector<std::string> out;
        out.reserve(symbols.size());

        std::size_t i = 0;
        while (i < symbols.size()) {
            if (
                i < symbols.size() - 1
                && symbols[i] == pair.first
                && symbols[i + 1] == pair.second
            ) {
                out.push_back(pair.first + pair.second);
                i += 2;
            } else {
                out.push_back(symbols[i]);
                ++i;
            }
        }
        return out;
    }
}
