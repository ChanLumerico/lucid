 #include "tokenizers.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace lucid::tokenizers::core {

    std::size_t WordPieceTokenizer::PairHash::operator()(
        const std::pair<std::string, std::string>& p
    ) const noexcept {
        const std::size_t h1 = std::hash<std::string>{}(p.first);
        const std::size_t h2 = std::hash<std::string>{}(p.second);

        return hash_combine(h1, h2);
    }

    WordPieceTokenizer::WordPieceTokenizer(
        std::optional<Vocab> vocab,
        std::optional<std::filesystem::path> vocab_file,

        std::string unk_token,
        std::string pad_token,

        std::optional<std::string> bos_token,
        std::optional<std::string> eos_token,

        std::string wordpieces_prefix,
        std::size_t max_input_chars_per_word,

        bool lowercase,
        bool clean_text
    ) {
        if (vocab.has_value() && vocab_file.has_value()) {
            throw std::invalid_argument("Provide only one of 'vocab' or 'vocab_file'.");
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

        wordpieces_prefix_ = std::move(wordpieces_prefix);
        max_input_chars_per_word_ = max_input_chars_per_word;
        lowercase_ = lowercase;
        clean_text_ = clean_text;

        ensure_special_tokens();
        rebuild_id_to_tokens();
    }

    std::size_t WordPieceTokenizer::vocab_size() const {
        return vocab_.size();
    }

    std::vector<std::string> WordPieceTokenizer::tokenize(std::string_view text) const {
        std::string input(text);
        if (clean_text_) input = clean_text_impl(input);

        const auto basic_tokens = basic_tokenize(input);
        std::vector<std::string> out;

        for(const auto& tok : basic_tokens) {
            const auto wp = wordpiece_tokenize(tok);
            out.insert(out.end(), wp.begin(), wp.end());
        }
        return out;
    }

    int32_t WordPieceTokenizer::convert_token_to_id(std::string_view token) const {
        auto it = vocab_.find(std::string(token));
        if (it != vocab_.end()) return it->second;
        return unk_id();
    }

    std::vector<int32_t> WordPieceTokenizer::convert_tokens_to_ids(
        const std::vector<std::string>& tokens
    ) const {
        std::vector<int32_t> out;
        out.reserve(tokens.size());

        for (const auto& t : tokens) {
            out.push_back(convert_token_to_id(t));
        }
        return out;
    }

    std::string WordPieceTokenizer::convert_id_to_token(int32_t id) const {
        return id_to_token_impl(id);
    }

    std::vector<std::string> WordPieceTokenizer::convert_ids_to_tokens(
        const std::vector<int32_t>& ids
    ) const {
        std::vector<std::string> out;
        out.reserve(ids.size());

        for (int32_t id : ids){
            out.push_back(convert_id_to_token(id));
        }
        return out;
    }

    std::string WordPieceTokenizer::convert_tokens_to_string(
        const std::vector<std::string>& tokens
    ) const {
        std::string out;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) out += " ";
            out += tokens[i];
        }

        const std::string pref = " " + wordpieces_prefix_;
        std::size_t pos = 0;
        while ((pos = out.find(pref, pos)) != std::string::npos) {
            out.erase(pos, pref.size());
        }

        const std::vector<std::string> right_attach = {
            ".", ",", "!", "?", ";", ":", "%", ")", "]", "}"
        };
        for (const auto& p : right_attach) {
            const std::string needle = " " + p;
            std::size_t rpos = 0;
            while ((rpos = out.find(needle, rpos)) != std::string::npos) {
                out.replace(rpos, needle.size(), p);
                rpos += p.size();
            }
        }

        const std::vector<std::string> left_attach = {"(", "[", "{"};
        for (const auto& p : left_attach) {
            const std::string needle = p + " ";
            std::size_t lpos = 0;
            while ((lpos = out.find(needle, lpos)) != std::string::npos) {
                out.replace(lpos, needle.size(), p);
                lpos += p.size();
            }
        }

        while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) {
            out.erase(out.begin());
        }
        while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) {
            out.pop_back();
        }

        return out;
    }

    WordPieceTokenizer& WordPieceTokenizer::fit(
        const std::vector<std::string>& texts,
        std::size_t target_vocab_size,
        std::size_t min_frequency
    ) {
        vocab_ = train_vocab(texts, target_vocab_size, min_frequency);
        rebuild_id_to_tokens();
        return *this;
    }

    WordPieceTokenizer::Vocab WordPieceTokenizer::load_vocab(
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
        int32_t idx = 0;
        while (std::getline(fin, line)) {
            if (!line.empty()) vocab[line] = idx;
            ++idx;
        }
        return vocab;
    }

    void WordPieceTokenizer::ensure_special_tokens() {
        const auto specials = all_special_tokens();
        for (const auto& token : specials) {
            if (vocab_.find(token) == vocab_.end()) {
                vocab_[token] = static_cast<int32_t>(vocab_.size());
            }
        }
    }

    void WordPieceTokenizer::rebuild_id_to_tokens() {
        int32_t max_id = -1;
        for (const auto& [token, id] : vocab_) {
            (void)token;
            if (id > max_id) max_id = id;
        }
        if (max_id < 0) {
            ids_to_tokens_.clear();
            return;
        }

        ids_to_tokens_.assign(static_cast<std::size_t>(max_id) + 1, "");
        for (const auto& [token, id] : vocab_) {
            if (id >= 0) ids_to_tokens_[static_cast<std::size_t>(id)] = token;
        }
    }

    int32_t WordPieceTokenizer::unk_id() const {
        const auto it = vocab_.find(unk_token_);
        if (it == vocab_.end()) {
            throw std::runtime_error(
                "Unknown token '" + unk_token_ + "' is not in vocabulary."
            );
        }
        return it->second;
    }

    std::string WordPieceTokenizer::id_to_token_impl(int32_t id) const {
        if (id < 0 || static_cast<std::size_t>(id) >= ids_to_tokens_.size() ){
            return unk_token_;
        }
        const auto& tok = ids_to_tokens_[static_cast<std::size_t>(id)];
        return tok.empty() ? unk_token_ : tok;
    }

    std::vector<std::string> WordPieceTokenizer::basic_tokenize(
        const std::string& text
    ) const {
        std::vector<std::string> out;
        std::string current;

        auto flush_current = [&]() {
            if (current.empty()) return;
            if (lowercase_) {
                for (char& ch : current) {
                    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
                }
            }
            const auto split = split_on_punctuation(current);
            out.insert(out.end(), split.begin(), split.end());
            current.clear();
        };
        for (char ch : text) {
            if (is_whitespace(ch)) flush_current();
            else current.push_back(ch);
        }
        flush_current();
        return out;
    }

    std::vector<std::string> WordPieceTokenizer::wordpiece_tokenize(
        const std::string& token
    ) const {
        if (token.size() > max_input_chars_per_word_) return {unk_token_};

        std::vector<std::string> out;
        std::size_t start = 0;
        while (start < token.size()) {
            std::size_t end = token.size();
            bool found = false;
            std::string best;

            while (start < end) {
                std::string sub = token.substr(start, end - start);
                if (start > 0) sub = wordpieces_prefix_ + sub;
                
                if (vocab_.find(sub) != vocab_.end()) {
                    best = std::move(sub);
                    found = true;
                    break;
                }
                --end;
            }
            if (!found) return {unk_token_};

            out.push_back(std::move(best));
            start = end;
        }
        return out;
    }

    std::vector<std::string> WordPieceTokenizer::split_on_punctuation(
        const std::string& token
    ) {
        std::vector<std::string> out;
        std::string current;

        for (char ch : token) {
            if (is_punctuation(ch)) {
                if (!current.empty()) {
                    out.push_back(std::move(current));
                    current.clear();
                }
                out.emplace_back(1, ch);
            } else {
                current.push_back(ch);
            }
        }
        if (!current.empty()) out.push_back(std::move(current));
        return out;
    }

    bool WordPieceTokenizer::is_whitespace(char ch) {
        return std::isspace(static_cast<unsigned char>(ch)) != 0;
    }

    bool WordPieceTokenizer::is_control(char ch) {
        if (ch == '\t' || ch == '\n' || ch == '\r') return false;
        return std::iscntrl(static_cast<unsigned char>(ch)) != 0;
    }

    bool WordPieceTokenizer::is_punctuation(char ch) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        const int cp = static_cast<int>(uch);
        if (
            (33 <= cp && cp <= 47) 
            || (58 <= cp && cp <= 64) 
            || (91 <= cp && cp <= 96) 
            || (123 <= cp && cp <= 126)
        ) return true;
        return std::ispunct(uch) != 0;
    }

    std::string WordPieceTokenizer::clean_text_impl(const std::string& text) {
        std::string out;
        out.reserve(text.size());

        for (unsigned char uch : text) {
            const char ch = static_cast<char>(uch);
            if (uch == 0 || is_control(ch)) continue;

            if (is_whitespace(ch)) out.push_back(' ');
            else out.push_back(ch);
        }
        return out;
    }

    WordPieceTokenizer::Vocab WordPieceTokenizer::train_vocab(
        const std::vector<std::string>& texts,
        std::size_t target_vocab_size,
        std::size_t min_frequency
    ) const {
        const auto special_tokens = all_special_tokens();
        if (target_vocab_size < special_tokens.size()) {
            throw std::invalid_argument(
                "'vocab_size' must be >= number of special tokens ("
                + std::to_string(special_tokens.size()) + ")."
            );
        }
        if (min_frequency < 1) {
            throw std::invalid_argument(
                "'min_frequency' must be >= 1."
            );
        }
        const auto word_freq = build_word_frequency(texts);
        if (word_freq.empty()) throw std::invalid_argument(
            "Cannot train WordPiece vocabulary from empty corpus."
        );

        auto [word_splits, token_set] = initialize_splits(word_freq);
        std::vector<std::string> merged_tokens;
        std::unordered_map<std::string, bool> merged_seen;

        const std::size_t token_target = target_vocab_size - special_tokens.size();

        while ((token_set.size() + merged_tokens.size()) < token_target) {
            const auto best_pair = select_best_pair(word_splits, word_freq);
            if (!best_pair.has_value()) break;

            const std::string merged = merge_pair(*best_pair, word_splits);
            if (
                token_set.find(merged) == token_set.end() 
                && merged_seen.find(merged) == merged_seen.end()
            ) {
                merged_tokens.push_back(merged);
                merged_seen[merged] = true;
            }
        }

        std::vector<std::string> token_list;
        token_list.reserve(token_set.size());

        for (const auto& [tok, seen] : token_set) {
            if (seen) token_list.push_back(tok);
        }
        std::sort(token_list.begin(), token_list.end());
        token_list.insert(token_list.end(), merged_tokens.begin(), merged_tokens.end());

        std::vector<std::string> vocab_tokens;
        vocab_tokens.reserve(std::min(token_target, token_list.size()));

        std::unordered_map<std::string, bool> seen;
        for (const auto& tok : token_list) {
            if (seen.find(tok) != seen.end()) continue;
            seen[tok] = true;

            vocab_tokens.push_back(tok);
            if (vocab_tokens.size() >= token_target) break;
        }

        std::vector<std::string> full_vocab;
        
        full_vocab.reserve(special_tokens.size() + vocab_tokens.size());
        full_vocab.insert(full_vocab.end(), special_tokens.begin(), special_tokens.end());
        full_vocab.insert(full_vocab.end(), vocab_tokens.begin(), vocab_tokens.end());

        Vocab trained;
        for (std::size_t i = 0; i < full_vocab.size(); ++i) {
            trained[full_vocab[i]] = static_cast<int32_t>(i);
        }
        return trained;
    }

    std::unordered_map<std::string, std::size_t> WordPieceTokenizer::build_word_frequency(
        const std::vector<std::string>& texts
    ) const {
        std::unordered_map<std::string, std::size_t> word_freq;
        for (const auto& raw_text: texts) {
            std::string text = raw_text;
            if (clean_text_) text = clean_text_impl(text);

            const auto tokens = basic_tokenize(text);
            for (const auto& token : tokens) {
                if (!token.empty()) ++word_freq[token];
            }
        }
        return word_freq;
    }

    std::pair<
        std::unordered_map<std::string, std::vector<std::string>>,
        std::unordered_map<std::string, bool>
    > WordPieceTokenizer::initialize_splits(
        const std::unordered_map<std::string, std::size_t>& word_freq
    ) const {
        std::unordered_map<std::string, std::vector<std::string>> word_splits;
        std::unordered_map<std::string, bool> token_set;

        for (const auto& [word, freq] : word_freq) {
            (void)freq;
            if (word.empty()) continue;

            std::vector<std::string> split;
            split.reserve(word.size());
            split.emplace_back(1, word[0]);

            for (std::size_t i = 1; i < word.size(); ++i) {
                split.push_back(wordpieces_prefix_ + std::string(1, word[i]));
            }

            word_splits[word] = split;
            for (const auto& tok : split) token_set[tok] = true;
        }
        return {std::move(word_splits), std::move(token_set)};
    }

    std::optional<std::pair<std::string, std::string>> WordPieceTokenizer::select_best_pair(
        const std::unordered_map<std::string, std::vector<std::string>>& word_splits,
        const std::unordered_map<std::string, std::size_t>& word_freq
    ) const {
        // not implemented
    }

    std::string WordPieceTokenizer::merge_pair(
        const std::pair<std::string, std::string>& pair,
        std::unordered_map<std::string, std::vector<std::string>>& word_splits
    ) const {
        // not implemented
    }
}
