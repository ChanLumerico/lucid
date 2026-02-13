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
        // not implemented
    }

    WordPieceTokenizer::Vocab WordPieceTokenizer::load_vocab(
        const std::filesystem::path& vocab_file
    ) {
        // not implemented
    }

    void WordPieceTokenizer::ensure_special_tokens() {
        // not imeplemented
    }

    void WordPieceTokenizer::rebuild_id_to_tokens() {
        // not implemented
    }

    // still a lot left to implement ...
}
