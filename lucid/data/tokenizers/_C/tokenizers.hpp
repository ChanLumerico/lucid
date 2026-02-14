#pragma once

#include "base.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>


namespace lucid::tokenizers::C {

    class WordPieceTokenizer final : public TokenizerBase {
        public:
            using Vocab = std::unordered_map<std::string, int32_t>;

            WordPieceTokenizer(
                std::optional<Vocab> vocab = std::nullopt,
                std::optional<std::filesystem::path> vocab_file = std::nullopt,

                std::string unk_token = "[UNK]",
                std::string pad_token = "[PAD]",

                std::optional<std::string> bos_token = std::nullopt,
                std::optional<std::string> eos_token = std::nullopt,

                std::string wordpieces_prefix = "##",
                std::size_t max_input_chars_per_word = 100,

                bool lowercase = true,
                bool clean_text = true
            );

            std::size_t vocab_size() const override;
            std::vector<std::string> tokenize(std::string_view text) const override;

            int32_t convert_token_to_id(std::string_view token) const override;
            std::vector<int32_t> convert_tokens_to_ids(
                const std::vector<std::string>& tokens
            ) const override;

            std::string convert_id_to_token(int32_t id) const override;
            std::vector<std::string> convert_ids_to_tokens(
                const std::vector<int32_t>& ids
            ) const override;

            std::string convert_tokens_to_string(
                const std::vector<std::string>& tokens
            ) const override;

            WordPieceTokenizer& fit(
                const std::vector<std::string>& texts,
                std::size_t vocab_size,
                std::size_t min_frequency
            ) override;
        
        private:
            struct PairHash {
                std::size_t operator()(
                    const std::pair<std::string, std::string>& p
                ) const noexcept;
            };

            static constexpr std::size_t kMagic = 0x9e3779b97f4a7c15ULL;
            static std::size_t hash_combine(std::size_t h1, std::size_t h2) noexcept {
                return h1 ^ (h2 + kMagic + (h1 << 6) + (h1 >> 2));
            }

            static Vocab load_vocab(const std::filesystem::path& vocab_file);
            void ensure_special_tokens();
            void rebuild_id_to_tokens();

            int32_t unk_id() const;
            std::string id_to_token_impl(int32_t id) const;
            
            std::vector<std::string> basic_tokenize(const std::string& text) const;
            std::vector<std::string> wordpiece_tokenize(const std::string& token) const;
            static std::vector<std::string> split_on_punctuation(const std::string& token);

            static bool is_whitespace(char ch);
            static bool is_control(char ch);
            static bool is_punctuation(char ch);
            static std::string clean_text_impl(const std::string& text);

            Vocab train_vocab(
                const std::vector<std::string>& texts,
                std::size_t vocab_size,
                std::size_t min_frequency
            ) const;

            std::unordered_map<std::string, std::size_t> build_word_frequency(
                const std::vector<std::string>& texts
            ) const;

            std::pair<
                std::unordered_map<std::string, std::vector<std::string>>,
                std::unordered_map<std::string, bool>
            > initialize_splits (
                const std::unordered_map<std::string, std::size_t>& word_freq
            ) const;
            
            std::optional<std::pair<std::string, std::string>> select_best_pair(
                const std::unordered_map<std::string, std::vector<std::string>>& word_splits,
                const std::unordered_map<std::string, std::size_t>& word_freq
            ) const;

            std::string merge_pair(
                const std::pair<std::string, std::string>& pair,
                std::unordered_map<std::string, std::vector<std::string>>& word_splits
            ) const;
        
        private:
            Vocab vocab_;
            std::vector<std::string> ids_to_tokens_;

            std::string wordpieces_prefix_ = "##";
            std::size_t max_input_chars_per_word_ = 100;

            bool lowercase_ = true;
            bool clean_text_ = true;
    };
}
