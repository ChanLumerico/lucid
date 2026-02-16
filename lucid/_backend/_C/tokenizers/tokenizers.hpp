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


namespace lucid::tokenizers::fast {
    namespace detail {
        using TokenPair = std::pair<std::string, std::string>;

        inline std::size_t hash_combine(std::size_t h1, std::size_t h2) noexcept {
            constexpr std::size_t kMagic = 0x9e3779b97f4a7c15ULL;
            return h1 ^ (h2 + kMagic + (h1 << 6) + (h1 >> 2));
        }

        struct TokenPairHash {
            std::size_t operator()(const TokenPair& p) const noexcept {
                const auto h1 = std::hash<std::string>{}(p.first);
                const auto h2 = std::hash<std::string>{}(p.second);
                return hash_combine(h1, h2);
            }
        };

        bool is_whitespace(char ch);
        bool is_control(char ch);
        bool is_punctuation(char ch);

        std::string clean_text(std::string_view text);
        std::vector<std::string> split_on_punctuation(std::string_view token);
        std::vector<std::string> basic_tokenize(std::string_view text, bool lowercase);
    }

    class WordPieceTokenizer : public TokenizerBase {
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
            static Vocab load_vocab(const std::filesystem::path& vocab_file);
            void ensure_special_tokens();
            void rebuild_id_to_tokens();

            int32_t unk_id() const;
            std::string id_to_token_impl(int32_t id) const;
            
            std::vector<std::string> wordpiece_tokenize(const std::string& token) const;

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

    class BPETokenizer : public TokenizerBase {
        public: 
            using Vocab = std::unordered_map<std::string, int32_t>;
            using Merge = detail::TokenPair;

            BPETokenizer(
                std::optional<Vocab> vocab = std::nullopt,
                std::optional<std::vector<Merge>> merges = std::nullopt,

                std::optional<std::filesystem::path> vocab_file = std::nullopt,
                std::optional<std::filesystem::path> merges_file = std::nullopt,

                std::string unk_token = "[UNK]",
                std::string pad_token = "[PAD]",
                
                std::optional<std::string> bos_token = std::nullopt,
                std::optional<std::string> eos_token = std::nullopt,

                bool lowercase = true,
                bool clean_text = true,
                
                std::string end_of_word_suffix = "</w>"
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

            const Vocab& vocab() const noexcept;
            const std::vector<Merge>& merges() const noexcept;

            BPETokenizer& fit(
                const std::vector<std::string>& texts,
                std::size_t vocab_size,
                std::size_t min_frequency
            ) override;
        
        private:
            static Vocab load_vocab(const std::filesystem::path& vocab_file);
            static std::vector<Merge> load_merges(const std::filesystem::path& merges_file);

            void ensure_special_tokens();
            void rebuild_id_to_tokens();
            void rebuild_merge_ranks();

            int32_t unk_id() const;
            std::string id_to_token_impl(int32_t id) const;

            std::vector<std::string> bpe_tokenize(const std::string& token) const;

            std::unordered_map<std::string, std::size_t> build_word_frequency(
                const std::vector<std::string>& texts
            ) const;

            static std::vector<std::string> merge_word_once(
                const std::vector<std::string>& symbols,
                const Merge& pair
            );
        
        private:
            Vocab vocab_;
            std::vector<std::string> ids_to_tokens_;

            std::vector<Merge> merges_;
            std::unordered_map<Merge, std::size_t, detail::TokenPairHash> merge_ranks_;
            
            bool lowercase_ = true;
            bool clean_text_ = true;
            std::string end_of_word_suffix_ = "</w>";
    };
}
