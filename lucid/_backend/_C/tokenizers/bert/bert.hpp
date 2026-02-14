#pragma once

#include "../base.hpp"
#include "../tokenizers.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lucid::tokenizers::fast {

    struct BERTEncodePairResult {
        std::vector<int32_t> input_ids;
        std::vector<int32_t> token_type_ids;
        std::vector<int32_t> attention_mask;
    };

    class BERTTokenizer : public WordPieceTokenizer {
        public: 
            BERTTokenizer(
                std::optional<Vocab> vocab = std::nullopt,
                std::optional<std::filesystem::path> vocab_file = std::nullopt,

                std::string unk_token = "[UNK]",
                std::string pad_token = "[PAD]",
                std::string cls_token = "[CLS]",
                std::string mask_token = "[MASK]",

                std::string wordpieces_prefix = "##",
                std::size_t max_input_chars_per_word = 100,

                bool lowercase = true,
                bool clean_text = true
            );

            std::vector<std::string> build_inputs_with_special_tokens(
                const std::vector<std::string>& tokens
            ) const override;

            std::vector<std::string> build_inputs_with_special_tokens_pair(
                const std::vector<std::string>& tokens_a,
                const std::vector<std::string>& tokens_b
            ) const;

            std::vector<int32_t> create_token_type_ids_from_sequences(
                const std::vector<std::string>& tokens_a,
                const std::optional<std::vector<std::string>>& tokens_b = std::nullopt
            ) const;

            BERTEncodePairResult encode_plus(
                std::string_view text_a,
                std::optional<std::string_view> text_b = std::nullopt
            ) const;

            const std::string& cls_token() const { return cls_token_; }
            const std::string& sep_token() const { return sep_token_; };
            const std::string& mask_token() const { return mask_token_; };
        
        private:
            std::string cls_token_ = "[CLS]";
            std::string sep_token_ = "[SEP]";
            std::string mask_token_ = "[MASK]";
    };
}
