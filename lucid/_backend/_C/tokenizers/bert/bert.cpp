#include "bert.hpp"

#include <utility>

namespace lucid::tokenizers::fast {

    BERTTokenizer::BERTTokenizer(
        std::optional<Vocab> vocab,
        std::optional<std::filesystem::path> vocab_file,

        std::string unk_token,
        std::string pad_token,
        std::string cls_token,
        std::string mask_token,

        std::string wordpieces_prefix,
        std::size_t max_input_chars_per_word,

        bool lowercase,
        bool clean_text
    ) 
        : WordPieceTokenizer(
            std::move(vocab),
            std::move(vocab_file),

            std::move(unk_token),
            std::move(pad_token),
            cls_token,
            std::string("[SEP]"),

            std::move(wordpieces_prefix),
            max_input_chars_per_word,
            lowercase,
            clean_text
        ),
        cls_token_(std::move(cls_token)),
        sep_token_("[SEP]"),
        mask_token_(std::move(mask_token)) {}
    
    std::vector<std::string> BERTTokenizer::build_inputs_with_special_tokens(
        const std::vector<std::string>& tokens
    ) const {
        std::vector<std::string> out;
        out.reserve(tokens.size() + 2);

        out.push_back(cls_token_);
        out.insert(out.end(), tokens.begin(), tokens.end());
        out.push_back(sep_token_);

        return out;
    }

    std::vector<std::string> BERTTokenizer::build_inputs_with_special_tokens_pair(
        const std::vector<std::string>& tokens_a,
        const std::vector<std::string>& tokens_b
    ) const {
        std::vector<std::string> out;
        out.reserve(tokens_a.size() + tokens_b.size() + 3);

        out.push_back(cls_token_);
        out.insert(out.end(), tokens_a.begin(), tokens_a.end());
        out.push_back(sep_token_);
        out.insert(out.end(), tokens_b.begin(), tokens_b.end());
        out.push_back(sep_token_);

        return out;
    }

    std::vector<int32_t> BERTTokenizer::create_token_type_ids_from_sequences(
        const std::vector<std::string>& tokens_a,
        const std::optional<std::vector<std::string>>& token_b
    ) const {
        std::vector<int32_t> out(1 + tokens_a.size() + 1, 0);
        if (token_b.has_value()) {
            out.insert(out.end(), token_b->size() + 1, 1);
        }
        return out;
    }

    BERTEncodePairResult BERTTokenizer::encode_plus(
        std::string_view text_a,
        std::optional<std::string_view> text_b
    ) const {
        const auto tokens_a = tokenize(text_a);

        std::vector<std::string> input_tokens;
        std::vector<int32_t> token_type_ids;

        if (text_b.has_value()) {
            const auto tokens_b = tokenize(*text_b);
            input_tokens = build_inputs_with_special_tokens_pair(tokens_a, tokens_b);
            token_type_ids = create_token_type_ids_from_sequences(tokens_a, tokens_b);
        } else {
            input_tokens = build_inputs_with_special_tokens(tokens_a);
            token_type_ids = create_token_type_ids_from_sequences(tokens_a, std::nullopt);
        }

        auto input_ids = convert_tokens_to_ids(input_tokens);
        std::vector<int32_t> attention_mask(input_ids.size(), 1);

        return BERTEncodePairResult{
            std::move(input_ids),
            std::move(token_type_ids),
            std::move(attention_mask),
        };
    }
}