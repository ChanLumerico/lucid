#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <optional>

namespace lucid::tokenizers::C {

    struct EncodeOptions {
        bool add_special_tokens = true;
    };

    struct BatchEncodeOptions {
        bool add_special_tokens = true;
        bool padding = false;
        bool truncation = false;

        std::size_t max_length = 0;
    };

    class TokenizerBase {
        public:
            virtual ~TokenizerBase() = default;

            virtual std::size_t vocab_size() const = 0;
            virtual std::vector<std::string> tokenize(std::string_view text) const = 0;

            virtual int32_t convert_token_to_id(std::string_view token) const = 0;
            virtual std::vector<int32_t> convert_tokens_to_ids(
                const std::vector<std::string>& tokens
            ) const = 0;

            virtual std::string convert_id_to_token(int32_t id) const = 0;
            virtual std::vector<std::string> convert_ids_to_tokens(
                const std::vector<std::int32_t>& ids
            ) const = 0;

            virtual std::string convert_tokens_to_string(
                const std::vector<std::string>& tokens
            ) const = 0;

            virtual std::vector<std::string> build_inputs_with_special_tokens(
                const std::vector<std::string>& tokens
            ) const;

            virtual std::vector<int32_t> encode(
                std::string_view text, const EncodeOptions& opts = {}
            ) const;

            virtual std::string decode(
                const std::vector<int32_t>& ids, bool skip_special_tokens = true
            ) const;

            virtual std::vector<std::vector<int32_t>> batch_encode(
                const std::vector<std::string>& texts,
                const BatchEncodeOptions& opts = {}
            ) const;

            virtual std::vector<std::string> batch_decode(
                const std::vector<std::vector<int32_t>>& batch_ids,
                bool skip_special_tokens = true
            ) const;

            virtual TokenizerBase& fit(
                const std::vector<std::string>& texts,
                std::size_t vocab_size,
                std::size_t min_frequency = 2
            );

            const std::string& unk_token() const { return unk_token_; }
            const std::string& pad_token() const { return pad_token_; }

            const std::optional<std::string>& bos_token() const { return bos_token_; }
            const std::optional<std::string>& eos_token() const { return eos_token_; }

            void set_special_tokens(
                std::string unk, 
                std::string pad, 
                
                std::optional<std::string> bos = std::nullopt, 
                std::optional<std::string> eos = std::nullopt
            );

            std::vector<std::string> all_special_tokens() const;
        
        protected:
            std::string unk_token_ = "[UNK]";
            std::string pad_token_ = "[PAD]";

            std::optional<std::string> bos_token_;
            std::optional<std::string> eos_token_;
    };
}
