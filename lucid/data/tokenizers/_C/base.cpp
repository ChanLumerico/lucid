#include "base.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace lucid::tokenizers::C {

    std::vector<std::string> TokenizerBase::build_inputs_with_special_tokens(
        const std::vector<std::string>& tokens
    ) const {
        std::vector<std::string> out;
        out.reserve(
            tokens.size() 
            + (bos_token_.has_value() ? 1 : 0) 
            + (eos_token_.has_value() ? 1 : 0)
        );
        
        if (bos_token_.has_value()) {
            out.push_back(*bos_token_);
        }
        out.insert(out.end(), tokens.begin(), tokens.end());
        if (eos_token_.has_value()) {
            out.push_back(*eos_token_);
        }
        
        return out;
    }

    std::vector<int32_t> TokenizerBase::encode(
        std::string_view text, const EncodeOptions& opts
    ) const {
        auto tokens = tokenize(text);
        if (opts.add_special_tokens) {
            tokens = build_inputs_with_special_tokens(tokens);
        }
        return convert_tokens_to_ids(tokens);
    }

    std::string TokenizerBase::decode(
        const std::vector<int32_t>& ids, bool skip_special_tokens
    ) const {
        auto tokens = convert_ids_to_tokens(ids);

        if (skip_special_tokens) {
            const auto specials = all_special_tokens();
            const std::unordered_set<std::string> special_set(
                specials.begin(), specials.end()
            );

            std::vector<std::string> filtered;
            filtered.reserve(tokens.size());
            
            for (const auto& t : tokens) {
                if (special_set.find(t) == special_set.end()) {
                    filtered.push_back(t);
                }
            }
            tokens = std::move(filtered);
        }
        return convert_tokens_to_string(tokens);
    }

    std::vector<std::vector<int32_t>> TokenizerBase::batch_encode(
        const std::vector<std::string>& texts,
        const BatchEncodeOptions& opts
    ) const {
        std::vector<std::vector<int32_t>> out;
        out.reserve(texts.size());

        EncodeOptions enc_opts{opts.add_special_tokens};
        for (const auto& text : texts) {
            out.push_back(encode(text, enc_opts));
        }

        if (opts.truncation && opts.max_length > 0) {
            for (auto& ids : out) {
                if (ids.size() > opts.max_length) ids.resize(opts.max_length);
            }
        }
        if (opts.padding) {
            std::size_t target_len = opts.max_length;
            if (target_len == 0) {
                for (const auto& ids : out) {
                    target_len = std::max(target_len, ids.size());
                }
            }
            const int32_t pad_id = convert_token_to_id(pad_token_);
            for (auto& ids : out) {
                if (ids.size() > target_len) ids.resize(target_len);
                else if (ids.size() < target_len) {
                    ids.insert(ids.end(), target_len - ids.size(), pad_id);
                }
            }
        }
        return out;
    }

    std::vector<std::string> TokenizerBase::batch_decode(
        const std::vector<std::vector<int32_t>>& batch_ids,
        bool skip_special_tokens
    ) const {
        std::vector<std::string> out;
        out.reserve(batch_ids.size());

        for (const auto& ids : batch_ids) {
            out.push_back(decode(ids, skip_special_tokens));
        }
        return out;
    }

    TokenizerBase& TokenizerBase::fit(
        const std::vector<std::string>&, std::size_t, std::size_t
    ) {
        throw std::runtime_error("This tokenizer does not support 'fit()'.");
    }

    void TokenizerBase::set_special_tokens(
        std::string unk,
        std::string pad,
        std::optional<std::string> bos,
        std::optional<std::string> eos
    ) {
        unk_token_ = std::move(unk);
        pad_token_ = std::move(pad);
        bos_token_ = std::move(bos);
        eos_token_ = std::move(eos);
    }
    
    std::vector<std::string> TokenizerBase::all_special_tokens() const {
        std::vector<std::string> out;
        out.reserve(4);

        out.push_back(unk_token_);
        out.push_back(pad_token_);

        if (bos_token_.has_value()) out.push_back(*bos_token_);
        if (eos_token_.has_value()) out.push_back(*eos_token_);

        return out;
    }
}