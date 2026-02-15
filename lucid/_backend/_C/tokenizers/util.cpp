#include "tokenizers.hpp"

#include <cctype>

namespace lucid::tokenizers::fast::detail {

    bool is_whitespace(char ch) {
        return std::isspace(static_cast<unsigned char>(ch)) != 0;
    }

    bool is_control(char ch) {
        if (ch == '\t' || ch == '\n' || ch == '\r') return false;
        return std::iscntrl(static_cast<unsigned char>(ch)) != 0;
    }

    bool is_punctuation(char ch) {
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

    std::string clean_text(std::string_view text) {
        std::string out;
        out.reserve(text.size());
        for (char ch : text) {
            const unsigned char uch = static_cast<unsigned char>(ch);
            if (uch == 0 || is_control(ch)) continue;
            if (is_whitespace(ch)) out.push_back(' ');
            else out.push_back(ch);
        }
        return out;
    }

    std::vector<std::string> split_on_punctuation(std::string_view token) {
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

    std::vector<std::string> basic_tokenize(
        std::string_view text,
        bool lowercase
    ) {
        std::vector<std::string> out;
        std::string current;

        auto flush_current = [&]() {
            if (current.empty()) return;
            if (lowercase) {
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
}

