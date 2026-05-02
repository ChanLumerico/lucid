#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"

namespace lucid::einops_detail {

struct Token;
using TokenVar = std::variant<std::string, std::vector<Token>, std::int64_t>;

struct Token {
    TokenVar v;

    bool is_name() const noexcept { return v.index() == 0; }
    bool is_group() const noexcept { return v.index() == 1; }
    bool is_literal() const noexcept { return v.index() == 2; }

    const std::string& name() const { return std::get<0>(v); }
    const std::vector<Token>& group() const { return std::get<1>(v); }
    std::int64_t literal() const { return std::get<2>(v); }
};

inline std::vector<Token> parse_side(const std::string& s) {
    std::vector<Token> out;
    std::size_t i = 0;
    auto skip_ws = [&]() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
            ++i;
    };
    while (i < s.size()) {
        skip_ws();
        if (i >= s.size())
            break;
        char c = s[i];
        if (c == '(') {
            std::size_t depth = 1, j = i + 1;
            while (j < s.size() && depth > 0) {
                if (s[j] == '(')
                    ++depth;
                else if (s[j] == ')')
                    --depth;
                if (depth > 0)
                    ++j;
            }
            if (depth != 0)
                ErrorBuilder("einops pattern").fail("unmatched '('");
            std::string inner = s.substr(i + 1, j - i - 1);
            Token tk;
            tk.v = parse_side(inner);
            out.push_back(std::move(tk));
            i = j + 1;
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            std::size_t j = i;
            while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j])))
                ++j;
            Token tk;
            tk.v = static_cast<std::int64_t>(std::stoll(s.substr(i, j - i)));
            out.push_back(std::move(tk));
            i = j;
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_'))
                ++j;
            Token tk;
            tk.v = s.substr(i, j - i);
            out.push_back(std::move(tk));
            i = j;
        } else {
            ErrorBuilder("einops pattern").fail(std::string("unexpected char '") + c + "'");
        }
    }
    return out;
}

inline std::vector<std::string> flat_axes(const std::vector<Token>& tokens) {
    std::vector<std::string> out;
    for (const auto& tk : tokens) {
        if (tk.is_name())
            out.push_back(tk.name());
        else if (tk.is_group()) {
            auto inner = flat_axes(tk.group());
            out.insert(out.end(), inner.begin(), inner.end());
        }
    }
    return out;
}

inline std::pair<std::string, std::string> split_arrow(const std::string& pat) {
    auto pos = pat.find("->");
    if (pos == std::string::npos)
        ErrorBuilder("einops pattern").fail("must contain '->'");
    auto trim = [](std::string x) {
        while (!x.empty() && (x.front() == ' ' || x.front() == '\t'))
            x.erase(x.begin());
        while (!x.empty() && (x.back() == ' ' || x.back() == '\t'))
            x.pop_back();
        return x;
    };
    return {trim(pat.substr(0, pos)), trim(pat.substr(pos + 2))};
}

}  // namespace lucid::einops_detail
