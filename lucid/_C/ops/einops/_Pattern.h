// lucid/_C/ops/einops/_Pattern.h
//
// Parser and intermediate representation for einops pattern strings of the
// form "lhs -> rhs", e.g. "b c h w -> b (c h w)".
//
// The grammar supported here is:
//   side     ::= (token SP)*
//   token    ::= name | '(' inner ')' | literal
//   inner    ::= (name SP)*
//   name     ::= [A-Za-z_][A-Za-z0-9_]*
//   literal  ::= [0-9]+
//
// Parentheses create a "group" token whose children are the inner names.
// A group on the left-hand side represents axis decomposition (a merged
// dimension is split into its constituent sizes); a group on the right-hand
// side represents axis merging (multiple dimensions are collapsed into one).
//
// All functions are inline; this header is included directly by the .cpp
// files for Rearrange, Reduce, Repeat, and Einsum.

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

// A token in an einops pattern can be one of three things:
//   std::string           — a named axis (e.g. "b", "c", "height")
//   std::vector<Token>    — a group of axes inside parentheses
//   std::int64_t          — a literal integer size (e.g. "1", "3")
//
// The ordering of alternatives in the variant encodes the token kind index
// used by Token::is_name(), is_group(), and is_literal().  Do not reorder.
using TokenVar = std::variant<std::string, std::vector<Token>, std::int64_t>;

// Single token in a parsed einops pattern side.
//
// A Token carries the result of parsing one whitespace-delimited unit from
// one side of an einops pattern:
//   - is_name(): a free axis label — appears in both lhs and rhs and has a
//     size determined either from the input shape or from axes_lengths.
//   - is_group(): a parenthesised axis list — on the lhs it means the input
//     dimension is the product of the listed axis sizes (decomposition); on
//     the rhs it means those axes will be merged into a single dimension.
//   - is_literal(): a fixed integer size constraint — the corresponding input
//     dimension must match exactly.  Literal tokens do not introduce free axes
//     and are not included in flat_axes().
//
// The three predicates provide a readable API over the raw variant index.
struct Token {
    TokenVar v;

    // True if this token is a named axis label.
    bool is_name() const noexcept { return v.index() == 0; }
    // True if this token is a parenthesised group of axis names.
    bool is_group() const noexcept { return v.index() == 1; }
    // True if this token is an integer literal size constraint.
    bool is_literal() const noexcept { return v.index() == 2; }

    // Access the name (precondition: is_name()).
    const std::string& name() const { return std::get<0>(v); }
    // Access the group children (precondition: is_group()).
    const std::vector<Token>& group() const { return std::get<1>(v); }
    // Access the literal size (precondition: is_literal()).
    std::int64_t literal() const { return std::get<2>(v); }
};

// Parse one side of an einops pattern string (before or after "->") into a
// sequence of Tokens.
//
// Whitespace is consumed between tokens.  Parenthesised groups are parsed
// recursively so nested parentheses are handled correctly (though the einops
// grammar does not require nesting; this is just a natural consequence of
// the depth-tracking loop).  Raises an error on unmatched '(' or any
// character that is not alphanumeric, underscore, parenthesis, or whitespace.
//
// The three scanning branches correspond to the three token kinds:
//   '('        — start of a group; depth-tracked scan to the matching ')'.
//   digit      — a literal integer; digits consumed until the first non-digit.
//   alpha/'_'  — an axis name; alphanumerics and underscores consumed.
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
            // Walk forward until the matching ')' tracking nesting depth.
            std::size_t depth = 1, j = i + 1;
            while (j < s.size() && depth > 0) {
                if (s[j] == '(')
                    ++depth;
                else if (s[j] == ')')
                    --depth;
                // Advance j only while still inside the group.
                if (depth > 0)
                    ++j;
            }
            if (depth != 0)
                ErrorBuilder("einops pattern").fail("unmatched '('");
            // Recursively parse the interior of the group, excluding the parens.
            std::string inner = s.substr(i + 1, j - i - 1);
            Token tk;
            tk.v = parse_side(inner);
            out.push_back(std::move(tk));
            i = j + 1;  // step past the closing ')'
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            // Consume all consecutive digit characters.
            std::size_t j = i;
            while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j])))
                ++j;
            Token tk;
            tk.v = static_cast<std::int64_t>(std::stoll(s.substr(i, j - i)));
            out.push_back(std::move(tk));
            i = j;
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            // Consume an axis name: starts with alpha or '_', continues with
            // alphanumerics or '_'.
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

// Flatten a token sequence to the ordered list of named axes it represents,
// recursively expanding group tokens.  Literal tokens are skipped because they
// do not correspond to free axis names that need tracking in the size map.
//
// The returned vector is used to:
//   - Determine the number of free axes on each side of the pattern.
//   - Build permutation vectors by comparing the flat lhs vs flat rhs order.
//   - Construct intermediate reshape targets.
inline std::vector<std::string> flat_axes(const std::vector<Token>& tokens) {
    std::vector<std::string> out;
    for (const auto& tk : tokens) {
        if (tk.is_name())
            out.push_back(tk.name());
        else if (tk.is_group()) {
            // Recursively expand nested group children.
            auto inner = flat_axes(tk.group());
            out.insert(out.end(), inner.begin(), inner.end());
        }
        // Literal tokens (integers) do not contribute free axis names.
    }
    return out;
}

// Split a full pattern string at the "->" arrow and return {lhs, rhs} with
// leading and trailing whitespace trimmed from each half.
//
// The two-character arrow "->" is consumed; neither half retains it.
// Only the outermost whitespace is trimmed; interior spaces within a side
// serve as token delimiters and are retained for parse_side.
// Raises an error if the arrow is absent, since all supported einops ops
// require explicit input and output side specifications.
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
    // pos+2 skips the two characters '-' and '>'.
    return {trim(pat.substr(0, pos)), trim(pat.substr(pos + 2))};
}

}  // namespace lucid::einops_detail
