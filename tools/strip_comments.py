#!/usr/bin/env python3
"""Strip all C++ // and /* */ comments from source files.
Preserves string literals, char literals, and raw string literals.
Usage: python3 tools/strip_comments.py lucid/_C
"""
import sys
import os
import re


def strip_cpp_comments(text: str) -> str:
    result = []
    i = 0
    n = len(text)

    while i < n:
        # Raw string literal: R"delim(...)delim"
        if (
            text[i] == "R"
            and i + 1 < n
            and text[i + 1] == '"'
            and i + 2 < n
            and text[i + 2] != "("
        ):
            j = i + 2
            while j < n and text[j] != "(":
                j += 1
            delimiter = text[i + 2 : j]
            end_marker = ")" + delimiter + '"'
            end_pos = text.find(end_marker, j)
            if end_pos == -1:
                result.append(text[i:])
                break
            end_pos += len(end_marker)
            result.append(text[i:end_pos])
            i = end_pos

        # String literal
        elif text[i] == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            result.append(text[i:j])
            i = j

        # Char literal
        elif text[i] == "'":
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                elif text[j] == "'":
                    j += 1
                    break
                else:
                    j += 1
            result.append(text[i:j])
            i = j

        # Line comment
        elif text[i : i + 2] == "//":
            j = i + 2
            while j < n and text[j] != "\n":
                j += 1
            i = j  # skip comment; newline (if any) appended on next iter

        # Block comment
        elif text[i : i + 2] == "/*":
            j = i + 2
            while j < n - 1:
                if text[j : j + 2] == "*/":
                    j += 2
                    break
                j += 1
            # Preserve newlines inside block comment so line numbers stay sane
            newlines = text[i:j].count("\n")
            result.append("\n" * newlines)
            i = j

        else:
            result.append(text[i])
            i += 1

    return "".join(result)


def collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines down to 1."""
    return re.sub(r"\n{3,}", "\n\n", text)


def strip_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.split("\n"))


def process_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        original = f.read()
    stripped = strip_cpp_comments(original)
    stripped = strip_trailing_whitespace(stripped)
    stripped = collapse_blank_lines(stripped)
    if stripped == original:
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(stripped)
    return True


EXTENSIONS = {".cpp", ".h", ".mm"}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    changed = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1] in EXTENSIONS:
                path = os.path.join(dirpath, fname)
                if process_file(path):
                    print(f"  stripped: {path}")
                    changed += 1
    print(f"\n{changed} files modified.")


if __name__ == "__main__":
    main()
