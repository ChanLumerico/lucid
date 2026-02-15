import unicodedata


def is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch) in {"Cc", "Cf"}


def is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def clean_text(text: str) -> str:
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in (0, 0xFFFD) or is_control(ch):
            continue
        if is_whitespace(ch):
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def split_on_punctuation(token: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for ch in token:
        if is_punctuation(ch):
            if current:
                out.append("".join(current))
                current = []
            out.append(ch)
        else:
            current.append(ch)
    if current:
        out.append("".join(current))
    return out


def basic_tokenize(text: str, lowercase: bool = True) -> list[str]:
    tokens: list[str] = []
    for token in text.strip().split():
        if lowercase:
            token = token.lower()
        tokens.extend(split_on_punctuation(token))
    return tokens
