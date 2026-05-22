import Link from "next/link";
import { cn } from "@/lib/utils";
import { getTypeLinkMap } from "@/lib/type-links";

interface TypeAnnotationProps {
  annotation: string | null;
  className?: string;
}

// Primitive types are *not* linked — they're language built-ins and
// usually not in the Lucid docs surface.  Coloured only.
const _PRIMITIVE_BLUE = new Set([
  "int", "float", "bool", "str", "bytes", "None", "Any",
]);
const _CONTAINER_GREY = new Set([
  "list", "dict", "tuple", "set", "frozenset", "Sequence", "Iterable",
  "Callable", "Optional", "Union",
]);

/** Tokenise a type annotation into renderable chunks.
 *
 *  The tokenizer is intentionally simple — Python type annotations have
 *  enough syntactic variety (``Tensor | None``, ``Optional[Tensor]``,
 *  ``tuple[int, ...]``, ``dict[str, Tensor]``) that a full parser
 *  would be overkill.  We split on the punctuation that separates
 *  identifiers (``[]`` / ``,`` / ``|`` / whitespace) and emit each
 *  identifier as either a link, a primitive, a container, or plain text.
 */
type Token =
  | { kind: "punct"; text: string }
  | { kind: "primitive"; text: string }
  | { kind: "container"; text: string }
  | { kind: "linked"; text: string; href: string }
  | { kind: "plain"; text: string };

function _tokenize(
  annotation: string,
  linkMap: ReadonlyMap<string, string>,
): Token[] {
  const tokens: Token[] = [];
  // Identifier vs everything-else: identifiers are
  // ``[A-Za-z_][A-Za-z0-9_.]*`` (dotted forms like ``lucid.dtype``
  // count as one), and everything else is treated as punctuation.
  const re = /([A-Za-z_][A-Za-z0-9_.]*)|([^A-Za-z_]+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(annotation)) !== null) {
    if (m[1] !== undefined) {
      const ident = m[1];
      if (_PRIMITIVE_BLUE.has(ident)) {
        tokens.push({ kind: "primitive", text: ident });
        continue;
      }
      if (_CONTAINER_GREY.has(ident)) {
        tokens.push({ kind: "container", text: ident });
        continue;
      }
      // Try the bare name first (``Tensor``), then strip dotted prefix
      // (``lucid.Tensor`` → ``Tensor``) to catch fully-qualified refs.
      const href =
        linkMap.get(ident) ??
        linkMap.get(ident.split(".").pop() ?? ident) ??
        null;
      if (href) {
        tokens.push({ kind: "linked", text: ident, href });
      } else {
        tokens.push({ kind: "plain", text: ident });
      }
    } else {
      tokens.push({ kind: "punct", text: m[2] });
    }
  }
  return tokens;
}

export function TypeAnnotation({ annotation, className }: TypeAnnotationProps) {
  if (!annotation) return null;
  const linkMap = getTypeLinkMap();
  const tokens = _tokenize(annotation, linkMap);

  return (
    <code className={cn("font-mono text-xs text-lucid-blue-light", className)}>
      {tokens.map((tok, i) => {
        switch (tok.kind) {
          case "primitive":
            return (
              <span key={i} className="text-lucid-blue-light">
                {tok.text}
              </span>
            );
          case "container":
            return (
              <span key={i} className="text-lucid-text-mid">
                {tok.text}
              </span>
            );
          case "linked":
            // ``hover:underline`` is what makes the link affordance
            // visible — without it, the token looks like every other
            // identifier and users miss that it's clickable.
            return (
              <Link
                key={i}
                href={tok.href}
                className="text-lucid-primary-light hover:text-lucid-primary hover:underline decoration-dotted underline-offset-2"
              >
                {tok.text}
              </Link>
            );
          case "plain":
            return (
              <span key={i} className="text-lucid-primary-light">
                {tok.text}
              </span>
            );
          case "punct":
            return (
              <span key={i} className="text-lucid-text-low">
                {tok.text}
              </span>
            );
        }
      })}
    </code>
  );
}
