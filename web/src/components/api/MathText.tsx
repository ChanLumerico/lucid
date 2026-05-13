import katex from "katex";

// ---------------------------------------------------------------------------
// Segment parser — splits text into plain text / inline math / block math / code
// ---------------------------------------------------------------------------

type Segment =
  | { type: "text"; content: string }
  | { type: "inline"; latex: string }
  | { type: "block"; latex: string }
  | { type: "code"; content: string };

function parseSegments(text: string): Segment[] {
  const segments: Segment[] = [];
  const RE = /(\$\$[\s\S]+?\$\$|\$(?!\$)[^$\n]+?\$|`[^`]+`)/g;
  let last = 0;

  for (const match of text.matchAll(RE)) {
    const idx = match.index ?? 0;
    if (idx > last) {
      segments.push({ type: "text", content: text.slice(last, idx) });
    }
    const m = match[0];
    if (m.startsWith("$$")) {
      segments.push({ type: "block", latex: m.slice(2, -2).trim() });
    } else if (m.startsWith("$")) {
      segments.push({ type: "inline", latex: m.slice(1, -1) });
    } else {
      segments.push({ type: "code", content: m.slice(1, -1) });
    }
    last = idx + m.length;
  }

  if (last < text.length) {
    segments.push({ type: "text", content: text.slice(last) });
  }

  return segments;
}

function renderSegments(segments: Segment[]) {
  return segments.map((seg, i) => {
    if (seg.type === "text") {
      // Preserve line breaks within a paragraph
      const lines = seg.content.split("\n");
      return (
        <span key={i}>
          {lines.map((line, j) => (
            <span key={j}>
              {j > 0 && <br />}
              {line}
            </span>
          ))}
        </span>
      );
    }

    if (seg.type === "code") {
      return (
        <code
          key={i}
          className="font-mono text-[0.875em] bg-lucid-elevated text-lucid-text-high px-1 py-px rounded"
        >
          {seg.content}
        </code>
      );
    }

    if (seg.type === "block") {
      const html = katex.renderToString(seg.latex, {
        throwOnError: false,
        displayMode: true,
        output: "html",
      });
      return (
        <span
          key={i}
          className="my-3 block overflow-x-auto text-center"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    }

    // inline math
    const html = katex.renderToString(seg.latex, {
      throwOnError: false,
      displayMode: false,
      output: "html",
    });
    return (
      <span
        key={i}
        className="mx-0.5"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  });
}

// ---------------------------------------------------------------------------
// MathText — renders a docstring text field with math and inline-code support
// ---------------------------------------------------------------------------

interface MathTextProps {
  text: string;
  className?: string;
  /** Render as block (p) rather than inline (span) */
  block?: boolean;
}

export function MathText({ text, className, block = false }: MathTextProps) {
  if (!text) return null;

  // Multi-paragraph: split on double newlines and render each separately
  const paragraphs = text.split(/\n{2,}/).filter((p) => p.trim());

  if (!block || paragraphs.length <= 1) {
    const Tag = block ? "p" : "span";
    return (
      <Tag className={className}>
        {renderSegments(parseSegments(text))}
      </Tag>
    );
  }

  return (
    <div className={className}>
      {paragraphs.map((para, i) => (
        <p key={i} className={i > 0 ? "mt-3" : ""}>
          {renderSegments(parseSegments(para))}
        </p>
      ))}
    </div>
  );
}
