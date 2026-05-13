import katex from "katex";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Segment types
// ---------------------------------------------------------------------------

type Segment =
  | { type: "text";   content: string }
  | { type: "inline"; latex: string }
  | { type: "block";  latex: string }
  | { type: "code";   content: string }
  | { type: "bold";   content: string }
  | { type: "italic"; content: string };

// ---------------------------------------------------------------------------
// Segment parser
// Priority: $$block$$ → **bold** → *italic* → $inline$ → `code`
// ---------------------------------------------------------------------------

const SEG_RE =
  /(\$\$[\s\S]+?\$\$|\*\*[^*\n]+?\*\*|\*[^*\n]+?\*|\$(?!\$)[^$\n]+?\$|`[^`]+`)/g;

function parseSegments(text: string): Segment[] {
  const segs: Segment[] = [];
  let last = 0;

  for (const match of text.matchAll(SEG_RE)) {
    const idx = match.index ?? 0;
    if (idx > last) segs.push({ type: "text", content: text.slice(last, idx) });
    const m = match[0];
    if (m.startsWith("$$")) {
      segs.push({ type: "block",  latex: m.slice(2, -2).trim() });
    } else if (m.startsWith("**")) {
      segs.push({ type: "bold",   content: m.slice(2, -2) });
    } else if (m.startsWith("*")) {
      segs.push({ type: "italic", content: m.slice(1, -1) });
    } else if (m.startsWith("$")) {
      segs.push({ type: "inline", latex: m.slice(1, -1) });
    } else {
      segs.push({ type: "code",   content: m.slice(1, -1) });
    }
    last = idx + m.length;
  }
  if (last < text.length) segs.push({ type: "text", content: text.slice(last) });
  return segs;
}

// ---------------------------------------------------------------------------
// Render a segment list → React nodes
// ---------------------------------------------------------------------------

function renderSegs(segs: Segment[]): React.ReactNode[] {
  return segs.map((seg, i) => {
    switch (seg.type) {
      case "text": {
        const lines = seg.content.split("\n");
        return (
          <span key={i}>
            {lines.map((l, j) => (
              <span key={j}>{j > 0 && <br />}{l}</span>
            ))}
          </span>
        );
      }
      case "bold":
        return (
          <strong key={i} className="font-semibold text-lucid-text-high">
            {seg.content}
          </strong>
        );
      case "italic":
        return <em key={i} className="italic">{seg.content}</em>;
      case "code":
        return (
          <code
            key={i}
            className="font-mono text-[0.875em] bg-lucid-elevated text-lucid-primary-light px-1 py-px rounded"
          >
            {seg.content}
          </code>
        );
      case "block": {
        const html = katex.renderToString(seg.latex, {
          throwOnError: false, displayMode: true, output: "html",
        });
        return (
          <span
            key={i}
            className="my-3 block overflow-x-auto text-center"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }
      case "inline": {
        const html = katex.renderToString(seg.latex, {
          throwOnError: false, displayMode: false, output: "html",
        });
        return (
          <span
            key={i}
            className="mx-0.5"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        );
      }
    }
  });
}

// ---------------------------------------------------------------------------
// Paragraph renderer — handles bullet lists
// ---------------------------------------------------------------------------

function renderParagraph(para: string, key: number, pClass?: string) {
  const lines = para.split("\n");
  const isBullet = (l: string) => /^\s*[-*]\s/.test(l);

  if (lines.some(isBullet)) {
    return (
      <ul key={key} className={cn("space-y-1 pl-1", pClass)}>
        {lines.map((line, j) => {
          if (!line.trim()) return null;
          const bullet = isBullet(line);
          const content = bullet ? line.replace(/^\s*[-*]\s/, "") : line.trim();
          return (
            <li
              key={j}
              className={cn(
                "flex items-start gap-2 text-sm",
                !bullet && "list-none",
              )}
            >
              {bullet && (
                <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-lucid-text-disabled" />
              )}
              <span>{renderSegs(parseSegments(content))}</span>
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <p key={key} className={pClass}>
      {renderSegs(parseSegments(para))}
    </p>
  );
}

// ---------------------------------------------------------------------------
// MathText — public component
// ---------------------------------------------------------------------------

interface MathTextProps {
  text: string;
  className?: string;
  block?: boolean;
}

export function MathText({ text, className, block = false }: MathTextProps) {
  if (!text) return null;

  const paragraphs = text.split(/\n{2,}/).filter((p) => p.trim());

  if (!block || paragraphs.length <= 1) {
    const Tag = block ? "p" : "span";
    return (
      <Tag className={className}>
        {renderSegs(parseSegments(text))}
      </Tag>
    );
  }

  return (
    <div className={className}>
      {paragraphs.map((para, i) =>
        renderParagraph(para, i, i > 0 ? "mt-3" : undefined),
      )}
    </div>
  );
}
