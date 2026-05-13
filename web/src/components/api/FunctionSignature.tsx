import Link from "next/link";
import { ExternalLink } from "lucide-react";
import { highlight } from "@/lib/shiki";
import { TypeAnnotation } from "./TypeAnnotation";
import { ParameterTable, RaisesTable } from "./ParameterTable";
import { ExampleBlock } from "./ExampleBlock";
import { AutoKindBadge } from "./ApiKindBadge";
import { getMemberNameColor } from "@/lib/api-kind-utils";
import { MathText } from "./MathText";
import type { ApiFunction, ApiMethod } from "@/lib/types";
import { cn } from "@/lib/utils";

interface FunctionSignatureProps {
  fn: ApiFunction | ApiMethod;
  headingLevel?: "h2" | "h3";
  className?: string;
  /** If true, show the full doc (params, examples). Otherwise show compact card. */
  expanded?: boolean;
}

export async function FunctionSignature({
  fn,
  headingLevel: Tag = "h3",
  className,
  expanded = true,
}: FunctionSignatureProps) {
  const sigHtml = fn.signature ? await highlight(fn.signature, "python") : null;
  const labels = fn.labels ?? [];
  const nameColor = getMemberNameColor(fn.name, labels);

  return (
    <article
      id={fn.name}
      className={cn(
        "scroll-mt-20 rounded-2xl border border-lucid-border bg-lucid-surface",
        "overflow-hidden",
        className,
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-4 border-b border-lucid-border px-5 py-4">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <AutoKindBadge labels={labels} name={fn.name} fallback="fn" />
            <Tag className={cn("font-mono text-base font-semibold", nameColor)}>
              {fn.name}
            </Tag>
            {fn.returns?.annotation && (
              <>
                <span className="text-lucid-text-disabled text-sm">→</span>
                <TypeAnnotation annotation={fn.returns.annotation} />
              </>
            )}
          </div>

          {/* Signature */}
          {sigHtml && (
            <div
              className={cn(
                "text-xs rounded-lg border border-lucid-border bg-lucid-elevated mt-2",
                "[&_pre]:px-3 [&_pre]:py-2 [&_pre]:overflow-x-auto [&_pre]:!bg-transparent",
                "[&_code]:font-mono [&_code]:text-xs [&_code]:leading-relaxed",
              )}
              dangerouslySetInnerHTML={{ __html: sigHtml }}
            />
          )}
        </div>

        {fn.source && (
          <a
            href={fn.source}
            target="_blank"
            rel="noopener noreferrer"
            className="shrink-0 flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-lucid-text-disabled hover:text-lucid-text-mid transition-colors border border-lucid-border hover:border-lucid-primary/40"
            aria-label="View source"
          >
            <ExternalLink className="h-3 w-3" />
            <span>source</span>
          </a>
        )}
      </div>

      {/* Body */}
      {expanded && (
        <div className="px-5 py-5 space-y-6">
          {fn.summary && (
            <MathText text={fn.summary} block className="text-sm text-lucid-text-mid leading-relaxed" />
          )}
          {fn.extended && (
            <MathText text={fn.extended} block className="text-sm text-lucid-text-low leading-relaxed" />
          )}

          {fn.parameters.length > 0 && <ParameterTable parameters={fn.parameters} />}

          {fn.returns && fn.returns.description && (
            <section className="space-y-2">
              <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
                Returns
              </h4>
              <div className="rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3">
                <TypeAnnotation annotation={fn.returns.annotation} className="block mb-1" />
                <MathText text={fn.returns.description} block className="text-sm text-lucid-text-low" />
              </div>
            </section>
          )}

          {fn.raises.length > 0 && <RaisesTable raises={fn.raises} />}

          {fn.notes.length > 0 && (
            <section className="space-y-2">
              <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
                Notes
              </h4>
              <div className="rounded-xl border-l-2 border-lucid-blue bg-lucid-blue/5 px-4 py-3">
                {fn.notes.map((note, i) => (
                  <MathText key={i} text={note} block className="text-sm text-lucid-text-low leading-relaxed" />
                ))}
              </div>
            </section>
          )}

          {fn.examples.length > 0 && <ExampleBlock examples={fn.examples} />}
        </div>
      )}
    </article>
  );
}

/** Compact one-line card for module overview lists */
interface FunctionCardProps {
  fn: ApiFunction;
  moduleSlug: string;
}

export function FunctionCard({ fn, moduleSlug }: FunctionCardProps) {
  const labels = fn.labels ?? [];
  const nameColor = getMemberNameColor(fn.name, labels);

  return (
    <Link
      href={`/api/${moduleSlug}/${fn.name}`}
      className={cn(
        "group flex items-start justify-between gap-4",
        "rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3.5",
        "transition-all hover:border-api-fn/40 hover:bg-lucid-elevated",
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <AutoKindBadge labels={labels} name={fn.name} fallback="fn" />
          <code className={cn("text-sm font-mono font-medium", nameColor)}>
            {fn.name}
          </code>
          {fn.returns?.annotation && (
            <span className="text-[11px] font-mono text-lucid-text-disabled">
              → {fn.returns.annotation}
            </span>
          )}
        </div>
        {fn.summary && (
          <p className="text-xs text-lucid-text-low leading-relaxed line-clamp-2">
            {fn.summary}
          </p>
        )}
      </div>
    </Link>
  );
}
