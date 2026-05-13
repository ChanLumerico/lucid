import Link from "next/link";
import { ExternalLink } from "lucide-react";
import { highlight } from "@/lib/shiki";
import { TypeAnnotation } from "./TypeAnnotation";
import { ParameterTable, AttributeTable, RaisesTable } from "./ParameterTable";
import { ExampleBlock } from "./ExampleBlock";
import { FunctionSignature } from "./FunctionSignature";
import { ClassBadge, AutoKindBadge } from "./ApiKindBadge";
import { getClassNameColor, getClassHoverBorder } from "@/lib/api-kind-utils";
import { MathText } from "./MathText";
import type { ApiClass, ApiMethod, ApiClassKind } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ClassDocProps {
  cls: ApiClass;
  /** If provided, only render this method. Otherwise render the full class. */
  methodName?: string;
}

export async function ClassDoc({ cls, methodName }: ClassDocProps) {
  if (methodName) {
    const method = cls.methods.find((m) => m.name === methodName);
    if (!method) return (
      <p className="text-lucid-text-low">Method <code>{methodName}</code> not found.</p>
    );
    return <FunctionSignature fn={method} headingLevel="h2" />;
  }

  const sigHtml = cls.signature ? await highlight(cls.signature, "python") : null;
  const clsKind: ApiClassKind = cls.class_kind ?? "regular";
  const nameColor = getClassNameColor(clsKind);

  // Group methods: init/call first, then public, then dunder
  const FIRST   = ["__init__", "__call__"];
  const DUNDERS = cls.methods.filter((m) => m.name.startsWith("__") && !FIRST.includes(m.name));
  const PUBLIC  = cls.methods.filter((m) => !m.name.startsWith("_"));
  const ordered: ApiMethod[] = [
    ...FIRST.map((n) => cls.methods.find((m) => m.name === n)).filter(Boolean) as ApiMethod[],
    ...PUBLIC,
    ...DUNDERS,
  ];

  return (
    <article>
      {/* Class header */}
      <div className="mb-8 rounded-2xl border border-lucid-border bg-lucid-surface overflow-hidden">
        <div className="flex items-start justify-between gap-4 border-b border-lucid-border px-5 py-5">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <ClassBadge kind={clsKind} />
              <h1 className={cn("font-mono text-2xl font-bold", nameColor)}>
                {cls.name}
              </h1>
            </div>
            {/* Base classes */}
            {cls.bases && cls.bases.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 mb-2">
                <span className="text-[11px] text-lucid-text-disabled tracking-wide">
                  extends
                </span>
                {cls.bases.map((base) => (
                  <code
                    key={base}
                    className="text-[11px] font-mono text-lucid-text-low bg-lucid-elevated px-1.5 py-0.5 rounded border border-lucid-border"
                  >
                    {base}
                  </code>
                ))}
              </div>
            )}
            {sigHtml && (
              <div
                className={cn(
                  "text-xs rounded-lg border border-lucid-border bg-lucid-elevated",
                  "[&_pre]:px-3 [&_pre]:py-2.5 [&_pre]:overflow-x-auto [&_pre]:!bg-transparent",
                  "[&_code]:font-mono [&_code]:text-xs [&_code]:leading-relaxed",
                )}
                dangerouslySetInnerHTML={{ __html: sigHtml }}
              />
            )}
          </div>
          {cls.source && (
            <a
              href={cls.source}
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

        <div className="px-5 py-5 space-y-6">
          {cls.summary && (
            <MathText text={cls.summary} block className="text-sm text-lucid-text-mid leading-relaxed" />
          )}
          {cls.extended && (
            <MathText text={cls.extended} block className="text-sm text-lucid-text-low leading-relaxed" />
          )}
          {cls.parameters.length > 0 && <ParameterTable parameters={cls.parameters} />}
          {cls.attributes.length > 0 && <AttributeTable attributes={cls.attributes} />}
          {cls.raises.length > 0 && <RaisesTable raises={cls.raises} />}
          {cls.notes.length > 0 && (
            <section className="space-y-2">
              <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
                Notes
              </h4>
              <div className="rounded-xl border-l-2 border-lucid-blue bg-lucid-blue/5 px-4 py-3">
                {cls.notes.map((note, i) => (
                  <MathText key={i} text={note} block className="text-sm text-lucid-text-low leading-relaxed" />
                ))}
              </div>
            </section>
          )}
          {cls.examples.length > 0 && <ExampleBlock examples={cls.examples} />}
        </div>
      </div>

      {/* Methods — blue accent (functions within a class) */}
      {ordered.length > 0 && (
        <section>
          <h2 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase mb-4">
            Methods ({ordered.length})
          </h2>
          <div className="space-y-4">
            {ordered.map((method) => (
              <FunctionSignature
                key={method.name}
                fn={method}
                headingLevel="h3"
              />
            ))}
          </div>
        </section>
      )}
    </article>
  );
}

/** Compact card for module overview */
interface ClassCardProps {
  cls: ApiClass;
  moduleSlug: string;
}

export function ClassCard({ cls, moduleSlug }: ClassCardProps) {
  const clsKind: ApiClassKind = cls.class_kind ?? "regular";
  const nameColor = getClassNameColor(clsKind);
  const hoverBorder = getClassHoverBorder(clsKind);

  return (
    <Link
      href={`/api/${moduleSlug}/${cls.name}`}
      className={cn(
        "group flex items-start justify-between gap-4",
        "rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3.5",
        "transition-all hover:bg-lucid-elevated",
        hoverBorder,
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <ClassBadge kind={clsKind} />
          <code className={cn("text-sm font-mono font-medium", nameColor)}>
            {cls.name}
          </code>
          {cls.methods.length > 0 && (
            <span className="text-[10px] text-lucid-text-disabled">
              {cls.methods.length} methods
            </span>
          )}
        </div>
        {cls.summary && (
          <p className="text-xs text-lucid-text-low leading-relaxed line-clamp-2">
            {cls.summary}
          </p>
        )}
      </div>
    </Link>
  );
}
