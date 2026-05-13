import type { DocstringParameter, DocstringAttribute, DocstringRaise } from "@/lib/types";
import { TypeAnnotation } from "./TypeAnnotation";
import { MathText } from "./MathText";
import { cn } from "@/lib/utils";

interface ParameterTableProps {
  parameters: DocstringParameter[];
  className?: string;
}

export function ParameterTable({ parameters, className }: ParameterTableProps) {
  if (!parameters.length) return null;

  return (
    <section className={cn("space-y-2", className)}>
      <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
        Parameters
      </h4>
      <div className="divide-y divide-lucid-border rounded-xl border border-lucid-border overflow-hidden">
        {parameters.map((p) => (
          <div key={p.name} className="flex flex-col sm:flex-row gap-1 sm:gap-4 px-4 py-3 bg-lucid-surface hover:bg-lucid-elevated transition-colors">
            <div className="shrink-0 flex flex-col gap-0.5 sm:w-48">
              <code className="text-xs font-mono font-semibold text-lucid-text-high">
                {p.name}
              </code>
              {p.annotation && (
                <TypeAnnotation annotation={p.annotation} />
              )}
              {p.default !== null && (
                <code className="text-[10px] font-mono text-lucid-text-disabled">
                  = {p.default}
                </code>
              )}
            </div>
            <div className="text-sm text-lucid-text-mid leading-relaxed flex-1">
              {p.description
                ? <MathText text={p.description} />
                : <span className="italic opacity-50">No description.</span>}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

interface AttributeTableProps {
  attributes: DocstringAttribute[];
  className?: string;
}

export function AttributeTable({ attributes, className }: AttributeTableProps) {
  if (!attributes.length) return null;

  return (
    <section className={cn("space-y-2", className)}>
      <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
        Attributes
      </h4>
      <div className="divide-y divide-lucid-border rounded-xl border border-lucid-border overflow-hidden">
        {attributes.map((a) => (
          <div key={a.name} className="flex flex-col sm:flex-row gap-1 sm:gap-4 px-4 py-3 bg-lucid-surface hover:bg-lucid-elevated transition-colors">
            <div className="shrink-0 flex flex-col gap-0.5 sm:w-48">
              <code className="text-xs font-mono font-semibold text-lucid-text-high">
                {a.name}
              </code>
              {a.annotation && <TypeAnnotation annotation={a.annotation} />}
            </div>
            <div className="text-sm text-lucid-text-mid leading-relaxed flex-1">
              <MathText text={a.description} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

interface RaisesTableProps {
  raises: DocstringRaise[];
  className?: string;
}

export function RaisesTable({ raises, className }: RaisesTableProps) {
  if (!raises.length) return null;

  return (
    <section className={cn("space-y-2", className)}>
      <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
        Raises
      </h4>
      <div className="divide-y divide-lucid-border rounded-xl border border-lucid-border overflow-hidden">
        {raises.map((r, i) => (
          <div key={i} className="flex flex-col sm:flex-row gap-1 sm:gap-4 px-4 py-3 bg-lucid-surface">
            <div className="shrink-0 sm:w-48">
              {r.annotation && (
                <code className="text-xs font-mono font-semibold text-lucid-error">
                  {r.annotation}
                </code>
              )}
            </div>
            <div className="text-sm text-lucid-text-low leading-relaxed flex-1">
              <MathText text={r.description} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
