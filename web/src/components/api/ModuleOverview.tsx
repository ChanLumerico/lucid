import { Badge } from "@/components/ui/badge";
import { FunctionCard } from "./FunctionSignature";
import { ClassCard, ClassDoc } from "./ClassDoc";
import type { ApiModule, ApiClassModule, ApiClass } from "@/lib/types";
import { isApiClass, isApiFunction } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ModuleOverviewProps {
  data: ApiModule | ApiClassModule;
}

export async function ModuleOverview({ data }: ModuleOverviewProps) {
  if (data.kind === "class-module") {
    // The module IS a class (e.g. lucid.tensor → Tensor).  Rebuild it as an
    // ApiClass and delegate to ClassDoc so the full docstring body —
    // extended description, parameters, attributes, notes, examples — is
    // rendered identically to any other class detail page.
    const cls: ApiClass = {
      name:        data.name,
      path:        data.path,
      kind:        "class",
      class_kind:  data.class_kind,
      bases:       data.bases,
      labels:      data.labels,
      signature:   data.signature,
      source:      data.source,
      methods:     data.methods,
      summary:     data.summary,
      extended:    data.extended,
      parameters:  data.parameters,
      returns:     data.returns,
      raises:      data.raises,
      examples:    data.examples,
      notes:       data.notes,
      attributes:  data.attributes,
      warns:       data.warns,
    };
    return <ClassDoc cls={cls} />;
  }

  const classes   = data.members.filter(isApiClass);
  const functions = data.members.filter(isApiFunction);

  return (
    <div>
      <ModuleHeader
        name={data.name}
        path={data.path}
        summary={data.summary}
        kind="module"
        count={data.members.length}
      />
      {classes.length > 0 && (
        <MemberSection title="Classes" slug={data.slug} accent="primary">
          {classes.map((cls) => (
            <ClassCard key={cls.name} cls={cls} moduleSlug={data.slug} />
          ))}
        </MemberSection>
      )}
      {functions.length > 0 && (
        <MemberSection title="Functions" slug={data.slug} accent="blue">
          {functions.map((fn) => (
            <FunctionCard key={fn.name} fn={fn} moduleSlug={data.slug} />
          ))}
        </MemberSection>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface ModuleHeaderProps {
  name: string;
  path: string;
  summary: string | null;
  kind: "module" | "class";
  count: number;
}

function ModuleHeader({ name, path, summary, kind, count }: ModuleHeaderProps) {
  const kindColor = kind === "class" ? "text-api-class/60" : "text-lucid-text-disabled";
  const nameColor = kind === "class" ? "text-api-class" : "text-lucid-text-high";
  return (
    <header className="mb-10">
      <div className="flex flex-wrap items-center gap-3 mb-2">
        <span className={cn("text-xs font-semibold tracking-widest uppercase", kindColor)}>
          {kind}
        </span>
        <h1 className={cn("font-mono text-3xl font-bold", nameColor)}>{name}</h1>
        <Badge variant="secondary" className="font-mono text-[11px]">
          {count} members
        </Badge>
      </div>
      <code className="text-sm text-lucid-text-low font-mono">{path}</code>
      {summary && (
        <p className="mt-3 text-base text-lucid-text-mid leading-relaxed max-w-3xl">
          {summary}
        </p>
      )}
    </header>
  );
}

interface MemberSectionProps {
  title: string;
  slug: string;
  children: React.ReactNode;
  accent?: "primary" | "blue";
}

function MemberSection({ title, children }: MemberSectionProps) {
  return (
    <section className="mb-10">
      <h2 className="text-xs font-semibold tracking-widest uppercase mb-3 text-lucid-text-disabled">
        {title}
      </h2>
      <div className="space-y-2">{children}</div>
    </section>
  );
}
