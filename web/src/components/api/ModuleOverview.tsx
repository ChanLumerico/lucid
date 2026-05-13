import { Badge } from "@/components/ui/badge";
import { FunctionCard } from "./FunctionSignature";
import { ClassCard } from "./ClassDoc";
import type { ApiModule, ApiClassModule, ApiFunction, ApiClass } from "@/lib/types";
import { isApiClass, isApiFunction } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ModuleOverviewProps {
  data: ApiModule | ApiClassModule;
}

export function ModuleOverview({ data }: ModuleOverviewProps) {
  if (data.kind === "class-module") {
    return (
      <div>
        <ModuleHeader
          name={data.name}
          path={data.path}
          summary={data.summary}
          kind="class"
          count={data.methods.length}
        />
        <section className="mb-10">
          <h2 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase mb-3">
            Methods ({data.methods.length})
          </h2>
          <div className="space-y-2">
            {data.methods.map((method) => (
              <FunctionCard key={method.name} fn={method as ApiFunction} moduleSlug={data.slug} />
            ))}
          </div>
        </section>
      </div>
    );
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
