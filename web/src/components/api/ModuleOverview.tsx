import { Badge } from "@/components/ui/badge";
import { FunctionCard } from "./FunctionSignature";
import { ClassCard, ClassDoc } from "./ClassDoc";
import { MathText } from "./MathText";
import type { ApiModule, ApiClassModule, ApiClass, FamilyGroup } from "@/lib/types";
import { isApiClass, isApiFunction, isApiModule } from "@/lib/types";
import { loadApiData } from "@/lib/api-loader";
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
      {data.family_groups && data.family_groups.length > 0 && (
        <FamilyGroups groups={data.family_groups} />
      )}
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
// Family groups (used by lucid.models to surface vision/text/generative)
// ---------------------------------------------------------------------------

interface FamilyGroupsProps {
  groups: FamilyGroup[];
}

interface FamilyCardData extends FamilyGroup {
  canonicalName: string;
  citation: string | null;
  theory: string | null;
  fallbackSummary: string | null;
  count: number;
  unit: "members" | "families";
}

function FamilyGroups({ groups }: FamilyGroupsProps) {
  const cards: FamilyCardData[] = groups.map((g) => {
    let canonicalName = g.label;
    let citation: string | null = null;
    let theory: string | null = null;
    let fallbackSummary: string | null = null;
    let count = 0;
    let unit: "members" | "families" = "members";
    try {
      const data = loadApiData(g.slug);
      fallbackSummary = data.summary ?? null;
      if (isApiModule(data)) {
        if (data.canonical_name && data.canonical_name.length > 0) {
          canonicalName = data.canonical_name;
        }
        if (data.citation && data.citation.length > 0) {
          citation = data.citation;
        }
        if (data.theory && data.theory.length > 0) {
          theory = data.theory;
        }
        if (data.family_groups && data.family_groups.length > 0) {
          count = data.family_groups.length;
          unit = "families";
        } else {
          count = data.members.length;
        }
      }
    } catch {
      // family JSON missing — render with placeholders
    }
    return { ...g, canonicalName, citation, theory, fallbackSummary, count, unit };
  });

  return (
    <section className="mb-10">
      <h2 className="text-xs font-semibold tracking-widest uppercase mb-3 text-lucid-text-disabled">
        Model Families
      </h2>
      <div className="space-y-4">
        {cards.map((c) => (
          <FamilyCard key={c.slug} card={c} />
        ))}
      </div>
    </section>
  );
}

function FamilyCard({ card }: { card: FamilyCardData }) {
  // Body text priority: ``theory`` (rST+math) > module summary > nothing.
  // ``theory`` renders through MathText; the summary fallback stays a
  // plain paragraph because it's never rST-formatted.
  const hasTheory = card.theory !== null;
  const hasFallback = !hasTheory && card.fallbackSummary !== null;
  return (
    <a
      href={`/api/${card.slug}`}
      className="group block rounded-lg border border-lucid-border bg-lucid-surface/40 hover:bg-lucid-surface hover:border-lucid-primary/40 transition-colors"
    >
      <header className="flex flex-wrap items-center gap-3 px-5 pt-4 pb-3 border-b border-lucid-border/60">
        <span className="text-[10px] font-semibold tracking-widest uppercase text-api-class/70">
          family
        </span>
        <h3 className="font-mono text-lg font-semibold text-api-class group-hover:text-lucid-primary transition-colors">
          {card.canonicalName}
        </h3>
        <Badge variant="secondary" className="font-mono text-[10px]">
          {card.count} {card.unit}
        </Badge>
        <code className="text-xs text-lucid-text-low font-mono ml-auto">
          {card.slug}
        </code>
      </header>
      <div className="px-5 py-4 space-y-5">
        {/* Citation FIRST. */}
        {card.citation && (
          <div>
            <div className="text-[10px] font-semibold tracking-widest uppercase text-lucid-text-disabled mb-1.5">
              Paper
            </div>
            <p className="text-xs text-lucid-text-mid leading-relaxed italic border-l-2 border-lucid-border pl-3">
              {card.citation}
            </p>
          </div>
        )}
        {/* Theory SECOND — rST + math via MathText. */}
        {hasTheory && (
          <div>
            <div className="text-[10px] font-semibold tracking-widest uppercase text-lucid-text-disabled mb-1.5">
              Overview
            </div>
            <div className="text-sm text-lucid-text-mid leading-relaxed prose-family">
              <MathText text={card.theory!} />
            </div>
          </div>
        )}
        {hasFallback && (
          <p className="text-sm text-lucid-text-mid leading-relaxed">
            {card.fallbackSummary}
          </p>
        )}
      </div>
    </a>
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
