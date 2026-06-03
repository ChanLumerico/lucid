import type * as React from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/Card";
import { SectionHeading } from "@/components/ui/SectionHeading";
import { FunctionCard } from "./FunctionSignature";
import { ClassCard, ClassDoc } from "./ClassDoc";
import { MathText } from "./MathText";
import { AnchorLink } from "./AnchorLink";
import { ViewModeToggle } from "./ViewModeToggle";
import type { ApiModule, ApiClassModule, ApiClass, FamilyGroup } from "@/lib/types";
import { isApiClass, isApiFunction, isApiModule } from "@/lib/types";
import { loadApiData, getAllModuleSlugs } from "@/lib/api-loader";
import { packageLabel } from "@/lib/labels";
import { cn } from "@/lib/utils";

interface ModuleOverviewProps {
  data: ApiModule | ApiClassModule;
}

/** Direct documented sub-packages of a module slug: every slug exactly one
 *  dotted level deeper that is itself a documented module.  This is the single
 *  rule that surfaces sub-packages as cards in the main content for ANY package
 *  with children — Model Zoo (Vision/Text/Generative → families), Utils
 *  (Data/Tokenizers/Transforms), NN (Functional/Init/…), etc.  No per-package
 *  special-casing; the card content adapts (rich model-family card when the
 *  child carries ``@model_family_meta``, plain package card otherwise). */
function directSubpackages(parentSlug: string): FamilyGroup[] {
  const prefix = `${parentSlug}.`;
  return getAllModuleSlugs()
    .filter((s) => s.startsWith(prefix) && !s.slice(prefix.length).includes("."))
    .map((slug) => ({ slug, label: packageLabel(slug) }))
    .sort((a, b) => a.label.localeCompare(b.label));
}

interface SubpackageSection {
  title: string;
  groups: FamilyGroup[];
}

/** Group a module's sub-package cards into titled sections.  The general rule
 *  is one alphabetical "Sub-packages" section.  Model Zoo is the one curated
 *  exception: the content families (Vision → Text → Generative, in that order)
 *  sit up top under "Model Families", and the ``lucid.models.weights``
 *  infrastructure package gets its own "Infrastructure" section below — it's
 *  porting substrate, not a model family. */
function sectionSubpackages(
  parentSlug: string,
  children: FamilyGroup[],
): SubpackageSection[] {
  if (children.length === 0) return [];

  if (parentSlug === "lucid.models") {
    const FAMILY_ORDER = [
      "lucid.models.vision",
      "lucid.models.text",
      "lucid.models.generative",
    ];
    const rank = (slug: string) => {
      const i = FAMILY_ORDER.indexOf(slug);
      return i === -1 ? FAMILY_ORDER.length : i;
    };
    const families = children
      .filter((c) => FAMILY_ORDER.includes(c.slug))
      .sort((a, b) => rank(a.slug) - rank(b.slug));
    const infra = children.filter((c) => !FAMILY_ORDER.includes(c.slug));
    const sections: SubpackageSection[] = [];
    if (families.length) sections.push({ title: "Model Families", groups: families });
    if (infra.length) sections.push({ title: "Infrastructure", groups: infra });
    return sections;
  }

  const title = parentSlug.startsWith("lucid.models") ? "Model Families" : "Sub-packages";
  return [{ title, groups: children }];
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

  // Sub-package cards (Model Zoo families, Utils sub-packages, …).  Computed
  // generically from the slug hierarchy — no per-package config — then split
  // into titled sections (Model Zoo separates families from infrastructure).
  const subpackageSections = sectionSubpackages(
    data.slug,
    directSubpackages(data.slug),
  );

  // Family-leaf pages (e.g. lucid.models.vision.resnet) carry the full
  // ``citation`` + ``theory`` extracted from ``@model_family_meta``.
  // Render that above the member sections so the leaf page reads like
  // a paper-style brief with the implementation listing underneath.
  const hasFamilyMeta = Boolean(data.citation || data.theory);
  const displayName = data.canonical_name && data.canonical_name.length > 0
    ? data.canonical_name
    : data.name;

  // ``<article>`` is the heading-scope ``PageTableOfContents`` scans —
  // wrapping the overview in one is what lets the right rail pick up
  // the per-subcategory ``<h2 id>`` anchors below.  The
  // ``data-module-overview`` marker is the hook ``ViewModeToggle`` uses
  // to find this article and stamp ``data-view`` on it, controlling
  // the compact/detailed CSS variants applied to member cards.
  return (
    <article data-module-overview>
      <ModuleHeader
        name={displayName}
        path={data.path}
        summary={data.summary}
        extended={data.extended}
        notes={data.notes}
        kind="module"
        count={data.members.length}
        toolbar={<ViewModeToggle />}
      />
      {hasFamilyMeta && (
        <section className="mb-10 space-y-5">
          {data.tasks && data.tasks.length > 0 && (
            <TaskTagRow tasks={data.tasks} />
          )}
          {data.citation && (
            <div>
              <div className="text-[10px] font-semibold tracking-widest uppercase text-lucid-text-disabled mb-1.5">
                Paper
              </div>
              <p className="text-sm text-lucid-text-mid leading-relaxed italic border-l-2 border-lucid-border pl-3">
                {data.citation}
              </p>
            </div>
          )}
          {data.theory && (
            <div>
              <div className="text-[10px] font-semibold tracking-widest uppercase text-lucid-text-disabled mb-1.5">
                Overview
              </div>
              <div className="text-sm text-lucid-text-mid">
                <MathText text={data.theory} block />
              </div>
            </div>
          )}
        </section>
      )}
      {subpackageSections.map((sec) => (
        <SubpackageGrid key={sec.title} groups={sec.groups} title={sec.title} />
      ))}
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
    </article>
  );
}

// ---------------------------------------------------------------------------
// Family groups (used by lucid.models to surface vision/text/generative)
// ---------------------------------------------------------------------------

interface SubpackageGridProps {
  groups: FamilyGroup[];
  title: string;
}

interface FamilyCardData extends FamilyGroup {
  canonicalName: string;
  citation: string | null;
  theory: string | null;
  fallbackSummary: string | null;
  tasks: string[];
  count: number;
  unit: "members" | "families";
}

function SubpackageGrid({ groups, title }: SubpackageGridProps) {
  const cards: FamilyCardData[] = groups.map((g) => {
    let canonicalName = g.label;
    let citation: string | null = null;
    let theory: string | null = null;
    let fallbackSummary: string | null = null;
    let tasks: string[] = [];
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
        if (data.tasks && data.tasks.length > 0) {
          tasks = data.tasks;
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
    return { ...g, canonicalName, citation, theory, fallbackSummary, tasks, count, unit };
  });

  return (
    <section className="mb-10">
      <SectionHeading>{title}</SectionHeading>
      <div className="space-y-4">
        {cards.map((c) => (
          <FamilyCard key={c.slug} card={c} />
        ))}
      </div>
    </section>
  );
}

/** Extract the first paragraph of a (markdown-converted) theory body.
 *  Stops at the first blank line so block-math fenced by ``$$ ... $$``
 *  on its own paragraph never gets sliced mid-formula. */
function firstParagraph(text: string): string {
  const stripped = text.trim();
  const idx = stripped.search(/\n\s*\n/);
  return idx === -1 ? stripped : stripped.slice(0, idx);
}

// Pretty-print a task identifier: ``image-classification`` →
// ``Image Classification``.  Specific multi-word acronyms get manual
// overrides so casing matches paper convention.
const TASK_LABEL_OVERRIDES: Record<string, string> = {
  "fill-mask":                  "Fill-Mask",
  "text2text-generation":       "Text-to-Text Generation",
  "image-to-image":             "Image-to-Image",
  "next-sentence-prediction":   "Next Sentence Prediction",
  "masked-image-modeling":      "Masked Image Modeling",
  "image-classification":       "Image Classification",
  "object-detection":           "Object Detection",
  "instance-segmentation":      "Instance Segmentation",
  "semantic-segmentation":      "Semantic Segmentation",
  "panoptic-segmentation":      "Panoptic Segmentation",
  "image-generation":           "Image Generation",
  "text-generation":            "Text Generation",
  "text-classification":        "Text Classification",
  "token-classification":       "Token Classification",
  "question-answering":         "Question Answering",
  "multiple-choice":            "Multiple Choice",
  "pretraining":                "Pre-training",
};

// Task → swatch slug.  The swatch values themselves live in globals.css
// as ``--color-task-<slug>`` — see the "Task-tag palette" block there.
// To re-theme a category, edit the CSS swatches.  To add a new task, add
// one row here pointing at an existing swatch (or define a new swatch in
// CSS first).  Adjacent swatches are intentionally separated within each
// category so neighbouring task tags on the same card never look alike.
const TASK_SWATCH: Record<string, string> = {
  // ── Vision (cool spectrum) ────────────────────────────────────────────────
  "image-classification":     "vision-1",
  "object-detection":         "vision-2",
  "instance-segmentation":    "vision-3",
  "semantic-segmentation":    "vision-4",
  "panoptic-segmentation":    "vision-5",
  "image-to-image":           "vision-7",
  "masked-image-modeling":    "vision-6",
  // ── Text (warm spectrum) ──────────────────────────────────────────────────
  "fill-mask":                "text-1",
  "text-generation":          "text-2",
  "text2text-generation":     "text-3",
  "text-classification":      "text-4",
  "token-classification":     "text-5",
  "question-answering":       "text-6",
  "next-sentence-prediction": "text-7",
  "multiple-choice":          "text-8",
  // ── Generative ────────────────────────────────────────────────────────────
  "image-generation":         "generative-1",
  // ── Multi / meta ──────────────────────────────────────────────────────────
  "pretraining":              "meta-1",
};

const TASK_SWATCH_DEFAULT = "meta-1";

/** Build inline style for a task pill from its CSS swatch.
 *  Single rendering template — opacities composed via ``color-mix`` so
 *  the bg / text / border share one source of truth (the swatch). */
function taskTagStyle(task: string): React.CSSProperties {
  const swatch = TASK_SWATCH[task] ?? TASK_SWATCH_DEFAULT;
  const color = `var(--color-task-${swatch})`;
  return {
    color,
    backgroundColor: `color-mix(in srgb, ${color} 10%, transparent)`,
    borderColor: `color-mix(in srgb, ${color} 35%, transparent)`,
  };
}

function taskLabel(task: string): string {
  if (task in TASK_LABEL_OVERRIDES) return TASK_LABEL_OVERRIDES[task];
  return task
    .split("-")
    .map((w) => (w.length > 0 ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");
}

function TaskTag({ task }: { task: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5",
        "text-[11px] font-medium font-mono leading-snug",
      )}
      style={taskTagStyle(task)}
    >
      {taskLabel(task)}
    </span>
  );
}

function TaskTagRow({ tasks }: { tasks: string[] }) {
  if (!tasks || tasks.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {tasks.map((t) => (
        <TaskTag key={t} task={t} />
      ))}
    </div>
  );
}

function FamilyCard({ card }: { card: FamilyCardData }) {
  // Compact overview: only the first paragraph of theory on the card —
  // the full body is rendered on the family-leaf page itself.  Citation
  // first, intro second.  Card itself stays clickable.
  const theoryIntro = card.theory ? firstParagraph(card.theory) : null;
  const hasFallback = !theoryIntro && card.fallbackSummary;
  return (
    <Card href={`/api/${card.slug}`}>
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
      <div className="px-5 py-4 space-y-4">
        {card.tasks.length > 0 && <TaskTagRow tasks={card.tasks} />}
        {card.citation && (
          <p className="text-xs text-lucid-text-mid leading-relaxed italic border-l-2 border-lucid-border pl-3">
            {card.citation}
          </p>
        )}
        {theoryIntro && (
          <div className="text-sm text-lucid-text-mid">
            <MathText text={theoryIntro} block />
          </div>
        )}
        {hasFallback && (
          <p className="text-sm text-lucid-text-mid leading-relaxed">
            {card.fallbackSummary}
          </p>
        )}
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface ModuleHeaderProps {
  name: string;
  path: string;
  summary: string | null;
  /** Long-form prose from the module's top-of-file docstring.  Surfaced
   *  underneath the summary so architectural context (layering rules,
   *  op categories, gotchas) the author put in the module docstring
   *  reaches the docs reader rather than only the source reader. */
  extended?: string | null;
  /** ``Notes`` blocks parsed from the module docstring — rendered in
   *  the same blue-bordered callout style as class / function-level
   *  notes for visual consistency. */
  notes?: string[];
  kind: "module" | "class";
  count: number;
  /** Right-aligned slot for header-level controls — currently used by
   *  ``ModuleOverview`` to mount the ``ViewModeToggle``.  Optional so
   *  ``ClassDoc`` and other callers don't have to opt in. */
  toolbar?: React.ReactNode;
}

function ModuleHeader({
  name,
  path,
  summary,
  extended,
  notes,
  kind,
  count,
  toolbar,
}: ModuleHeaderProps) {
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
        {toolbar && <div className="ml-auto">{toolbar}</div>}
      </div>
      <code className="text-sm text-lucid-text-low font-mono">{path}</code>
      {summary && (
        <p className="mt-3 text-base text-lucid-text-mid leading-relaxed max-w-3xl">
          {summary}
        </p>
      )}
      {extended && (
        <div className="mt-4 max-w-3xl">
          <MathText
            text={extended}
            block
            className="text-sm text-lucid-text-mid leading-relaxed"
          />
        </div>
      )}
      {notes && notes.length > 0 && (
        <section className="mt-5 max-w-3xl space-y-2">
          <h2 className="text-[11px] font-semibold tracking-widest text-lucid-text-disabled uppercase">
            Notes
          </h2>
          <div className="rounded-xl border-l-2 border-lucid-blue bg-lucid-blue/5 px-4 py-3 space-y-2">
            {notes.map((note, i) => (
              <MathText
                key={i}
                text={note}
                block
                className="text-sm text-lucid-text-mid leading-relaxed"
              />
            ))}
          </div>
        </section>
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

/** Convert a section title to a URL-safe ToC anchor id. */
function _sectionId(title: string): string {
  return (
    "section-" +
    title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
  );
}

function MemberSection({ title, children }: MemberSectionProps) {
  const id = _sectionId(title);
  return (
    <section id={id} className="group mb-10 scroll-mt-24">
      <SectionHeading id={id} className="flex items-center gap-2">
        {title}
        <AnchorLink id={id} />
      </SectionHeading>
      <div className="space-y-2">{children}</div>
    </section>
  );
}
