import fs from "fs";
import path from "path";
import { BASE_PATH } from "@/lib/type-links";

/**
 * Architecture tab — interactive archify diagrams of how Lucid works.
 *
 * Source of truth is ``content/architecture/``: each diagram is an archify
 * JSON spec plus the standalone HTML viewer archify delivered from it.
 * ``scripts/sync-archify.mjs`` copies the viewers into ``public/archify/`` and
 * links them to ``host.js``, which keeps an embedded viewer on the site's
 * theme and palette.
 *
 * Everything the pages show about a diagram comes from its spec — title,
 * diagram type, guided chapters, key-fact cards.  The catalog holds only what
 * a spec cannot say: the URL slug, a short nav label, the reading order and
 * grouping, and a one-line summary for the overview cards.
 */

export type DiagramKind = "architecture" | "sequence" | "workflow" | "dataflow" | "lifecycle";

export type FactTone = "blue" | "success" | "primary" | "warning" | "error" | "neutral";

export interface DiagramChapter {
  id: string;
  label: string;
  note?: string;
}

export interface DiagramFacts {
  tone: FactTone;
  title: string;
  items: string[];
}

export interface Diagram {
  slug: string;
  label: string;
  group: string;
  summary: string;
  title: string;
  kind: DiagramKind;
  /** What the diagram is made of, by kind — "12 components", "8 states". */
  extent: string;
  chapters: DiagramChapter[];
  facts: DiagramFacts[];
  /** Commit the diagram's source links are pinned to (architecture only). */
  sourceRevision: string | null;
  sourceLinks: number;
  viewerSrc: string;
}

const CATALOG: ReadonlyArray<{
  slug: string;
  file: string;
  label: string;
  group: string;
  summary: string;
}> = [
  {
    slug: "layer-map",
    file: "lucid-layers",
    label: "Layer map",
    group: "Structure",
    summary:
      "Every layer an op crosses: the Python API, pybind11, ops and kernels, the backend dispatcher, and the two Apple stream backends.",
  },
  {
    slug: "forward-op",
    file: "lucid-forward-add",
    label: "Forward op",
    group: "Execution",
    summary:
      "One a + b traced end to end — dtype promotion, SchemaGuard, vDSP or MLX, and the AddBackward node it leaves behind.",
  },
  {
    slug: "backward",
    file: "lucid-backward",
    label: "Backward",
    group: "Execution",
    summary:
      "What loss.backward() actually runs: the ones seed, the reverse-DFS walk, version checks and gradient accumulation.",
  },
  {
    slug: "training-step",
    file: "lucid-train-step",
    label: "Training step",
    group: "Training",
    summary:
      "A full iteration from DataLoader to optimizer.step, including the AMP branch that skips a step on overflow.",
  },
  {
    slug: "tensor-data",
    file: "lucid-tensor-data",
    label: "Tensor data",
    group: "Data & state",
    summary:
      "Where tensor bytes enter and leave Lucid — the bridges, the three Storage variants, and which moves copy.",
  },
  {
    slug: "autograd-life",
    file: "lucid-autograd-state",
    label: "Autograd life",
    group: "Data & state",
    summary:
      "A tensor's autograd states, from recorded to walked to released, plus the version-mismatch recovery loop.",
  },
];

export const DIAGRAM_KIND_LABEL: Record<DiagramKind, string> = {
  architecture: "Architecture",
  sequence: "Sequence",
  workflow: "Workflow",
  dataflow: "Data flow",
  lifecycle: "Lifecycle",
};

/** archify card dot colour → site tone. */
const DOT_TONE: Record<string, FactTone> = {
  cyan: "blue",
  emerald: "success",
  violet: "primary",
  amber: "warning",
  orange: "warning",
  rose: "error",
  slate: "neutral",
};

/** Kept as literal class names so Tailwind's scanner emits them. */
export const FACT_TONE_DOT: Record<FactTone, string> = {
  blue: "bg-lucid-blue",
  success: "bg-lucid-success",
  primary: "bg-lucid-primary",
  warning: "bg-lucid-warning",
  error: "bg-lucid-error",
  neutral: "bg-lucid-text-low",
};

interface Spec {
  diagram_type: DiagramKind;
  meta: {
    title: string;
    repository?: { revision?: string };
    views?: { id: string; label: string; note?: string }[];
  };
  cards?: { dot: string; title: string; items: string[] }[];
  components?: { sources?: unknown[] }[];
  participants?: unknown[];
  messages?: unknown[];
  nodes?: unknown[];
  states?: unknown[];
}

const SPEC_DIR = path.join(process.cwd(), "content", "architecture");

function readSpec(file: string): Spec {
  const name = fs.existsSync(SPEC_DIR)
    ? fs.readdirSync(SPEC_DIR).find((f) => f.startsWith(`${file}.`) && f.endsWith(".json"))
    : undefined;
  // Fail the build rather than ship a page whose viewer 404s.
  if (!name) throw new Error(`architecture: no archify spec for "${file}" in content/architecture/`);
  return JSON.parse(fs.readFileSync(path.join(SPEC_DIR, name), "utf-8")) as Spec;
}

function plural(n: number, noun: string): string {
  return `${n} ${noun}${n === 1 ? "" : "s"}`;
}

function extentOf(spec: Spec): string {
  switch (spec.diagram_type) {
    case "architecture":
      return plural(spec.components?.length ?? 0, "component");
    case "sequence":
      return `${plural(spec.participants?.length ?? 0, "participant")} · ${plural(spec.messages?.length ?? 0, "message")}`;
    case "workflow":
      return plural(spec.nodes?.length ?? 0, "step");
    case "dataflow":
      return plural(spec.nodes?.length ?? 0, "node");
    case "lifecycle":
      return plural(spec.states?.length ?? 0, "state");
  }
}

let cache: Diagram[] | null = null;

export function getDiagrams(): Diagram[] {
  if (cache) return cache;
  cache = CATALOG.map((entry) => {
    const spec = readSpec(entry.file);
    return {
      slug: entry.slug,
      label: entry.label,
      group: entry.group,
      summary: entry.summary,
      title: spec.meta.title,
      kind: spec.diagram_type,
      extent: extentOf(spec),
      chapters: (spec.meta.views ?? []).map(({ id, label, note }) => ({ id, label, note })),
      facts: (spec.cards ?? []).map(({ dot, title, items }) => ({
        tone: DOT_TONE[dot] ?? "neutral",
        title,
        items,
      })),
      sourceRevision: spec.meta.repository?.revision ?? null,
      sourceLinks: (spec.components ?? []).reduce((n, c) => n + (c.sources?.length ?? 0), 0),
      viewerSrc: `${BASE_PATH}/archify/${entry.file}.html`,
    };
  });
  return cache;
}

export function getDiagram(slug: string): Diagram | undefined {
  return getDiagrams().find((d) => d.slug === slug);
}

/** Groups in first-appearance order, catalog order within each group. */
export function getDiagramGroups(): [string, Diagram[]][] {
  const groups = new Map<string, Diagram[]>();
  for (const d of getDiagrams()) {
    const list = groups.get(d.group) ?? [];
    list.push(d);
    groups.set(d.group, list);
  }
  return [...groups];
}
