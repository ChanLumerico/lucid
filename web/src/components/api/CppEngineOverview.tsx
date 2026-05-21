import type { LucideIcon } from "lucide-react";
import {
  Boxes,
  Cpu,
  Dices,
  GitBranch,
  Layers,
  Network,
  Plug,
  Sigma,
  TrendingDown,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { MathText } from "./MathText";
import type { ApiMember, ApiModule } from "@/lib/types";
import { cn } from "@/lib/utils";

interface CategoryDef {
  /** First segment of the member's ``subcategory`` ("ops/ufunc" → "ops"). */
  slug: string;
  title: string;
  description: string;
  icon: LucideIcon;
}

const CATEGORIES: CategoryDef[] = [
  { slug: "core",      title: "Core",              description: "Storage, dtype, device — the runtime substrate every tensor sits on.",        icon: Cpu          },
  { slug: "tensor",    title: "Tensor",            description: "TensorImpl + OpEntry — Lucid's reference-counted value type.",                 icon: Boxes        },
  { slug: "backend",   title: "Backends",          description: "Stream dispatch — CPU (Accelerate), GPU (MLX), Metal Performance Shaders.",   icon: Layers       },
  { slug: "kernel",    title: "Kernels",           description: "Low-level compute primitives — gather, scatter, im2col, broadcast reduce.",    icon: Zap          },
  { slug: "ops",       title: "Tensor Operations", description: "Unary / binary / generic / composite ops + linalg, FFT, einops, complex.",     icon: Sigma        },
  { slug: "nn",        title: "Neural Networks",   description: "Layer kernels — Conv, BatchNorm, attention, RNN, pooling, normalization.",     icon: Network      },
  { slug: "autograd",  title: "Autograd",          description: "Backward node graph + saved-tensor machinery for every primitive op.",         icon: GitBranch    },
  { slug: "optim",     title: "Optimizers",        description: "SGD, Adam, AdamW, ASGD, Adadelta, Adagrad, RMSprop and friends.",               icon: TrendingDown },
  { slug: "random",    title: "Random",            description: "RNG state + sampling primitives shared by initializers and distributions.",    icon: Dices        },
  { slug: "bindings",  title: "Bindings",          description: "pybind11 entry points exposing the C++ surface to Python.",                    icon: Plug         },
];

const ACCENT = "var(--color-lucid-primary)";

interface SubBucket {
  slug: string;
  label: string;
  count: number;
}

/** Second-segment sub-bucket map for nested categories.  Only categories
 *  with a meaningful sub-tree get pills — everything else just shows a
 *  count badge. */
const SUB_LABELS: Record<string, Record<string, string>> = {
  backend: {
    cpu: "CPU (Accelerate)",
    gpu: "GPU (MLX)",
    "gpu/mps": "MPS",
  },
  ops: {
    ufunc: "Unary",
    bfunc: "Binary",
    gfunc: "Generic",
    composite: "Composite",
    linalg: "Linear Algebra",
    fft: "FFT",
    einops: "Einops",
    complex: "Complex",
    utils: "Utils",
  },
  kernel: {
    primitives: "Primitives",
  },
};

function countSubBuckets(
  members: ApiMember[],
  catSlug: string,
): SubBucket[] {
  const sublabels = SUB_LABELS[catSlug];
  if (!sublabels) return [];
  const counts = new Map<string, number>();
  for (const m of members) {
    const sub = m.subcategory ?? "";
    if (!sub.startsWith(`${catSlug}/`)) continue;
    const rest = sub.slice(catSlug.length + 1);
    // Match the longest declared sub-label prefix.
    let matched: string | null = null;
    for (const key of Object.keys(sublabels)) {
      if (rest === key || rest.startsWith(`${key}/`)) {
        if (matched === null || key.length > matched.length) {
          matched = key;
        }
      }
    }
    if (matched === null) continue;
    counts.set(matched, (counts.get(matched) ?? 0) + 1);
  }
  return Object.entries(sublabels)
    .map(([slug, label]) => ({ slug, label, count: counts.get(slug) ?? 0 }))
    .filter((b) => b.count > 0);
}

interface CppEngineOverviewProps {
  data: ApiModule;
}

export function CppEngineOverview({ data }: CppEngineOverviewProps) {
  // First-segment count map for every present category.
  const catCounts = new Map<string, number>();
  for (const m of data.members) {
    const seg = (m.subcategory ?? "").split("/")[0];
    if (!seg) continue;
    catCounts.set(seg, (catCounts.get(seg) ?? 0) + 1);
  }

  const visible = CATEGORIES
    .map((c) => ({
      def: c,
      count: catCounts.get(c.slug) ?? 0,
      subs: countSubBuckets(data.members, c.slug),
    }))
    .filter((c) => c.count > 0);

  return (
    <div>
      <Hero
        name={data.name}
        path={data.path}
        summary={data.summary}
        extended={data.extended}
        memberCount={data.members.length}
      />

      <section className="mb-10">
        <h2 className="text-xs font-semibold tracking-widest uppercase mb-3 text-lucid-text-disabled">
          Surfaces
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {visible.map((c) => (
            <CategoryCard
              key={c.def.slug}
              def={c.def}
              count={c.count}
              subs={c.subs}
            />
          ))}
        </div>
      </section>

      <p className="mt-12 text-xs text-lucid-text-disabled leading-relaxed">
        Browse the full surface — every class, free function, backward node,
        and backend kernel — through the left-hand sidebar.  This page lists
        only the top-level surfaces for orientation.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Hero
// ---------------------------------------------------------------------------

interface HeroProps {
  name: string;
  path: string;
  summary: string | null;
  extended: string | null;
  memberCount: number;
}

function Hero({ name, path, summary, extended, memberCount }: HeroProps) {
  return (
    <header className="mb-10">
      <div className="flex flex-wrap items-center gap-3 mb-2">
        <span className="text-xs font-semibold tracking-widest uppercase text-lucid-text-disabled">
          module
        </span>
        <h1 className="font-mono text-3xl font-bold text-lucid-text-high">
          C++ Engine
        </h1>
        <Badge variant="secondary" className="font-mono text-[11px]">
          {memberCount} members
        </Badge>
      </div>
      <code className="text-sm text-lucid-text-low font-mono">{path}</code>
      {(summary || extended) && (
        <div className="mt-4 max-w-3xl space-y-3">
          {summary && (
            <p className="text-base text-lucid-text-mid leading-relaxed">
              {summary}
            </p>
          )}
          {extended && (
            <div className="text-sm text-lucid-text-low leading-relaxed">
              <MathText text={extended} block />
            </div>
          )}
        </div>
      )}
    </header>
  );
}

// ---------------------------------------------------------------------------
// Category card
// ---------------------------------------------------------------------------

interface CategoryCardProps {
  def: CategoryDef;
  count: number;
  subs: SubBucket[];
}

function CategoryCard({ def, count, subs }: CategoryCardProps) {
  const Icon = def.icon;
  const bg = `color-mix(in srgb, ${ACCENT} 14%, transparent)`;
  const ring = `color-mix(in srgb, ${ACCENT} 38%, transparent)`;
  return (
    <div
      className={cn(
        "group block rounded-xl border border-lucid-border bg-lucid-surface/40",
        "transition-colors hover:bg-lucid-surface hover:border-lucid-primary/40",
      )}
    >
      <div
        className={cn(
          "px-5 pt-4 flex items-start gap-3",
          subs.length > 0 ? "pb-3" : "pb-4",
        )}
      >
        <span
          className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-lg border"
          style={{ backgroundColor: bg, borderColor: ring, color: ACCENT }}
          aria-hidden
        >
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <code
              className="font-mono text-sm font-semibold"
              style={{ color: ACCENT }}
            >
              {def.title}
            </code>
            <Badge variant="secondary" className="font-mono text-[10px]">
              {count}
            </Badge>
          </div>
          <p className="mt-1 text-xs text-lucid-text-low leading-relaxed">
            {def.description}
          </p>
        </div>
      </div>
      {subs.length > 0 && (
        <div className="pl-[4.25rem] pr-5 pb-4 flex flex-wrap gap-1.5">
          {subs.map((s) => (
            <span
              key={s.slug}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5",
                "text-[10px] font-mono text-lucid-text-mid border-lucid-border bg-lucid-elevated/50",
              )}
            >
              {s.label}
              <span className="text-lucid-text-disabled">{s.count}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
