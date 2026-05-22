import Link from "next/link";
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

const ENGINE_SLUG = "lucid._C.engine";

/** First member whose ``subcategory`` matches the supplied surface
 *  slug.  Used to point a surface card at a concrete detail page — the
 *  user lands on a real symbol from which they can branch out via the
 *  sidebar tree.  Returns ``null`` when the surface contains no public
 *  members (treated as a non-clickable card). */
function firstMemberFor(members: ApiMember[], surfaceSlug: string): string | null {
  for (const m of members) {
    const sub = m.subcategory ?? "";
    if (sub === surfaceSlug || sub.startsWith(`${surfaceSlug}/`)) return m.name;
  }
  return null;
}

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
      firstMember: firstMemberFor(data.members, c.slug),
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
              firstMember={c.firstMember}
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
          engine
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
  /** Destination for the card's click — the first member that lives
   *  under this surface.  Cards without a target render as a
   *  non-interactive ``<div>`` (rare — would mean an empty surface). */
  firstMember: string | null;
}

/** Single-row surface card.  Mirrors ``/api/page.tsx::ModuleCardLink``
 *  so the C++ engine landing and the top-level API Reference share one
 *  visual contract — icon halo + identifier + count badge + one-line
 *  description, full card clickable.  Sub-bucket pills (CPU/GPU/MPS,
 *  Unary/Binary/...) intentionally removed: the same drill-down is
 *  available via the left-hand sidebar tree, and keeping the cards
 *  uniform across surfaces reads cleanly on the grid. */
function CategoryCard({ def, count, firstMember }: CategoryCardProps) {
  const Icon = def.icon;
  const bg = `color-mix(in srgb, ${ACCENT} 14%, transparent)`;
  const ring = `color-mix(in srgb, ${ACCENT} 38%, transparent)`;

  const cardClass = cn(
    "group block rounded-xl border border-lucid-border bg-lucid-surface/40",
    "transition-colors hover:bg-lucid-surface hover:border-lucid-primary/40",
  );
  const inner = (
    <div className="px-5 pt-4 pb-4 flex items-start gap-3">
      <span
        className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-lg border"
        style={{ backgroundColor: bg, borderColor: ring, color: ACCENT }}
        aria-hidden
      >
        <Icon className="h-4 w-4" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <code className="font-mono text-sm font-semibold" style={{ color: ACCENT }}>
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
  );

  return firstMember ? (
    <Link href={`/api/${ENGINE_SLUG}/${firstMember}`} className={cardClass}>
      {inner}
    </Link>
  ) : (
    <div className={cardClass}>{inner}</div>
  );
}
