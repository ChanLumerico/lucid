import type { Metadata } from "next";
import Link from "next/link";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { getAllModuleSlugs, loadApiData } from "@/lib/api-loader";
import { isApiModule, isApiClassModule } from "@/lib/types";

export const metadata: Metadata = {
  title: "API Reference",
  description: "Complete API reference for the Lucid ML framework.",
};

// ---------------------------------------------------------------------------
// Display labels + categorisation — kept in sync with the sidebar
// (web/src/app/api/layout.tsx).  The bare `lucid` slug is intentionally
// absent from the build manifest, so it never appears here either.
// ---------------------------------------------------------------------------

const PACKAGE_LABELS: Record<string, string> = {
  "lucid.tensor":        "lucid.Tensor",
  "lucid.creation":      "Tensor Creation",
  "lucid.ops":           "Tensor Operations",
  "lucid.ops.composite": "Composite Ops",
  "lucid.nn":            "Neural Networks",
  "lucid.nn.functional": "Functional",
  "lucid.nn.init":       "Init",
  "lucid.nn.utils":      "NN Utils",
  "lucid.optim":         "Optimizers",
  "lucid.autograd":      "Autograd",
  "lucid.func":          "Functional Transforms",
  "lucid.linalg":        "Linear Algebra",
  "lucid.fft":           "FFT",
  "lucid.signal":        "Signal",
  "lucid.special":       "Special Functions",
  "lucid.distributions": "Distributions",
  "lucid.einops":        "Einops",
  "lucid.amp":           "Mixed Precision",
  "lucid.profiler":      "Profiler",
  "lucid.serialization": "Serialization",
  "lucid.utils.data":    "Data",
};

const CATEGORY_ORDER = [
  "Core",
  "Neural Networks",
  "Optimization",
  "Differentiation",
  "Math",
  "Probabilistic",
  "Tooling",
] as const;

function categoryFor(slug: string): string {
  if (slug === "lucid.tensor" || slug === "lucid.creation" || slug === "lucid.ops" || slug === "lucid.ops.composite") {
    return "Core";
  }
  if (slug === "lucid.nn" || slug.startsWith("lucid.nn.")) return "Neural Networks";
  if (slug === "lucid.optim" || slug.startsWith("lucid.optim.")) return "Optimization";
  if (slug === "lucid.autograd" || slug === "lucid.func") return "Differentiation";
  if (["lucid.linalg", "lucid.fft", "lucid.special", "lucid.signal", "lucid.einops"].includes(slug)) {
    return "Math";
  }
  if (slug === "lucid.distributions") return "Probabilistic";
  return "Tooling";
}

function titleize(slug: string): string {
  return slug
    .replace(/^lucid\./, "")
    .split(/[._-]/)
    .map((s) => (s ? s[0].toUpperCase() + s.slice(1) : s))
    .join(" ");
}

function labelFor(slug: string): string {
  return PACKAGE_LABELS[slug] ?? titleize(slug);
}

// ---------------------------------------------------------------------------
// Card rendering
// ---------------------------------------------------------------------------

interface CardData {
  slug: string;
  label: string;
  summary: string;
  count: number;
}

function buildCards(): Map<string, CardData[]> {
  const slugs = getAllModuleSlugs();
  const byCategory = new Map<string, CardData[]>();

  for (const slug of slugs) {
    let summary = "";
    let count = 0;
    try {
      const data = loadApiData(slug);
      summary = data.summary ?? "";
      if (isApiModule(data)) count = data.members?.length ?? 0;
      else if (isApiClassModule(data)) count = data.methods?.length ?? 0;
    } catch {
      // skip — missing JSON
      continue;
    }
    const cat = categoryFor(slug);
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat)!.push({ slug, label: labelFor(slug), summary, count });
  }

  // Sort: `lucid.tensor` first in Core, rest alphabetical by label
  for (const cards of byCategory.values()) {
    cards.sort((a, b) => {
      const rank = (s: string) => (s === "lucid.tensor" ? 0 : 1);
      return rank(a.slug) - rank(b.slug) || a.label.localeCompare(b.label);
    });
  }
  return byCategory;
}

export default function ApiPage() {
  const byCategory = buildCards();

  return (
    <FadeIn>
      <div className="max-w-4xl">
        <h1 className="text-3xl font-bold text-lucid-text-high mb-3">
          API Reference
        </h1>
        <p className="text-lucid-text-mid mb-12 text-base leading-relaxed">
          Complete reference documentation, auto-generated from Lucid source code.
        </p>

        <div className="space-y-10">
          {CATEGORY_ORDER.map((category) => {
            const cards = byCategory.get(category);
            if (!cards?.length) return null;
            return (
              <section key={category}>
                <FadeIn>
                  <h2 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase mb-3">
                    {category}
                  </h2>
                </FadeIn>
                <FadeInStagger staggerDelay={0.04} className="space-y-2">
                  {cards.map(({ slug, label, summary, count }) => (
                    <Link
                      key={slug}
                      href={`/api/${slug}`}
                      className="group flex items-start justify-between rounded-xl border border-lucid-border bg-lucid-surface px-4 py-3.5 transition-all hover:border-lucid-primary/40 hover:bg-lucid-elevated"
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <span className="text-sm font-medium text-lucid-text-high">
                            {label}
                          </span>
                          <code className="text-[11px] font-mono text-lucid-text-disabled">
                            {slug}
                          </code>
                          {count > 0 && (
                            <Badge variant="secondary" className="font-mono text-[10px]">
                              {count}
                            </Badge>
                          )}
                        </div>
                        {summary && (
                          <p className="text-xs text-lucid-text-low leading-relaxed line-clamp-2">
                            {summary}
                          </p>
                        )}
                      </div>
                    </Link>
                  ))}
                </FadeInStagger>
              </section>
            );
          })}
        </div>
      </div>
    </FadeIn>
  );
}
