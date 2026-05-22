import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getAllModuleSlugs, loadApiData, findMember } from "@/lib/api-loader";
import { isApiClass, isApiFunction, isApiModule, isApiClassModule } from "@/lib/types";
import { ModuleOverview } from "@/components/api/ModuleOverview";
import { CppEngineOverview } from "@/components/api/CppEngineOverview";
import { ClassDoc } from "@/components/api/ClassDoc";
import { FunctionSignature } from "@/components/api/FunctionSignature";
import { PageTableOfContents } from "@/components/layout/TableOfContents";
import { FadeIn } from "@/components/motion/FadeIn";

// ---------------------------------------------------------------------------
// Static generation
// ---------------------------------------------------------------------------

export async function generateStaticParams(): Promise<{ slug: string[] }[]> {
  const slugs = getAllModuleSlugs();
  const params: { slug: string[] }[] = [];

  // Module overview pages: /api/lucid.fft, /api/lucid.nn, …
  for (const slug of slugs) {
    params.push({ slug: [slug] });
  }

  // Member detail pages: /api/lucid.nn/Linear, /api/lucid.fft/fft, …
  for (const slug of slugs) {
    try {
      const data = loadApiData(slug);
      if (isApiModule(data)) {
        for (const member of data.members) {
          params.push({ slug: [slug, member.name] });
        }
      } else if (isApiClassModule(data)) {
        for (const method of data.methods) {
          params.push({ slug: [slug, method.name] });
        }
      }
    } catch {
      // skip missing data
    }
  }

  return params;
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const [moduleSlug, memberName] = slug;

  try {
    const data = loadApiData(moduleSlug);
    if (memberName) {
      return {
        title: `${memberName} — ${data.name}`,
        description: `API reference for ${memberName} in ${data.path}`,
      };
    }
    return {
      title: data.name,
      description: data.summary ?? `API reference for ${data.path}`,
    };
  } catch {
    return { title: "API Reference" };
  }
}

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default async function ApiSlugPage({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const [moduleSlug, memberName] = slug;

  let data;
  try {
    data = loadApiData(moduleSlug);
  } catch {
    notFound();
  }

  // Memoise once per request — both detail branches need the same set
  // to build their breadcrumbs.
  const validSlugs = new Set(getAllModuleSlugs());

  // ── Member detail page ────────────────────────────────────────────────────
  if (memberName) {
    const breadcrumbParts = buildBreadcrumb(moduleSlug, memberName, validSlugs);

    // Class-module (Tensor) method
    if (isApiClassModule(data)) {
      const method = data.methods.find((m) => m.name === memberName);
      if (!method) notFound();
      return (
        <FadeIn className="flex flex-col gap-10 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 max-w-4xl">
            <Breadcrumb parts={breadcrumbParts} />
            <FunctionSignature
              fn={method}
              headingLevel="h2"
              className="mt-6"
              moduleSlug={moduleSlug}
            />
          </div>
          <PageTableOfContents />
        </FadeIn>
      );
    }

    // Regular module member
    if (isApiModule(data)) {
      const member = findMember(data, memberName);
      if (!member) notFound();

      return (
        <FadeIn className="flex flex-col gap-10 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0 max-w-4xl">
            <Breadcrumb parts={breadcrumbParts} />
            <div className="mt-6">
              {isApiClass(member) ? (
                <ClassDoc cls={member} moduleSlug={moduleSlug} />
              ) : isApiFunction(member) ? (
                <FunctionSignature fn={member} headingLevel="h2" moduleSlug={moduleSlug} />
              ) : null}
            </div>
          </div>
          <PageTableOfContents />
        </FadeIn>
      );
    }

    notFound();
  }

  // ── Module overview page ───────────────────────────────────────────────────
  // The C++ engine module gets a curated landing treatment — 813 member
  // cards would be unscannable, so ``CppEngineOverview`` surfaces only the
  // top-level surfaces (Core / Tensor / Backends / Ops / NN / Autograd / …)
  // and relies on the sidebar tree for the full drill-down.
  if (moduleSlug === "lucid._C.engine" && isApiModule(data)) {
    return (
      <FadeIn className="max-w-4xl">
        <CppEngineOverview data={data} />
      </FadeIn>
    );
  }
  return (
    <FadeIn className="flex flex-col gap-10 xl:flex-row xl:items-start xl:justify-between">
      <div className="min-w-0 max-w-4xl">
        <ModuleOverview data={data} />
      </div>
      <PageTableOfContents minEntries={2} />
    </FadeIn>
  );
}

// ---------------------------------------------------------------------------
// Breadcrumb
// ---------------------------------------------------------------------------

interface BreadcrumbPart {
  label: string;
  href?: string;
}

/** Friendly label overrides for slugs whose raw last-segment doesn't
 *  read well on its own (e.g. ``engine`` → ``C++ Engine``).  Only the
 *  step matching the FULL slug gets renamed; intermediate segments
 *  stay as-is.  Layered on top of the default ``segs[i]`` label in
 *  ``buildBreadcrumb``. */
const BREADCRUMB_LABEL_OVERRIDES: Record<string, string> = {
  "lucid._C.engine": "C++ Engine",
};

/** Build a multi-step breadcrumb from a slug like ``lucid.nn.functional``.
 *
 *  Each dot-separated prefix that has its own docs page becomes a
 *  clickable step ( ``lucid`` → ``lucid.nn`` → ``lucid.nn.functional`` );
 *  prefixes that aren't documented (e.g. the synthetic ``lucid._C``
 *  intermediate above ``lucid._C.engine``) are skipped to avoid dead
 *  links.  When the page renders a specific member (class / function),
 *  the member name is appended as the final non-link step. */
function buildBreadcrumb(
  moduleSlug: string,
  memberName: string | undefined,
  validSlugs: Set<string>,
): BreadcrumbPart[] {
  const segs = moduleSlug.split(".");
  const parts: BreadcrumbPart[] = [];
  for (let i = 0; i < segs.length; i++) {
    const prefix = segs.slice(0, i + 1).join(".");
    if (!validSlugs.has(prefix)) continue;
    const isFinalSlugStep = i === segs.length - 1;
    parts.push({
      label: BREADCRUMB_LABEL_OVERRIDES[prefix] ?? segs[i],
      // The terminal slug step links to itself only when there's a
      // deeper member step after it — otherwise it's the active page
      // and shouldn't be a link.
      href: !isFinalSlugStep || memberName ? `/api/${prefix}` : undefined,
    });
  }
  if (memberName) parts.push({ label: memberName });
  return parts;
}

interface BreadcrumbProps {
  parts: BreadcrumbPart[];
}

function Breadcrumb({ parts }: BreadcrumbProps) {
  return (
    <nav className="flex items-center gap-1.5 text-sm text-lucid-text-low mb-2" aria-label="Breadcrumb">
      {parts.map((part, i) => (
        <span key={i} className="flex items-center gap-1.5">
          {/* Separators sit BETWEEN parts; the first step renders bare. */}
          {i > 0 && <span className="text-lucid-text-disabled">/</span>}
          {part.href ? (
            <a href={part.href} className="font-mono hover:text-lucid-primary transition-colors">
              {part.label}
            </a>
          ) : (
            <span className="font-mono text-lucid-text-high">{part.label}</span>
          )}
        </span>
      ))}
    </nav>
  );
}
