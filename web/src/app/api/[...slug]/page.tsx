import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { getAllModuleSlugs, loadApiData, findMember } from "@/lib/api-loader";
import { isApiClass, isApiFunction, isApiModule, isApiClassModule } from "@/lib/types";
import { ModuleOverview } from "@/components/api/ModuleOverview";
import { ClassDoc } from "@/components/api/ClassDoc";
import { FunctionSignature } from "@/components/api/FunctionSignature";
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

  // ── Member detail page ────────────────────────────────────────────────────
  if (memberName) {
    // Class-module (Tensor) method
    if (isApiClassModule(data)) {
      const method = data.methods.find((m) => m.name === memberName);
      if (!method) notFound();
      return (
        <FadeIn className="max-w-4xl">
          <Breadcrumb parts={[
            { label: data.name, href: `/api/${moduleSlug}` },
            { label: memberName },
          ]} />
          <FunctionSignature fn={method} headingLevel="h2" className="mt-6" />
        </FadeIn>
      );
    }

    // Regular module member
    if (isApiModule(data)) {
      const member = findMember(data, memberName);
      if (!member) notFound();

      return (
        <FadeIn className="max-w-4xl">
          <Breadcrumb parts={[
            { label: data.name, href: `/api/${moduleSlug}` },
            { label: memberName },
          ]} />
          <div className="mt-6">
            {isApiClass(member) ? (
              <ClassDoc cls={member} />
            ) : isApiFunction(member) ? (
              <FunctionSignature fn={member} headingLevel="h2" />
            ) : null}
          </div>
        </FadeIn>
      );
    }

    notFound();
  }

  // ── Module overview page ───────────────────────────────────────────────────
  return (
    <FadeIn className="max-w-4xl">
      <ModuleOverview data={data} />
    </FadeIn>
  );
}

// ---------------------------------------------------------------------------
// Breadcrumb
// ---------------------------------------------------------------------------

interface BreadcrumbProps {
  parts: Array<{ label: string; href?: string }>;
}

function Breadcrumb({ parts }: BreadcrumbProps) {
  return (
    <nav className="flex items-center gap-1.5 text-sm text-lucid-text-low mb-2" aria-label="Breadcrumb">
      <a href="/api" className="hover:text-lucid-text-mid transition-colors">
        API
      </a>
      {parts.map((part, i) => (
        <span key={i} className="flex items-center gap-1.5">
          <span className="text-lucid-text-disabled">/</span>
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
