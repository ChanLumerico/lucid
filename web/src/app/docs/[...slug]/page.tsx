import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { getAllDocSlugs, compileDoc } from "@/lib/mdx-compile";
import { getMDXComponents } from "@/components/mdx";
import { DocTableOfContents } from "@/components/layout/TableOfContents";

interface PageProps {
  params: Promise<{ slug: string[] }>;
}

export async function generateStaticParams() {
  const slugs = getAllDocSlugs();
  return slugs.map((slug) => ({ slug }));
}

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const doc = await compileDoc(slug);
  if (!doc) return {};
  return {
    title: doc.frontmatter.title,
    description: doc.frontmatter.description,
  };
}

export default async function DocSlugPage({ params }: PageProps) {
  const { slug } = await params;
  const doc = await compileDoc(slug, getMDXComponents());
  if (!doc) notFound();

  return (
    <div className="flex min-w-0 gap-10">
      <article className="min-w-0 flex-1">
        <header className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight text-lucid-text-high">
            {doc.frontmatter.title}
          </h1>
          {doc.frontmatter.description && (
            <p className="mt-2 text-base text-lucid-text-mid leading-relaxed">
              {doc.frontmatter.description}
            </p>
          )}
        </header>
        <div className="prose-lucid">{doc.content}</div>
      </article>
      <DocTableOfContents />
    </div>
  );
}
