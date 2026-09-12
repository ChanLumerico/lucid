import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { DiagramViewer } from "@/components/architecture/DiagramViewer";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/Card";
import { SectionHeading } from "@/components/ui/SectionHeading";
import {
  DIAGRAM_KIND_LABEL,
  FACT_TONE_DOT,
  getDiagram,
  getDiagrams,
  type Diagram,
} from "@/lib/architecture";
import { cn } from "@/lib/utils";

type Params = Promise<{ slug: string }>;

export const dynamicParams = false;

export function generateStaticParams() {
  return getDiagrams().map((d) => ({ slug: d.slug }));
}

export async function generateMetadata({ params }: { params: Params }): Promise<Metadata> {
  const diagram = getDiagram((await params).slug);
  return diagram ? { title: diagram.title, description: diagram.summary } : {};
}

function DiagramSwitcher({ current, all }: { current: Diagram; all: Diagram[] }) {
  return (
    <nav aria-label="Diagrams" className="flex flex-wrap gap-1.5">
      {all.map((d) => {
        const active = d.slug === current.slug;
        return (
          <Link
            key={d.slug}
            href={`/architecture/${d.slug}`}
            aria-current={active ? "page" : undefined}
            className={cn(
              "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors",
              active
                ? "border-lucid-primary/30 bg-lucid-primary/10 text-lucid-primary"
                : "border-lucid-border text-lucid-text-mid hover:border-lucid-primary/40 hover:text-lucid-text-high",
            )}
          >
            {d.label}
          </Link>
        );
      })}
    </nav>
  );
}

function NeighbourCard({ diagram, direction }: { diagram: Diagram; direction: "previous" | "next" }) {
  const next = direction === "next";
  return (
    <Card
      href={`/architecture/${diagram.slug}`}
      className={cn("px-5 py-4", next && "text-right sm:col-start-2")}
    >
      <span
        className={cn(
          "flex items-center gap-1.5 text-xs text-lucid-text-low",
          next && "justify-end",
        )}
      >
        {!next && <ArrowLeft className="h-3.5 w-3.5" aria-hidden />}
        {next ? "Next" : "Previous"}
        {next && <ArrowRight className="h-3.5 w-3.5" aria-hidden />}
      </span>
      <span className="mt-1 block text-sm font-semibold text-lucid-text-high transition-colors group-hover:text-lucid-primary">
        {diagram.title}
      </span>
    </Card>
  );
}

export default async function DiagramPage({ params }: { params: Params }) {
  const diagram = getDiagram((await params).slug);
  if (!diagram) notFound();

  const all = getDiagrams();
  const index = all.findIndex((d) => d.slug === diagram.slug);
  const previous = all[index - 1];
  const next = all[index + 1];

  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main id="main-content" tabIndex={-1} className="flex-1 pt-14 focus:outline-none">
        <div className="mx-auto max-w-screen-2xl px-4 sm:px-6">
          <header className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4 pt-8 pb-5">
            <div className="min-w-0">
              <nav
                aria-label="Breadcrumb"
                className="mb-2 flex items-center gap-1.5 text-xs text-lucid-text-low"
              >
                <Link href="/architecture" className="transition-colors hover:text-lucid-text-high">
                  Architecture
                </Link>
                <span aria-hidden className="text-lucid-text-disabled">
                  /
                </span>
                <span className="text-lucid-text-mid">{diagram.group}</span>
              </nav>
              <div className="flex flex-wrap items-center gap-3">
                <h1 className="text-3xl font-bold text-lucid-text-high">{diagram.title}</h1>
                <Badge variant="secondary" className="font-mono text-[10px]">
                  {DIAGRAM_KIND_LABEL[diagram.kind]}
                </Badge>
              </div>
              <p className="mt-3 max-w-3xl text-base text-lucid-text-mid leading-relaxed">
                {diagram.summary}
              </p>
            </div>
            <DiagramSwitcher current={diagram} all={all} />
          </header>

          <DiagramViewer
            src={diagram.viewerSrc}
            title={`${diagram.title} — interactive diagram`}
            className="h-[75dvh] min-h-[560px] lg:h-[calc(100dvh-15rem)]"
          />
          <p className="mt-2 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-lucid-text-disabled">
            <span>
              {diagram.extent} · {diagram.chapters.length} guided chapters
              {diagram.sourceRevision && (
                <>
                  {" · "}
                  {diagram.sourceLinks} source links pinned to{" "}
                  <a
                    href={`https://github.com/ChanLumerico/lucid/tree/${diagram.sourceRevision}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-lucid-text-low transition-colors hover:text-lucid-primary"
                  >
                    {diagram.sourceRevision.slice(0, 7)}
                  </a>
                </>
              )}
            </span>
          </p>

          <div className="mx-auto max-w-5xl space-y-12 py-12">
            {diagram.facts.length > 0 && (
              <section>
                <SectionHeading>Key facts</SectionHeading>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
                  {diagram.facts.map((fact) => (
                    <div
                      key={fact.title}
                      className="rounded-xl border border-lucid-border bg-lucid-surface/40 px-5 py-4"
                    >
                      <div className="mb-2 flex items-center gap-2">
                        <span
                          aria-hidden
                          className={cn("h-1.5 w-1.5 shrink-0 rounded-full", FACT_TONE_DOT[fact.tone])}
                        />
                        <h3 className="text-sm font-semibold text-lucid-text-high">{fact.title}</h3>
                      </div>
                      <ul className="space-y-1.5 text-sm leading-relaxed text-lucid-text-mid">
                        {fact.items.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {diagram.chapters.length > 0 && (
              <section>
                <SectionHeading>Guided chapters</SectionHeading>
                <ol className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  {diagram.chapters.map((chapter, n) => (
                    <li
                      key={chapter.id}
                      className="flex gap-3 rounded-xl border border-lucid-border bg-lucid-surface/40 px-4 py-3"
                    >
                      <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-lucid-primary/30 bg-lucid-primary/10 font-mono text-[10px] font-semibold text-lucid-primary">
                        {n + 1}
                      </span>
                      <div className="min-w-0">
                        <p className="text-sm font-semibold text-lucid-text-high">{chapter.label}</p>
                        {chapter.note && (
                          <p className="mt-0.5 text-xs leading-relaxed text-lucid-text-low">
                            {chapter.note}
                          </p>
                        )}
                      </div>
                    </li>
                  ))}
                </ol>
              </section>
            )}

            {(previous || next) && (
              <nav aria-label="More diagrams" className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {previous && <NeighbourCard diagram={previous} direction="previous" />}
                {next && <NeighbourCard diagram={next} direction="next" />}
              </nav>
            )}
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
