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

/** Columns per item count that never leave a card alone on a row. */
const BALANCED_COLS: Record<number, string> = {
  1: "grid-cols-1",
  2: "grid-cols-1 sm:grid-cols-2",
  3: "grid-cols-1 lg:grid-cols-3",
  4: "grid-cols-1 sm:grid-cols-2 xl:grid-cols-4",
};

function balancedCols(n: number): string {
  return BALANCED_COLS[n] ?? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3";
}

/** Equal segments, one per diagram — the same slots on every diagram page. */
function DiagramSwitcher({
  current,
  all,
  className,
}: {
  current: Diagram;
  all: Diagram[];
  className?: string;
}) {
  return (
    <nav
      aria-label="Diagrams"
      className={cn(
        "grid grid-cols-3 gap-1 rounded-xl border border-lucid-border bg-lucid-surface/40 p-1 md:grid-cols-6",
        className,
      )}
    >
      {all.map((d) => {
        const active = d.slug === current.slug;
        return (
          <Link
            key={d.slug}
            href={`/architecture/${d.slug}`}
            aria-current={active ? "page" : undefined}
            className={cn(
              "truncate rounded-lg px-3 py-1.5 text-center text-sm font-medium transition-colors",
              active
                ? "bg-lucid-primary/10 text-lucid-primary ring-1 ring-inset ring-lucid-primary/30"
                : "text-lucid-text-mid hover:bg-lucid-surface hover:text-lucid-text-high",
            )}
          >
            {d.label}
          </Link>
        );
      })}
    </nav>
  );
}

function NeighbourCard({
  href,
  label,
  title,
  direction,
}: {
  href: string;
  label: string;
  title: string;
  direction: "previous" | "next";
}) {
  const next = direction === "next";
  return (
    <Card href={href} className="h-full">
      <div className={cn("flex h-full flex-col px-5 py-4", next && "items-end text-right")}>
        <span className="flex items-center gap-1.5 text-sm text-lucid-text-low">
          {!next && <ArrowLeft className="h-4 w-4" aria-hidden />}
          {label}
          {next && <ArrowRight className="h-4 w-4" aria-hidden />}
        </span>
        <span className="mt-1 text-base font-semibold text-lucid-text-high transition-colors group-hover:text-lucid-primary">
          {title}
        </span>
      </div>
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
          {/* The switcher closes the header, right above the viewer; from xl it
              moves up beside the title, so the viewer starts as high as it can. */}
          <header className="grid grid-cols-1 pt-6 pb-4 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center xl:gap-x-8">
            <nav
              aria-label="Breadcrumb"
              className="mb-2 flex items-center gap-1.5 text-sm text-lucid-text-low xl:col-span-2"
            >
              <Link href="/architecture" className="transition-colors hover:text-lucid-text-high">
                Architecture
              </Link>
              <span aria-hidden className="text-lucid-text-disabled">
                /
              </span>
              <span className="text-lucid-text-mid">{diagram.group}</span>
            </nav>
            <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2 xl:col-start-1 xl:row-start-2">
              <h1 className="text-3xl font-bold text-lucid-text-high">{diagram.title}</h1>
              <Badge variant="secondary" className="font-mono text-[11px]">
                {DIAGRAM_KIND_LABEL[diagram.kind]}
              </Badge>
            </div>
            {/* Two lines reserved, so moving between diagrams never shifts the viewer. */}
            <p className="mt-3 max-w-3xl text-lg leading-relaxed text-lucid-text-mid md:min-h-[3.65625rem] xl:col-span-2 xl:row-start-3">
              {diagram.summary}
            </p>
            <DiagramSwitcher
              current={diagram}
              all={all}
              className="mt-5 xl:col-start-2 xl:row-start-2 xl:mt-0"
            />
          </header>

          <DiagramViewer
            src={diagram.viewerSrc}
            title={`${diagram.title} — interactive diagram`}
            className="h-[75dvh] min-h-[560px] lg:h-[calc(100dvh-19.5rem)] xl:h-[calc(100dvh-16rem)]"
          />
          <div className="mt-3 flex flex-wrap items-center justify-between gap-x-6 gap-y-1 text-sm text-lucid-text-low">
            <span>
              {diagram.extent} · {diagram.chapters.length} guided chapters
            </span>
            {diagram.sourceRevision && (
              <span>
                {diagram.sourceLinks} source links pinned to{" "}
                <a
                  href={`https://github.com/ChanLumerico/lucid/tree/${diagram.sourceRevision}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-mono text-lucid-text-mid transition-colors hover:text-lucid-primary"
                >
                  {diagram.sourceRevision.slice(0, 7)}
                </a>
              </span>
            )}
          </div>

          <div className="space-y-12 pt-12 pb-16">
            {diagram.facts.length > 0 && (
              <section>
                <SectionHeading className="text-sm">Key facts</SectionHeading>
                <div className={cn("grid gap-4", balancedCols(diagram.facts.length))}>
                  {diagram.facts.map((fact) => (
                    <div
                      key={fact.title}
                      className="rounded-xl border border-lucid-border bg-lucid-surface/40 p-5"
                    >
                      <div className="mb-3 flex items-center gap-2.5">
                        <span aria-hidden className="flex w-2 shrink-0 justify-center">
                          <span className={cn("h-2 w-2 rounded-full", FACT_TONE_DOT[fact.tone])} />
                        </span>
                        <h3 className="text-base font-semibold text-lucid-text-high">{fact.title}</h3>
                      </div>
                      {/* Bullets share the tone dot's column, so item text lines up with the title. */}
                      <ul className="space-y-2 text-[15px] leading-relaxed text-lucid-text-mid">
                        {fact.items.map((item) => (
                          <li key={item} className="flex gap-2.5">
                            <span aria-hidden className="flex w-2 shrink-0 justify-center pt-2.5">
                              <span className="h-1 w-1 rounded-full bg-lucid-text-disabled" />
                            </span>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {diagram.chapters.length > 0 && (
              <section>
                <SectionHeading className="text-sm">Guided chapters</SectionHeading>
                <ol className={cn("grid gap-4", balancedCols(diagram.chapters.length))}>
                  {diagram.chapters.map((chapter, n) => (
                    <li
                      key={chapter.id}
                      className="rounded-xl border border-lucid-border bg-lucid-surface/40 p-5"
                    >
                      <div className="flex items-center gap-2.5">
                        <span className="inline-flex h-6 w-7 shrink-0 items-center justify-center rounded-md border border-lucid-primary/30 bg-lucid-primary/10 font-mono text-xs font-semibold text-lucid-primary">
                          {String(n + 1).padStart(2, "0")}
                        </span>
                        <h3 className="text-base font-semibold text-lucid-text-high">{chapter.label}</h3>
                      </div>
                      {/* Indented past the number (w-7 + gap-2.5), so the note lines up with the title. */}
                      {chapter.note && (
                        <p className="mt-2 pl-[2.375rem] text-[15px] leading-relaxed text-lucid-text-mid">
                          {chapter.note}
                        </p>
                      )}
                    </li>
                  ))}
                </ol>
              </section>
            )}

            {/* Both ends always filled: the first and last diagram lead back to the overview. */}
            <nav aria-label="More diagrams" className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <NeighbourCard
                direction="previous"
                href={previous ? `/architecture/${previous.slug}` : "/architecture"}
                label={previous ? "Previous" : "Overview"}
                title={previous ? previous.title : "All diagrams"}
              />
              <NeighbourCard
                direction="next"
                href={next ? `/architecture/${next.slug}` : "/architecture"}
                label={next ? "Next" : "Overview"}
                title={next ? next.title : "All diagrams"}
              />
            </nav>
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
