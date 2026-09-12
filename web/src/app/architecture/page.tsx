import type { Metadata } from "next";
import type { LucideIcon } from "lucide-react";
import { GitCommitHorizontal, HardDrive, Layers, Repeat, Zap } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/Card";
import { SectionHeading } from "@/components/ui/SectionHeading";
import {
  DIAGRAM_KIND_LABEL,
  getDiagramGroups,
  type Diagram,
  type DiagramKind,
} from "@/lib/architecture";
import { cn } from "@/lib/utils";

export const metadata: Metadata = {
  title: "Architecture",
  description: "Interactive diagrams of how Lucid works, traced from the source.",
};

const KIND_ICON: Record<DiagramKind, LucideIcon> = {
  architecture: Layers,
  sequence: Zap,
  workflow: Repeat,
  dataflow: HardDrive,
  lifecycle: GitCommitHorizontal,
};

const ACCENT = "var(--color-lucid-primary)";

function DiagramCard({ diagram }: { diagram: Diagram }) {
  const Icon = KIND_ICON[diagram.kind];
  return (
    <Card href={`/architecture/${diagram.slug}`}>
      <div className="px-5 pt-4 pb-4 flex items-start gap-3">
        <span
          className="shrink-0 inline-flex h-9 w-9 items-center justify-center rounded-lg border"
          style={{
            backgroundColor: `color-mix(in srgb, ${ACCENT} 14%, transparent)`,
            borderColor: `color-mix(in srgb, ${ACCENT} 38%, transparent)`,
            color: ACCENT,
          }}
          aria-hidden
        >
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-2">
            <h3 className="text-sm font-semibold text-lucid-text-high transition-colors group-hover:text-lucid-primary">
              {diagram.title}
            </h3>
            <Badge variant="secondary" className="font-mono text-[10px]">
              {DIAGRAM_KIND_LABEL[diagram.kind]}
            </Badge>
          </div>
          <p className="mt-1 text-xs text-lucid-text-low leading-relaxed">{diagram.summary}</p>
          <p className="mt-2 font-mono text-[10px] text-lucid-text-disabled">
            {diagram.extent} · {diagram.chapters.length} chapters
            {diagram.sourceLinks > 0 && ` · ${diagram.sourceLinks} source links`}
          </p>
        </div>
      </div>
    </Card>
  );
}

export default function ArchitecturePage() {
  const groups = getDiagramGroups();
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main id="main-content" tabIndex={-1} className="flex-1 pt-14 focus:outline-none">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 py-12">
          <FadeIn>
            <header className="mb-12">
              <div className="flex flex-wrap items-center gap-3 mb-3">
                <span className="text-xs font-semibold tracking-widest uppercase text-lucid-text-disabled">
                  Architecture
                </span>
                <h1 className="text-3xl font-bold text-lucid-text-high">How Lucid works</h1>
              </div>
              <p className="max-w-3xl text-base text-lucid-text-mid leading-relaxed">
                Six diagrams traced from the source rather than drawn from memory — how a call
                reaches a kernel, what backward really does, where tensor bytes live. Each opens
                in a full interactive viewer: guided chapters, search, focus, route tracing, and
                export all work in place.
              </p>
            </header>

            <div className="space-y-10">
              {groups.map(([group, diagrams]) => (
                <section key={group}>
                  <SectionHeading>{group}</SectionHeading>
                  <FadeInStagger
                    staggerDelay={0.04}
                    className={cn(
                      "grid gap-3",
                      diagrams.length === 1 ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2",
                    )}
                  >
                    {diagrams.map((d) => (
                      <DiagramCard key={d.slug} diagram={d} />
                    ))}
                  </FadeInStagger>
                </section>
              ))}
            </div>
          </FadeIn>
        </div>
      </main>
      <Footer />
    </div>
  );
}
