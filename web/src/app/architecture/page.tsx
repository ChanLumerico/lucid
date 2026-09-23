import type { Metadata } from "next";
import type { LucideIcon } from "lucide-react";
import { GitCommitHorizontal, HardDrive, Layers, Repeat, Zap } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/Card";
import {
  DIAGRAM_KIND_LABEL,
  getDiagrams,
  type Diagram,
  type DiagramKind,
} from "@/lib/architecture";

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

/** Every card has the same rows — heading, summary, footer — so a grid row lines up. */
function DiagramCard({ diagram }: { diagram: Diagram }) {
  const Icon = KIND_ICON[diagram.kind];
  return (
    <Card href={`/architecture/${diagram.slug}`} className="h-full">
      <div className="flex h-full flex-col p-5">
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-lucid-primary/30 bg-lucid-primary/10 text-lucid-primary"
          >
            <Icon className="h-[18px] w-[18px]" />
          </span>
          <div className="min-w-0">
            <p className="text-xs font-semibold uppercase tracking-widest text-lucid-text-disabled">
              {diagram.ordinal} · {diagram.group}
            </p>
            <h2 className="text-lg font-semibold leading-snug text-lucid-text-high transition-colors group-hover:text-lucid-primary">
              {diagram.title}
            </h2>
          </div>
        </div>
        <p className="mt-3 flex-1 text-[15px] leading-relaxed text-lucid-text-mid">
          {diagram.summary}
        </p>
        <div className="mt-4 flex items-center justify-between gap-3 border-t border-lucid-border pt-3">
          <Badge variant="secondary" className="font-mono text-[11px]">
            {DIAGRAM_KIND_LABEL[diagram.kind]}
          </Badge>
          <span className="truncate font-mono text-[13px] text-lucid-text-low">{diagram.extent}</span>
        </div>
      </div>
    </Card>
  );
}

export default function ArchitecturePage() {
  const diagrams = getDiagrams();
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main id="main-content" tabIndex={-1} className="flex-1 pt-14 focus:outline-none">
        <div className="mx-auto max-w-5xl px-4 sm:px-6 py-12">
          <FadeIn>
            <header className="mb-10">
              <p className="mb-2 text-sm font-semibold uppercase tracking-widest text-lucid-text-disabled">
                Architecture
              </p>
              <h1 className="text-3xl font-bold text-lucid-text-high">How Lucid works</h1>
              <p className="mt-3 max-w-3xl text-lg leading-relaxed text-lucid-text-mid">
                Six diagrams traced from the source rather than drawn from memory — how a call
                reaches a kernel, what backward really does, where tensor bytes live. Each opens
                in a full interactive viewer: guided chapters, search, focus, route tracing, and
                export all work in place.
              </p>
            </header>

            {/* One grid in reading order, equal rows — the numbers carry the sequence. */}
            <FadeInStagger
              staggerDelay={0.04}
              className="grid grid-cols-1 gap-4 sm:auto-rows-fr sm:grid-cols-2"
            >
              {diagrams.map((d) => (
                <DiagramCard key={d.slug} diagram={d} />
              ))}
            </FadeInStagger>
          </FadeIn>
        </div>
      </main>
      <Footer />
    </div>
  );
}
