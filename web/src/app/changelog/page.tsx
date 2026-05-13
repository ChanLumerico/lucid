import type { Metadata } from "next";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";

export const metadata: Metadata = {
  title: "Changelog",
  description: "What's new in Lucid.",
};

export default function ChangelogPage() {
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main className="flex-1 pt-14">
        <div className="mx-auto max-w-3xl px-4 sm:px-6 py-12">
          <FadeIn>
            <h1 className="text-3xl font-bold text-lucid-text-high mb-3">
              Changelog
            </h1>
            <p className="text-lucid-text-mid mb-12">
              A record of all notable changes to Lucid.
            </p>

            {/* Placeholder entry */}
            <div className="relative pl-6 before:absolute before:left-0 before:top-0 before:bottom-0 before:w-px before:bg-lucid-border">
              <div className="absolute left-[-4px] top-1.5 h-2 w-2 rounded-full bg-lucid-primary ring-2 ring-lucid-bg" />
              <div className="mb-1.5 flex flex-wrap items-center gap-2">
                <time className="text-sm font-semibold text-lucid-text-high">
                  v3.0.0
                </time>
                <Badge variant="default">Latest</Badge>
                <span className="text-xs text-lucid-text-disabled">2026-05-12</span>
              </div>
              <p className="text-sm text-lucid-text-mid leading-relaxed">
                Full changelog will be populated in Phase 4 (MDX content).
              </p>
            </div>
          </FadeIn>
        </div>
      </main>
      <Footer />
    </div>
  );
}
