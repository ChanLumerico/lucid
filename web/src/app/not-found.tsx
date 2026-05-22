import Link from "next/link";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { Button } from "@/components/ui/button";
import { FadeIn } from "@/components/motion/FadeIn";

export const metadata = {
  title: "Page not found",
};

export default function NotFound() {
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main
        id="main-content"
        tabIndex={-1}
        className="flex-1 flex items-center justify-center px-4 pt-14 focus:outline-none"
      >
        <FadeIn>
          <div className="mx-auto max-w-md text-center">
            {/* Background glow */}
            <div className="pointer-events-none absolute inset-x-0 top-1/3 -z-10" aria-hidden>
              <div className="mx-auto h-[400px] w-[600px] -translate-y-1/2 rounded-full bg-lucid-primary/[0.04] blur-[100px]" />
            </div>

            <p className="mb-3 font-mono text-[11px] font-semibold tracking-widest uppercase text-lucid-text-disabled">
              404 · not found
            </p>
            <h1 className="font-mono text-5xl sm:text-6xl font-bold mb-4">
              <span className="gradient-hero-text">No such symbol</span>
            </h1>
            <p className="text-sm text-lucid-text-mid leading-relaxed mb-8">
              The slug you followed isn't a documented module, class, function,
              or guide.  It may have been renamed, removed, or never existed —
              the API surface is built from <code className="text-lucid-text-low">lucid/</code> at
              every prebuild, so stale links die fast.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Button asChild>
                <Link href="/api">API Reference</Link>
              </Button>
              <Button variant="secondary" asChild>
                <Link href="/docs/quickstart">Quickstart</Link>
              </Button>
            </div>
          </div>
        </FadeIn>
      </main>
      <Footer />
    </div>
  );
}
