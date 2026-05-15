import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn, FadeInStagger } from "@/components/motion/FadeIn";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";

const FEATURES = [
  {
    icon: "⚡",
    title: "MLX Native GPU",
    description:
      "Metal-accelerated compute on every forward and backward pass. No CUDA, no compromise — purpose-built for Apple Silicon.",
  },
  {
    icon: "🧠",
    title: "Accelerate CPU Kernels",
    description:
      "vDSP, vForce, and BLAS/LAPACK from Apple's Accelerate framework power the CPU stream. Zero third-party dependencies.",
  },
  {
    icon: "🔧",
    title: "PyTorch-compatible API",
    description:
      "Familiar interface — nn.Module, autograd, optim, DataLoader — so you can focus on the model, not the framework.",
  },
  {
    icon: "📦",
    title: "314 Top-level Ops",
    description:
      "Comprehensive operator coverage: linalg, fft, einops, distributions, signal, special math, and more.",
  },
] as const;

const INSTALL_CMD = "pip install lucid";

function InstallSnippet() {
  return (
    <div className="relative overflow-hidden rounded-xl border border-lucid-border bg-lucid-surface px-5 py-3.5">
      <div className="flex items-center gap-3">
        <span className="select-none text-lucid-text-disabled text-sm">$</span>
        <code className="font-mono text-sm text-lucid-text-high">{INSTALL_CMD}</code>
      </div>
      <div className="absolute right-3 top-1/2 -translate-y-1/2">
        <Badge variant="secondary" className="text-[10px]">Python 3.14+</Badge>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />

      <main className="flex-1">
        {/* Hero */}
        <section className="relative flex min-h-[calc(100dvh-3.5rem)] items-center justify-center overflow-hidden pt-14">
          {/* Background glow */}
          <div
            className="pointer-events-none absolute inset-0 -z-10"
            aria-hidden="true"
          >
            <div className="absolute left-1/2 top-1/3 -translate-x-1/2 -translate-y-1/2 h-[600px] w-[900px] rounded-full bg-lucid-primary/[0.06] blur-[120px]" />
            <div className="absolute left-1/3 top-2/3 -translate-x-1/2 -translate-y-1/2 h-[400px] w-[600px] rounded-full bg-lucid-blue/[0.05] blur-[100px]" />
          </div>

          <div className="mx-auto max-w-4xl px-4 sm:px-6 text-center">
            <FadeIn delay={0}>
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-lucid-primary/25 bg-lucid-primary/10 px-4 py-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-lucid-primary animate-pulse" />
                <span className="text-xs font-medium text-lucid-primary tracking-wide">
                  Lucid 3.0 — Now available
                </span>
              </div>
            </FadeIn>

            <FadeIn delay={0.05}>
              <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.08] mb-6">
                <span className="gradient-hero-text">Production ML</span>
                <br />
                <span className="text-lucid-text-high">for Apple Silicon</span>
              </h1>
            </FadeIn>

            <FadeIn delay={0.1}>
              <p className="mx-auto max-w-2xl text-lg sm:text-xl text-lucid-text-mid leading-relaxed mb-10">
                MLX + Accelerate native backend. PyTorch-compatible API.
                Zero third-party dependencies in the compute path.
              </p>
            </FadeIn>

            <FadeIn delay={0.15}>
              <div className="flex flex-col items-center gap-4">
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <Button size="lg" asChild>
                    <Link href="/docs/quickstart">Get Started</Link>
                  </Button>
                  <Button variant="secondary" size="lg" asChild>
                    <Link href="/api">API Reference</Link>
                  </Button>
                </div>
                <div className="w-full max-w-sm">
                  <InstallSnippet />
                </div>
              </div>
            </FadeIn>
          </div>
        </section>

        {/* Features */}
        <section className="mx-auto max-w-screen-xl px-4 sm:px-6 py-24">
          <FadeIn inView>
            <div className="mb-12 text-center">
              <h2 className="text-3xl sm:text-4xl font-bold text-lucid-text-high mb-3">
                Built for the hardware
              </h2>
              <p className="text-lucid-text-low max-w-xl mx-auto">
                Every layer of the stack is optimized for Apple Silicon's unified memory architecture.
              </p>
            </div>
          </FadeIn>

          <FadeInStagger
            inView
            staggerDelay={0.08}
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            {FEATURES.map(({ icon, title, description }) => (
              <div
                key={title}
                className="group relative overflow-hidden rounded-2xl border border-lucid-border bg-lucid-surface p-6 transition-all duration-200 hover:border-lucid-primary/40 hover:bg-lucid-elevated"
              >
                <div className="mb-4 text-3xl">{icon}</div>
                <h3 className="mb-2 text-base font-semibold text-lucid-text-high">
                  {title}
                </h3>
                <p className="text-sm text-lucid-text-low leading-relaxed">
                  {description}
                </p>
                <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-br from-lucid-primary/[0.03] to-transparent opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
              </div>
            ))}
          </FadeInStagger>
        </section>
      </main>

      <Footer />
    </div>
  );
}
