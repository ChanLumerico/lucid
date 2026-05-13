import type { Metadata } from "next";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn } from "@/components/motion/FadeIn";
import { Badge } from "@/components/ui/badge";

export const metadata: Metadata = {
  title: "Changelog",
  description: "What's new in Lucid.",
};

interface ChangeEntry {
  version: string;
  date: string;
  tag?: "latest" | "major";
  items: { kind: "feat" | "fix" | "refactor" | "chore"; text: string }[];
}

const CHANGELOG: ChangeEntry[] = [
  {
    version: "v3.0.0",
    date: "2026-05-12",
    tag: "latest",
    items: [
      { kind: "feat", text: "Full MLX GPU + Apple Accelerate CPU backend bifurcation" },
      { kind: "feat", text: "314 top-level ops — linalg, fft, signal, special, distributions, einops, amp, profiler" },
      { kind: "feat", text: "Model zoo: 10+ vision families (ViT, Swin-V2, EfficientNetV2, CaiT, DeiT, NFNet, …)" },
      { kind: "feat", text: "Functional transforms: grad, vmap, vjp, jvp, jacrev, jacfwd, hessian" },
      { kind: "feat", text: "Full parity test suite — 1169 tests passing" },
      { kind: "refactor", text: "Hard Rules H1–H10 enforced throughout — strict type hints, no forward refs" },
      { kind: "chore", text: "Sphinx docs retired — replaced with Next.js + Griffe site" },
    ],
  },
  {
    version: "v2.1.0",
    date: "2026-03-15",
    items: [
      { kind: "feat", text: "torch.fft 22-function rollout — rfft, irfft, fftshift, fftfreq, …" },
      { kind: "feat", text: "AMP (autocast + GradScaler) — experimental" },
      { kind: "fix", text: "MobileNetV2 head always 1280 channels (not width-scaled)" },
    ],
  },
  {
    version: "v2.0.0",
    date: "2026-01-01",
    tag: "major",
    items: [
      { kind: "feat", text: "Initial Apple Silicon–only release. CPU = Accelerate, GPU = MLX" },
      { kind: "feat", text: "nn.Module, autograd engine, Adam / SGD optimizers" },
      { kind: "feat", text: "ResNet, MobileNet, VGG model families" },
    ],
  },
];

const KIND_COLOR = {
  feat: "text-lucid-success",
  fix: "text-lucid-warning",
  refactor: "text-lucid-blue",
  chore: "text-lucid-text-low",
} as const;

export default function ChangelogPage() {
  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main className="flex-1 pt-14">
        <div className="mx-auto max-w-2xl px-4 sm:px-6 py-12">
          <FadeIn>
            <h1 className="text-3xl font-bold text-lucid-text-high mb-2">
              Changelog
            </h1>
            <p className="text-lucid-text-mid mb-12 text-sm leading-relaxed">
              A record of all notable changes to Lucid.
            </p>

            <div className="relative space-y-10 before:absolute before:left-0 before:top-2 before:bottom-2 before:w-px before:bg-lucid-border">
              {CHANGELOG.map(({ version, date, tag, items }) => (
                <div key={version} className="relative pl-6">
                  <div className="absolute left-[-3.5px] top-1.5 h-2 w-2 rounded-full bg-lucid-primary ring-2 ring-lucid-bg" />
                  <div className="mb-3 flex flex-wrap items-center gap-2">
                    <span className="text-base font-bold text-lucid-text-high">
                      {version}
                    </span>
                    {tag === "latest" && <Badge variant="default">Latest</Badge>}
                    {tag === "major" && <Badge variant="secondary">Major</Badge>}
                    <time className="text-xs text-lucid-text-disabled">{date}</time>
                  </div>
                  <ul className="space-y-1.5">
                    {items.map(({ kind, text }, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm">
                        <span
                          className={`shrink-0 font-mono text-[10px] font-semibold uppercase tracking-wider mt-[3px] w-16 ${KIND_COLOR[kind]}`}
                        >
                          {kind}
                        </span>
                        <span className="text-lucid-text-mid leading-relaxed">
                          {text}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </FadeIn>
        </div>
      </main>
      <Footer />
    </div>
  );
}
