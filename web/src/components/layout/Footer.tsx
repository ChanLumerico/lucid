import Link from "next/link";
import { Separator } from "@/components/ui/separator";

const FOOTER_LINKS = [
  {
    title: "Documentation",
    links: [
      { label: "Quickstart", href: "/docs/quickstart" },
      { label: "Installation", href: "/docs/installation" },
      { label: "Autograd", href: "/docs/autograd" },
      { label: "Metal Device", href: "/docs/metal-device" },
    ],
  },
  {
    title: "API Reference",
    links: [
      { label: "lucid.Tensor", href: "/api/lucid.tensor" },
      { label: "lucid.nn", href: "/api/lucid.nn" },
      { label: "lucid.optim", href: "/api/lucid.optim" },
      { label: "lucid.autograd", href: "/api/lucid.autograd" },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "GitHub", href: "https://github.com/ChanLumerico/lucid", external: true },
      { label: "Changelog", href: "/changelog" },
    ],
  },
] as const;

export function Footer() {
  return (
    <footer
      className="border-t border-lucid-border bg-lucid-bg mt-auto"
      aria-label="Site footer"
    >
      <div className="mx-auto max-w-screen-xl px-4 sm:px-6 py-12">
        <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {/* Brand */}
          <div className="lg:col-span-1">
            <Link
              href="/"
              className="inline-flex items-center gap-2 text-base font-bold text-lucid-text-high hover:text-lucid-primary transition-colors"
              aria-label="Lucid home"
            >
              Lucid
              <span className="rounded-md border border-lucid-primary/35 bg-lucid-primary/10 px-1.5 py-0.5 text-[9px] font-bold tracking-[0.12em] text-lucid-primary uppercase">
                3.0
              </span>
            </Link>
            <p className="mt-3 text-sm text-lucid-text-low leading-relaxed max-w-xs">
              Production-grade ML framework for Apple Silicon.
              MLX + Accelerate native backend.
            </p>
          </div>

          {/* Links */}
          {FOOTER_LINKS.map(({ title, links }) => (
            <div key={title}>
              <h3 className="text-xs font-semibold tracking-widest text-lucid-text-low uppercase mb-3">
                {title}
              </h3>
              <ul className="space-y-2">
                {links.map(({ label, href, ...rest }) => {
                  const isExternal = "external" in rest && rest.external;
                  return (
                    <li key={href}>
                      <Link
                        href={href}
                        target={isExternal ? "_blank" : undefined}
                        rel={isExternal ? "noopener noreferrer" : undefined}
                        className="text-sm text-lucid-text-low hover:text-lucid-text-mid transition-colors"
                      >
                        {label}
                        {isExternal && (
                          <span className="sr-only">(opens in new tab)</span>
                        )}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>

        <Separator className="my-8 bg-lucid-border" />

        <div className="flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
          <p className="text-xs text-lucid-text-disabled">
            © {new Date().getFullYear()} Lucid. Built for Apple Silicon.
          </p>
          <p className="text-xs text-lucid-text-disabled">
            Python 3.14+ · macOS arm64
          </p>
        </div>
      </div>
    </footer>
  );
}
