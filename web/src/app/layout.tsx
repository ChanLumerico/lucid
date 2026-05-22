import type { Metadata, Viewport } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";
import { SkipLink } from "@/components/layout/SkipLink";
import { ScrollProgress } from "@/components/layout/ScrollProgress";
import { BackToTop } from "@/components/layout/BackToTop";
import { ReducedMotionProvider } from "@/components/motion/ReducedMotionProvider";
import { ThemeBoot, ThemeProvider } from "@/components/layout/ThemeProvider";
import { RecentPageTracker } from "@/components/layout/RecentPageTracker";
import { InlineCodeCopy } from "@/components/layout/InlineCodeCopy";
import { RouteProgress } from "@/components/layout/RouteProgress";

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)",  color: "#0d0c15" },
    { media: "(prefers-color-scheme: light)", color: "#fdfcfb" },
  ],
  colorScheme: "dark light",
};

const DOCS_ORIGIN =
  process.env.LUCID_DOCS_ORIGIN ??
  process.env.NEXT_PUBLIC_LUCID_DOCS_ORIGIN ??
  "https://lucid.docs.local";

export const metadata: Metadata = {
  // metadataBase is used by Next to absolutise any relative URL — most
  // importantly the auto-generated ``opengraph-image`` / ``twitter-image``
  // routes — when emitting social-preview tags.  Without it, crawlers
  // see relative ``/og.png`` paths and skip them.
  metadataBase: new URL(DOCS_ORIGIN),
  title: {
    default: "Lucid — Apple Silicon ML Framework",
    template: "%s | Lucid",
  },
  description:
    "Production-grade machine learning framework for Apple Silicon. MLX + Accelerate native backend with a PyTorch-compatible API.",
  keywords: [
    "machine learning",
    "Apple Silicon",
    "MLX",
    "Accelerate",
    "ML framework",
    "Python",
    "deep learning",
  ],
  authors: [{ name: "Chan Lee" }],
  creator: "Chan Lee",
  openGraph: {
    type: "website",
    locale: "en_US",
    title: "Lucid — Apple Silicon ML Framework",
    description:
      "Production-grade ML framework for Apple Silicon. MLX + Accelerate native backend.",
    siteName: "Lucid Docs",
    url: DOCS_ORIGIN,
  },
  twitter: {
    card: "summary_large_image",
    title: "Lucid — Apple Silicon ML Framework",
    description:
      "Production-grade ML framework for Apple Silicon. MLX + Accelerate native backend.",
  },
  robots: {
    index: true,
    follow: true,
  },
  alternates: {
    canonical: "/",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* ThemeBoot runs SYNCHRONOUSLY before React hydrates so the
            user's persisted theme is applied without a dark/light flash. */}
        <ThemeBoot />
      </head>
      <body>
        <SkipLink />
        <ScrollProgress />
        <RecentPageTracker />
        <InlineCodeCopy />
        <RouteProgress />
        <ThemeProvider>
          <ReducedMotionProvider>{children}</ReducedMotionProvider>
        </ThemeProvider>
        <BackToTop />
      </body>
    </html>
  );
}
