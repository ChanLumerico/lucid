import type { Metadata, Viewport } from "next";
import "./globals.css";
import "katex/dist/katex.min.css";

export const viewport: Viewport = {
  themeColor: "#0d0c15",
  colorScheme: "dark",
};

export const metadata: Metadata = {
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
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
