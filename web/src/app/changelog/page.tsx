import fs from "fs";
import path from "path";
import type { Metadata } from "next";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import { FadeIn } from "@/components/motion/FadeIn";
import { ChangelogAccordion, type ChangelogVersion } from "@/components/layout/ChangelogAccordion";

export const metadata: Metadata = {
  title: "Changelog",
  description: "What's new in Lucid.",
};

function parseChangelog(raw: string): ChangelogVersion[] {
  const content = raw.replace(/^\[[^\]]+\]:\s*https?:\/\/.+$/gm, "").trim();
  const sections = content.split(/^(?=## )/m).filter((s) => s.startsWith("## "));

  return sections.map((section) => {
    const lines = section.split("\n");
    const headerLine = lines[0].replace(/^## /, "").trim();

    const m = headerLine.match(/^\[?([^\]\n]+)\]?(?:\s*[—–-]+\s*(.+))?/);
    const version = m?.[1]?.trim() ?? headerLine;
    const date = m?.[2]?.trim() ?? null;
    const isUnreleased = version.toLowerCase() === "unreleased";
    const isPreRelease = version.toLowerCase().startsWith("pre");

    const body = lines.slice(1).join("\n");
    const catParts = body.split(/^### /m);
    const description = catParts[0].replace(/^---+$/m, "").trim();

    const categories: ChangelogVersion["categories"] = [];
    for (const catPart of catParts.slice(1)) {
      const catLines = catPart.split("\n");
      const catName = catLines[0].trim();
      const items: string[] = [];
      let current = "";
      for (const line of catLines.slice(1)) {
        if (line.startsWith("- ")) {
          if (current) items.push(current.trim());
          current = line.slice(2);
        } else if (current && line.trim() && !line.startsWith("#")) {
          current += " " + line.trim();
        }
      }
      if (current) items.push(current.trim());
      if (items.length > 0) categories.push({ name: catName, items });
    }

    return { version, date, isUnreleased, isPreRelease, description, categories };
  });
}

export default async function ChangelogPage() {
  const filePath = path.join(process.cwd(), "..", "CHANGELOG.md");
  const raw = fs.existsSync(filePath) ? fs.readFileSync(filePath, "utf-8") : "";
  const entries = parseChangelog(raw);

  return (
    <div className="flex min-h-dvh flex-col">
      <Header />
      <main className="flex-1 pt-14">
        <div className="mx-auto max-w-2xl px-4 sm:px-6 py-12">
          <FadeIn>
            <h1 className="text-3xl font-bold text-lucid-text-high mb-1">
              Changelog
            </h1>
            <p className="mb-10 text-sm text-lucid-text-low">
              All notable changes to Lucid.{" "}
              <a
                href="https://keepachangelog.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-lucid-primary/80 underline underline-offset-4 transition-colors hover:text-lucid-primary"
              >
                Keep a Changelog
              </a>{" "}
              format.
            </p>

            <ChangelogAccordion entries={entries} />
          </FadeIn>
        </div>
      </main>
      <Footer />
    </div>
  );
}
