import path from "path";
import fs from "fs";
import matter from "gray-matter";
import { compileMDX } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeShiki from "@shikijs/rehype";
import type { MDXComponents } from "mdx/types";

const CONTENT_DIR = path.join(process.cwd(), "content");

export interface DocFrontmatter {
  title: string;
  description?: string;
  category?: string;
  order?: number;
}

export interface DocMeta extends DocFrontmatter {
  slug: string;
  href: string;
}

function resolveDocPath(slug: string[]): string {
  return path.join(CONTENT_DIR, ...slug) + ".mdx";
}

export function getAllDocSlugs(): string[][] {
  const slugs: string[][] = [];

  function walk(dir: string, prefix: string[]) {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir)) {
      const full = path.join(dir, entry);
      if (fs.statSync(full).isDirectory()) {
        walk(full, [...prefix, entry]);
      } else if (entry.endsWith(".mdx")) {
        slugs.push([...prefix, entry.replace(/\.mdx$/, "")]);
      }
    }
  }

  walk(CONTENT_DIR, []);
  return slugs;
}

export function getAllDocMeta(): DocMeta[] {
  return getAllDocSlugs().map((slug) => {
    const filePath = resolveDocPath(slug);
    const raw = fs.readFileSync(filePath, "utf-8");
    const { data } = matter(raw);
    const fm = data as Partial<DocFrontmatter>;
    return {
      title: fm.title ?? slug[slug.length - 1],
      description: fm.description,
      category: fm.category,
      order: fm.order ?? 999,
      slug: slug.join("/"),
      href: "/docs/" + slug.join("/"),
    };
  });
}

export async function compileDoc(
  slug: string[],
  components: MDXComponents = {},
) {
  const filePath = resolveDocPath(slug);

  if (!fs.existsSync(filePath)) return null;

  const raw = fs.readFileSync(filePath, "utf-8");
  const { content, data } = matter(raw);
  const fm = data as Partial<DocFrontmatter>;

  const { content: mdxContent } = await compileMDX<DocFrontmatter>({
    source: content,
    components,
    options: {
      parseFrontmatter: false,
      mdxOptions: {
        remarkPlugins: [remarkGfm, remarkMath],
        rehypePlugins: [
          rehypeKatex,
          [
            rehypeShiki,
            {
              theme: "one-dark-pro",
            },
          ],
        ],
      },
    },
  });

  return {
    content: mdxContent,
    frontmatter: {
      title: fm.title ?? slug[slug.length - 1],
      description: fm.description,
      category: fm.category,
      order: fm.order ?? 999,
    } satisfies DocFrontmatter,
  };
}
