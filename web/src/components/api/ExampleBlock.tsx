import { highlight } from "@/lib/shiki";
import { cn } from "@/lib/utils";

interface ExampleBlockProps {
  examples: string[];
  className?: string;
}

export async function ExampleBlock({ examples, className }: ExampleBlockProps) {
  if (!examples.length) return null;

  const rendered = await Promise.all(
    examples.map((ex) => highlight(ex, "python")),
  );

  return (
    <div className={cn("space-y-3", className)}>
      <h4 className="text-xs font-semibold tracking-widest text-lucid-text-disabled uppercase">
        Examples
      </h4>
      {rendered.map((html, i) => (
        <div
          key={i}
          className={cn(
            "rounded-xl border border-lucid-border bg-lucid-surface",
            "overflow-hidden text-sm",
            "[&_pre]:p-4 [&_pre]:overflow-x-auto [&_pre]:!bg-transparent",
            "[&_code]:font-mono [&_code]:text-xs [&_code]:leading-relaxed",
          )}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ))}
    </div>
  );
}
