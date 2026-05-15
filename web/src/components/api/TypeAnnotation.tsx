import { cn } from "@/lib/utils";

interface TypeAnnotationProps {
  annotation: string | null;
  className?: string;
}

export function TypeAnnotation({ annotation, className }: TypeAnnotationProps) {
  if (!annotation) return null;

  // Colorise known primitive types
  const colored = annotation
    .replace(/\b(int|float|bool|str|bytes|None|Any)\b/g, '<span class="text-lucid-blue-light">$1</span>')
    .replace(/\b(Tensor)\b/g, '<span class="text-lucid-primary-light">$1</span>')
    .replace(/\b(list|dict|tuple|set|frozenset|Sequence|Iterable|Callable|Optional|Union)\b/g,
             '<span class="text-lucid-text-mid">$1</span>');

  return (
    <code
      className={cn(
        "font-mono text-xs text-lucid-blue-light",
        className,
      )}
      dangerouslySetInnerHTML={{ __html: colored }}
    />
  );
}
