import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  cppFor,
  pythonFor,
  cppHref,
  pythonHref,
  type CrossLinkCppRef,
  type CrossLinkPythonRef,
} from "@/lib/cross-links";

const ENGINE_SLUG = "lucid._C.engine";

interface CrossLinkPanelProps {
  /** Slug of the module the current member lives in (e.g. ``"lucid.nn"`` or
   *  ``"lucid._C.engine"``).  The slug determines which direction of the
   *  cross-link table we consult and how the panel is labelled. */
  moduleSlug: string;
  /** Bare member name as it appears in the detail-page URL.  Combined with
   *  ``moduleSlug`` to form the Python path used as a lookup key. */
  memberName: string;
}

/** Show the corresponding symbols on the other side of the pybind11
 *  boundary — a Python class points at its C++ backward node + free
 *  function, a C++ engine class points at the Python wrappers (often
 *  several, e.g. ``Conv2dBackward`` → ``Conv2d`` + ``conv2d``).
 *
 *  Renders nothing when the cross-link table has no match — better to
 *  hide the panel than show a misleadingly empty rail. */
export function CrossLinkPanel({ moduleSlug, memberName }: CrossLinkPanelProps) {
  const isEngine = moduleSlug === ENGINE_SLUG;

  if (isEngine) {
    const pyRefs = pythonFor(memberName);
    if (pyRefs.length === 0) return null;
    return (
      <Panel
        title="Python wrappers"
        accent="primary"
        helpText={`Public Python APIs implemented by this engine symbol${pyRefs.length > 1 ? "s" : ""}.`}
      >
        {pyRefs.map((pyRef) => (
          <PythonRefRow key={pyRef.path} pyRef={pyRef} />
        ))}
      </Panel>
    );
  }

  const cppRefs = cppFor(`${moduleSlug}.${memberName}`);
  if (cppRefs.length === 0) return null;
  return (
    <Panel
      title="Implementing kernel"
      accent="blue"
      helpText="C++ engine symbols that back this Python API."
    >
      {cppRefs.map((ref) => (
        <CppRefRow key={ref.name} cppRef={ref} />
      ))}
    </Panel>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface PanelProps {
  title: string;
  helpText: string;
  /** ``primary`` (violet) for cpp→python (the user is on a C++ page),
   *  ``blue`` for python→cpp.  Subtle visual signal that this panel
   *  bridges two surfaces. */
  accent: "primary" | "blue";
  children: React.ReactNode;
}

function Panel({ title, helpText, accent, children }: PanelProps) {
  const borderClass = accent === "primary"
    ? "border-lucid-primary/30"
    : "border-lucid-blue/30";
  const dotClass = accent === "primary"
    ? "bg-lucid-primary"
    : "bg-lucid-blue";
  return (
    <section
      className={cn(
        "rounded-xl border bg-lucid-surface/40 px-4 py-3",
        borderClass,
      )}
      aria-label={`Cross-link: ${title}`}
    >
      <div className="flex items-center gap-2 mb-2">
        <span
          className={cn("h-1.5 w-1.5 shrink-0 rounded-full", dotClass)}
          aria-hidden
        />
        <h3 className="text-[11px] font-semibold tracking-widest uppercase text-lucid-text-disabled">
          {title}
        </h3>
        <span className="sr-only">{helpText}</span>
      </div>
      <ul className="space-y-1.5">{children}</ul>
    </section>
  );
}

function CppRefRow({ cppRef }: { cppRef: CrossLinkCppRef }) {
  const kindLabel = cppRef.kind === "backward_node" ? "class" : "free fn";
  return (
    <li>
      <Link
        href={cppHref(cppRef.name)}
        className={cn(
          "group flex items-center justify-between gap-3 rounded-md px-2 py-1.5 -mx-1",
          "transition-colors hover:bg-lucid-elevated/60",
        )}
      >
        <span className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-[10px] tracking-wide text-lucid-text-disabled uppercase">
            C++
          </span>
          <code className="font-mono text-sm text-lucid-blue truncate">
            {cppRef.name}
          </code>
          <span className="font-mono text-[10px] text-lucid-text-disabled">
            {kindLabel}
          </span>
        </span>
        <ArrowRight
          className="h-3.5 w-3.5 shrink-0 text-lucid-text-disabled transition-colors group-hover:text-lucid-blue"
          aria-hidden
        />
      </Link>
    </li>
  );
}

function PythonRefRow({ pyRef }: { pyRef: CrossLinkPythonRef }) {
  return (
    <li>
      <Link
        href={pythonHref(pyRef)}
        className={cn(
          "group flex items-center justify-between gap-3 rounded-md px-2 py-1.5 -mx-1",
          "transition-colors hover:bg-lucid-elevated/60",
        )}
      >
        <span className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-[10px] tracking-wide text-lucid-text-disabled uppercase">
            py
          </span>
          <code className="font-mono text-sm text-lucid-primary truncate">
            {pyRef.path}
          </code>
          <span className="font-mono text-[10px] text-lucid-text-disabled">
            {pyRef.kind}
          </span>
        </span>
        <ArrowRight
          className="h-3.5 w-3.5 shrink-0 text-lucid-text-disabled transition-colors group-hover:text-lucid-primary"
          aria-hidden
        />
      </Link>
    </li>
  );
}
