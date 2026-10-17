import Image from "next/image";
import { brandAsset } from "@/lib/site";
import { cn } from "@/lib/utils";

// Logo artwork from the repository brand kit (``../assets``), served from
// ``/brand/`` by ``scripts/sync-brand.mjs``.  Size it with a height class
// (``h-6``) — width follows the artwork's ratio.  The kit's rules apply to
// every use: no recolouring outside its palette, no rotation, stretching,
// strokes or shadows, and clear space of half the mark's height around it.
//
// Everything loads eagerly: the header puts both lockup variants above the
// fold on every page, so the footer's copies are cache hits, and the marks
// are a few KB each.  (A lazy footer copy also makes Next's dev-mode LCP
// check, which tracks images by src, misreport the header logo as lazy.)

/** Intrinsic sizes (the SVG viewBoxes, rounded).  They reserve the box
 *  before the file arrives, so the header never shifts. */
const LOCKUP = { width: 542, height: 152 };
const MARK = { width: 48, height: 48 };

const THEMES = ["dark", "light"] as const;

/** The horizontal lockup in the variant for the site's theme.  Both variants
 *  are in the markup and a rule in ``globals.css`` hides the one whose
 *  ``data-theme-only`` doesn't match ``html[data-theme]`` — the attribute the
 *  site's own toggle sets before first paint — so the logo follows that
 *  toggle, not just the OS ``prefers-color-scheme``, and never flashes. */
export function BrandLockup({ className }: { className?: string }) {
  return (
    <>
      {THEMES.map((theme) => (
        <Image
          key={theme}
          src={brandAsset(`lucid-lockup-horizontal-on-${theme}.svg`)}
          alt="Lucid"
          width={LOCKUP.width}
          height={LOCKUP.height}
          unoptimized
          loading="eager"
          draggable={false}
          data-theme-only={theme}
          className={cn("w-auto select-none", className)}
        />
      ))}
    </>
  );
}

/** The mark alone.  ``duotone`` is the kit's variant for hero sections; the
 *  violet mark reads on both themes, so neither needs a theme pair. */
export function BrandMark({
  variant = "violet",
  className,
}: {
  variant?: "violet" | "duotone";
  className?: string;
}) {
  return (
    <Image
      src={brandAsset(variant === "duotone" ? "lucid-mark-duotone.svg" : "lucid-mark.svg")}
      alt=""
      aria-hidden
      width={MARK.width}
      height={MARK.height}
      unoptimized
      loading="eager"
      draggable={false}
      className={cn("select-none", className)}
    />
  );
}

/** The release-line pill that sits beside the lockup. */
export function VersionPill() {
  return (
    <span
      className={cn(
        "rounded-md border border-lucid-primary/35 bg-lucid-primary/10",
        "px-1.5 py-0.5 text-[9px] font-bold tracking-[0.12em] text-lucid-primary uppercase",
      )}
    >
      3.0
    </span>
  );
}
