import { BrandMark } from "@/components/brand/Brand";

/** Route-level loading state.  Shown by Next.js's streaming routing
 *  while a server-rendered page is still resolving its data — the
 *  Griffe-fed JSON loaders run server-side at request time on
 *  ``dev`` builds, so a couple of frames of skeleton avoids a janky
 *  blank flash. */
export default function Loading() {
  return (
    <div className="min-h-dvh pt-14 flex items-center justify-center">
      <div className="flex items-center gap-3 text-lucid-text-low">
        <BrandMark className="h-5 w-5 animate-pulse" />
        <span className="font-mono text-sm">Loading…</span>
      </div>
    </div>
  );
}
