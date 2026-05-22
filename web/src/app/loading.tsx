/** Route-level loading state.  Shown by Next.js's streaming routing
 *  while a server-rendered page is still resolving its data — the
 *  Griffe-fed JSON loaders run server-side at request time on
 *  ``dev`` builds, so a couple of frames of skeleton avoids a janky
 *  blank flash. */
export default function Loading() {
  return (
    <div className="min-h-dvh pt-14 flex items-center justify-center">
      <div className="flex items-center gap-3 text-lucid-text-low">
        <span
          className="inline-block h-2.5 w-2.5 rounded-full bg-lucid-primary animate-pulse"
          aria-hidden
        />
        <span className="font-mono text-sm">Loading…</span>
      </div>
    </div>
  );
}
