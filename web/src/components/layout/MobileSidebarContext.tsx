"use client";

import * as React from "react";
import type { SidebarItem } from "./Sidebar";

interface MobileSidebarValue {
  items: SidebarItem[] | null;
}

/** Carries the docs / api sidebar tree from the route-level layout
 *  down to the (route-shared) ``MobileMenu`` inside the ``Header``.
 *  When ``items`` is ``null`` the mobile drawer falls back to the
 *  top-nav links only — the same surface marketing / landing pages
 *  see.
 *
 *  Implemented as a Context (not a top-level layout slot) so the
 *  ``Header`` doesn't need a per-route conditional in its JSX. */
const MobileSidebarContext = React.createContext<MobileSidebarValue>({ items: null });

export function MobileSidebarProvider({
  items,
  children,
}: {
  items: SidebarItem[];
  children: React.ReactNode;
}) {
  const value = React.useMemo(() => ({ items }), [items]);
  return (
    <MobileSidebarContext.Provider value={value}>
      {children}
    </MobileSidebarContext.Provider>
  );
}

export function useMobileSidebarItems(): SidebarItem[] | null {
  return React.useContext(MobileSidebarContext).items;
}
