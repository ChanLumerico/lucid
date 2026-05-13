"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

interface TabProps {
  label: string;
  children: React.ReactNode;
}

export function Tab({ children }: TabProps) {
  return <>{children}</>;
}

interface TabsProps {
  children: React.ReactNode;
  defaultValue?: string;
}

function getTabChildren(
  children: React.ReactNode,
): { label: string; content: React.ReactNode }[] {
  return React.Children.toArray(children)
    .filter(
      (child): child is React.ReactElement<TabProps> =>
        React.isValidElement(child),
    )
    .map((child) => ({
      label: (child.props as TabProps).label ?? "Tab",
      content: (child.props as TabProps).children,
    }));
}

export function Tabs({ children, defaultValue }: TabsProps) {
  const tabs = getTabChildren(children);
  const [active, setActive] = React.useState(
    defaultValue ?? tabs[0]?.label ?? "",
  );

  if (tabs.length === 0) return null;

  const activeTab = tabs.find((t) => t.label === active) ?? tabs[0];

  return (
    <div className="my-6">
      <div className="flex gap-0.5 rounded-xl border border-lucid-border bg-lucid-surface p-1 w-fit">
        {tabs.map(({ label }) => (
          <button
            key={label}
            onClick={() => setActive(label)}
            className={cn(
              "rounded-lg px-3.5 py-1.5 text-sm font-medium transition-all duration-150 outline-none",
              "focus-visible:ring-2 focus-visible:ring-lucid-primary",
              active === label
                ? "bg-lucid-elevated text-lucid-text-high"
                : "text-lucid-text-low hover:text-lucid-text-mid",
            )}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="mt-3">{activeTab.content}</div>
    </div>
  );
}
