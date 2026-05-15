import type { ApiLabel, ApiClassKind } from "@/lib/types";

/** Tailwind text-color class for a method/function based on its labels + name. */
export function getMemberNameColor(name: string, labels: ApiLabel[]): string {
  if (labels.includes("property"))       return "text-api-prop";
  if (labels.includes("classmethod"))    return "text-api-cls";
  if (labels.includes("staticmethod"))   return "text-api-static";
  if (labels.includes("abstractmethod")) return "text-api-abstract";
  if (name.startsWith("__") && name.endsWith("__")) return "text-api-dunder";
  return "text-api-fn";
}

/** Tailwind text-color class for a class name based on its kind. */
export function getClassNameColor(classKind: ApiClassKind): string {
  if (classKind === "abstract")  return "text-api-class-abstract";
  if (classKind === "dataclass") return "text-api-class-dataclass";
  return "text-api-class";
}

/** Tailwind hover-border class for a class card based on its kind. */
export function getClassHoverBorder(classKind: ApiClassKind): string {
  if (classKind === "abstract")  return "hover:border-api-class-abstract/40";
  if (classKind === "dataclass") return "hover:border-api-class-dataclass/40";
  return "hover:border-api-class/40";
}
