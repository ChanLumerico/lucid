// Auto-generated JSON schema types for Griffe API data
// Mirrors the shape produced by scripts/build-api-data.py

export interface DocstringParameter {
  name: string;
  annotation: string | null;
  description: string;
  default: string | null;
}

export interface DocstringReturn {
  annotation: string | null;
  description: string;
}

export interface DocstringRaise {
  annotation: string | null;
  description: string;
}

export interface DocstringAttribute {
  name: string;
  annotation: string | null;
  description: string;
}

export interface DocstringParsed {
  summary: string | null;
  extended: string | null;
  parameters: DocstringParameter[];
  returns: DocstringReturn | null;
  raises: DocstringRaise[];
  examples: string[];
  notes: string[];
  attributes: DocstringAttribute[];
  warns: DocstringRaise[];
}

// ---------------------------------------------------------------------------
// Labels / decorator kinds
// ---------------------------------------------------------------------------

export type ApiLabel =
  | "property"
  | "staticmethod"
  | "classmethod"
  | "abstractmethod"
  | "writable";

// ---------------------------------------------------------------------------
// API members
// ---------------------------------------------------------------------------

export interface ApiFunction extends DocstringParsed {
  name: string;
  path: string;
  kind: "function";
  labels: ApiLabel[];
  signature: string | null;
  source: string | null;
}

export interface ApiMethod extends DocstringParsed {
  name: string;
  path: string;
  kind: "function";
  labels: ApiLabel[];
  signature: string | null;
  source: string | null;
}

export type ApiClassKind = "regular" | "abstract" | "dataclass";

export interface ApiClass extends DocstringParsed {
  name: string;
  path: string;
  kind: "class";
  class_kind: ApiClassKind;
  bases: string[];
  labels: ApiLabel[];
  signature: string | null;
  source: string | null;
  methods: ApiMethod[];
}

export type ApiMember = ApiFunction | ApiClass;

// ---------------------------------------------------------------------------
// Module / top-level data files
// ---------------------------------------------------------------------------

export interface ApiModule extends DocstringParsed {
  slug: string;
  name: string;
  path: string;
  kind: "module";
  source: string | null;
  members: ApiMember[];
}

/** Special kind for lucid.tensor — a single class exposed as a module */
export interface ApiClassModule extends DocstringParsed {
  slug: string;
  name: string;
  path: string;
  kind: "class-module";
  signature: string | null;
  source: string | null;
  methods: ApiMethod[];
}

export type ApiData = ApiModule | ApiClassModule;

// ---------------------------------------------------------------------------
// Helper type guards
// ---------------------------------------------------------------------------

export function isApiModule(data: ApiData): data is ApiModule {
  return data.kind === "module";
}

export function isApiClassModule(data: ApiData): data is ApiClassModule {
  return data.kind === "class-module";
}

export function isApiClass(member: ApiMember): member is ApiClass {
  return member.kind === "class";
}

export function isApiFunction(member: ApiMember): member is ApiFunction {
  return member.kind === "function";
}

// ---------------------------------------------------------------------------
// Sidebar / navigation
// ---------------------------------------------------------------------------

export interface ModuleEntry {
  slug: string;
  name: string;
  memberCount: number;
  summary: string | null;
}

export const MODULE_SLUGS = [
  "lucid",
  "lucid.tensor",
  "lucid.nn",
  "lucid.nn.functional",
  "lucid.nn.init",
  "lucid.nn.utils",
  "lucid.optim",
  "lucid.autograd",
  "lucid.func",
  "lucid.linalg",
  "lucid.fft",
  "lucid.signal",
  "lucid.special",
  "lucid.distributions",
  "lucid.utils.data",
  "lucid.amp",
  "lucid.profiler",
  "lucid.einops",
  "lucid.serialization",
] as const;

export type ModuleSlug = (typeof MODULE_SLUGS)[number];
