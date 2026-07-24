// Generate the Python API reference MDX from the griffe JSON produced by
// `fumapy-generate monoprop` (see the `gen-api` recipe in the justfile).
//
// fumadocs-python's `convert`/`write` do the heavy lifting; this script only
//  1. prunes the griffe dump to the public surface (the modules that the old
//     docs reference documented, minus private `_`-prefixed members),
//  2. fixes the package-name segment that `convert` puts in cross-links but
//     `write` strips from file paths, and
//  3. emits a `meta.json` so the API section is ordered like the old reference.
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { convert, write, frontmatter } from 'fumadocs-python';

const JSON_PATH = path.resolve('monoprop.json');
const OUT_DIR = path.resolve('content/docs/api');
const BASE_URL = '/api';

// Public modules, in the order they should appear in the sidebar
const MODULES = [
  ['majorana_propagator', 'Majorana Propagator'],
  ['pauli_propagator', 'Pauli Propagator'],
  ['monomial_propagator', 'Shared propagator surface'],
  ['circuit', 'Circuits and gates'],
  ['fermi', 'Fermionic & Majorana operators'],
  ['pauli', 'Qubit (Pauli) operators'],
  ['qiskit_conversion', 'Qiskit conversion'],
  ['integral_conversion', 'Integral conversion'],
  ['utils', 'Utilities'],
  ['majorana', 'Internal representations'],
  ['quantum_data', 'Protocols'],
  ['exceptions', 'Exceptions'],
];
const KEEP = new Set(MODULES.map(([name]) => name));

/** Drop single-underscore (non-dunder) members from a {name: node} map. */
function prunePrivate(map) {
  for (const name of Object.keys(map)) {
    if (name.startsWith('_') && !name.startsWith('__')) delete map[name];
  }
}

/** Recursively strip private classes/functions/submodules from a module node. */
function pruneModule(mod) {
  prunePrivate(mod.classes ?? {});
  prunePrivate(mod.functions ?? {});
  for (const cls of Object.values(mod.classes ?? {})) prunePrivate(cls.functions ?? {});
  for (const sub of Object.values(mod.modules ?? {})) pruneModule(sub);
}

/**
 * Turn a fully-qualified symbol path into the URL of the page that documents
 * it. Mirrors fumadocs-python's `getHref`, then drops the leading `monoprop`
 * segment that `write` strips from file paths (the same realignment applied to
 * the rendered content below).
 */
function pageUrl(dottedPath) {
  const url =
    '/' +
    [...BASE_URL.split('/'), ...dottedPath.split('.')].filter((v) => v.length > 0).join('/');
  return url.replace(`${BASE_URL}/monoprop`, BASE_URL);
}

/**
 * Build a map from fully-qualified symbol path to the doc URL that documents it.
 *
 * Classes and modules get their own page. Functions, methods and attributes are
 * rendered inline on their parent's page; the `getMDXComponents` wrappers give
 * each `PyFunction`/`PyAttribute` card an `id` equal to its member name, so those
 * members resolve to their parent page with a `#<name>` fragment that jumps to
 * the definition.
 */
function buildXrefMap(pkg) {
  const map = new Map();

  // A module or class page: itself at the page top, its inline members at `#<name>`.
  const addScope = (node, pageUrl) => {
    map.set(node.path, pageUrl);
    for (const fn of Object.values(node.functions ?? {})) map.set(fn.path, `${pageUrl}#${fn.name}`);
    for (const attr of node.attributes ?? [])
      map.set(`${node.path}.${attr.name}`, `${pageUrl}#${attr.name}`);
  };

  const visit = (mod) => {
    addScope(mod, pageUrl(mod.path));
    for (const cls of Object.values(mod.classes ?? {})) addScope(cls, pageUrl(cls.path));
    for (const sub of Object.values(mod.modules ?? {})) visit(sub);
  };
  visit(pkg);
  return map;
}

/**
 * Derive the fully-qualified scope a rendered page documents, from its file
 * path (before `write` strips the leading package segment):
 *   `monoprop/circuit/index.mdx`   -> `monoprop.circuit`            (module)
 *   `monoprop/circuit/ExpGate.mdx` -> `monoprop.circuit.ExpGate`    (class)
 *   `monoprop/index.mdx`           -> `monoprop`                    (package)
 */
function scopeFromPath(filePath) {
  return filePath
    .replace(/\/index\.mdx$/, '')
    .replace(/\.mdx$/, '')
    .split('/')
    .join('.');
}

/**
 * Resolve a cross-reference target to a page URL, or `null`. A fully-qualified
 * target hits `xref` directly; an unqualified (or class-relative) target is
 * resolved against the current page's scope, walking outward from the nearest
 * enclosing symbol to the package root. Nearest scope wins, so resolution is
 * deterministic -- a bare name only ever binds to something reachable from its
 * own scope chain, never to a same-named symbol in an unrelated module.
 */
function resolve(name, scope, xref) {
  if (xref.has(name)) return xref.get(name);
  const parts = scope.split('.');
  for (let i = parts.length; i > 0; i--) {
    const url = xref.get(`${parts.slice(0, i).join('.')}.${name}`);
    if (url) return url;
  }
  return null;
}

/**
 * Resolve cross-references in a rendered MDX page into real links. `scope` is
 * the page's fully-qualified scope (see `scopeFromPath`).
 *
 * The input is the griffe/mkdocstrings Markdown reference-link form. The target
 * is a Python path, resolved fully-qualified or relative to `scope`:
 *   `[Circuit][monoprop.circuit.Circuit]`  ->  `[Circuit](/api/circuit/Circuit)`
 *   `[ExpGate][]`  (on a page scoped to `monoprop.circuit`)  ->  `[ExpGate](/api/circuit/ExpGate)`
 * `[X][]` is shorthand for target == display. The display text is rendered as a
 * code span (symbol names read as code). Unresolved targets are collected in
 * `unresolved` (surfaced by the caller) and left untouched.
 */
function resolveXrefs(content, xref, unresolved, scope) {
  // Wrap the display text in a code span, avoiding a double-wrap if the
  // docstring already backticked it.
  const code = (text) => (/^`.*`$/.test(text) ? text : `\`${text}\``);

  return content.replaceAll(/\[([^\]]+)\]\[([^\]]*)\]/g, (match, display, target) => {
    const name = (target || display).trim().replace(/^`|`$/g, '');
    const url = resolve(name, scope, xref);
    if (url) return `[${code(display)}](${url})`;
    unresolved.add(name);
    return match;
  });
}

/** Collapse internal whitespace/newlines to a single line. */
function foldToOneLine(text) {
  return String(text).replace(/\s*\n\s*/g, ' ').replace(/\s+/g, ' ').trim();
}

/**
 * Unescape braces within LaTeX/KaTeX math expressions.
 *
 * Upstream markdown serialization escapes braces (`\{` and `\}`), but KaTeX
 * requires unescaped braces for proper grouping (subscripts, superscripts, and
 * command arguments). This function selectively unescapes braces only inside
 * `$...$` (inline) and `$$...$$` (display) math spans.
 *
 * Examples:
 * - `$r_\{12\}$` -> `$r_{12}$`
 * - `$$^\{-1\}$$` -> `$$^{-1}$$`
 * - `$\mathrm\{d\}$` -> `$\mathrm{d}$`
 */
function normalizeMathGroupingEscapes(content) {
  const normalizeMathSegment = (math) =>
    math
      .replace(/\\\{/g, '{')
      .replace(/\\\}/g, '}');

  return content.replace(/\$\$[\s\S]*?\$\$|\$(?:\\.|[^$\\])+\$/g, (math) =>
    normalizeMathSegment(math),
  );
}

/**
 * Extract the raw `Returns:` block from a function source docstring.
 * We prefer source extraction because some upstream parsed `returns.description`
 * values are already truncated to the first wrapped line.
 */
function extractReturnsFromSource(source) {
  if (typeof source !== 'string' || source.length === 0) return null;

  // Find a Returns section inside the function source, ending at the next
  // section header (e.g. Args:, Raises:, Note:) or the closing docstring.
  const match = source.match(
    /\n\s+Returns:\n([\s\S]*?)(?=\n\s+[A-Z][A-Za-z_ ]+:\n|\n\s+["']{3})/,
  );
  if (!match) return null;

  const text = match[1]
    .split('\n')
    .map((line) => line.replace(/^\s+/, ''))
    .join(' ')
    .trim();

  return text || null;
}

/**
 * griffe preserves wrapped descriptions in parsed docstring sections. The
 * fumadocs-python renderer currently truncates multi-line Returns text in some
 * cases, so we normalize function Returns descriptions to one line before
 * conversion.
 */
function normalizeFunctionReturnsSection(funcNode) {
  const fromSource = extractReturnsFromSource(funcNode?.source);
  if (fromSource && funcNode?.returns && typeof funcNode.returns === 'object') {
    funcNode.returns.description = foldToOneLine(fromSource);
  }

  const parsed = funcNode?.docstring?.parsed;
  if (!Array.isArray(parsed)) return;

  for (const section of parsed) {
    if (section?.kind !== 'returns') continue;

    // Depending on parser shape, `value` can be a list of return entries or a
    // single entry-like object.
    const values = Array.isArray(section.value) ? section.value : [section.value];
    for (const item of values) {
      if (item && typeof item.description === 'string') {
        item.description = foldToOneLine(item.description);
      }
    }
  }
}

/** Recursively visit modules/classes and normalize function Returns text. */
function normalizeFunctionReturns(mod) {
  if (!mod || typeof mod !== 'object') return;

  for (const fn of Object.values(mod.functions ?? {})) {
    normalizeFunctionReturnsSection(fn);
  }

  for (const cls of Object.values(mod.classes ?? {})) {
    for (const fn of Object.values(cls.functions ?? {})) {
      normalizeFunctionReturnsSection(fn);
    }
  }

  for (const sub of Object.values(mod.modules ?? {})) {
    normalizeFunctionReturns(sub);
  }
}

async function main() {
  const pkg = JSON.parse(await fs.readFile(JSON_PATH, 'utf8'));

  // Keep only the documented public submodules.
  pkg.modules = Object.fromEntries(
    Object.entries(pkg.modules ?? {}).filter(([name]) => KEEP.has(name)),
  );
  pkg.attributes = []; // drop the package-level __all__ dump
  pruneModule(pkg);
  normalizeFunctionReturns(pkg);

  const xref = buildXrefMap(pkg);
  const unresolved = new Set();

  const files = convert(pkg, { baseUrl: BASE_URL });

  for (const file of files) {
    // Resolve Markdown `[text][path]` cross-references into real links before
    // other processing. Bare/relative targets resolve against the page's own scope.
    file.content = resolveXrefs(file.content, xref, unresolved, scopeFromPath(file.path));

    // `convert` keeps the package name ("monoprop") in hrefs, but `write`
    // strips that leading segment from file paths. Realign the links.
    file.content = file.content.replaceAll(`${BASE_URL}/monoprop`, BASE_URL);

    // Fumadocs parameter rendering can escape braces in math arguments
    // (e.g. `\mathrm\{d\}`), which breaks KaTeX grouping.
    file.content = normalizeMathGroupingEscapes(file.content);

    // Add titles to the frontmatter for the modules we documented. The title
    // is used in the sidebar and in the page's <title> tag.
    const m = file.path.match(/^monoprop\/([^/]+)\/index\.mdx$/);
    if (m) {
      const entry = MODULES.find(([name]) => name === m[1]);
      if (entry) file.frontmatter.title = entry[0]; // use entry[1] for the human-readable title if you want it in the sidebar
    }
    if (file.path === 'monoprop/index.mdx') {
      file.frontmatter.title = 'Python API';
      file.frontmatter.description = 'Generated reference for the public monoprop Python package.';
    }
  }

  if (unresolved.size > 0) {
    console.warn(
      `Warning: ${unresolved.size} cross-reference target(s) did not resolve and were left as-is:\n  ` +
        [...unresolved].sort().join('\n  '),
    );
  }

  await fs.rm(OUT_DIR, { recursive: true, force: true });
  await write(files, { outDir: OUT_DIR });

  // Sidebar ordering for the API section. `index.mdx` is intentionally omitted:
  // fumadocs treats it as the folder's own index (so the "Python API" title
  // links to `/api`); listing it would add a redundant child duplicating
  // the section title in the sidebar.
  await fs.writeFile(
    path.join(OUT_DIR, 'meta.json'),
    JSON.stringify({ title: 'Python API', pages: [...MODULES.map(([n]) => n)] }, null, 2),
  );

  console.log(`Wrote ${files.length} API pages to ${OUT_DIR}`);
}

await main();
