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
 * Convert Sphinx-style cross-reference markup (e.g., `:meth:`name``) to markdown.
 * Patterns include `:meth:`, `:class:`, `:func:`, `:attr:` with optional ~ prefix
 * for module paths (e.g., `:meth:`~full.path.method_name``).
 *
 * The converted format wraps the name in backticks, which fumadocs will try to
 * resolve as a cross-reference. Explicit links are preserved.
 */
function convertSphinxMarkup(content) {
  // Match `:role:`~?name`` or `:role:`~?path.name``
  // Captures: role (meth/class/func/attr), optional ~, name/path
  return content.replaceAll(/:\w+:`([^`]+)`/g, (match, inner) => {
    // Remove leading ~ if present (used in Sphinx for full qualified paths)
    const cleanName = inner.replace(/^~/, '');
    // Wrap in backticks for cross-reference resolution
    return `\`${cleanName}\``;
  });
}

/**
 * fumadocs-python renders only the first line of a docstring section item's
 * description (Returns, Args, Raises, …). Collapse each item's description to
 * a single line in the griffe JSON *before* passing it to `convert()` so the
 * full text survives.
 */
function normalizeDocstringDescriptions(node) {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node)) {
    for (const item of node) normalizeDocstringDescriptions(item);
    return;
  }

  // Collapse multi-line item descriptions inside parsed docstring sections.
  if (Array.isArray(node.docstring?.parsed)) {
    for (const section of node.docstring.parsed) {
      if (Array.isArray(section.value)) {
        for (const item of section.value) {
          if (typeof item.description === 'string') {
            item.description = item.description.replace(/\n\s*/g, ' ').trim();
          }
        }
      }
    }
  }

  // Recurse into classes, functions, and submodules.
  for (const key of ['classes', 'functions', 'modules']) {
    if (node[key] && typeof node[key] === 'object') {
      for (const child of Object.values(node[key])) normalizeDocstringDescriptions(child);
    }
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

  // Collapse wrapped descriptions in Returns/Args/… sections before convert()
  // sees the JSON, so fumadocs-python renders the full text instead of only the
  // first line of each item.
  normalizeDocstringDescriptions(pkg);

  const files = convert(pkg, { baseUrl: BASE_URL });

  for (const file of files) {
    // Convert Sphinx-style markup to markdown before other processing
    file.content = convertSphinxMarkup(file.content);

    // `convert` keeps the package name ("monoprop") in hrefs, but `write`
    // strips that leading segment from file paths. Realign the links.
    file.content = file.content.replaceAll(`${BASE_URL}/monoprop`, BASE_URL);

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
