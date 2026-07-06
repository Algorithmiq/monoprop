// Generate the Python API reference MDX from the griffe JSON produced by
// `fumapy-generate monoprop` (see the `gen-api` recipe in the justfile).
//
// fumadocs-python's `convert`/`write` do the heavy lifting; this script only
//  1. prunes the griffe dump to the public surface (the modules that the old
//     Sphinx `python-api.rst` documented, minus private `_`-prefixed members),
//  2. fixes the package-name segment that `convert` puts in cross-links but
//     `write` strips from file paths, and
//  3. emits a `meta.json` so the API section is ordered like the old reference.
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import { convert, write, frontmatter } from 'fumadocs-python';

const JSON_PATH = path.resolve('monoprop.json');
const OUT_DIR = path.resolve('content/docs/api');
const BASE_URL = '/docs/api';

// Public modules, in the order they should appear in the sidebar
const MODULES = [
  ['monomial_propagator', 'Monomial Propagator'],
  ['fermi_data', 'Fermionic & Majorana operators'],
  ['pauli_data', 'Qubit (Pauli) operators'],
  ['qiskit_conversion', 'Qiskit conversion'],
  ['integral_conversion', 'Integral conversion'],
  ['utils', 'Utilities'],
  ['monomial_data', 'Internal representations'],
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

async function main() {
  const pkg = JSON.parse(await fs.readFile(JSON_PATH, 'utf8'));

  // Keep only the documented public submodules.
  pkg.modules = Object.fromEntries(
    Object.entries(pkg.modules ?? {}).filter(([name]) => KEEP.has(name)),
  );
  pkg.attributes = []; // drop the package-level __all__ dump
  pruneModule(pkg);

  const files = convert(pkg, { baseUrl: BASE_URL });

  for (const file of files) {
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
  // links to `/docs/api`); listing it would add a redundant child duplicating
  // the section title in the sidebar.
  await fs.writeFile(
    path.join(OUT_DIR, 'meta.json'),
    JSON.stringify({ title: 'Python API', pages: [...MODULES.map(([n]) => n)] }, null, 2),
  );

  console.log(`Wrote ${files.length} API pages to ${OUT_DIR}`);
}

await main();
