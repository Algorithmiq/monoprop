// Remark plugin: resolve mkdocstrings-style cross-references in prose MDX.
//
// Undefined markdown reference links stay as literal *text* nodes (micromark does
// not emit a `linkReference` without a matching definition), so we operate on
// text nodes via `mdast-util-find-and-replace`. Real `[text](url)` links and
// `[@citation]` nodes do not match the two-bracket pattern and are left untouched.
import * as fs from 'node:fs';
import * as path from 'node:path';
import { findAndReplace } from 'mdast-util-find-and-replace';
import { buildXrefMap, resolve } from './xref.mjs';

// Prose is scoped to the package root, so a bare `[MajoranaPropagator][]`
// resolves via the scope walk in `resolve`.
const SCOPE = 'monoprop';
// Same pattern as `resolveXrefs` in generate-api.mjs.
const XREF_PATTERN = /\[([^\]]+)\]\[([^\]]*)\]/g;

const JSON_PATH = path.resolve('monoprop.json');

let xref = null;
let byLeaf = null;
try {
  xref = buildXrefMap(JSON.parse(fs.readFileSync(JSON_PATH, 'utf8')));
  // The public classes/functions live in submodules (e.g.
  // `monoprop.majorana_propagator.MajoranaPropagator`) but authors know them by the
  // bare name they import, so index every symbol by its leaf name as a fallback --
  // dropping any leaf that maps to more than one URL.
  byLeaf = new Map();
  const clashed = new Set();
  for (const [dotted, url] of xref) {
    const leaf = dotted.split('.').pop();
    if (clashed.has(leaf)) continue;
    if (byLeaf.has(leaf) && byLeaf.get(leaf) !== url) {
      byLeaf.delete(leaf);
      clashed.add(leaf);
    } else {
      byLeaf.set(leaf, url);
    }
  }
} catch (err) {
  console.warn(
    `remark-xref: could not read ${JSON_PATH} (${err.code ?? err.message}); ` +
      'prose cross-references will be left as-is. Run `just gen-api` first.',
  );
}

const unbacktick = (text) => text.replace(/^`(.*)`$/, '$1');

export default function remarkXref() {
  return (tree, file) => {
    if (!xref) return;
    findAndReplace(tree, [
      [
        XREF_PATTERN,
        (_match, display, target) => {
          const name = unbacktick((target || display).trim());
          const url = resolve(name, SCOPE, xref) ?? byLeaf.get(name) ?? null;
          if (!url) {
            // `return false` leaves the text untouched. `console.warn`, not a
            // vfile message -- fumadocs/next does not print those.
            const where = file.path ? path.relative(process.cwd(), file.path) : 'prose';
            console.warn(`remark-xref: unresolved cross-reference [${display}][${target}] in ${where}`);
            return false;
          }
          // Symbol names read as code, matching the API pages' styling.
          return {
            type: 'link',
            url,
            children: [{ type: 'inlineCode', value: unbacktick(display.trim()) }],
          };
        },
      ],
    ]);
  };
}
