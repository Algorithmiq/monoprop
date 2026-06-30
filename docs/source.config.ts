import { defineConfig, defineDocs } from 'fumadocs-mdx/config';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeCitation from 'rehype-citation';

// You can customize Zod schemas for frontmatter and `meta.json` here
// see https://fumadocs.dev/docs/mdx/collections
export const docs = defineDocs({
  dir: 'content/docs',
  docs: {
    schema: pageSchema,
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export default defineConfig({
  mdxOptions: {
    // Tutorial pages inline figures as base64 `data:` URIs; keep them as plain
    // `<img src>` instead of turning images into static imports.
    remarkImageOptions: { useImport: false },
    remarkPlugins: [remarkMath],
    // `rehypeCitation` resolves `[@key]` references against bibliography.bib
    // (replacing sphinxcontrib-bibtex); `rehypeKatex` renders the math nodes
    // produced by `remarkMath`. Both run before the default fumadocs plugins.
    rehypePlugins: (plugins) => [
      rehypeKatex,
      [
        rehypeCitation,
        {
          bibliography: './bibliography.bib',
          linkCitations: true,
          // The references page hand-writes the formatted bibliography (with
          // `id="..."` anchors matching the bib keys), so suppress the
          // per-page bibliography that rehype-citation would otherwise append
          // to every page that uses an inline `[@key]` citation.
          suppressBibliography: true,
        },
      ],
      ...plugins,
    ],
  },
});
