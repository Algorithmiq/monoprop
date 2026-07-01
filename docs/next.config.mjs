import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// Production GitHub Pages serves the site from the `/monoprop` sub-path, but
// other hosts (e.g. Cloudflare Pages PR previews, served at the domain root)
// need no prefix. `DOCS_BASE_PATH` overrides the default — set it to an empty
// string to serve from the root. `next dev` keeps the root for local preview.
const basePath =
  process.env.DOCS_BASE_PATH ??
  (process.env.NODE_ENV === 'production' ? '/monoprop' : '');

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  basePath: basePath || undefined,
  images: { unoptimized: true },
};

export default withMDX(config);
