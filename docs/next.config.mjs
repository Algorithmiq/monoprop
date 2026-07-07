import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

// GitHub Pages, Cloudflare Pages, and local dev/preview all serve the site
// under `/monoprop/` — matches GitHub Pages' default project-site path
// (`<org>.github.io/<repo>/`), so every host stays consistent with it.
/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  basePath: '/monoprop',
  images: { unoptimized: true },
};

export default withMDX(config);
