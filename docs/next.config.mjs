import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  output: 'export',
  reactStrictMode: true,
  // Served from https://docs.algorithmiq.fi/monoprop/ (GitHub Pages sub-path)
  // in production. Skipped in `next dev` so local preview lives at the root.
  basePath: process.env.NODE_ENV === 'production' ? '/monoprop' : undefined,
  images: { unoptimized: true },
};

export default withMDX(config);
