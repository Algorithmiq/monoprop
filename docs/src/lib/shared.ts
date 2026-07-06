export const appName = 'monoprop';
export const docsRoute = '/docs';
export const docsImageRoute = '/og/docs';
export const docsContentRoute = '/llms.mdx/docs';

export const gitConfig = {
  user: 'Algorithmiq',
  repo: 'monoprop',
  branch: process.env.GITHUB_HEAD_REF ?? process.env.GITHUB_REF_NAME ?? 'main',
};
