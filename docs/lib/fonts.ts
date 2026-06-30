import localFont from 'next/font/local';

/**
 * Algorithmiq's brand typeface, GT-Planar (self-hosted woff2).
 *
 * Only the weights used on algorithmiq.fi are bundled (thin/light/regular);
 * heavier UI weights are synthesised by the browser from 400.
 */
export const gtPlanar = localFont({
  src: [
    {
      path: '../public/fonts/gt-planar/GT-Planar-Thin.woff2',
      weight: '100',
      style: 'normal',
    },
    {
      path: '../public/fonts/gt-planar/GT-Planar-Light.woff2',
      weight: '300',
      style: 'normal',
    },
    {
      path: '../public/fonts/gt-planar/GT-Planar-Regular.woff2',
      weight: '400',
      style: 'normal',
    },
  ],
  variable: '--font-gt-planar',
  display: 'swap',
  fallback: ['ui-sans-serif', 'system-ui', 'sans-serif'],
});
