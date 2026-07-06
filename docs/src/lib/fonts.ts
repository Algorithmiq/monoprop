import localFont from 'next/font/local';

export const gtPlanar = localFont({
  src: [
    {
      path: '../../public/fonts/gt-planar/GT-Planar-Thin.woff2',
      weight: '100',
      style: 'normal',
    },
    {
      path: '../../public/fonts/gt-planar/GT-Planar-Light.woff2',
      weight: '300',
      style: 'normal',
    },
    {
      path: '../../public/fonts/gt-planar/GT-Planar-Regular.woff2',
      weight: '400',
      style: 'normal',
    },
  ],
  display: 'swap',
  fallback: ['system-ui', 'sans-serif'],
});
