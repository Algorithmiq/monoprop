import { RootProvider } from 'fumadocs-ui/provider/next';
import { gtPlanar } from '@/lib/fonts';
import './global.css';
// KaTeX styles for math rendered by rehype-katex.
import 'katex/dist/katex.min.css';

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={gtPlanar.className} suppressHydrationWarning>
      <head>
        <script
          defer
          src="https://cloud.umami.is/script.js"
          data-website-id="aef9f54a-22cb-4455-ad63-ff2d82f908ad"
        />
      </head>
      <body className="flex flex-col min-h-screen">
        <RootProvider
          // The docs site is exported as static files, so search must load the
          // prebuilt Orama index instead of calling the dynamic query API.
          search={{ options: { type: 'static', api: '/api/search' } }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
