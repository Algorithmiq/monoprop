import { RootProvider } from 'fumadocs-ui/provider/next';
import { gtPlanar } from '@/lib/fonts';
import './global.css';

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html lang="en" className={gtPlanar.className} suppressHydrationWarning>
      <body className="flex flex-col min-h-screen">
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
