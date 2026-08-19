import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = {
  title: "Colorectal Cancer Assistant | مساعد سرطان القولون والمستقيم",
  description: "Bilingual colorectal cancer guideline assistant grounded on NICE",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ar"><body>{children}</body></html>;
}
