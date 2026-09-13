import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NVMe Failure Predictor",
  description: "SMART telemetry failure-risk console for NVMe drives.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col bg-console text-ink antialiased">
        {children}
      </body>
    </html>
  );
}
