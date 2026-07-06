import "./globals.css";

export const metadata = {
  title: "Factory Operator Dashboard",
  description: "Production-floor operator presence, activity, and cycle review dashboard."
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
