import { build } from "esbuild";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

// A classic inline script works over file:// without a development server,
// cross-origin ES-module loading, fonts, CDNs, or network dependencies.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const result = await build({
  absWorkingDir: root,
  entryPoints: ["src/main.jsx"],
  outfile: "offline/app.js",
  bundle: true,
  write: false,
  minify: true,
  format: "iife",
  target: ["chrome100", "firefox100", "safari15.4"],
  define: { "process.env.NODE_ENV": '"production"' },
  legalComments: "none",
});
const svg = await readFile(path.join(root, "public/source-assets/favicon.svg"), "utf8");
const icon = `data:image/svg+xml;base64,${Buffer.from(svg).toString("base64")}`;
const js = result.outputFiles.find((file) => file.path.endsWith(".js")).text
  .replaceAll("/source-assets/favicon.svg", icon)
  .replaceAll(/<\/script/gi, "<\\/script");
const css = result.outputFiles.find((file) => file.path.endsWith(".css"))?.text ?? "";
const notices = await Promise.all(["react", "react-dom", "scheduler", "lz-string", "@heroicons/react"]
  .map(async (name) => `${name}\n${await readFile(path.join(root, "node_modules", name, "LICENSE"), "utf8")}`));
const noticeText = notices.join("\n\n--------------------\n\n");
const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Barricade · 本地对战与回放</title><link rel="icon" href="${icon}"><style>${css}</style></head>
<body><div id="root"></div><script>${js}</script><script type="text/plain" id="third-party-notices">${noticeText.replaceAll(/<\/script/gi, "<\\/script")}</script></body></html>\n`;
await mkdir(path.join(root, "offline"), { recursive: true });
await writeFile(path.join(root, "offline/index.html"), html);
await writeFile(path.join(root, "THIRD_PARTY_NOTICES.txt"), noticeText);
console.log(`Offline HTML: frontend/offline/index.html (${Buffer.byteLength(html)} bytes)`);
