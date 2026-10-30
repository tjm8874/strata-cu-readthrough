import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const root = process.cwd();
const srcDir = path.join(root, 'diagrams');
const outDir = path.join(root, 'diagrams', 'light');
const pngDir = path.join(outDir, 'png');
fs.mkdirSync(pngDir, { recursive: true });

const MAP = {
  '#07090d': '#ffffff', '#0d1218': '#faf8f2', '#0b1119': '#ffffff',
  '#101823': '#f4f1e7', '#0a1a22': '#f4f1e7', '#05080c': '#ffffff',
  '#1d2a36': '#d8d2c2', '#00f0ff': '#1f4e79', '#ff2a6d': '#9e2b25',
  '#d6ff2e': '#4f5d00', '#ffe066': '#7a5c12', '#c9d4e0': '#232323',
  '#8fa3b8': '#6a6a6a'
};
const lum = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
};

const svgs = fs.readdirSync(srcDir).filter(f => f.endsWith('.svg')).sort();
const made = [];
for (const f of svgs) {
  let s = fs.readFileSync(path.join(srcDir, f), 'utf8');
  // drop scanline overlay rects
  s = s.replace(/[ \t]*<rect[^>]*fill="url\(#scan\)"[^>]*\/>\n?/g, '');
  s = s.replace(/[ \t]*<pattern id="scan"[\s\S]*?<\/pattern>\n?/g, '');
  // map known colors
  s = s.replace(/#[0-9a-fA-F]{6}\b/g, (hex) => {
    const k = hex.toLowerCase();
    if (MAP[k]) return MAP[k];
    if (lum(k) < 0.28) return '#f4f1e7';
    return hex;
  });
  // neon-ish glows / shadows are not present as filters; ensure no leftover opacity trickery on backgrounds
  const outSvg = path.join(outDir, f);
  fs.writeFileSync(outSvg, s);
  const m = s.match(/viewBox="0 0 (\d+) (\d+)"/);
  const w = m ? +m[1] : 1000, h = m ? +m[2] : 700;
  const shell = fs.readFileSync(path.join(process.env.HOME, '.cache-paperfig-head.html'), 'utf8').replace('__SVG__', outSvg);
  const html = path.join('/tmp', 'pf-' + f.replace('.svg', '.html'));
  fs.writeFileSync(html, shell);
  const png = path.join(pngDir, f.replace(/\.svg$/, '.png'));
  execFileSync(path.join(process.env.HOME, 'Library/Caches/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-mac-arm64/chrome-headless-shell'), [
    '--no-sandbox', '--disable-gpu', '--hide-scrollbars', '--headless',
    '--force-device-scale-factor=2',
    '--window-size=' + w + ',' + h,
    '--screenshot=' + png, 'file://' + html
  ], { stdio: 'ignore' });
  made.push([f, w, h, fs.statSync(png).size]);
}
for (const m of made) console.log(m.join(' '));
