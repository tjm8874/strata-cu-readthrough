import fs from 'node:fs';
import path from 'node:path';
import { marked } from '/Users/ts/DSH/node_modules/.pnpm/marked@18.0.11/node_modules/marked/lib/marked.esm.js';

const root = process.cwd();
const outDir = path.join(root, process.argv[2] || 'site');
const docsOut = path.join(outDir, 'docs');
const figOut = path.join(outDir, 'diagrams', 'light', 'png');
fs.mkdirSync(docsOut, { recursive: true });
fs.mkdirSync(figOut, { recursive: true });

const css = `:root{--paper:#fffdf8;--ink:#191510;--muted:#605a4e;--rule:#d6d0c0;--link:#1f4e79;--accent:#9e2b25;--tint:#f5f2e8}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:"Hiragino Mincho ProN","Yu Mincho",YuMincho,Georgia,"Times New Roman",serif;font-size:16.5px;line-height:1.95;margin:0;-webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:52px 26px 110px}
.masthead{font-family:"SF Mono",Menlo,monospace;font-size:11.5px;letter-spacing:.12em;color:var(--muted);border-bottom:1px solid var(--rule);padding-bottom:10px;display:flex;justify-content:space-between;gap:14px;margin-bottom:42px}
.masthead a{color:var(--muted);text-decoration:none;border:0}
.masthead a:hover{color:var(--link)}
h1{font-size:30px;line-height:1.45;font-weight:600;margin:0 0 .35em;letter-spacing:.01em}
h1 small{display:block;font-size:15px;font-weight:400;color:var(--muted);margin-top:.6em;letter-spacing:.02em}
h2{font-size:18px;font-weight:600;margin:2.6em 0 .8em;padding-bottom:.3em;border-bottom:1px solid var(--rule);letter-spacing:.02em}
h2 span{font-family:"SF Mono",Menlo,monospace;font-size:11px;color:var(--muted);letter-spacing:.14em;margin-left:.6em;font-weight:400}
h3{font-size:16.5px;font-weight:700;margin:2em 0 .5em;letter-spacing:.01em}
p{margin:.95em 0;text-align:justify}
a{color:var(--link);text-decoration:none;border-bottom:1px solid rgba(31,78,121,.28)}
a:hover{border-bottom-color:var(--link)}
strong{font-weight:700}
code{font-family:"SF Mono",Menlo,ui-monospace,monospace;font-size:13px;background:var(--tint);padding:.05em .32em;word-break:break-word}
pre{background:var(--tint);border:1px solid var(--rule);border-left:3px solid #b8b0a0;padding:13px 16px;overflow-x:auto;font-size:12.6px;line-height:1.68;margin:1.5em 0}
pre code{background:none;padding:0;font-size:12.6px}
blockquote{margin:1.5em 0;padding:.15em 0 .15em 1.1em;border-left:2px solid #c3bba9;color:#3b382f;text-align:left}
blockquote p{margin:.4em 0}
table{border-collapse:collapse;width:100%;margin:1.6em 0;font-size:13.8px;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}
th,td{border:0;border-bottom:1px solid #e4dfd1;padding:7px 12px 7px 0;text-align:left;vertical-align:top}
thead th{border-bottom:1.4px solid var(--ink);font-weight:700;white-space:nowrap;background:none}
ul,ol{padding-left:1.5em;margin:.9em 0}
li{margin:.32em 0}
hr{border:0;border-top:1px solid var(--rule);margin:2.4em 0}
.figrow{display:block;margin:2em -82px}
.figlink{display:block;border:0;margin:0 0 1.3em}
.figlink:last-child{margin-bottom:0}
.figlink img{width:100%;display:block;border:1px solid var(--rule)}
.figcap{display:block;font-size:12.6px;color:var(--muted);margin-top:.55em;text-align:center;font-family:"SF Mono",Menlo,monospace;letter-spacing:.02em}
.abstract{font-size:15.5px;color:#2c2a24;margin:1.5em 0 2.2em}
.meta{width:100%;border-collapse:collapse;font-size:13.4px;margin:0 0 2.8em;font-family:"SF Mono",Menlo,monospace;display:table}
.meta th{text-align:left;font-weight:400;color:var(--muted);width:9em;border-top:1px solid var(--rule);padding:7px 10px 7px 0;vertical-align:top;letter-spacing:.04em}
.meta td{border-top:1px solid var(--rule);padding:7px 0;vertical-align:top}
.toc{list-style:none;padding:0;margin:1.2em 0}
.toc li{border-top:1px solid #e8e3d5;padding:.9em 0}
.toc li:last-child{border-bottom:1px solid #e8e3d5}
.toc .n{font-family:"SF Mono",Menlo,monospace;font-size:11.5px;color:var(--muted);letter-spacing:.12em}
.toc .row{display:block;border:0;color:inherit;padding:0}
.toc .row:hover .t{color:var(--link);border-bottom:1px solid rgba(31,78,121,.3)}
.toc .t{display:inline-block;font-size:16.5px;margin:.1em 0 .18em;border-bottom:1px solid transparent}
.toc .d{display:block;font-size:14px;color:var(--muted);line-height:1.7}
.toc .s{font-family:"SF Mono",Menlo,monospace;font-size:11px;color:#8b8677;letter-spacing:.06em}
.gallery{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:1.6em 0}
.gallery a{display:block;border:0;color:inherit}
.gallery img{width:100%;display:block;border:1px solid var(--rule)}
.gallery .cap{font-size:12.6px;margin-top:.55em;font-family:"SF Mono",Menlo,monospace;color:var(--ink)}
.gallery .cd{font-size:12.4px;color:var(--muted);line-height:1.65;margin-top:.2em}
footer{margin-top:4em;padding-top:1.2em;border-top:1px solid var(--rule);font-size:13.4px;color:var(--muted)}
footer p{margin:.6em 0}
.pager{display:flex;justify-content:space-between;gap:16px;font-size:13.4px;margin-top:3.2em;padding-top:1.1em;border-top:1px solid var(--rule);color:var(--muted)}
@media(max-width:1100px){.figrow{margin:1.8em 0}}
@media(max-width:640px){body{font-size:16px}.wrap{padding:32px 18px 80px}h1{font-size:23px}.gallery{grid-template-columns:1fr}}
@media print{body{background:#fff;font-size:11pt}.masthead,.pager{display:none}a{color:#000;border:0}pre,table,.figlink{page-break-inside:avoid}}
`;

const FIG = /^>\s*図:\s*(.+)$/;
const FIGPATH = /\[([^\]]+)\]\(([^)]+\.svg)\)/g;

function figBlock(line) {
  const parts = [...line.matchAll(FIGPATH)];
  if (!parts.length) return null;
  const imgs = parts.map((m) => {
    const png = m[2].replace(/^(\.\.\/)?diagrams\//, '../diagrams/light/png/').replace(/\.svg$/, '.png');
    return '<a class="figlink" href="' + png + '"><img src="' + png + '" alt="' + m[1] + '"><span class="figcap">' + m[1] + '</span></a>';
  }).join('');
  return '<div class="figrow">' + imgs + '</div>';
}

const docs = fs.readdirSync(path.join(root, 'docs')).filter((f) => /^0\d_.*\.md$/.test(f)).sort();
const chapters = [];
for (const f of docs) {
  const md = fs.readFileSync(path.join(root, 'docs', f), 'utf8').split('\n');
  const out = []; let inFence = false;
  for (const ln of md) {
    if (/^`{3}/.test(ln)) inFence = !inFence;
    if (!inFence && FIG.test(ln)) { const b = figBlock(ln); if (b) { out.push(b); continue; } }
    out.push(ln);
  }
  const body = marked.parse(out.join('\n').replace(/\.md\)/g, '.html)'), { gfm: true, breaks: false });
  const m = body.match(/<h1[^>]*>([\s\S]*?)<\/h1>/);
  chapters.push({ file: f, htmlFile: f.replace(/\.md$/, '.html'), title: m ? m[1].replace(/<[^>]+>/g, '') : f, body });
}

function chapterShell(ch, i) {
  const prev = i > 0 ? chapters[i - 1] : null;
  const next = i < chapters.length - 1 ? chapters[i + 1] : null;
  const nav = '<div class="masthead"><a href="../index.html">← 目次 / 図一覧</a><span>STRATA 読解 · ' + ch.htmlFile.slice(0, 2) + ' / 08</span></div>';
  const pager = '<div class="pager">' + (prev ? '<a href="' + prev.htmlFile + '">← ' + prev.title + '</a>' : '<span></span>') +
    '<span>' + (next ? '<a href="' + next.htmlFile + '">' + next.title + ' →</a>' : '<a href="../index.html">目次へ</a>') + '</span></div>' +
    '<footer><p>出典: github.com/Niko1221/Strata HEAD 61b3fb5（v0.1.49）／ 本文中の行番号・定数はこのリビジョンのもの。図は同梱 PNG（2 倍解像度）。</p></footer>';
  return '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' +
    ch.title + ' — STRATA CUDA ソース読解</title>' +
    '<link rel="icon" href="../favicon.svg" type="image/svg+xml"><style>' + css + '</style></head><body><div class="wrap">' + nav + '<article>' + ch.body + '</article>' + pager + '</div></body></html>';
}

for (let i = 0; i < chapters.length; i++) fs.writeFileSync(path.join(docsOut, chapters[i].htmlFile), chapterShell(chapters[i], i));

// ---- figures ----
const figSrc = path.join(root, 'diagrams', 'light', 'png');
for (const f of fs.readdirSync(figSrc)) fs.copyFileSync(path.join(figSrc, f), path.join(figOut, f));

// ---- index ----
const cyber = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const sub = (cyber.match(/<p class="sub">([\s\S]*?)<\/p>/) || [, ''])[1].replace(/\s+/g, ' ').trim();
const cards = [...cyber.matchAll(/<a class="card" href="docs\/([^"]+)"><div class="num">(\d+)<\/div><div class="t">([^<]+)<\/div>\s*<div class="d">([^<]+)<\/div><div class="sz">([^<]+)<\/div><\/a>/g)];
const figs = [...cyber.matchAll(/<a class="fig" href="diagrams\/png\/([^"]+)"><img src="[^"]+" alt="([^"]+)">\s*<div class="cap">([^<]+)<\/div><div class="cd">([^<]+)<\/div><\/a>/g)];

const toc = cards.map((c) => '<li><a class="row" href="docs/' + c[1] + '"><span class="t">' + c[3] + '</span><span class="d">' + c[4] + '</span><span class="n">' + c[2] + ' · ' + c[5] + '</span></a></li>').join('');
const gal = figs.map((f) => '<a href="diagrams/light/png/' + f[1] + '"><img src="diagrams/light/png/' + f[1] + '" alt="' + f[2] + '"><div class="cap">' + f[3] + '</div><div class="cd">' + f[4] + '</div></a>').join('');

const index = '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +
  '<title>STRATA CUDA ソース読解 — Qwen3.8 125B MoE エンジンを読む</title><meta name="description" content="Strata（Qwen3.8-Flash-Next 125B MoE）の CUDA 実装 57 ファイル / 34,312 行の読解。9 章 + 図 7 枚。">' +
  '<link rel="icon" href="favicon.svg" type="image/svg+xml"><style>' + css + '</style></head><body><div class="wrap">' +
  '<div class="masthead"><span>STRATA CUDA ソース読解</span><span>v0.1.49 · HEAD 61b3fb5</span></div>' +
  '<h1>STRATA CUDA ソース読解<small>1 枚の GPU で 125B MoE を回すエンジンを、行番号から読む</small></h1>' +
  '<p class="abstract">' + sub + '</p>' +
  '<table class="meta"><tbody>' +
  '<tr><th>対象</th><td>Qwen3.8-Flash-Next 125B（48 層 × 512 expert = 24,576、top-10/token）を動かす Strata の CUDA/HIP 実装</td></tr>' +
  '<tr><th>規模</th><td>57 ファイル / 34,312 行（cuda 45 · prefill 9 · core 3）</td></tr>' +
  '<tr><th>構成</th><td>全 9 章 + 図 7 枚。各章に「なぜそう書くのか」の解説とメモリ移動の図</td></tr>' +
  '<tr><th>リビジョン</th><td>HEAD <code>61b3fb5</code>（v0.1.49）</td></tr>' +
  '<tr><th>読み方</th><td>まず 00（総覧）と 08（目録）→ 図で流れを掴む → 該当章 → ソース</td></tr>' +
  '</tbody></table>' +
  '<h2>目次 <span>00–08</span></h2><ul class="toc">' + toc + '</ul>' +
  '<h2>図一覧 <span>7 figures</span></h2><div class="gallery">' + gal + '</div>' +
  '<footer><p><b>出典</b>: github.com/Niko1221/Strata HEAD <code>61b3fb5</code>（v0.1.49）。行番号・定数はこのリビジョンのもの。</p>' +
  '<p><b>注意</b>: 公式ドキュメントではない。推測で埋めた箇所は本文に「推測」と明記した。図は各章の PNG を同梱している。</p></footer>' +
  '</div></body></html>';
fs.writeFileSync(path.join(outDir, 'index.html'), index);
fs.writeFileSync(path.join(outDir, 'favicon.svg'), '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="10" fill="#fffdf8"/><rect x="1" y="1" width="62" height="62" rx="9.5" fill="none" stroke="#d6d0c0"/><text x="32" y="44" text-anchor="middle" font-family="Georgia,serif" font-size="34" font-weight="700" fill="#1f4e79">S</text><rect x="14" y="50" width="36" height="2" fill="#9e2b25"/></svg>');
fs.writeFileSync(path.join(outDir, '.nojekyll'), '');
console.log('built', chapters.length, 'chapters ->', path.relative(root, outDir), '(cards', cards.length, 'figs', figs.length + ')');
