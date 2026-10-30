import fs from 'node:fs';
import path from 'node:path';
import { marked } from '/Users/ts/DSH/node_modules/.pnpm/marked@18.0.11/node_modules/marked/lib/marked.esm.js';

const root = process.cwd();
const docs = fs.readdirSync(path.join(root,'docs')).filter(f=>/^0\d_.*\.md$/.test(f)).sort();

const FIG = /^>\s*図:\s*(.+)$/;
function figBlock(line){
  const parts = [...line.matchAll(/\[([^\]]+)\]\(([^)]+\.svg)\)/g)];
  if(!parts.length) return null;
  const imgs = parts.map(m=>{
    const png = m[2].replace(/^(\.\.\/)?diagrams\//,'../diagrams/png/').replace(/\.svg$/,'.png');
    return '<a class="figlink" href="'+png+'"><img src="'+png+'" alt="'+m[1]+'"><span class="figcap">'+m[1]+'</span></a>';
  }).join('');
  return '<div class="figrow">'+imgs+'</div>';
}

const chapters = [];
for(const f of docs){
  let md = fs.readFileSync(path.join(root,'docs',f),'utf8');
  const lines = md.split('\n');
  const out = [];
  let inFence = false;
  for(const ln of lines){
    if(/^```/.test(ln)) inFence = !inFence;
    if(!inFence && FIG.test(ln)){ const b = figBlock(ln); if(b){ out.push(b); continue; } }
    out.push(ln);
  }
  md = out.join('\n').replace(/\.md\)/g,'.html)');
  const body = marked.parse(md, {gfm:true, breaks:false});
  const titleMatch = body.match(/<h1[^>]*>([\s\S]*?)<\/h1>/);
  const title = titleMatch ? titleMatch[1].replace(/<[^>]+>/g,'') : f;
  chapters.push({file:f, htmlFile:f.replace(/\.md$/,'.html'), title, body});
}

const css = [
':root{--cyan:#00f0ff;--mag:#ff2a6d;--acid:#d6ff2e;--ink:#07090d;--panel:#0b1119;--line:#1d2a36;--txt:#c9d4e0;--dim:#95a9bd;--lbl:#ffe066}',
'*{box-sizing:border-box}',
'html,body{margin:0;background:var(--ink);color:var(--txt);font-family:"JetBrains Mono","SF Mono",ui-monospace,Menlo,monospace;font-size:14px;line-height:1.85}',
'body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:9;background:repeating-linear-gradient(180deg,rgba(0,240,255,.04) 0 1px,transparent 1px 4px)}',
'.wrap{max-width:880px;margin:0 auto;padding:28px 24px 80px;position:relative;z-index:10}',
'.nav{display:flex;justify-content:space-between;gap:12px;font-size:13px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:26px}',
'.nav a{color:var(--dim);text-decoration:none;border-bottom:1px solid transparent}',
'.nav a:hover{color:var(--cyan);border-color:var(--cyan)}',
'h1{font-size:23px;line-height:1.5;color:var(--cyan);letter-spacing:1px;margin:6px 0 18px;text-shadow:0 0 16px rgba(0,240,255,.35)}',
'h2{font-size:17px;color:var(--lbl);margin:34px 0 10px;padding-left:10px;border-left:3px solid var(--mag);line-height:1.6}',
'h3{font-size:15px;color:var(--acid);margin:24px 0 8px}',
'p{margin:12px 0}',
'a{color:var(--cyan)}',
'strong{color:var(--lbl)}',
'code{background:#12202b;color:var(--acid);padding:1px 5px;border:1px solid var(--line);font-size:13px;word-break:break-word}',
'pre{background:#05080c;border:1px solid var(--line);padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.65}',
'pre code{background:none;border:0;padding:0;color:#d7e6f2;font-size:13px}',
'blockquote{margin:16px 0;padding:10px 16px;border-left:3px solid var(--cyan);background:rgba(0,240,255,.05);color:var(--txt)}',
'blockquote p{margin:4px 0}',
'table{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px;display:block;overflow-x:auto}',
'th,td{border:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}',
'th{background:#101c26;color:var(--lbl);white-space:nowrap}',
'tr:nth-child(even) td{background:#0a1219}',
'ul,ol{padding-left:22px}',
'li{margin:5px 0}',
'hr{border:0;height:1px;background:linear-gradient(90deg,var(--cyan),var(--mag) 60%,transparent);margin:30px 0}',
'.figrow{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}',
'.figlink{display:block;flex:1 1 320px;text-decoration:none;background:#05080c;border:1px solid var(--line);padding:8px}',
'.figlink img{width:100%;display:block}',
'.figcap{display:block;font-size:13px;color:var(--lbl);font-weight:700;margin-top:6px}',
'footer{margin-top:44px;padding-top:14px;border-top:1px solid var(--line);font-size:13px;color:var(--dim)}',
'footer a{color:var(--dim)}'
].join('\n');

function shell(ch, i){
  const prev = i>0 ? chapters[i-1] : null;
  const next = i<chapters.length-1 ? chapters[i+1] : null;
  const nav = '<div class="nav"><a href="../index.html">← 目次 / 図一覧</a><span>STRATA 読解 ' + ch.htmlFile.slice(0,2) + ' / 08</span></div>';
  const pager = '<footer>' + (prev ? '<a href="'+prev.htmlFile+'">← '+prev.title+'</a>' : '<span></span>') + ' ・ ' + (next ? '<a href="'+next.htmlFile+'">'+next.title+' →</a>' : '<a href="../index.html">目次へ</a>') + '<br>出典: strata:~/Strata HEAD 61b3fb5 (v0.1.49) ／ 図は PNG（1000px 幅）</footer>';
  return '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+ch.title+'</title><style>'+css+'</style></head><body><div class="wrap">'+nav+'<article>'+ch.body+'</article>'+pager+'</div></body></html>';
}

chapters.forEach((ch,i)=>{
  fs.writeFileSync(path.join(root,'docs',ch.htmlFile), shell(ch,i));
  console.log(ch.htmlFile, ch.title);
});